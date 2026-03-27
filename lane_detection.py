import cv2
import numpy as np
import os
import time

def average_lines(frame, lines):
    left, right = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if slope < -0.3:
            left.append(line[0])
        elif slope > 0.3:
            right.append(line[0])

    def fit_line(points, frame):
        h = frame.shape[0]
        x = np.array([[p[0], p[2]] for p in points]).flatten()
        y = np.array([[p[1], p[3]] for p in points]).flatten()
        m, b = np.polyfit(x, y, 1)
        y_bottom = h
        y_top = h // 2
        x_bottom = int((y_bottom - b) / m)
        x_top = int((y_top - b) / m)
        return (x_bottom, y_bottom, x_top, y_top)

    left_line = fit_line(left, frame) if left else None
    right_line = fit_line(right, frame) if right else None
    return left_line, right_line


def draw_lane_fill(frame, left_line, right_line):
    if left_line is None or right_line is None:
        return frame
    overlay = frame.copy()
    pts = np.array([[
        (left_line[0], left_line[1]),
        (left_line[2], left_line[3]),
        (right_line[2], right_line[3]),
        (right_line[0], right_line[1])
    ]], dtype=np.int32)
    cv2.fillPoly(overlay, pts, (0, 255, 0))
    return cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)


def draw_lines_on_frame(frame, left_line, right_line):
    line_image = np.zeros_like(frame)
    if left_line:
        cv2.line(line_image, (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]), (0, 255, 0), 10)
    if right_line:
        cv2.line(line_image, (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]), (0, 255, 0), 10)
    return line_image


def get_departure_warning(frame, left_line, right_line):
    if left_line is None or right_line is None:
        return "NO LANE DETECTED", (0, 0, 255)
    w = frame.shape[1]
    frame_center = w // 2
    lane_center = (left_line[0] + right_line[0]) // 2
    offset = lane_center - frame_center
    if abs(offset) < 50:
        return "Lane Centered", (0, 255, 0)
    elif offset > 0:
        return "WARNING: Drifting Right!", (0, 0, 255)
    else:
        return "WARNING: Drifting Left!", (0, 0, 255)


def draw_harris_corners(frame, gray):
    # Harris needs float32 input
    gray_float = np.float32(gray)
    # blockSize=2: neighbourhood size, ksize=3: Sobel aperture, k=0.04: Harris detector free parameter
    corners = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    # dilate to make corner points more visible
    corners = cv2.dilate(corners, None)
    # threshold: only keep strong corners (top 1% response)
    frame[corners > 0.01 * corners.max()] = [0, 0, 255]
    return frame


def draw_optical_flow(frame, prev_gray, curr_gray, prev_points):
    if prev_points is None or len(prev_points) == 0:
        return frame, prev_points

    # Lucas-Kanade optical flow — tracks prev_points into the new frame
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None
    )

    # status=1 means the point was successfully tracked
    good_prev = prev_points[status == 1]
    good_curr = curr_points[status == 1]

    # draw arrows from old position to new position
    for p0, p1 in zip(good_prev, good_curr):
        x0, y0 = p0.ravel().astype(int)
        x1, y1 = p1.ravel().astype(int)
        cv2.arrowedLine(frame, (x0, y0), (x1, y1), (255, 100, 0), 1, tipLength=0.4)

    return frame, good_curr.reshape(-1, 1, 2) if len(good_curr) > 0 else None


def draw_hud(frame, warning, warning_color, confidence, fps=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Confidence: {confidence}%", (w // 2 - 100, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(frame, warning, (w - text_size[0] - 10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)

    return frame


def process_frame(frame, fps=None, prev_gray=None, prev_points=None):
    # HLS masking for white and yellow lanes
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, (0, 200, 0), (255, 255, 255))
    yellow_mask = cv2.inRange(hls, (15, 100, 100), (35, 255, 255))
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape
    triangle = np.array([[(0, h), (w, h), (w // 2, h // 2)]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, triangle, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 2, np.pi/180, 100,
                             minLineLength=40, maxLineGap=5)

    left_line, right_line = average_lines(frame, lines) if lines is not None else (None, None)

    line_count = len(lines) if lines is not None else 0
    confidence = min(int((line_count / 20) * 100), 100)

    # build result frame
    result = draw_lane_fill(frame, left_line, right_line)
    line_image = draw_lines_on_frame(result, left_line, right_line)
    result = cv2.addWeighted(result, 0.8, line_image, 1, 0)

    # Harris corners on the full grayscale (not masked) — more corners visible
    full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = draw_harris_corners(result, full_gray)

    # Optical flow (only for video — needs previous frame)
    curr_points = None
    if prev_gray is not None and prev_points is not None:
        result, curr_points = draw_optical_flow(result, prev_gray, full_gray, prev_points)

    warning, warning_color = get_departure_warning(frame, left_line, right_line)
    result = draw_hud(result, warning, warning_color, confidence, fps)

    return result, full_gray, curr_points


def get_initial_points(gray):
    # Shi-Tomasi corner points — good starting points to track with optical flow
    points = cv2.goodFeaturesToTrack(gray, maxCorners=100,
                                     qualityLevel=0.3, minDistance=7)
    return points


# --- setup output folders ---
os.makedirs("output_images", exist_ok=True)
os.makedirs("output_videos", exist_ok=True)

# --- process all images ---
for filename in os.listdir("test_images"):
    if filename.endswith((".jpg", ".png")):
        img = cv2.imread(f"test_images/{filename}")
        # images have no previous frame so optical flow is skipped
        output, _, _ = process_frame(img)
        cv2.imwrite(f"output_images/{filename}", output)
        print(f"Saved output_images/{filename}")

# --- process all videos ---
for filename in os.listdir("test_videos"):
    if filename.endswith((".mp4", ".avi")):
        cap = cv2.VideoCapture(f"test_videos/{filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            f"output_videos/{filename}",
            fourcc, 25,
            (int(cap.get(3)), int(cap.get(4)))
        )

        prev_gray = None
        prev_points = None
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            result, curr_gray, curr_points = process_frame(
                frame, fps, prev_gray, prev_points
            )

            # refresh tracking points every 30 frames or when lost
            if prev_points is None or curr_points is None or len(curr_points) < 10:
                prev_points = get_initial_points(curr_gray)
            else:
                prev_points = curr_points

            prev_gray = curr_gray
            out.write(result)

        cap.release()
        out.release()
        print(f"Saved output_videos/{filename}")

print("All done!")