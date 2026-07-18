import json
import cv2
import numpy as np
from .video import load_video_frame

def draw_single_roi(image, roi_name, existing_rois=None):
    existing_rois = existing_rois or {}
    points = []
    window_name = f"Draw ROI: {roi_name}"

    def callback(event, x, y, flags, parameter):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, callback)

    while True:
        canvas = image.copy()

        for name, polygon in existing_rois.items():
            contour = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [contour], True, (0, 255, 255), 2)
            cv2.putText(
                canvas,
                name,
                polygon[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        for point in points:
            cv2.circle(canvas, point, 5, (0, 0, 255), -1)

        if len(points) >= 2:
            contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [contour], False, (255, 0, 255), 2)

        if len(points) >= 3:
            cv2.line(canvas, points[-1], points[0], (255, 0, 255), 1)

        instructions = [
            f"ROI: {roi_name}",
            "Left click: add vertex",
            "Right click or U: undo",
            "R: reset",
            "Enter / Space: save",
            "Esc / Q: cancel",
        ]

        for index, text in enumerate(instructions):
            cv2.putText(
                canvas,
                text,
                (15, 25 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("u"), ord("U")) and points:
            points.pop()
        elif key in (ord("r"), ord("R")):
            points = []
        elif key in (13, 32) and len(points) >= 3:
            cv2.destroyWindow(window_name)
            return points.copy()
        elif key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            return None

def define_rois_interactively(video_path, roi_names, frame_number=0):
    frame = load_video_frame(video_path, frame_number)
    rois = {}

    for name in roi_names:
        polygon = draw_single_roi(frame, name, rois)
        if polygon is not None:
            rois[name] = polygon

    cv2.destroyAllWindows()
    return rois

def save_rois(rois, output_path):
    serializable = {
        name: [[int(x), int(y)] for x, y in polygon]
        for name, polygon in rois.items()
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)

def load_rois(input_path):
    with open(input_path, "r", encoding="utf-8") as file:
        loaded = json.load(file)

    return {
        name: [(int(x), int(y)) for x, y in polygon]
        for name, polygon in loaded.items()
    }
