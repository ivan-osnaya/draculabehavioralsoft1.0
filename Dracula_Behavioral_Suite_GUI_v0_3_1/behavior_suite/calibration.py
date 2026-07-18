import cv2
import math
from .video import load_video_frame

def draw_scale_line(video_path, frame_number=0, known_cm=None):
    frame = load_video_frame(video_path, frame_number)
    points = []
    window_name = "Spatial Calibration"

    def callback(event, x, y, flags, parameter):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, callback)

    while True:
        canvas = frame.copy()

        for point in points:
            cv2.circle(canvas, point, 6, (0, 255, 0), -1)

        if len(points) == 2:
            cv2.line(canvas, points[0], points[1], (0, 255, 255), 3)
            distance_px = math.hypot(
                points[1][0] - points[0][0],
                points[1][1] - points[0][1],
            )
            cv2.putText(
                canvas,
                f"{distance_px:.1f} px",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        instructions = [
            "Left click: select two endpoints",
            "Right click or U: undo",
            "R: reset",
            "Enter / Space: accept",
            "Esc / Q: cancel",
        ]

        for index, text in enumerate(instructions):
            cv2.putText(
                canvas,
                text,
                (20, 25 + index * 25),
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
        elif key in (13, 32) and len(points) == 2:
            break
        elif key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError("Calibration canceled")

    cv2.destroyWindow(window_name)

    distance_px = math.hypot(
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
    )

    if known_cm is None:
        while True:
            try:
                known_cm = float(input("Real length of selected line (cm): "))
                if known_cm <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Enter a positive numeric value.")

    return distance_px / float(known_cm), float(known_cm) / distance_px
