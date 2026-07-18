import cv2

def detect_available_cameras(max_index=10):
    available = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(index)
        cap.release()

    return available

def preview_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(
            frame,
            "Press Q or Esc to close",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(f"Camera Preview {camera_index}", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

def record_camera(camera_index, output_path, duration_s=None):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    start_tick = cv2.getTickCount()
    tick_frequency = cv2.getTickFrequency()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        elapsed = (cv2.getTickCount() - start_tick) / tick_frequency

        cv2.putText(
            frame,
            f"REC {elapsed:.1f}s | Q/Esc to stop",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        writer.write(frame)
        cv2.imshow("Live Recording", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break

        if duration_s is not None and elapsed >= duration_s:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
