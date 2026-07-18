import cv2

def get_video_information(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps,
    }

def load_video_frame(video_path, frame_number=0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number}")

    return frame
