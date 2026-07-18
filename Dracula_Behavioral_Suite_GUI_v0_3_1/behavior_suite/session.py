from .video import get_video_information

def seconds_to_frames(video_path, start_s=0.0, end_s=None):
    info = get_video_information(video_path)
    fps = info["fps"]

    start_frame = max(0, int(round(float(start_s) * fps)))

    if end_s is None:
        end_frame = None
    else:
        end_frame = min(
            int(round(float(end_s) * fps)),
            info["frame_count"] - 1,
        )

    if end_frame is not None and end_frame <= start_frame:
        raise ValueError("End time must be greater than start time.")

    return start_frame, end_frame
