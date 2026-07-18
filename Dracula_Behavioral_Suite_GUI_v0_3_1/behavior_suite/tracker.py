from pathlib import Path
import cv2
import math
import numpy as np
import pandas as pd
from .video import get_video_information

def _odd(value):
    value = max(1, int(value))
    return value if value % 2 else value + 1

def _preprocess(frame, blur_kernel):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = _odd(blur_kernel)
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)

def _get_bbox(rois, width, height, padding):
    points = [
        point
        for polygon in rois.values()
        for point in polygon
    ]

    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x0 = max(0, min(xs) - padding)
    y0 = max(0, min(ys) - padding)
    x1 = min(width - 1, max(xs) + padding)
    y1 = min(height - 1, max(ys) + padding)

    return (
        int(x0),
        int(y0),
        int(x1 - x0 + 1),
        int(y1 - y0 + 1),
    )

def _transform_rois(rois, crop, factor):
    x0 = crop[0] if crop else 0
    y0 = crop[1] if crop else 0

    return {
        name: [
            (
                int((x - x0) * factor),
                int((y - y0) * factor),
            )
            for x, y in polygon
        ]
        for name, polygon in rois.items()
    }

def _make_mask(shape, rois):
    mask = np.zeros(shape, dtype=np.uint8)

    for polygon in rois.values():
        contour = np.array(
            polygon,
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.fillPoly(mask, [contour], 255)

    return mask

def _inside(x, y, polygon):
    contour = np.array(
        polygon,
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    return cv2.pointPolygonTest(
        contour,
        (float(x), float(y)),
        False,
    ) >= 0

def _build_reference(
    video_path,
    config,
    start_frame,
    end_frame,
    crop,
):
    info = get_video_information(video_path)

    last = (
        info["frame_count"] - 1
        if end_frame is None
        else min(end_frame, info["frame_count"] - 1)
    )

    indices = np.linspace(
        start_frame,
        last,
        config.reference_frames,
        dtype=int,
    )

    cap = cv2.VideoCapture(str(video_path))
    frames = []

    for frame_index in indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        ok, frame = cap.read()

        if not ok:
            continue

        if crop:
            x, y, width, height = crop
            frame = frame[
                y:y + height,
                x:x + width,
            ]

        if config.resize_factor != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=config.resize_factor,
                fy=config.resize_factor,
                interpolation=cv2.INTER_AREA,
            )

        frames.append(
            _preprocess(
                frame,
                config.blur_kernel,
            )
        )

    cap.release()

    if len(frames) < 2:
        raise RuntimeError(
            "Not enough frames to create reference background."
        )

    return np.median(
        np.stack(frames),
        axis=0,
    ).astype(np.uint8)

def _detect(
    frame,
    reference,
    config,
    analysis_mask,
    previous,
):
    gray = _preprocess(
        frame,
        config.blur_kernel,
    )

    if config.method == "dark":
        difference = cv2.subtract(
            reference,
            gray,
        )
    elif config.method == "light":
        difference = cv2.subtract(
            gray,
            reference,
        )
    else:
        difference = cv2.absdiff(
            gray,
            reference,
        )

    _, binary = cv2.threshold(
        difference,
        config.threshold,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = np.ones(
        (
            _odd(config.morphology_kernel),
            _odd(config.morphology_kernel),
        ),
        dtype=np.uint8,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    if analysis_mask is not None:
        binary = cv2.bitwise_and(
            binary,
            analysis_mask,
        )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if (
            config.min_area_px
            <= area
            <= config.max_area_px
        ):
            moments = cv2.moments(contour)

            if moments["m00"] > 0:
                x = moments["m10"] / moments["m00"]
                y = moments["m01"] / moments["m00"]
                candidates.append(
                    (contour, area, x, y)
                )

    if not candidates:
        return None, frame.copy()

    if previous is None:
        selected = max(
            candidates,
            key=lambda item: item[1],
        )
    else:
        selected = min(
            candidates,
            key=lambda item:
                (item[2] - previous[0]) ** 2
                + (item[3] - previous[1]) ** 2,
        )

        jump = math.hypot(
            selected[2] - previous[0],
            selected[3] - previous[1],
        )

        if jump > config.max_jump_px:
            return None, frame.copy()

    contour, _, x, y = selected
    annotated = frame.copy()

    cv2.drawContours(
        annotated,
        [contour],
        -1,
        (0, 255, 0),
        2,
    )

    cv2.circle(
        annotated,
        (int(x), int(y)),
        5,
        (0, 0, 255),
        -1,
    )

    return (x, y), annotated

def track_video(
    video_path,
    rois,
    output_csv,
    output_video,
    scale_cm_per_px,
    config,
    start_frame=0,
    end_frame=None,
):
    info = get_video_information(video_path)

    crop = (
        _get_bbox(
            rois,
            info["width"],
            info["height"],
            config.roi_crop_padding_px,
        )
        if config.crop_to_roi_bounds and rois
        else None
    )

    local_rois = _transform_rois(
        rois,
        crop,
        config.resize_factor,
    )

    reference = _build_reference(
        video_path,
        config,
        start_frame,
        end_frame,
        crop,
    )

    analysis_mask = (
        _make_mask(
            reference.shape[:2],
            local_rois,
        )
        if config.analyze_only_inside_rois
        and local_rois
        else None
    )

    cap = cv2.VideoCapture(str(video_path))
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    stop_frame = (
        info["frame_count"]
        if end_frame is None
        else min(
            end_frame + 1,
            info["frame_count"],
        )
    )

    writer = None
    previous = None
    previous_valid = None
    cumulative_resized_px = 0.0
    trajectory = []
    rows = []

    x_offset = crop[0] if crop else 0
    y_offset = crop[1] if crop else 0

    while True:
        frame_index = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        )

        if frame_index >= stop_frame:
            break

        ok, frame = cap.read()

        if not ok:
            break

        if crop:
            x, y, width, height = crop
            frame = frame[
                y:y + height,
                x:x + width,
            ]

        if config.resize_factor != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=config.resize_factor,
                fy=config.resize_factor,
                interpolation=cv2.INTER_AREA,
            )

        position, annotated = _detect(
            frame,
            reference,
            config,
            analysis_mask,
            previous,
        )

        if output_video and writer is None:
            height, width = annotated.shape[:2]

            writer = cv2.VideoWriter(
                str(output_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                info["fps"],
                (width, height),
            )

        detected = position is not None
        time_s = (
            frame_index - start_frame
        ) / info["fps"]

        if detected:
            resized_x, resized_y = position
            previous = position

            trajectory.append(
                (
                    int(resized_x),
                    int(resized_y),
                )
            )

            if previous_valid is None:
                step_resized_px = 0.0
            else:
                step_resized_px = math.hypot(
                    resized_x - previous_valid[0],
                    resized_y - previous_valid[1],
                )

            previous_valid = position
            cumulative_resized_px += step_resized_px

            x_px = (
                resized_x / config.resize_factor
                + x_offset
            )

            y_px = (
                resized_y / config.resize_factor
                + y_offset
            )

            step_px = (
                step_resized_px
                / config.resize_factor
            )

        else:
            resized_x = resized_y = np.nan
            x_px = y_px = np.nan
            step_px = np.nan

        total_px = (
            cumulative_resized_px
            / config.resize_factor
        )

        if (
            scale_cm_per_px is not None
            and detected
        ):
            step_cm = (
                step_px
                * scale_cm_per_px
            )

            total_cm = (
                total_px
                * scale_cm_per_px
            )

            speed_cm_s = (
                step_cm
                * info["fps"]
            )

        else:
            step_cm = total_cm = speed_cm_s = np.nan

        row = {
            "frame": frame_index,
            "time_s": time_s,
            "detected": detected,
            "x_px": x_px,
            "y_px": y_px,
            "distance_frame_px": step_px,
            "distance_total_px": total_px,
            "distance_frame_cm": step_cm,
            "distance_total_cm": total_cm,
            "speed_cm_s": speed_cm_s,
        }

        for name, polygon in local_rois.items():
            inside = (
                _inside(
                    resized_x,
                    resized_y,
                    polygon,
                )
                if detected
                else False
            )

            row[f"roi_{name}"] = inside

            contour = np.array(
                polygon,
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            cv2.polylines(
                annotated,
                [contour],
                True,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                name,
                polygon[0],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

        rows.append(row)

        if (
            config.draw_trajectory
            and len(trajectory) > 1
        ):
            first_index = max(
                1,
                len(trajectory)
                - config.trajectory_tail_frames,
            )

            for index in range(
                first_index,
                len(trajectory),
            ):
                cv2.line(
                    annotated,
                    trajectory[index - 1],
                    trajectory[index],
                    (255, 0, 255),
                    2,
                )

        if writer is not None:
            writer.write(annotated)

        if len(rows) % 500 == 0:
            print(
                f"Processed {len(rows)} frames"
            )

    cap.release()

    if writer is not None:
        writer.release()

    data = pd.DataFrame(rows)

    Path(output_csv).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_csv,
        index=False,
    )

    return data
