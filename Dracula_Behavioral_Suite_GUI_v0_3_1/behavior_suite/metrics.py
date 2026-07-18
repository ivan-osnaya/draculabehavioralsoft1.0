import pandas as pd

def summarize_session(data, fps, immobility_speed_cm_s=1.0):
    if data.empty:
        return pd.DataFrame()

    summary = {
        "duration_s": float(data["time_s"].iloc[-1] - data["time_s"].iloc[0]),
        "frames": int(len(data)),
        "detection_percent": float(100 * data["detected"].mean()),
        "distance_total_px": float(data["distance_total_px"].max()),
    }

    if data["distance_total_cm"].notna().any():
        summary["distance_total_cm"] = float(data["distance_total_cm"].max())
        summary["mean_speed_cm_s"] = float(data["speed_cm_s"].dropna().mean())
        summary["immobility_time_s"] = float(
            (
                data["speed_cm_s"].fillna(float("inf"))
                <= immobility_speed_cm_s
            ).sum() / fps
        )

    for column in [
        column for column in data.columns if column.startswith("roi_")
    ]:
        previous = data[column].shift(1, fill_value=False)
        summary[f"time_{column}_s"] = float(data[column].sum() / fps)
        summary[f"entries_{column}"] = int(
            ((data[column] == True) & (previous == False)).sum()
        )

    return pd.DataFrame([summary])
