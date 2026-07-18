import pandas as pd

def analyze_open_field(data, fps, center_roi_name="center"):
    column = f"roi_{center_roi_name}"

    if column not in data.columns:
        raise KeyError(f"Open Field analysis requires ROI '{center_roi_name}'.")

    center = data[column]
    previous = center.shift(1, fill_value=False)
    center_time_s = float(center.sum() / fps)
    total_time_s = float(len(data) / fps)

    return pd.DataFrame([{
        "center_time_s": center_time_s,
        "periphery_time_s": total_time_s - center_time_s,
        "center_percent": 100 * center_time_s / total_time_s,
        "center_entries": int(
            ((center == True) & (previous == False)).sum()
        ),
    }])
