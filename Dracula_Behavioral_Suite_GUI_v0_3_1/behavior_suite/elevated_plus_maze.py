import pandas as pd

def analyze_elevated_plus_maze(
    data,
    fps,
    open_arm_names=("open_1", "open_2"),
    closed_arm_names=("closed_1", "closed_2"),
    center_name="center",
):
    required = [*open_arm_names, *closed_arm_names, center_name]
    missing = [
        name for name in required
        if f"roi_{name}" not in data.columns
    ]

    if missing:
        raise KeyError(f"Missing EPM ROIs: {missing}")

    result = {}

    for name in required:
        series = data[f"roi_{name}"]
        previous = series.shift(1, fill_value=False)

        result[f"{name}_time_s"] = float(series.sum() / fps)
        result[f"{name}_entries"] = int(
            ((series == True) & (previous == False)).sum()
        )

    open_time = sum(
        result[f"{name}_time_s"] for name in open_arm_names
    )
    closed_time = sum(
        result[f"{name}_time_s"] for name in closed_arm_names
    )
    open_entries = sum(
        result[f"{name}_entries"] for name in open_arm_names
    )
    closed_entries = sum(
        result[f"{name}_entries"] for name in closed_arm_names
    )

    result["open_arm_time_s"] = open_time
    result["closed_arm_time_s"] = closed_time
    result["open_arm_entries"] = open_entries
    result["closed_arm_entries"] = closed_entries
    result["open_time_percent"] = (
        100 * open_time / (open_time + closed_time)
        if open_time + closed_time > 0 else 0
    )
    result["open_entry_percent"] = (
        100 * open_entries / (open_entries + closed_entries)
        if open_entries + closed_entries > 0 else 0
    )

    return pd.DataFrame([result])
