from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLOBAL_METRICS = {
    "test_duration_s": "Test duration",
    "detection_percent": "Detection percentage",
    "total_distance_cm": "Total distance traveled",
    "average_speed_cm_s": "Average speed",
    "maximum_speed_cm_s": "Maximum speed",
    "time_moving_s": "Time moving",
    "time_immobile_s": "Time immobile",
}

ROI_METRICS = {
    "roi_time_s": "Time in ROI",
    "roi_time_percent": "Percentage of time in ROI",
    "roi_entries": "Entries into ROI",
    "roi_latency_s": "Latency to first ROI entry",
    "roi_distance_cm": "Distance traveled in ROI",
    "roi_average_speed_cm_s": "Average speed in ROI",
}


def default_metric_selection() -> List[str]:
    return [
        "test_duration_s",
        "detection_percent",
        "total_distance_cm",
        "average_speed_cm_s",
        "time_immobile_s",
        "roi_time_s",
        "roi_time_percent",
        "roi_entries",
        "roi_latency_s",
    ]


def equal_stages(
    total_duration_s: float,
    number_of_stages: int,
    prefix: str = "Stage",
) -> List[Dict]:
    if number_of_stages < 1:
        raise ValueError("Number of stages must be at least 1.")

    boundaries = np.linspace(
        0.0,
        float(total_duration_s),
        number_of_stages + 1,
    )

    return [
        {
            "name": f"{prefix} {index + 1}",
            "start_s": float(boundaries[index]),
            "end_s": float(boundaries[index + 1]),
        }
        for index in range(number_of_stages)
    ]


def validate_stages(stages: Iterable[Dict]) -> List[Dict]:
    cleaned = []

    for index, stage in enumerate(stages):
        name = str(stage.get("name", "")).strip() or f"Stage {index + 1}"
        start_s = float(stage["start_s"])
        end_s = float(stage["end_s"])

        if start_s < 0:
            raise ValueError(f"{name}: start time cannot be negative.")

        if end_s <= start_s:
            raise ValueError(f"{name}: end time must be greater than start time.")

        cleaned.append({
            "name": name,
            "start_s": start_s,
            "end_s": end_s,
        })

    return cleaned


def _slice_stage(
    data: pd.DataFrame,
    stage: Dict,
    include_end: bool = False,
) -> pd.DataFrame:
    if include_end:
        mask = (
            (data["time_s"] >= stage["start_s"])
            & (data["time_s"] <= stage["end_s"])
        )
    else:
        mask = (
            (data["time_s"] >= stage["start_s"])
            & (data["time_s"] < stage["end_s"])
        )

    return data.loc[mask].copy()


def _entries(series: pd.Series) -> int:
    previous = series.shift(1, fill_value=False)
    return int(((series == True) & (previous == False)).sum())


def _latency(series: pd.Series, times: pd.Series) -> float:
    indices = series[series == True].index

    if len(indices) == 0:
        return np.nan

    first_index = indices[0]
    return float(times.loc[first_index] - times.iloc[0])


def _distance_for_subset(
    subset: pd.DataFrame,
    distance_column: str,
) -> float:
    if distance_column not in subset.columns:
        return np.nan

    values = subset[distance_column].dropna()

    if values.empty:
        return np.nan

    return float(values.sum())


def calculate_results(
    data: pd.DataFrame,
    fps: float,
    selected_metrics: List[str],
    stages: Optional[List[Dict]] = None,
    immobility_speed_cm_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected = set(selected_metrics)

    if stages:
        stage_definitions = validate_stages(stages)
    else:
        stage_definitions = [{
            "name": "Entire session",
            "start_s": float(data["time_s"].min()),
            "end_s": float(data["time_s"].max() + (1.0 / fps)),
        }]

    summary_rows = []
    roi_rows = []

    roi_columns = [
        column
        for column in data.columns
        if column.startswith("roi_")
    ]

    for stage_index, stage in enumerate(stage_definitions):
        stage_data = _slice_stage(
            data,
            stage,
            include_end=stage_index == len(stage_definitions) - 1,
        )

        if stage_data.empty:
            continue

        stage_duration_s = float(len(stage_data) / fps)
        valid = stage_data[stage_data["detected"] == True]
        speed = stage_data["speed_cm_s"].dropna()

        row = {
            "stage": stage["name"],
            "stage_start_s": stage["start_s"],
            "stage_end_s": stage["end_s"],
        }

        if "test_duration_s" in selected:
            row["test_duration_s"] = stage_duration_s

        if "detection_percent" in selected:
            row["detection_percent"] = float(
                100.0 * stage_data["detected"].mean()
            )

        if "total_distance_cm" in selected:
            row["total_distance_cm"] = _distance_for_subset(
                stage_data,
                "distance_frame_cm",
            )

        if "average_speed_cm_s" in selected:
            row["average_speed_cm_s"] = (
                float(speed.mean()) if not speed.empty else np.nan
            )

        if "maximum_speed_cm_s" in selected:
            row["maximum_speed_cm_s"] = (
                float(speed.max()) if not speed.empty else np.nan
            )

        if "time_moving_s" in selected:
            row["time_moving_s"] = float(
                (stage_data["speed_cm_s"] > immobility_speed_cm_s).sum()
                / fps
            )

        if "time_immobile_s" in selected:
            row["time_immobile_s"] = float(
                (
                    stage_data["speed_cm_s"].notna()
                    & (
                        stage_data["speed_cm_s"]
                        <= immobility_speed_cm_s
                    )
                ).sum()
                / fps
            )

        summary_rows.append(row)

        for roi_column in roi_columns:
            roi_name = roi_column.replace("roi_", "", 1)
            roi_state = stage_data[roi_column].astype(bool)
            roi_data = stage_data.loc[roi_state]

            roi_row = {
                "stage": stage["name"],
                "roi": roi_name,
                "stage_start_s": stage["start_s"],
                "stage_end_s": stage["end_s"],
            }

            if "roi_time_s" in selected:
                roi_row["time_s"] = float(roi_state.sum() / fps)

            if "roi_time_percent" in selected:
                roi_row["time_percent"] = float(
                    100.0 * roi_state.mean()
                )

            if "roi_entries" in selected:
                roi_row["entries"] = _entries(roi_state)

            if "roi_latency_s" in selected:
                roi_row["latency_s"] = _latency(
                    roi_state,
                    stage_data["time_s"],
                )

            if "roi_distance_cm" in selected:
                roi_row["distance_cm"] = _distance_for_subset(
                    roi_data,
                    "distance_frame_cm",
                )

            if "roi_average_speed_cm_s" in selected:
                roi_speed = roi_data["speed_cm_s"].dropna()
                roi_row["average_speed_cm_s"] = (
                    float(roi_speed.mean())
                    if not roi_speed.empty
                    else np.nan
                )

            roi_rows.append(roi_row)

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(roi_rows),
    )


def create_long_results(
    stage_results: pd.DataFrame,
    roi_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if not stage_results.empty:
        id_columns = [
            column
            for column in (
                "stage",
                "stage_start_s",
                "stage_end_s",
            )
            if column in stage_results.columns
        ]

        value_columns = [
            column
            for column in stage_results.columns
            if column not in id_columns
        ]

        melted = stage_results.melt(
            id_vars=id_columns,
            value_vars=value_columns,
            var_name="measure",
            value_name="value",
        )

        melted["roi"] = ""
        rows.append(melted)

    if not roi_results.empty:
        id_columns = [
            column
            for column in (
                "stage",
                "roi",
                "stage_start_s",
                "stage_end_s",
            )
            if column in roi_results.columns
        ]

        value_columns = [
            column
            for column in roi_results.columns
            if column not in id_columns
        ]

        melted = roi_results.melt(
            id_vars=id_columns,
            value_vars=value_columns,
            var_name="measure",
            value_name="value",
        )

        rows.append(melted)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True, sort=False)

    units = {
        "test_duration_s": "s",
        "detection_percent": "%",
        "total_distance_cm": "cm",
        "average_speed_cm_s": "cm/s",
        "maximum_speed_cm_s": "cm/s",
        "time_moving_s": "s",
        "time_immobile_s": "s",
        "time_s": "s",
        "time_percent": "%",
        "entries": "count",
        "latency_s": "s",
        "distance_cm": "cm",
    }

    result["unit"] = result["measure"].map(units).fillna("")
    return result


def export_results_workbook(
    output_path: str | Path,
    stage_results: pd.DataFrame,
    roi_results: pd.DataFrame,
    metadata: Dict,
    selected_metrics: List[str],
    stages: List[Dict],
) -> None:
    output_path = Path(output_path)

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        stage_results.to_excel(
            writer,
            sheet_name="Stage Results",
            index=False,
        )

        roi_results.to_excel(
            writer,
            sheet_name="ROI Results",
            index=False,
        )

        create_long_results(
            stage_results,
            roi_results,
        ).to_excel(
            writer,
            sheet_name="Long Format",
            index=False,
        )

        pd.DataFrame([metadata]).to_excel(
            writer,
            sheet_name="Metadata",
            index=False,
        )

        pd.DataFrame({
            "selected_metric": selected_metrics,
        }).to_excel(
            writer,
            sheet_name="Selected Metrics",
            index=False,
        )

        pd.DataFrame(stages).to_excel(
            writer,
            sheet_name="Stages",
            index=False,
        )


def _safe_filename(text: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in ("_", "-")
        else "_"
        for character in text
    ).strip("_")


def generate_result_graphs(
    stage_results: pd.DataFrame,
    roi_results: pd.DataFrame,
    output_folder: str | Path,
    dpi: int = 300,
) -> List[Path]:
    """
    Generate graphs only from valid numeric result columns.

    Text metadata columns such as animal ID, group, treatment,
    experimenter, paradigm, and notes are ignored automatically.
    Empty or entirely nonnumeric columns are also skipped.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    def numeric_values(
        dataframe: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        return pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    if not stage_results.empty and "stage" in stage_results.columns:
        excluded = {
            "stage",
            "stage_start_s",
            "stage_end_s",
            "experiment_name",
            "animal_id",
            "group",
            "treatment",
            "sex",
            "age",
            "weight",
            "experimenter",
            "notes",
            "paradigm",
            "camera_view",
        }

        for metric in [
            column
            for column in stage_results.columns
            if column not in excluded
        ]:
            values = numeric_values(
                stage_results,
                metric,
            )

            valid_mask = values.notna()

            if valid_mask.sum() == 0:
                continue

            labels = (
                stage_results.loc[
                    valid_mask,
                    "stage",
                ]
                .astype(str)
            )

            graph_values = values.loc[valid_mask]

            plt.figure(figsize=(8, 5))

            if len(graph_values) == 1:
                plt.bar(
                    labels,
                    graph_values,
                )
            else:
                plt.plot(
                    labels,
                    graph_values,
                    marker="o",
                    linewidth=2,
                )

            plt.xlabel("Stage")
            plt.ylabel(
                metric.replace("_", " ")
            )
            plt.title(
                metric.replace("_", " ").title()
            )
            plt.xticks(
                rotation=30,
                ha="right",
            )
            plt.tight_layout()

            path = (
                output_folder
                / f"stage_{_safe_filename(metric)}.png"
            )

            plt.savefig(
                path,
                dpi=dpi,
            )
            plt.close()
            created.append(path)

    if (
        not roi_results.empty
        and "roi" in roi_results.columns
        and "stage" in roi_results.columns
    ):
        excluded = {
            "stage",
            "roi",
            "stage_start_s",
            "stage_end_s",
            "experiment_name",
            "animal_id",
            "group",
            "treatment",
            "sex",
            "age",
            "weight",
            "experimenter",
            "notes",
            "paradigm",
            "camera_view",
        }

        for metric in [
            column
            for column in roi_results.columns
            if column not in excluded
        ]:
            numeric_metric = numeric_values(
                roi_results,
                metric,
            )

            if numeric_metric.notna().sum() == 0:
                continue

            graph_data = roi_results[
                [
                    "roi",
                    "stage",
                ]
            ].copy()

            graph_data[metric] = numeric_metric
            graph_data = graph_data.dropna(
                subset=[metric]
            )

            if graph_data.empty:
                continue

            pivot = graph_data.pivot_table(
                index="roi",
                columns="stage",
                values=metric,
                aggfunc="first",
            )

            if (
                pivot.empty
                or pivot.select_dtypes(
                    include=[np.number]
                ).empty
            ):
                continue

            axis = pivot.plot(
                kind="bar",
                figsize=(9, 5),
            )

            axis.set_xlabel("ROI")
            axis.set_ylabel(
                metric.replace("_", " ")
            )
            axis.set_title(
                f"{metric.replace('_', ' ').title()} by ROI"
            )

            plt.xticks(
                rotation=30,
                ha="right",
            )
            plt.tight_layout()

            path = (
                output_folder
                / f"roi_{_safe_filename(metric)}.png"
            )

            plt.savefig(
                path,
                dpi=dpi,
            )
            plt.close()
            created.append(path)

    return created

