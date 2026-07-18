from __future__ import annotations

from typing import Dict


WELCOME_MESSAGE = """
Hi, I am Batty — the Dracula Behavioral Suite assistant.

I can help with:
• tracking parameters
• ROI and calibration workflow
• stages and time bins
• result selection
• heatmap settings
• protocols
• camera acquisition
• common errors

Batty currently runs locally and does not send experiment data anywhere.
""".strip()


class BattyAssistant:
    def respond(
        self,
        question: str,
        context: Dict,
    ) -> str:
        text = question.lower().strip()

        if not text:
            return "Ask me a question about the current experiment or DBS workflow."

        if any(word in text for word in ("hello", "hi", "hey")):
            return (
                "Hello! I am Batty. Tell me what is not working "
                "or what you want to configure."
            )

        if "resize" in text or "compress" in text or "resolution" in text:
            factor = context.get("resize_factor", 0.5)
            return (
                f"Your current resize factor is {factor}. "
                "Increase it toward 0.75 or 1.0 when the animal becomes "
                "too small after downsampling. A value of 1.0 uses the "
                "original resolution but processes more slowly."
            )

        if "threshold" in text or "not detect" in text or "losing" in text:
            method = context.get("method", "dark")
            threshold = context.get("threshold", 30)
            minimum = context.get("min_area_px", 100)

            return (
                f"Current method: {method}; threshold: {threshold}; "
                f"minimum area: {minimum} px. "
                "For missed detections, first verify dark/light mode, "
                "then lower the threshold gradually and lower the minimum "
                "area. Review the annotated video after each test."
            )

        if "stage" in text or "epoch" in text or "time bin" in text:
            return (
                "Use the Results & Stages tab. Choose Entire session, "
                "Equal stages, or Custom stages. Stage times are relative "
                "to the beginning of the analyzed interval. Each stage is "
                "calculated independently, including distance and ROI metrics."
            )

        if "protocol" in text or "template" in text:
            return (
                "A DBS protocol (.dbp) saves tracking settings, result metrics, "
                "stage definitions, figure settings, paradigm, ROI coordinates, "
                "and calibration. Animal-specific metadata and video paths are "
                "not required, so the protocol can be reused across subjects."
            )

        if "heatmap" in text or "scale" in text or "color" in text:
            return (
                "Heatmap controls are in the Figures tab. For comparisons "
                "between animals, use the same bin count, color map, minimum, "
                "and maximum across every subject. Percent occupancy is useful "
                "when session durations differ."
            )

        if "roi" in text or "zone" in text:
            return (
                "ROIs define both the analysis mask and reported behavioral "
                "zones. Use clear, non-overlapping zones when entries and "
                "transitions matter. Save the setup as a protocol when the "
                "camera and apparatus remain fixed."
            )

        if "calibr" in text or "centimeter" in text or "distance" in text:
            return (
                "Draw the calibration line over a known arena dimension and "
                "enter its real length in centimeters. Calibration converts "
                "pixel displacement into distance and speed. Recalibrate if "
                "the camera position or zoom changes."
            )

        if "camera" in text or "webcam" in text or "live" in text:
            return (
                "Open the Live Camera tab, detect cameras, select the desired "
                "internal or external camera, preview it, and record. This beta "
                "records a video for immediate analysis; continuous live metrics "
                "are planned for a later release."
            )

        if "result" in text or "report" in text or "excel" in text:
            return (
                "In Results & Stages, select only the measurements you need. "
                "DBS exports stage results, ROI results, long-format data, "
                "metadata, selected metrics, and stages to one Excel workbook. "
                "It also generates a graph for every compatible selected result."
            )

        if "error" in text or "traceback" in text or "failed" in text:
            return (
                "Copy the complete error from the log and ask me about it. "
                "Also confirm that the video exists, ROIs are defined, "
                "calibration is complete, and the end time is after the start."
            )

        animal = context.get("animal_id", "").strip()
        suffix = f" for animal {animal}" if animal else ""

        return (
            f"I do not have a specific rule for that yet{suffix}. "
            "Try asking about tracking, thresholds, ROIs, calibration, stages, "
            "results, protocols, heatmaps, cameras, or an error message."
        )
