
import os
import sys
import subprocess
import traceback
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QInputDialog,
)

from PySide6.QtGui import QPixmap, QFont

from behavior_suite.batty import BattyAssistant, WELCOME_MESSAGE
from behavior_suite.calibration import draw_scale_line
from behavior_suite.camera import (
    detect_available_cameras,
    preview_camera,
    record_camera,
)
from behavior_suite.config import TrackingConfig
from behavior_suite.elevated_plus_maze import analyze_elevated_plus_maze
from behavior_suite.metadata import save_metadata
from behavior_suite.open_field import analyze_open_field
from behavior_suite.plots import save_heatmap, save_trajectory
from behavior_suite.protocols import load_protocol, save_protocol
from behavior_suite.results_builder import (
    GLOBAL_METRICS,
    ROI_METRICS,
    calculate_results,
    default_metric_selection,
    equal_stages,
    export_results_workbook,
    generate_result_graphs,
    validate_stages,
)
from behavior_suite.roi import (
    define_rois_interactively,
    load_rois,
    save_rois,
)
from behavior_suite.session import seconds_to_frames
from behavior_suite.tracker import track_video
from behavior_suite.video import get_video_information


class AnalysisWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            settings = self.settings
            video_path = Path(settings["video_path"])
            output_folder = Path(settings["output_folder"])
            output_folder.mkdir(parents=True, exist_ok=True)

            save_metadata(
                settings["metadata"],
                output_folder,
            )

            start_frame, end_frame = seconds_to_frames(
                str(video_path),
                start_s=settings["start_s"],
                end_s=settings["end_s"],
            )

            self.log.emit(
                f"Analysis interval: {settings['start_s']:.2f} s "
                f"to {'video end' if settings['end_s'] is None else f'{settings['end_s']:.2f} s'}"
            )

            config = TrackingConfig(
                threshold=settings["threshold"],
                min_area_px=settings["min_area_px"],
                max_area_px=settings["max_area_px"],
                blur_kernel=7,
                morphology_kernel=3,
                method=settings["method"],
                max_jump_px=settings["max_jump_px"],
                reference_frames=settings["reference_frames"],
                resize_factor=settings["resize_factor"],
                analyze_only_inside_rois=settings[
                    "analyze_only_inside_rois"
                ],
                crop_to_roi_bounds=settings["crop_to_roi_bounds"],
                roi_crop_padding_px=settings["roi_crop_padding_px"],
                draw_trajectory=settings["draw_trajectory"],
                trajectory_tail_frames=300,
            )

            annotated_video = (
                str(output_folder / "annotated_video.mp4")
                if settings["save_annotated_video"]
                else None
            )

            self.log.emit("Tracking video...")

            data = track_video(
                video_path=str(video_path),
                rois=settings["rois"],
                output_csv=str(output_folder / "tracking_frames.csv"),
                output_video=annotated_video,
                scale_cm_per_px=settings["scale_cm_per_px"],
                config=config,
                start_frame=start_frame,
                end_frame=end_frame,
            )

            fps = get_video_information(str(video_path))["fps"]

            stages = settings["stages"]

            if settings["stage_mode"] == "equal":
                analyzed_duration_s = float(len(data) / fps)
                stages = equal_stages(
                    analyzed_duration_s,
                    settings["equal_stage_count"],
                )

            elif settings["stage_mode"] == "entire":
                stages = []

            else:
                stages = validate_stages(stages)

            self.log.emit("Calculating selected results...")

            stage_results, roi_results = calculate_results(
                data=data,
                fps=fps,
                selected_metrics=settings["selected_metrics"],
                stages=stages,
                immobility_speed_cm_s=settings[
                    "immobility_speed_cm_s"
                ],
            )

            for key, value in settings["metadata"].items():
                stage_results[key] = value
                roi_results[key] = value

            stage_results.to_csv(
                output_folder / "selected_stage_results.csv",
                index=False,
            )

            roi_results.to_csv(
                output_folder / "selected_roi_results.csv",
                index=False,
            )

            workbook_stages = stages or [{
                "name": "Entire session",
                "start_s": 0.0,
                "end_s": float(len(data) / fps),
            }]

            export_results_workbook(
                output_path=output_folder / "results_report.xlsx",
                stage_results=stage_results,
                roi_results=roi_results,
                metadata=settings["metadata"],
                selected_metrics=settings["selected_metrics"],
                stages=workbook_stages,
            )

            graph_paths = generate_result_graphs(
                stage_results=stage_results,
                roi_results=roi_results,
                output_folder=output_folder / "result_graphs",
                dpi=settings["figure_dpi"],
            )

            self.log.emit(
                f"Generated {len(graph_paths)} result graphs."
            )

            if settings["paradigm"] == "Open Field":
                result = analyze_open_field(
                    data,
                    fps,
                    center_roi_name="center",
                )
                result.to_csv(
                    output_folder / "open_field_results.csv",
                    index=False,
                )

            elif settings["paradigm"] == "Elevated Plus Maze":
                result = analyze_elevated_plus_maze(data, fps)
                result.to_csv(
                    output_folder / "elevated_plus_maze_results.csv",
                    index=False,
                )

            save_trajectory(
                data,
                output_folder / "trajectory.png",
            )

            save_heatmap(
                data,
                output_folder / "heatmap.png",
                bins=settings["heatmap_bins"],
                cmap=settings["heatmap_cmap"],
                vmin=settings["heatmap_vmin"],
                vmax=settings["heatmap_vmax"],
                normalize_percent=settings["heatmap_percent"],
                log_scale=settings["heatmap_log"],
            )

            self.finished_ok.emit(str(output_folder))

        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dracula Behavioral Suite v0.3.1 — Batty Beta")
        self.resize(1120, 900)

        self.rois = {}
        self.scale_cm_per_px = None
        self.worker = None
        self.batty = BattyAssistant()
        self.metric_checks = {}

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        header = QHBoxLayout()

        title = QLabel("DRACULA Behavioral Suite v0.3.1")
        title.setStyleSheet(
            "font-size: 24px; font-weight: 700;"
        )
        header.addWidget(title)

        header.addStretch()

        save_protocol_button = QPushButton("Save Protocol")
        save_protocol_button.clicked.connect(self.save_protocol_file)
        header.addWidget(save_protocol_button)

        load_protocol_button = QPushButton("Load Protocol")
        load_protocol_button.clicked.connect(self.load_protocol_file)
        header.addWidget(load_protocol_button)

        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tabs.addTab(self._project_tab(), "Project")
        self.tabs.addTab(self._session_tab(), "Session")
        self.tabs.addTab(self._tracking_tab(), "Tracking")
        self.tabs.addTab(
            self._results_and_stages_tab(),
            "Results & Stages",
        )
        self.tabs.addTab(self._figures_tab(), "Figures")
        self.tabs.addTab(self._camera_tab(), "Live Camera")
        self.tabs.addTab(self._batty_tab(), "Batty AI")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(145)
        root.addWidget(self.log_box)

        actions = QHBoxLayout()

        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        actions.addWidget(self.run_button)

        open_output_button = QPushButton("Open Output Folder")
        open_output_button.clicked.connect(self.open_output_folder)
        actions.addWidget(open_output_button)

        root.addLayout(actions)

    def _project_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        files_box = QGroupBox("Files and Paradigm")
        form = QFormLayout(files_box)

        self.video_edit = QLineEdit()
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_edit)

        browse_video = QPushButton("Browse")
        browse_video.clicked.connect(self.select_video)
        video_row.addWidget(browse_video)
        form.addRow("Video", video_row)

        self.output_edit = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit)

        browse_output = QPushButton("Browse")
        browse_output.clicked.connect(self.select_output_folder)
        output_row.addWidget(browse_output)
        form.addRow("Output folder", output_row)

        self.paradigm_combo = QComboBox()
        self.paradigm_combo.addItems([
            "Generic",
            "Open Field",
            "Elevated Plus Maze",
            "Operant Chamber",
            "Wheel Task",
            "Custom Maze",
        ])
        form.addRow("Paradigm", self.paradigm_combo)

        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "Top view",
            "Side view",
            "Angled view",
        ])
        form.addRow("Camera view", self.view_combo)

        layout.addWidget(files_box)

        metadata_box = QGroupBox("Experiment Metadata")
        metadata_form = QFormLayout(metadata_box)

        self.experiment_name = QLineEdit()
        self.animal_id = QLineEdit()
        self.group = QLineEdit()
        self.treatment = QLineEdit()
        self.sex = QComboBox()
        self.sex.addItems([
            "",
            "Male",
            "Female",
            "Other/Not specified",
        ])
        self.age = QLineEdit()
        self.weight = QLineEdit()
        self.experimenter = QLineEdit()
        self.notes = QTextEdit()
        self.notes.setMaximumHeight(75)

        metadata_form.addRow("Experiment name", self.experiment_name)
        metadata_form.addRow("Animal ID", self.animal_id)
        metadata_form.addRow("Group", self.group)
        metadata_form.addRow("Treatment", self.treatment)
        metadata_form.addRow("Sex", self.sex)
        metadata_form.addRow("Age", self.age)
        metadata_form.addRow("Weight", self.weight)
        metadata_form.addRow("Experimenter", self.experimenter)
        metadata_form.addRow("Notes", self.notes)

        layout.addWidget(metadata_box)

        roi_box = QGroupBox("Regions of Interest")
        roi_layout = QVBoxLayout(roi_box)

        self.roi_list = QListWidget()
        roi_layout.addWidget(self.roi_list)

        roi_buttons = QHBoxLayout()

        draw_button = QPushButton("Draw ROIs")
        draw_button.clicked.connect(self.draw_rois)
        roi_buttons.addWidget(draw_button)

        load_button = QPushButton("Load ROIs")
        load_button.clicked.connect(self.load_rois_file)
        roi_buttons.addWidget(load_button)

        save_button = QPushButton("Save ROIs")
        save_button.clicked.connect(self.save_rois_file)
        roi_buttons.addWidget(save_button)

        roi_layout.addLayout(roi_buttons)
        layout.addWidget(roi_box)

        calibration_box = QGroupBox("Spatial Calibration")
        calibration_form = QFormLayout(calibration_box)

        self.calibration_label = QLabel("Not calibrated")
        calibration_form.addRow("Status", self.calibration_label)

        calibration_button = QPushButton("Draw Calibration Line")
        calibration_button.clicked.connect(self.calibrate)
        calibration_form.addRow(calibration_button)

        layout.addWidget(calibration_box)

        return tab

    def _session_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 1000000)
        self.start_spin.setDecimals(2)

        self.use_end_check = QCheckBox()

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 1000000)
        self.end_spin.setDecimals(2)

        form.addRow("Analysis start (seconds)", self.start_spin)
        form.addRow("Use custom end time", self.use_end_check)
        form.addRow("Analysis end (seconds)", self.end_spin)

        help_label = QLabel(
            "The video is not cut. DBS only analyzes the selected interval. "
            "Stages in the next tab are relative to the beginning of this interval."
        )
        help_label.setWordWrap(True)
        form.addRow(help_label)

        return tab

    def _tracking_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["dark", "light", "abs"])

        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 255)
        self.threshold_spin.setValue(30)

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 1000000)
        self.min_area_spin.setValue(100)

        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1, 5000000)
        self.max_area_spin.setValue(100000)

        self.resize_spin = QDoubleSpinBox()
        self.resize_spin.setRange(0.1, 1.0)
        self.resize_spin.setSingleStep(0.05)
        self.resize_spin.setValue(0.5)

        self.max_jump_spin = QDoubleSpinBox()
        self.max_jump_spin.setRange(1, 5000)
        self.max_jump_spin.setValue(80)

        self.reference_frames_spin = QSpinBox()
        self.reference_frames_spin.setRange(5, 500)
        self.reference_frames_spin.setValue(30)

        self.roi_only_check = QCheckBox()
        self.roi_only_check.setChecked(True)

        self.crop_check = QCheckBox()
        self.crop_check.setChecked(True)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 1000)
        self.padding_spin.setValue(20)

        self.annotated_check = QCheckBox()
        self.annotated_check.setChecked(True)

        self.trajectory_check = QCheckBox()
        self.trajectory_check.setChecked(True)

        self.immobility_spin = QDoubleSpinBox()
        self.immobility_spin.setRange(0, 100)
        self.immobility_spin.setDecimals(2)
        self.immobility_spin.setValue(1.0)

        form.addRow("Animal/background method", self.method_combo)
        form.addRow("Threshold", self.threshold_spin)
        form.addRow("Minimum object area", self.min_area_spin)
        form.addRow("Maximum object area", self.max_area_spin)
        form.addRow("Resize factor", self.resize_spin)
        form.addRow("Maximum jump", self.max_jump_spin)
        form.addRow("Reference frames", self.reference_frames_spin)
        form.addRow("Analyze only inside ROIs", self.roi_only_check)
        form.addRow("Crop to ROI bounds", self.crop_check)
        form.addRow("ROI crop padding", self.padding_spin)
        form.addRow("Save annotated video", self.annotated_check)
        form.addRow("Draw trajectory", self.trajectory_check)
        form.addRow(
            "Immobility threshold (cm/s)",
            self.immobility_spin,
        )

        return tab

    def _results_and_stages_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        metrics_box = QGroupBox("Select Results")
        metrics_layout = QVBoxLayout(metrics_box)

        general_label = QLabel("General movement")
        general_label.setStyleSheet("font-weight: 700;")
        metrics_layout.addWidget(general_label)

        defaults = set(default_metric_selection())

        for key, label in GLOBAL_METRICS.items():
            check = QCheckBox(label)
            check.setChecked(key in defaults)
            self.metric_checks[key] = check
            metrics_layout.addWidget(check)

        roi_label = QLabel("ROI measurements")
        roi_label.setStyleSheet("font-weight: 700;")
        metrics_layout.addWidget(roi_label)

        for key, label in ROI_METRICS.items():
            check = QCheckBox(label)
            check.setChecked(key in defaults)
            self.metric_checks[key] = check
            metrics_layout.addWidget(check)

        select_buttons = QHBoxLayout()

        select_all = QPushButton("Select All")
        select_all.clicked.connect(
            lambda: self.set_all_metrics(True)
        )
        select_buttons.addWidget(select_all)

        clear_all = QPushButton("Clear")
        clear_all.clicked.connect(
            lambda: self.set_all_metrics(False)
        )
        select_buttons.addWidget(clear_all)

        metrics_layout.addLayout(select_buttons)
        metrics_layout.addStretch()

        layout.addWidget(metrics_box)

        stages_box = QGroupBox("Experiment Stages")
        stages_layout = QVBoxLayout(stages_box)

        mode_form = QFormLayout()

        self.stage_mode_combo = QComboBox()
        self.stage_mode_combo.addItems([
            "Entire session",
            "Equal stages",
            "Custom stages",
        ])
        self.stage_mode_combo.currentTextChanged.connect(
            self.update_stage_mode
        )
        mode_form.addRow("Stage mode", self.stage_mode_combo)

        self.equal_stage_count = QSpinBox()
        self.equal_stage_count.setRange(1, 100)
        self.equal_stage_count.setValue(4)
        mode_form.addRow("Number of equal stages", self.equal_stage_count)

        stages_layout.addLayout(mode_form)

        self.stage_table = QTableWidget(0, 3)
        self.stage_table.setHorizontalHeaderLabels([
            "Stage name",
            "Start (s)",
            "End (s)",
        ])
        self.stage_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        stages_layout.addWidget(self.stage_table)

        stage_buttons = QHBoxLayout()

        add_stage = QPushButton("Add Stage")
        add_stage.clicked.connect(self.add_stage_row)
        stage_buttons.addWidget(add_stage)

        remove_stage = QPushButton("Remove Selected")
        remove_stage.clicked.connect(self.remove_stage_row)
        stage_buttons.addWidget(remove_stage)

        stages_layout.addLayout(stage_buttons)

        note = QLabel(
            "Custom stage times are relative to the beginning of the analyzed "
            "interval. Each stage receives independent distance, speed, "
            "immobility, and ROI calculations."
        )
        note.setWordWrap(True)
        stages_layout.addWidget(note)

        layout.addWidget(stages_box, 1)

        self.update_stage_mode()
        return tab

    def _figures_tab(self):
        tab = QWidget()
        form = QFormLayout(tab)

        self.heatmap_bins = QSpinBox()
        self.heatmap_bins.setRange(10, 500)
        self.heatmap_bins.setValue(60)

        self.heatmap_cmap = QComboBox()
        self.heatmap_cmap.addItems([
            "viridis",
            "inferno",
            "magma",
            "plasma",
            "cividis",
            "hot",
            "gray",
            "jet",
        ])

        self.use_vmin = QCheckBox()

        self.heatmap_vmin = QDoubleSpinBox()
        self.heatmap_vmin.setRange(0, 100000000)

        self.use_vmax = QCheckBox()

        self.heatmap_vmax = QDoubleSpinBox()
        self.heatmap_vmax.setRange(0, 100000000)
        self.heatmap_vmax.setValue(100)

        self.heatmap_percent = QCheckBox()
        self.heatmap_log = QCheckBox()

        self.figure_dpi = QSpinBox()
        self.figure_dpi.setRange(72, 1200)
        self.figure_dpi.setValue(300)

        form.addRow("Heatmap bins", self.heatmap_bins)
        form.addRow("Color map", self.heatmap_cmap)
        form.addRow("Use minimum scale", self.use_vmin)
        form.addRow("Minimum value", self.heatmap_vmin)
        form.addRow("Use maximum scale", self.use_vmax)
        form.addRow("Maximum value", self.heatmap_vmax)
        form.addRow(
            "Normalize to occupancy percent",
            self.heatmap_percent,
        )
        form.addRow("Logarithmic scale", self.heatmap_log)
        form.addRow("Result graph DPI", self.figure_dpi)

        graph_note = QLabel(
            "DBS generates stage graphs and ROI comparison graphs from the "
            "measurements selected in Results & Stages."
        )
        graph_note.setWordWrap(True)
        form.addRow(graph_note)

        return tab

    def _camera_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Camera Acquisition")
        form = QFormLayout(box)

        self.camera_combo = QComboBox()

        self.live_duration = QDoubleSpinBox()
        self.live_duration.setRange(0, 1000000)
        self.live_duration.setValue(600)

        form.addRow("Camera", self.camera_combo)
        form.addRow(
            "Recording duration (s, 0 = manual)",
            self.live_duration,
        )

        layout.addWidget(box)

        buttons = QHBoxLayout()

        detect_button = QPushButton("Detect Cameras")
        detect_button.clicked.connect(self.detect_cameras)
        buttons.addWidget(detect_button)

        preview_button = QPushButton("Preview Camera")
        preview_button.clicked.connect(self.preview_selected_camera)
        buttons.addWidget(preview_button)

        record_button = QPushButton("Record Video")
        record_button.clicked.connect(self.record_selected_camera)
        buttons.addWidget(record_button)

        layout.addLayout(buttons)

        note = QLabel(
            "This beta records from the selected camera. Real-time result "
            "calculation will be added in a later live-analysis release."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()

        return tab

    def _batty_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title = QLabel("🦇 Batty — DBS Assistant")
        title.setStyleSheet(
            "font-size: 18px; font-weight: 700;"
        )
        layout.addWidget(title)

        privacy = QLabel(
            "Batty currently works locally with built-in DBS guidance. "
            "Experiment data is not sent to an external service."
        )
        privacy.setWordWrap(True)
        layout.addWidget(privacy)

        self.batty_chat = QTextEdit()
        self.batty_chat.setReadOnly(True)
        self.batty_chat.setPlainText(WELCOME_MESSAGE)
        layout.addWidget(self.batty_chat)

        input_row = QHBoxLayout()

        self.batty_input = QLineEdit()
        self.batty_input.setPlaceholderText(
            "Ask Batty about tracking, stages, results, protocols, heatmaps..."
        )
        self.batty_input.returnPressed.connect(self.ask_batty)
        input_row.addWidget(self.batty_input)

        ask_button = QPushButton("Ask Batty")
        ask_button.clicked.connect(self.ask_batty)
        input_row.addWidget(ask_button)

        layout.addLayout(input_row)

        quick_row = QHBoxLayout()

        for label, question in [
            ("Mouse not detected", "The mouse is not detected. What should I change?"),
            ("Explain stages", "How do experiment stages work?"),
            ("Protocol help", "What is saved in a protocol?"),
            ("Heatmap help", "How should I set the heatmap scale?"),
        ]:
            button = QPushButton(label)
            button.clicked.connect(
                lambda checked=False, q=question: self.ask_batty(q)
            )
            quick_row.addWidget(button)

        layout.addLayout(quick_row)
        return tab

    def set_all_metrics(self, checked):
        for checkbox in self.metric_checks.values():
            checkbox.setChecked(checked)

    def update_stage_mode(self):
        mode = self.stage_mode_combo.currentText()
        equal_enabled = mode == "Equal stages"
        custom_enabled = mode == "Custom stages"

        self.equal_stage_count.setEnabled(equal_enabled)
        self.stage_table.setEnabled(custom_enabled)

    def add_stage_row(self):
        row = self.stage_table.rowCount()
        self.stage_table.insertRow(row)

        self.stage_table.setItem(
            row,
            0,
            QTableWidgetItem(f"Stage {row + 1}"),
        )
        self.stage_table.setItem(
            row,
            1,
            QTableWidgetItem(str(row * 300)),
        )
        self.stage_table.setItem(
            row,
            2,
            QTableWidgetItem(str((row + 1) * 300)),
        )

    def remove_stage_row(self):
        row = self.stage_table.currentRow()

        if row >= 0:
            self.stage_table.removeRow(row)

    def selected_metrics(self):
        return [
            key
            for key, checkbox in self.metric_checks.items()
            if checkbox.isChecked()
        ]

    def custom_stages(self):
        stages = []

        for row in range(self.stage_table.rowCount()):
            name_item = self.stage_table.item(row, 0)
            start_item = self.stage_table.item(row, 1)
            end_item = self.stage_table.item(row, 2)

            if not all((name_item, start_item, end_item)):
                continue

            stages.append({
                "name": name_item.text().strip(),
                "start_s": float(start_item.text()),
                "end_s": float(end_item.text()),
            })

        return stages

    def stage_mode_key(self):
        return {
            "Entire session": "entire",
            "Equal stages": "equal",
            "Custom stages": "custom",
        }[self.stage_mode_combo.currentText()]

    def log(self, text):
        self.log_box.append(text)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            "",
            "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )

        if path:
            self.video_edit.setText(path)

            output = (
                Path(path).parent
                / f"{Path(path).stem}_behavior_analysis"
            )
            self.output_edit.setText(str(output))

    def select_output_folder(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
        )

        if path:
            self.output_edit.setText(path)

    def required_roi_names(self):
        paradigm = self.paradigm_combo.currentText()

        if paradigm == "Open Field":
            return ["center"]

        if paradigm == "Elevated Plus Maze":
            return [
                "open_1",
                "open_2",
                "closed_1",
                "closed_2",
                "center",
            ]

        names, accepted = QInputDialog.getText(
            self,
            "ROI names",
            "Enter comma-separated ROI names:",
        )

        if not accepted:
            return []

        return [
            name.strip()
            for name in names.split(",")
            if name.strip()
        ]

    def draw_rois(self):
        video_path = self.video_edit.text().strip()

        if not video_path:
            QMessageBox.warning(
                self,
                "Video",
                "Select a video first.",
            )
            return

        names = self.required_roi_names()

        if not names:
            return

        self.rois = define_rois_interactively(
            video_path,
            names,
        )

        self.refresh_roi_list()

    def refresh_roi_list(self):
        self.roi_list.clear()

        for name, polygon in self.rois.items():
            self.roi_list.addItem(
                f"{name} ({len(polygon)} vertices)"
            )

    def load_rois_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load ROI file",
            "",
            "JSON files (*.json)",
        )

        if path:
            self.rois = load_rois(path)
            self.refresh_roi_list()

    def save_rois_file(self):
        if not self.rois:
            QMessageBox.warning(
                self,
                "ROIs",
                "No ROIs are defined.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save ROI file",
            "rois.json",
            "JSON files (*.json)",
        )

        if path:
            save_rois(self.rois, path)

    def calibrate(self):
        video_path = self.video_edit.text().strip()

        if not video_path:
            QMessageBox.warning(
                self,
                "Video",
                "Select a video first.",
            )
            return

        known_cm, accepted = QInputDialog.getDouble(
            self,
            "Known distance",
            "Real length of the line (cm):",
            10.0,
            0.001,
            100000,
            3,
        )

        if not accepted:
            return

        try:
            pixels_per_cm, cm_per_pixel = draw_scale_line(
                video_path,
                known_cm=known_cm,
            )

            self.scale_cm_per_px = cm_per_pixel

            self.calibration_label.setText(
                f"{cm_per_pixel:.6f} cm/pixel "
                f"({pixels_per_cm:.3f} px/cm)"
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Calibration error",
                str(error),
            )

    def detect_cameras(self):
        self.camera_combo.clear()
        cameras = detect_available_cameras()

        for index in cameras:
            self.camera_combo.addItem(
                f"Camera {index}",
                index,
            )

        self.log(f"Detected cameras: {cameras}")

    def preview_selected_camera(self):
        if self.camera_combo.count() == 0:
            self.detect_cameras()

        if self.camera_combo.count() == 0:
            QMessageBox.warning(
                self,
                "Camera",
                "No camera was detected.",
            )
            return

        preview_camera(
            self.camera_combo.currentData()
        )

    def record_selected_camera(self):
        if self.camera_combo.count() == 0:
            self.detect_cameras()

        if self.camera_combo.count() == 0:
            QMessageBox.warning(
                self,
                "Camera",
                "No camera was detected.",
            )
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save camera recording",
            "live_recording.mp4",
            "MP4 video (*.mp4)",
        )

        if not output_path:
            return

        duration = self.live_duration.value()
        duration = None if duration <= 0 else duration

        record_camera(
            self.camera_combo.currentData(),
            output_path,
            duration_s=duration,
        )

        self.video_edit.setText(output_path)

        self.output_edit.setText(
            str(
                Path(output_path).parent
                / f"{Path(output_path).stem}_behavior_analysis"
            )
        )

    def metadata(self):
        return {
            "experiment_name": self.experiment_name.text().strip(),
            "animal_id": self.animal_id.text().strip(),
            "group": self.group.text().strip(),
            "treatment": self.treatment.text().strip(),
            "sex": self.sex.currentText(),
            "age": self.age.text().strip(),
            "weight": self.weight.text().strip(),
            "experimenter": self.experimenter.text().strip(),
            "notes": self.notes.toPlainText().strip(),
            "paradigm": self.paradigm_combo.currentText(),
            "camera_view": self.view_combo.currentText(),
        }

    def collect_protocol(self):
        return {
            "name": self.experiment_name.text().strip() or "DBS Protocol",
            "paradigm": self.paradigm_combo.currentText(),
            "camera_view": self.view_combo.currentText(),
            "tracking": {
                "method": self.method_combo.currentText(),
                "threshold": self.threshold_spin.value(),
                "min_area_px": self.min_area_spin.value(),
                "max_area_px": self.max_area_spin.value(),
                "resize_factor": self.resize_spin.value(),
                "max_jump_px": self.max_jump_spin.value(),
                "reference_frames": self.reference_frames_spin.value(),
                "analyze_only_inside_rois": self.roi_only_check.isChecked(),
                "crop_to_roi_bounds": self.crop_check.isChecked(),
                "roi_crop_padding_px": self.padding_spin.value(),
                "save_annotated_video": self.annotated_check.isChecked(),
                "draw_trajectory": self.trajectory_check.isChecked(),
                "immobility_speed_cm_s": self.immobility_spin.value(),
            },
            "analysis_window": {
                "start_s": self.start_spin.value(),
                "use_end": self.use_end_check.isChecked(),
                "end_s": self.end_spin.value(),
            },
            "results": {
                "selected_metrics": self.selected_metrics(),
                "stage_mode": self.stage_mode_key(),
                "equal_stage_count": self.equal_stage_count.value(),
                "custom_stages": self.custom_stages(),
            },
            "figures": {
                "heatmap_bins": self.heatmap_bins.value(),
                "heatmap_cmap": self.heatmap_cmap.currentText(),
                "use_vmin": self.use_vmin.isChecked(),
                "heatmap_vmin": self.heatmap_vmin.value(),
                "use_vmax": self.use_vmax.isChecked(),
                "heatmap_vmax": self.heatmap_vmax.value(),
                "heatmap_percent": self.heatmap_percent.isChecked(),
                "heatmap_log": self.heatmap_log.isChecked(),
                "figure_dpi": self.figure_dpi.value(),
            },
            "rois": self.rois,
            "scale_cm_per_px": self.scale_cm_per_px,
        }

    def apply_protocol(self, protocol):
        self.paradigm_combo.setCurrentText(
            protocol.get("paradigm", "Generic")
        )
        self.view_combo.setCurrentText(
            protocol.get("camera_view", "Top view")
        )

        tracking = protocol.get("tracking", {})
        self.method_combo.setCurrentText(
            tracking.get("method", "dark")
        )
        self.threshold_spin.setValue(
            int(tracking.get("threshold", 30))
        )
        self.min_area_spin.setValue(
            int(tracking.get("min_area_px", 100))
        )
        self.max_area_spin.setValue(
            int(tracking.get("max_area_px", 100000))
        )
        self.resize_spin.setValue(
            float(tracking.get("resize_factor", 0.5))
        )
        self.max_jump_spin.setValue(
            float(tracking.get("max_jump_px", 80))
        )
        self.reference_frames_spin.setValue(
            int(tracking.get("reference_frames", 30))
        )
        self.roi_only_check.setChecked(
            bool(tracking.get("analyze_only_inside_rois", True))
        )
        self.crop_check.setChecked(
            bool(tracking.get("crop_to_roi_bounds", True))
        )
        self.padding_spin.setValue(
            int(tracking.get("roi_crop_padding_px", 20))
        )
        self.annotated_check.setChecked(
            bool(tracking.get("save_annotated_video", True))
        )
        self.trajectory_check.setChecked(
            bool(tracking.get("draw_trajectory", True))
        )
        self.immobility_spin.setValue(
            float(tracking.get("immobility_speed_cm_s", 1.0))
        )

        analysis_window = protocol.get("analysis_window", {})
        self.start_spin.setValue(
            float(analysis_window.get("start_s", 0))
        )
        self.use_end_check.setChecked(
            bool(analysis_window.get("use_end", False))
        )
        self.end_spin.setValue(
            float(analysis_window.get("end_s", 0))
        )

        results = protocol.get("results", {})
        selected = set(
            results.get(
                "selected_metrics",
                default_metric_selection(),
            )
        )

        for key, checkbox in self.metric_checks.items():
            checkbox.setChecked(key in selected)

        mode_map = {
            "entire": "Entire session",
            "equal": "Equal stages",
            "custom": "Custom stages",
        }

        self.stage_mode_combo.setCurrentText(
            mode_map.get(
                results.get("stage_mode", "entire"),
                "Entire session",
            )
        )

        self.equal_stage_count.setValue(
            int(results.get("equal_stage_count", 4))
        )

        self.stage_table.setRowCount(0)

        for stage in results.get("custom_stages", []):
            row = self.stage_table.rowCount()
            self.stage_table.insertRow(row)

            for column, value in enumerate([
                stage.get("name", f"Stage {row + 1}"),
                stage.get("start_s", 0),
                stage.get("end_s", 0),
            ]):
                self.stage_table.setItem(
                    row,
                    column,
                    QTableWidgetItem(str(value)),
                )

        figures = protocol.get("figures", {})
        self.heatmap_bins.setValue(
            int(figures.get("heatmap_bins", 60))
        )
        self.heatmap_cmap.setCurrentText(
            figures.get("heatmap_cmap", "viridis")
        )
        self.use_vmin.setChecked(
            bool(figures.get("use_vmin", False))
        )
        self.heatmap_vmin.setValue(
            float(figures.get("heatmap_vmin", 0))
        )
        self.use_vmax.setChecked(
            bool(figures.get("use_vmax", False))
        )
        self.heatmap_vmax.setValue(
            float(figures.get("heatmap_vmax", 100))
        )
        self.heatmap_percent.setChecked(
            bool(figures.get("heatmap_percent", False))
        )
        self.heatmap_log.setChecked(
            bool(figures.get("heatmap_log", False))
        )
        self.figure_dpi.setValue(
            int(figures.get("figure_dpi", 300))
        )

        loaded_rois = protocol.get("rois", {})

        self.rois = {
            name: [
                (int(point[0]), int(point[1]))
                for point in polygon
            ]
            for name, polygon in loaded_rois.items()
        }
        self.refresh_roi_list()

        self.scale_cm_per_px = protocol.get("scale_cm_per_px")

        if self.scale_cm_per_px is not None:
            self.calibration_label.setText(
                f"{float(self.scale_cm_per_px):.6f} cm/pixel "
                "(loaded from protocol)"
            )
        else:
            self.calibration_label.setText("Not calibrated")

        self.update_stage_mode()

    def save_protocol_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save DBS protocol",
            "experiment_protocol.dbp",
            "DBS Protocol (*.dbp)",
        )

        if not path:
            return

        saved_path = save_protocol(
            path,
            self.collect_protocol(),
        )

        self.log(f"Protocol saved: {saved_path}")

    def load_protocol_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load DBS protocol",
            "",
            "DBS Protocol (*.dbp)",
        )

        if not path:
            return

        try:
            protocol = load_protocol(path)
            self.apply_protocol(protocol)
            self.log(f"Protocol loaded: {path}")

        except Exception as error:
            QMessageBox.critical(
                self,
                "Protocol error",
                str(error),
            )

    def batty_context(self):
        return {
            "animal_id": self.animal_id.text(),
            "method": self.method_combo.currentText(),
            "threshold": self.threshold_spin.value(),
            "min_area_px": self.min_area_spin.value(),
            "resize_factor": self.resize_spin.value(),
            "stage_mode": self.stage_mode_combo.currentText(),
            "selected_metrics": self.selected_metrics(),
            "roi_names": list(self.rois),
        }

    def ask_batty(self, predefined_question=None):
        if isinstance(predefined_question, str):
            question = predefined_question
        else:
            question = self.batty_input.text().strip()

        if not question:
            return

        response = self.batty.respond(
            question,
            self.batty_context(),
        )

        self.batty_chat.append(f"\nYou: {question}\n")
        self.batty_chat.append(f"Batty: {response}\n")
        self.batty_input.clear()

    def collect_settings(self):
        end_s = (
            self.end_spin.value()
            if self.use_end_check.isChecked()
            else None
        )

        return {
            "video_path": self.video_edit.text().strip(),
            "output_folder": self.output_edit.text().strip(),
            "paradigm": self.paradigm_combo.currentText(),
            "metadata": self.metadata(),
            "rois": self.rois,
            "scale_cm_per_px": self.scale_cm_per_px,
            "start_s": self.start_spin.value(),
            "end_s": end_s,
            "method": self.method_combo.currentText(),
            "threshold": self.threshold_spin.value(),
            "min_area_px": self.min_area_spin.value(),
            "max_area_px": self.max_area_spin.value(),
            "max_jump_px": self.max_jump_spin.value(),
            "reference_frames": self.reference_frames_spin.value(),
            "resize_factor": self.resize_spin.value(),
            "analyze_only_inside_rois":
                self.roi_only_check.isChecked(),
            "crop_to_roi_bounds": self.crop_check.isChecked(),
            "roi_crop_padding_px": self.padding_spin.value(),
            "save_annotated_video": self.annotated_check.isChecked(),
            "draw_trajectory": self.trajectory_check.isChecked(),
            "immobility_speed_cm_s": self.immobility_spin.value(),
            "selected_metrics": self.selected_metrics(),
            "stage_mode": self.stage_mode_key(),
            "equal_stage_count": self.equal_stage_count.value(),
            "stages": self.custom_stages(),
            "heatmap_bins": self.heatmap_bins.value(),
            "heatmap_cmap": self.heatmap_cmap.currentText(),
            "heatmap_vmin": (
                self.heatmap_vmin.value()
                if self.use_vmin.isChecked()
                else None
            ),
            "heatmap_vmax": (
                self.heatmap_vmax.value()
                if self.use_vmax.isChecked()
                else None
            ),
            "heatmap_percent": self.heatmap_percent.isChecked(),
            "heatmap_log": self.heatmap_log.isChecked(),
            "figure_dpi": self.figure_dpi.value(),
        }

    def validate_settings(self, settings):
        if not settings["video_path"]:
            return "Select a video."

        if not Path(settings["video_path"]).exists():
            return "The selected video does not exist."

        if not settings["output_folder"]:
            return "Select an output folder."

        if not settings["rois"]:
            return "Define or load at least one ROI."

        if settings["scale_cm_per_px"] is None:
            return "Perform spatial calibration or load a calibrated protocol."

        if (
            settings["end_s"] is not None
            and settings["end_s"] <= settings["start_s"]
        ):
            return "Analysis end time must be greater than start time."

        if not settings["selected_metrics"]:
            return "Select at least one result measurement."

        if (
            settings["stage_mode"] == "custom"
            and not settings["stages"]
        ):
            return "Add at least one custom stage."

        if settings["stage_mode"] == "custom":
            try:
                validate_stages(settings["stages"])
            except Exception as error:
                return str(error)

        required = []

        if settings["paradigm"] == "Open Field":
            required = ["center"]

        elif settings["paradigm"] == "Elevated Plus Maze":
            required = [
                "open_1",
                "open_2",
                "closed_1",
                "closed_2",
                "center",
            ]

        missing = [
            name
            for name in required
            if name not in settings["rois"]
        ]

        if missing:
            return f"Missing required ROIs: {missing}"

        return None

    def run_analysis(self):
        try:
            settings = self.collect_settings()
        except ValueError as error:
            QMessageBox.warning(
                self,
                "Invalid value",
                str(error),
            )
            return

        error = self.validate_settings(settings)

        if error:
            QMessageBox.warning(
                self,
                "Cannot run",
                error,
            )
            return

        self.run_button.setEnabled(False)
        self.log("Analysis started.")

        self.worker = AnalysisWorker(settings)
        self.worker.log.connect(self.log)
        self.worker.finished_ok.connect(
            self.analysis_finished
        )
        self.worker.failed.connect(
            self.analysis_failed
        )
        self.worker.start()

    def analysis_finished(self, folder):
        self.run_button.setEnabled(True)
        self.log(f"Analysis completed: {folder}")

        QMessageBox.information(
            self,
            "Analysis completed",
            "Results, Excel report, and graphs were saved to:\n"
            f"{folder}",
        )

    def analysis_failed(self, error):
        self.run_button.setEnabled(True)
        self.log(error)

        QMessageBox.critical(
            self,
            "Analysis failed",
            "The analysis failed. Review the log or ask Batty about the error.",
        )

    def open_output_folder(self):
        folder = self.output_edit.text().strip()

        if not folder:
            return

        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


if __name__ == "__main__":
    application = QApplication(sys.argv)
    application.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(application.exec())
