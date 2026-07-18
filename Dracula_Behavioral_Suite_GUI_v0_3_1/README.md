# Dracula Behavioral Suite GUI v0.3.1 — Batty Beta

## New in v0.3

### Results Builder
Users choose exactly which measurements appear in the report:

- test duration
- detection percentage
- total distance
- average and maximum speed
- time moving
- time immobile
- time and percentage in each ROI
- ROI entries
- latency to first ROI entry
- distance and average speed in each ROI

### Stages
Analyze:

- the entire session
- equal time stages
- custom named stages

Every stage is calculated independently.

### Automated report graphs
DBS generates:

- stage-by-stage line graphs
- ROI comparison bar graphs
- trajectory
- heatmap

### Excel report
`results_report.xlsx` includes:

- Stage Results
- ROI Results
- Long Format
- Metadata
- Selected Metrics
- Stages

### Protocols
Save and load `.dbp` protocol files containing:

- paradigm
- camera view
- tracking parameters
- analysis interval
- selected results
- stage configuration
- figure settings
- ROIs
- calibration

Animal metadata, video path, and output path remain session-specific.

### Batty assistant
Batty is included inside DBS and helps with:

- tracking parameters
- missed detections
- ROIs
- calibration
- stages
- results
- protocols
- heatmaps
- cameras
- common errors

This beta uses a local rule-based assistant and does not send experiment data anywhere.

## Install or update

Open Command Prompt in the extracted folder:

```cmd
py -m pip install -r requirements.txt
py app.py
```

Or double-click:

```text
run_gui.bat
```


## v0.3.1 fix

- Result graphs now include only numeric measurements.
- Text metadata columns are excluded automatically.
- Empty and nonnumeric columns are skipped.
- A single-stage result is displayed as a bar rather than failing.
- Graph generation no longer stops the complete analysis when metadata is present.
