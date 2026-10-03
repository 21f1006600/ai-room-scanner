# AI Room Scanner - Experimental Prototype

This repository contains experimental code for an AI-based room scanning prototype using depth sensors. 
It is **not** a completed room scanner or validated measurement system.

## Submission artifacts

- **Main demo:** `Demo-RoomPipeline.ps1` — runs the supplied-data single-frame pipeline; see the command in [Demo wrapper](#5-demo-wrapper-for-evaluator).
- **Technical report:** `docs/final_technical_report.md`
- **Reproduction commands:** `docs/reproduction.md`
- **Requirement/status matrix:** `docs/compliance_matrix.md`
- **Fix-loop evidence:** `output/fix_loop/fix_loop_report.md` and `output/fix_loop_threshold_experiment_corrected/`
- **Unified schema and example:** `schemas/room_output.schema.json` and `output/demo_single_scan_floor_only_1a8384c3f6_Frame0/room_output.json`

The report and matrix explicitly identify the assignment capabilities that remain incomplete.

## Current Capabilities

- Dataset inspection and metadata reporting (depth, confidence, odometry)
- Depth and confidence visualization
- Single-frame point cloud back-projection from depth images
- Exploratory plane detection using RANSAC
- Threshold sensitivity experiments for plane detection (with corrected disjoint plane masks)
- JSON and CSV output of experiment results
- Basic point cloud statistics (extents, percentiles, depth ranges)

## Known Limitations

- **Depth units**: Raw uint16 values; physical scale (e.g., meters per unit) is unknown and unverified.
- **Intrinsics assumption**: Depth resolution is assumed to share the same field-of-view as RGB, using scaled intrinsics—this assumption is **unverified**.
- **Odometry conventions**: Coordinate system and quaternion conventions from odometry are not verified.
- **No metric accuracy**: No verification against laser/tape ground truth; measurements are not claimed to be accurate.
- **No room reconstruction**: The pipeline processes single frames only; no multi-frame stitching, loop closure, or room plan generation.
- **No damage detection**: Surface damage detection, concealed-damage flags, or scope line items are not implemented.
- **No capture pipeline**: No hardware integration or capture procedure (iOS app or off‑the‑shelf protocol) is demonstrated.
- **No consumer app comparison**: No comparison with commercial scanning apps (e.g., Magicplan, poly.cam) has been performed.

## Environment and Setup

- **OS**: Windows 11 (development environment). The PowerShell demo wrapper is Windows-specific, and portability of the Python scripts has not been verified; several experiment scripts contain local dataset paths.
- **Python**: The code was developed with Python 3.12 (as evidenced by the cached bytecode files), but may work with other 3.x versions.
- **Dependencies** (based on import statements in the source):
  - OpenCV (`cv2`)
  - NumPy (`numpy`)
  - Matplotlib (`matplotlib`) – optional for visualization
  - Standard library: `json`, `csv`, `time`, `pathlib`, `argparse`, `sys`

  **Note**: No `requirements.txt` or lock file is provided. Install the above packages via `pip` as needed.
  Versions are not specified; the code was tested with the versions available in the development environment.

## Usage

Run the following commands from the repository root (`D:\brynz-applied-AI\ai-room-scanner`). 
The output-producing scripts described below refuse to overwrite non-empty output directories; use a fresh output location or preserve the existing artifacts.

### 1. Dataset Metadata and Audits

- **Full depth/audit processing** (all three datasets, ~2-3 minutes total):
  ```bash
  python -u src/full_depth_audit.py
  ```
- **Sampled audit** (first 100 frames of each dataset):
  ```bash
  python -u src/full_depth_audit.py --sample
  ```
- **Dataset-specific audit** (e.g., only the with-ceiling dataset):
  ```bash
  python -u src/full_depth_audit.py single_scan_with_ceiling/c7d28f72c6
  ```
- **Timestamp and odometry consistency check** (single_room dataset):
  ```bash
  python -u src/temporal_audit.py
  ```
- **Odometry rotation-only audit**:
  ```bash
  python -u src/odometry_rotation_audit.py
  ```
- **Depth/confidence visualization** (example frame):
  ```bash
  python -u src/visualize_point_cloud.py
  ```
- **Dataset comparison** (point cloud statistics across datasets):
  ```bash
  python -u src/dataset_comparison.py
  ```

### 2. Single-Frame Geometry Pipeline

- **Process a specific dataset and frame** (outputs to `output/single_room_pipeline/`):
  ```bash
  python -u src/single_room_pipeline.py --dataset single_scan_floor_only/1a8384c3f6 --frame 0
  ```
  - `--dataset`: Path relative to the base dataset directory (default: `single_scan_floor_only/1a8384c3f6`).
  - `--frame`: Zero-based frame index (default: 0).

### 3. Three-Dataset Single-Frame Benchmark

- **Benchmark the first frame of each dataset** (outputs to `output/benchmark_single_frame/`):
  ```bash
  python -u src/benchmark_single_frame.py
  ```

### 4. Corrected Threshold Sensitivity Experiment

- **Threshold sensitivity with disjoint plane masks** (outputs to `output/fix_loop_threshold_experiment_corrected/`):
  ```bash
  python -u src/fix_loop_threshold_experiment.py
  ```
  - Tests RANSAC plane distance thresholds at 0.5%, 1.0%, 1.5%, and 2.0% of the depth range.
  - Reports plane coverage, residuals, angles, and median runtime.
  - **Important**: This experiment does **not** claim metric accuracy or label planes as floor/wall/ceiling.

## Output Directory Map

- `output/single_room_pipeline/` – point cloud (`point_cloud.npy`), visualization (`room_visualization.png`), and results (`result.json`) from the single-room pipeline.
- `output/benchmark_single_frame/` – benchmark summary JSON and CSV for the first frame of each dataset.
- `output/fix_loop_threshold_experiment_corrected/` – threshold sensitivity experiment results (`threshold_sensitivity.json` and `threshold_sensitivity.csv`). The interpretation and synthetic regression evidence are in `output/fix_loop/fix_loop_report.md`.
- Other experiment outputs (e.g., `output/plane_detection_experiment/`, `output/plane_threshold_sensitivity/`, `output/multiple_plane_experiment/`) contain intermediate exploratory results.

## Reproducibility and Evidence

- All reported numbers are generated from the raw input datasets via the provided scripts.
- Where randomness is involved (RANSAC), a fixed seed (42) is used for deterministic results.
- Scripts refuse to overwrite existing non‑empty output directories to protect prior results.
- The codebase does **not** include a capture pipeline or ground‑truth data; it only processes the supplied datasets.
- For a detailed mapping of experiment to command, input, and output, see `docs/reproduction.md`.

## Assignment Compliance Status

An evidence‑based compliance matrix is available at:
`docs/compliance_matrix.md`

This matrix maps the requirements from the assignment PDF to the current implementation, indicating which items have been demonstrated, partially demonstrated, or not demonstrated.

---
**Disclaimer**: This prototype is for educational and exploratory purposes only. 
It is not suitable for production use or critical measurements without further validation, 
ground‑truth calibration, and integration with a proper capture pipeline.
## Fix‑Loop Deliverable
A reproducible engineering fix‑loop for the plane‑mask overlap bug is documented in `output/fix_loop/fix_loop_report.md`. It includes the observed failure, root cause, synthetic test setup, buggy versus corrected results, and exact commands to reproduce the test and the threshold‑sensitivity experiment.


## 5. Demo wrapper for evaluator
A minimal PowerShell wrapper is provided to run the single-room pipeline on a supplied capture directory and produce a structured output directory with the required files and metadata.

Usage:
```powershell
.\Demo-RoomPipeline.ps1 -Dataset <relative_dataset_path> -Frame <frame_number> [-AssignmentRoot <path>]
```

Example:
```powershell
.\Demo-RoomPipeline.ps1 -Dataset single_scan_floor_only/1a8384c3f6 -Frame 0
```

Notes:
- The script must be run from the repository root. If -AssignmentRoot is omitted, it defaults to a sibling directory relative to the repo root (assuming the standard layout where the assignment data resides in `../Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup`).
- It will create an output directory under `output/demo_<dataset>_Frame<frame>` (path separators in the dataset are replaced with underscores) containing:
  - `result.json`
  - `point_cloud.npy`
  - `room_visualization.png`
  - `run_metadata.json`
- The wrapper does not modify the geometry logic; it reuses `src/single_room_pipeline.py`.
- Metadata includes dataset identifier, selected frame, input dimensions, verification flags, units, and known limitations.
- The script refuses to overwrite an existing non‑empty output directory (including hidden files and subdirectories).
- The wrapper does not claim clean‑machine portability has been verified; it is designed to be portable but assumes the presence of Python, .NET (for image dimensions), and the dataset structure.

## 6. Unified Output Contract

A minimal unified JSON output contract is defined to capture available data while honestly representing limitations and unimplemented fields.

- **JSON Schema**: `schemas/room_output.schema.json`
- **Example output** (based on the demo single-frame pipeline): `output/demo_single_scan_floor_only_1a8384c3f6_Frame0/room_output.json`

With `jsonschema` already installed, validate the preserved example without changing outputs:
```powershell
@'
import json
from pathlib import Path
from jsonschema import Draft7Validator
schema = json.loads(Path('schemas/room_output.schema.json').read_text(encoding='utf-8-sig'))
example = json.loads(Path('output/demo_single_scan_floor_only_1a8384c3f6_Frame0/room_output.json').read_text(encoding='utf-8-sig'))
Draft7Validator.check_schema(schema)
errors = list(Draft7Validator(schema).iter_errors(example))
if errors:
    raise SystemExit('\n'.join(error.message for error in errors))
print('schema validation passed')
'@ | python -
```

See `docs/output_contract.md` for a description of the contract structure and design principles.

