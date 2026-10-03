# Applied AI Room Scanner Submission Bundle

This folder is a preserved submission bundle for the exploratory AI room-scanning assignment. It contains the documentation, reproducibility artifacts, regression test, unified output contract, selected source scripts, and preserved experiment outputs. It does **not** contain the original raw dataset.

## Exact demo command

Run from the bundle root in a fresh copy or another empty working copy, because the wrapper refuses to overwrite non-empty output directories:

```powershell
Set-Location "D:\brynz-applied-AI\ai-room-scanner\submission_bundle"
.\Demo-RoomPipeline.ps1 -Dataset single_scan_floor_only/1a8384c3f6 -Frame 0 -AssignmentRoot "D:\brynz-applied-AI\Assignment - YC Startup-20261001T152812Z-1-001\Assignment - YC Startup"
```

The preserved demonstration output is already in:

`output/demo_single_scan_floor_only_1a8384c3f6_Frame0/`

The wrapper creates `result.json`, `point_cloud.npy`, `room_visualization.png`, and `run_metadata.json` in the selected demo output directory. It does not create or overwrite the preserved output when that directory is non-empty.

## External dataset location

The raw assignment dataset is expected outside this bundle at:

`D:\brynz-applied-AI\Assignment - YC Startup-20261001T152812Z-1-001\Assignment - YC Startup`

The required dataset subdirectories are:

- `single_room/c00a170fe1`
- `single_scan_floor_only/1a8384c3f6`
- `single_scan_with_ceiling/c7d28f72c6`

Use `-AssignmentRoot` explicitly when the dataset is not in the wrapper's default sibling location.

## Bundle contents

- `README.md` — project overview, usage, limitations, and artifact map.
- `Demo-RoomPipeline.ps1` — supplied-data single-frame demo wrapper.
- `docs/` — final technical report, reproduction guide, compliance matrix, calibration findings, output contract, and extracted assignment requirements.
- `schemas/room_output.schema.json` — unified output schema.
- `output/fix_loop/` — fix-loop report, regression evidence, and preserved before-side artifacts.
- `output/fix_loop_threshold_experiment_corrected/` — corrected threshold-sensitivity JSON and CSV.
- `output/benchmark_single_frame/` — preserved three-dataset benchmark summaries.
- `output/demo_single_scan_floor_only_1a8384c3f6_Frame0/` — preserved demo output and metadata.
- `test_regression/` — synthetic mask-mapping regression tests.
- `src/` — only the Python scripts needed for the documented audits, geometry pipeline, diagnostics, benchmark, fix loop, and regression reproduction.

## Environment requirements

Development and artifact generation were performed on Windows 11 with Python 3.12. The selected scripts require:

- Python standard library
- NumPy
- OpenCV (`cv2`)
- Matplotlib for visualization-producing scripts
- `jsonschema` only for optional schema validation
- .NET `System.Drawing` is used by the PowerShell wrapper to read image dimensions; the wrapper records a limitation if this step is unavailable.

No `requirements.txt` or lock file is included. Dependencies have not been installed or verified on a clean machine.

## Key reproduction references

- Full experiment-to-command mapping: `docs/reproduction.md`
- Technical report: `docs/final_technical_report.md`
- Assignment status: `docs/compliance_matrix.md`
- Fix-loop explanation and evidence: `output/fix_loop/fix_loop_report.md`
- Schema description: `docs/output_contract.md`
- Schema: `schemas/room_output.schema.json`

## Known limitations

This bundle documents an exploratory single-frame RGB-D geometry prototype, not a complete validated room scanner. Depth values are raw uint16 units; depth scale, depth intrinsics, depth-to-RGB extrinsics, coordinate conventions, and odometry semantics are not independently verified. Same-FOV intrinsics scaling is an unverified assumption.

The bundle does not implement or verify the required capture tiers, metric floor plans, multi-room stitching and adjacency, damage classification, concealed-damage rules, surface-keyed scope, confidence intervals, laser/tape ground truth, accuracy or repeatability gates, drift correction ablation, consumer-app comparison, clean-machine setup, or walk-in evaluation. Point-cloud extents and plane residuals must not be interpreted as physical room measurements.

The fix-loop evidence proves disjoint plane-mask behavior on a deterministic synthetic fixture. It does not prove physical measurement accuracy. Clean-machine portability and end-to-end assignment completion are not claimed.
