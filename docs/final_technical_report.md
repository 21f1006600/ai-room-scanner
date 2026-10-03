# AI Room Scanner — Final Technical Report

**Status:** exploratory single-frame RGB-D geometry prototype; not a validated room-measurement or damage-assessment product.  
**Repository:** `D:\brynz-applied-AI\ai-room-scanner`  
**Dataset root:** `D:\brynz-applied-AI\Assignment - YC Startup-20261001T152812Z-1-001\Assignment - YC Startup`  
**Report scope:** work and artifacts present on disk as inspected on 2026-10-03.

## 1. Executive summary

The repository contains a reproducible exploratory pipeline for inspecting the supplied RGB-D captures, back-projecting one depth frame into a point cloud, detecting up to two candidate planes with RANSAC, and emitting JSON/CSV summaries and visualizations. It also contains calibration, timing, pose-convention, multiframe-concatenation, and threshold-sensitivity diagnostics. A unified JSON schema and a PowerShell demo wrapper are present.

The implementation does **not** meet the assignment's end-to-end acceptance requirements. In particular, no capture route or three-tier device matrix, metric scale, laser/tape ground truth, dimensioned floor plan, registered multi-room stitch, damage classifier, concealed-damage rule, surface-keyed scope, confidence intervals, repeatability/accuracy gates, consumer-app comparison, drift ablation, clean-machine test, or walk-in test is demonstrated. These are intentionally reported as incomplete rather than inferred from raw point-cloud extents.

The strongest completed engineering artifact is the fix-loop bundle: a deterministic synthetic regression test catches overlapping plane masks, and the shipped implementation restricts second-plane recomputation to the first-plane outlier subset. This proves mask disjointness for the fixture, not physical measurement accuracy.

## 2. Architecture and execution flow

The current flow is:

1. `src/data_loader.py` inspects dataset files, image properties, timestamps, IMU, and odometry.
2. `src/point_cloud.py` loads a uint16 depth PNG and RGB `camera_matrix.csv`; it optionally scales RGB intrinsics to the 256×192 depth image under an unverified same-FOV assumption, then applies the pinhole equations to produce `(X,Y,Z)` in raw depth units.
3. `src/single_room_pipeline.py` processes one selected frame, computes bounds and depth statistics, detects two candidate planes with seeded RANSAC, and writes a point cloud, visualization, and `result.json`.
4. `src/benchmark_single_frame.py` applies the same single-frame logic to the first frame of the three supplied datasets.
5. Diagnostic scripts examine timing, calibration evidence, quaternion conventions, rotation-only fusion, and unregistered multiframe concatenation.
6. `schemas/room_output.schema.json` and `output/demo_single_scan_floor_only_1a8384c3f6_Frame0/room_output.json` define a unified envelope. Unsupported product fields are explicitly `not_implemented`, `null`, or empty.

`Demo-RoomPipeline.ps1` selects a frame, refuses to overwrite a non-empty output directory, invokes the single-room pipeline, and writes `run_metadata.json`. It was inspected but not clean-machine or cwd-independent verified; the README still instructs running it from the repository root.

## 3. Dataset and calibration methodology

The three dataset identifiers are `single_room/c00a170fe1`, `single_scan_floor_only/1a8384c3f6`, and `single_scan_with_ceiling/c7d28f72c6`. The inspected data has 1920×1440 RGB video and 256×192 uint16 depth images. `camera_matrix.csv` is treated as RGB intrinsics; the audit reports focal lengths/principal point near 1600/1600/955/718. The repository has no independently verified depth intrinsics, depth-to-RGB extrinsics, depth scale, or coordinate-frame definition.

Consequently, the pipeline's scaled intrinsics and `Z = raw uint16 depth` are hypotheses. Coordinates follow the assumed camera convention X-right, Y-down, Z-forward. Odometry contains translations and quaternions, but their units, frame meaning, convention, and synchronization are not established. Matching file counts or frame IDs do not prove RGB/depth/pose synchronization. No result below is presented in centimetres, metres, or as a room dimension.

## 4. Experimental results

### 4.1 First-frame comparison

The existing `output/benchmark_single_frame/benchmark_summary.json` reports 49,152 valid points per frame. The following are **raw depth-derived units**; plane numbers are detection-order candidates, not floor/wall/ceiling labels.

| Dataset | X extent | Y extent | Z extent | Plane 1 | Plane 2 | Unassigned | Angle |
|---|---:|---:|---:|---:|---:|---:|---:|
| `single_room` | 1564.605 | 1206.665 | 1182 | 22,285 (45.34%) | 13,821 (28.12%) | 26.54% | 86.16° |
| `floor-only` | 1518.090 | 851.962 | 1688 | 24,319 (49.48%) | 15,105 (30.73%) | 19.79% | 88.37° |
| `with-ceiling` | 3044.360 | 2402.008 | 3730 | 26,741 (54.40%) | 13,766 (28.01%) | 17.59% | 76.50° |

The unified demo output for the floor-only frame records the same 49,152 points, raw depth range `[242, 849, 1930]` (minimum, median, maximum), raw axis-aligned XY proxy area `1,293,355.349`, and Z extent `1688`. The proxy is not a floor area or physical room measurement.

### 4.2 Unregistered multiframe experiment

`src/multiframe_geometry_experiment.py` back-projects frames 000000, 000030, 000060, and 000090 from `single_room` and concatenates them without odometry alignment. The artifact reports 196,608 total points, 195,226 unique rows after rounding to two decimals, combined extents `[1598.651, 1334.933, 1311]`, and approximately 0.160 seconds runtime. The unique-row count is only a duplicate diagnostic; this is not registered reconstruction, room coverage, drift measurement, or a physical footprint.

### 4.3 Pose/fusion diagnostic

For frames 000000 and 000001, the original camera-frame centroid distance is 4.7645 raw units. Rotation-only diagnostics produced centroid distances of 9.7584 (camera-to-world, offset 0), 8.8956 (camera-to-world, offset 1), 8.0416 (world-to-camera, offset 0), and 7.6223 (world-to-camera, offset 1). Translation was deliberately not applied because units and depth scale are unknown. These values cannot select a correct quaternion convention or prove alignment.

## 5. Reproducible fix loop

The fix-loop issue is a mask-accounting defect, not a measured benchmark gate. The buggy logic recomputed the second-plane mask over the full point cloud, allowing a point already assigned to plane 1 to be assigned again. The current `src/fix_loop_threshold_experiment.py` computes the second-plane mask only over `~refined_inliers1`, then maps it back to original indices. Its disjointness assertion rejects overlap or point-count mismatch.

The deterministic 11-point fixture in `test_regression/test_mask_mapping.py` uses two five-point planes and one ambiguous point with threshold 3.0. The preserved console artifact reports:

| Result | Fixed | Buggy full-cloud recomputation |
|---|---:|---:|
| Plane 1 inliers | 6 | 6 |
| Plane 2 inliers | 5 | 6 |
| Overlap | 0 | 1 |
| Unassigned | 0 | 0 |

All assertions pass. This isolates the index/mask bug; it does not establish scale, plane identity, accuracy, or a pass on the assignment's room gates. The original buggy source is unavailable because the directory has no Git history, so a source-level before/after diff cannot be claimed.

The corrected threshold experiment uses `single_scan_with_ceiling/c7d28f72c6`, frame 000000, seed 42, and three repetitions per threshold:

| Threshold | Raw threshold | P1 coverage | P2 coverage | Unassigned | P1 median residual | P2 median residual |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 18.65 | 49.34% | 21.90% | 28.76% | 2.66 | 2.58 |
| 1.0% | 37.30 | 54.40% | 28.08% | 17.52% | 4.52 | 15.71 |
| 1.5% | 55.95 | 58.00% | 28.54% | 13.46% | 9.27 | 16.80 |
| 2.0% | 74.60 | 60.86% | 28.15% | 10.99% | 11.67 | 17.47 |

Increasing the heuristic threshold assigns more points but increases residuals; it is not evidence of improved physical accuracy.

## 6. Assignment status and limitations

| Requirement area | Status in this repository |
|---|---|
| Three capture tiers, route, device matrix | Not implemented or demonstrated |
| Dimensioned per-room and stitched multi-room plans | Not implemented; no verified metric scale or adjacency |
| Damage regions/classes, concealed-damage rule, scope | Not implemented |
| Confidence interval on every measurement | Not implemented |
| Self-built benchmark, ground truth, repeated captures | Not present or verified |
| Opening, ceiling-height, repeatability, and drift gates | Not measured |
| Drift correction and on/off ablation | Not implemented |
| Consumer scan-app comparison | Not performed |
| One-command capture and clean-machine <15-minute setup | Not verified; wrapper operates on supplied data |
| JSON envelope and rendered exploratory output | Present, with unsupported fields explicitly marked |
| Reproducible mask fix loop | Present for the synthetic mask-mapping defect |

Known failure modes are unverified calibration and scale, unknown odometry conventions, lack of registration, single-frame plane ambiguity, heuristic threshold sensitivity, and the inability to distinguish architectural surfaces from arbitrary detected planes. The schema's validity, where used, means structural conformity only; it does not imply feature completeness or measurement validity.

## 7. Prioritized next steps

1. **Capture and calibration dependency:** choose and document the two permitted capture routes, implement photo/video/LiDAR ingestion, record a device matrix, and obtain independent depth intrinsics, extrinsics, scale, and coordinate conventions.
2. **Benchmark dependency:** capture three rooms plus a connector and a furnished staged-damage room at all tiers, repeat one room, and record laser/tape ground truth and raw exports.
3. **Geometry dependency:** register frames, estimate/validate poses, add drift correction and an ablation, identify walls/openings/ceiling, and produce metric plans with calibrated intervals and adjacency.
4. **Damage/output dependency:** implement damage classes and metric extents, an explicit concealed-damage rule, surface-keyed scope items, and confidence intervals in the unified contract.
5. **Evaluation dependency:** run all accuracy/repeatability gates, the two-room consumer-app comparison, the cold clean-machine test, and the walk-in test before making any product-level claim.

## 8. Exact reproduction commands

Run from `D:\brynz-applied-AI\ai-room-scanner` in a clean output location (or with a fresh copy of the repository); the scripts refuse non-empty destination directories:

```powershell
python -u src/benchmark_single_frame.py
python test_regression/test_mask_mapping.py
python -u src/fix_loop_threshold_experiment.py
python -u src/multiframe_geometry_experiment.py
python -u src/two_frame_fusion_check.py
.\Demo-RoomPipeline.ps1 -Dataset single_scan_floor_only/1a8384c3f6 -Frame 0
```

The first four commands reproduce the benchmark, synthetic regression, corrected threshold experiment, and unregistered multiframe artifact respectively. The fusion command reproduces the two-frame rotation-only diagnostic. The wrapper produces a new structured single-frame demo only when its target output directory is empty or absent; provide `-AssignmentRoot` explicitly if the standard sibling dataset layout is not available. Full artifact-to-command mapping is in `docs/reproduction.md`, and the requirement-by-requirement evidence map is in `docs/compliance_matrix.md`.
