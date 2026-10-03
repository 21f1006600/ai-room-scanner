"""Inspect RGB-D sensor inputs for frame 000000 without reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_ASSIGNMENT_ROOT = Path(
    "D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup"
)
DEFAULT_DATASET = "single_scan_floor_only/1a8384c3f6"
DEFAULT_OUTPUT = Path("output/frame0_sensor_inspection")
FRAME_ID = "000000"


def json_value(value: Any) -> Any:
    """Convert NumPy scalar/container values to JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def numeric_stats(image: np.ndarray, percentiles=(1, 25, 50, 75, 99)) -> dict[str, Any]:
    values = image.astype(np.float64, copy=False).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": int(values.size), "finite_count": 0}
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "min": json_value(np.min(finite)),
        "max": json_value(np.max(finite)),
        "mean": json_value(np.mean(finite)),
        "std": json_value(np.std(finite)),
        "percentiles": {
            str(p): json_value(np.percentile(finite, p)) for p in percentiles
        },
    }


def image_description(image: np.ndarray) -> dict[str, Any]:
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return {
        "shape": list(image.shape),
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "channels": channels,
        "dtype": str(image.dtype),
        "statistics": numeric_stats(image),
    }


def csv_row_for_frame(path: Path, frame_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None
        for raw_row in reader:
            row = {(key or "").strip(): (value or "").strip() for key, value in raw_row.items()}
            if row.get("frame") == frame_id or row.get("frame", "").zfill(6) == frame_id:
                converted: dict[str, Any] = {}
                for key, value in row.items():
                    if key == "frame":
                        converted[key] = value
                    else:
                        try:
                            converted[key] = float(value) if value else None
                        except ValueError:
                            converted[key] = value
                return converted
    return None


def read_frame0_rgb(rgb_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(rgb_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open RGB video: {rgb_path}")
    ok, bgr = capture.read()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Could not read RGB frame 0: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, {
        "path": str(rgb_path),
        "frame_id": FRAME_ID,
        "frame_index": 0,
        "frame_count_metadata": frame_count,
        "fps_metadata": fps,
        **image_description(rgb),
    }


def inspect_depth(depth_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Could not read depth image: {depth_path}")
    if depth.ndim != 2:
        raise RuntimeError(f"Expected single-channel depth image, got shape {depth.shape}")
    zeros = depth == 0
    finite = np.isfinite(depth)
    saturated = depth == np.iinfo(depth.dtype).max if np.issubdtype(depth.dtype, np.integer) else np.zeros_like(depth, dtype=bool)
    valid = finite & ~zeros
    valid_values = depth[valid]
    stats = numeric_stats(depth)
    stats["valid_nonzero_statistics"] = numeric_stats(valid_values) if valid_values.size else {"count": 0}
    stats.update({
        "path": str(depth_path),
        "frame_id": FRAME_ID,
        "zero_count": int(np.count_nonzero(zeros)),
        "zero_percentage": float(np.mean(zeros) * 100),
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "saturation_value": int(np.iinfo(depth.dtype).max) if np.issubdtype(depth.dtype, np.integer) else None,
        "saturation_count": int(np.count_nonzero(saturated)),
        "saturation_percentage": float(np.mean(saturated) * 100),
        "unique_value_count": int(np.unique(depth).size),
        "valid_nonzero_count": int(np.count_nonzero(valid)),
        "valid_nonzero_percentage": float(np.mean(valid) * 100),
        "units": "raw image values; metric scale unknown",
    })
    return depth, {**image_description(depth), **stats}


def inspect_optional_image(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"available": False, "path": str(path) if path else None}
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"available": False, "path": str(path), "error": "unreadable"}
    result = {"available": True, "path": str(path), **image_description(image)}
    result["unique_values"] = [json_value(v) for v in np.unique(image)[:100]]
    result["unique_values_truncated"] = bool(np.unique(image).size > 100)
    return result


def make_visualization(rgb: np.ndarray, depth: np.ndarray, output_path: Path) -> None:
    masked_depth = np.ma.masked_where((depth == 0) | ~np.isfinite(depth), depth)
    valid = depth[(depth != 0) & np.isfinite(depth)]
    if valid.size:
        vmin, vmax = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))
        if vmin == vmax:
            vmin, vmax = float(np.min(valid)), float(np.max(valid))
    else:
        vmin, vmax = 0.0, 1.0
    figure, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB frame 000000")
    axes[0].set_xlabel("RGB pixels; not assumed aligned to depth")
    axes[0].set_axis_off()
    image = axes[1].imshow(masked_depth, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Depth frame 000000")
    axes[1].set_xlabel("Raw uint16 depth values; not metric units")
    axes[1].set_axis_off()
    colorbar = figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("Raw depth value (uint16; metric scale unknown)")
    figure.suptitle("Frame 0 RGB-D Sensor Inspection")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def markdown_report(report: dict[str, Any], command: str) -> str:
    rgb = report["rgb"]
    depth = report["depth"]
    calibration = report["calibration"]
    pose = report["pose_and_timing"]
    checks = report["invalid_and_saturation_checks"]
    lines = [
        "# Frame 0 RGB-D Sensor Inspection",
        "",
        "## Scope",
        "",
        "This is a read-only inspection of RGB and depth sensor inputs for frame `000000`. It does not perform point-cloud generation, plane detection, room reconstruction, or metric dimension extraction.",
        "",
        f"- Dataset: `{report['dataset']['identifier']}`",
        f"- Dataset path: `{report['dataset']['path']}`",
        f"- Frame: `{report['frame']['id']}`",
        "",
        "## Inputs",
        "",
        f"- RGB source: `{rgb['path']}`",
        f"- Depth source: `{depth['path']}`",
        f"- Confidence source: `{report['confidence']['path']}` (available: `{report['confidence']['available']}`)",
        f"- Calibration source: `{calibration['path']}`",
        "",
        "## RGB frame",
        "",
        f"- Dimensions: {rgb['width']} x {rgb['height']} pixels",
        f"- Channels: {rgb['channels']}",
        f"- Data type: `{rgb['dtype']}`",
        f"- Video metadata frame count: {rgb['frame_count_metadata']}",
        f"- Video metadata FPS: {rgb['fps_metadata']}",
        "",
        "## Depth frame",
        "",
        f"- Dimensions: {depth['width']} x {depth['height']} pixels",
        f"- Data type: `{depth['dtype']}`",
        f"- Raw minimum / median / maximum: {depth['statistics']['min']} / {depth['statistics']['percentiles']['50']} / {depth['statistics']['max']}",
        f"- Unique value count: {depth['unique_value_count']}",
        f"- Nonzero valid pixels: {depth['valid_nonzero_count']} ({depth['valid_nonzero_percentage']:.4f}%)",
        f"- Zero pixels: {checks['zero_count']} ({checks['zero_percentage']:.4f}%)",
        f"- Nonfinite pixels: {checks['nonfinite_count']}",
        f"- Saturation value: {checks['saturation_value']}",
        f"- Saturated pixels: {checks['saturation_count']} ({checks['saturation_percentage']:.4f}%)",
        "",
        "Depth values are reported as raw image values. They are not assumed to be millimetres or metres.",
        "",
        "## Calibration and timing/pose",
        "",
        f"- RGB resolution: {calibration['rgb_resolution'][0]} x {calibration['rgb_resolution'][1]} pixels",
        f"- Depth resolution: {calibration['depth_resolution'][0]} x {calibration['depth_resolution'][1]} pixels",
        f"- Camera matrix dimensions: {calibration['matrix_dimensions']}",
        f"- Camera matrix values: `{json.dumps(calibration['matrix'])}`",
        f"- Explicit RGB/depth identification: {calibration['explicit_sensor_identification']}",
        f"- Absent or unverified calibration: {', '.join(calibration['absent_or_unverified'])}",
        f"- Interpretation: {calibration['interpretation']}",
        f"- Frame-0 odometry row available: `{pose['odometry_row_available']}`",
        f"- Frame-0 odometry/timestamp data: `{json.dumps(pose['odometry_row'])}`",
        "",
        "The available camera matrix is treated conservatively as likely RGB intrinsics. Depth intrinsics, lens distortion, RGB-to-depth extrinsics, and depth scale are not independently verified. Any scaling of this matrix to depth resolution would be a hypothesis, not verified calibration.",
        "",
        "## RGB/depth alignment assessment",
        "",
        f"{report['alignment_assessment']['conclusion']}",
        "",
        "The RGB video frame index 0 and `depth/000000.png` are candidate frame-0 inputs, not an established synchronized pair. Temporal synchronization remains unverified unless actual sensor timestamps or documented synchronization evidence establish it. Matching frame IDs or equal frame counts do not establish pixel-to-pixel registration. The two images have different resolutions, and no verified RGB/depth extrinsic calibration or registration map is present.",
        "",
        "## Visualization",
        "",
        f"The generated visualization is `{report['generated_files']['visualization']}`. Its depth colorbar is labelled raw uint16 values and uses a display percentile range; this display range does not change the reported raw statistics.",
        "",
        "## Smallest next experiment toward one physical distance",
        "",
        "Place a rigid target with independently measured dimensions or two independently measured points in a controlled capture. Obtain verified depth intrinsics, RGB-to-depth extrinsics if RGB/depth overlay is needed, and the sensor's depth scale. Compare a recovered depth-derived distance against the tape-measured reference over repeated positions. Report error and repeatability before attempting room dimensions. This would establish evidence for one distance only, not a complete room plan.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += ["", "## Reproduction", "", "```powershell", command, "```", ""]
    return "\n".join(lines)


def build_report(
    dataset_path: Path,
    output_dir: Path,
    dataset_identifier: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, str]:
    rgb_path = dataset_path / "rgb.mp4"
    depth_path = dataset_path / "depth" / f"{FRAME_ID}.png"
    confidence_path = dataset_path / "confidence" / f"{FRAME_ID}.png"
    camera_path = dataset_path / "camera_matrix.csv"
    odometry_path = dataset_path / "odometry.csv"
    imu_path = dataset_path / "imu.csv"
    required = [rgb_path, depth_path, camera_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    rgb, rgb_info = read_frame0_rgb(rgb_path)
    depth, depth_info = inspect_depth(depth_path)
    confidence_info = inspect_optional_image(confidence_path)
    matrix = np.loadtxt(camera_path, delimiter=",")
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 camera matrix, got {matrix.shape}")
    odometry_row = csv_row_for_frame(odometry_path, FRAME_ID)
    imu_available = imu_path.is_file()
    command = (
        f"python -u src/frame0_sensor_inspection.py --dataset {dataset_identifier} "
        f"--assignment-root \"{DEFAULT_ASSIGNMENT_ROOT}\" --output-dir output/frame0_sensor_inspection"
    )
    report = {
        "schema": {"name": "frame0_sensor_inspection", "version": "1.0"},
        "dataset": {"identifier": dataset_identifier, "path": str(dataset_path)},
        "frame": {"id": FRAME_ID, "index": 0},
        "rgb": rgb_info,
        "depth": depth_info,
        "confidence": confidence_info,
        "calibration": {
            "path": str(camera_path),
            "rgb_resolution": [rgb_info["width"], rgb_info["height"]],
            "depth_resolution": [depth_info["width"], depth_info["height"]],
            "shape": list(matrix.shape),
            "matrix_dimensions": "3x3",
            "matrix": matrix.tolist(),
            "finite": bool(np.isfinite(matrix).all()),
            "explicit_sensor_identification": "Not explicitly identified in camera_matrix.csv; repository context suggests it is likely RGB calibration, not verified depth calibration.",
            "interpretation": "Likely RGB camera intrinsics based on repository metadata; depth intrinsics are not independently identified or verified.",
            "absent_or_unverified": [
                "Explicit RGB-versus-depth calibration designation",
                "Verified depth intrinsics",
                "Depth scale",
                "RGB-to-depth extrinsics",
                "Distortion coefficients/model",
                "Cropping/binning relationship",
            ],
            "depth_intrinsics_verified": False,
            "rgb_to_depth_extrinsics_available": False,
            "distortion_coefficients_available": False,
            "depth_scale_verified": False,
        },
        "pose_and_timing": {
            "odometry_path": str(odometry_path),
            "odometry_file_available": odometry_path.is_file(),
            "odometry_row_available": odometry_row is not None,
            "odometry_row": odometry_row,
            "imu_path": str(imu_path),
            "imu_file_available": imu_available,
            "timestamp_alignment_verified": False,
        },
        "invalid_and_saturation_checks": {
            "zero_count": depth_info["zero_count"],
            "zero_percentage": depth_info["zero_percentage"],
            "nonfinite_count": depth_info["nonfinite_count"],
            "saturation_value": depth_info["saturation_value"],
            "saturation_count": depth_info["saturation_count"],
            "saturation_percentage": depth_info["saturation_percentage"],
        },
        "alignment_assessment": {
            "conclusion": "RGB video frame index 0 and depth/000000.png are candidate frame-0 inputs. Their temporal synchronization is unverified without sensor timestamps or documented synchronization evidence, and pixel-level overlay or correspondence is not safe to assume.",
            "pixel_alignment_safe": False,
            "temporal_synchronization_verified": False,
            "candidate_frame_inputs": [
                "RGB video frame index 0.",
                "Depth frame file depth/000000.png.",
            ],
            "evidence_for_capture_association": [],
            "missing_evidence": [
                "Sensor timestamps or documented synchronization evidence establishing temporal correspondence.",
                "Verified RGB-to-depth extrinsic calibration.",
                "Verified depth intrinsics and distortion model.",
                "A registration map or documented synchronized pixel correspondence.",
            ],
        },
        "limitations": [
            "Depth values remain raw image values; no millimetre/metre conversion is claimed.",
            "The camera matrix is not independently established as depth calibration.",
            "RGB and depth have different resolutions and are not assumed pixel-aligned.",
            "Matching filenames, counts, or timestamps do not prove geometric registration.",
            "No RGB-to-depth extrinsics, depth scale, or distortion calibration was found in this inspection.",
            "This artifact does not estimate room dimensions or validate physical measurement accuracy.",
        ],
        "generated_files": {
            "visualization": str(output_dir / "frame0_rgb_depth.png"),
            "json": str(output_dir / "inspection_report.json"),
            "markdown": str(output_dir / "inspection_report.md"),
        },
        "reproduction_command": command,
    }
    return report, rgb, depth, command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--assignment-root", type=Path, default=DEFAULT_ASSIGNMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset_path = args.assignment_root / args.dataset
    output_dir = args.output_dir
    if output_dir.exists():
        try:
            has_entries = next(output_dir.iterdir(), None) is not None
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect output directory {output_dir}: {exc}") from exc
        if has_entries:
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    report, rgb, depth, command = build_report(dataset_path, output_dir, args.dataset)
    output_dir.mkdir(parents=True, exist_ok=False)
    make_visualization(rgb, depth, output_dir / "frame0_rgb_depth.png")
    (output_dir / "inspection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "inspection_report.md").write_text(markdown_report(report, command), encoding="utf-8")
    print(json.dumps({"dataset": str(dataset_path), "output_dir": str(output_dir), "files": report["generated_files"]}, indent=2))


if __name__ == "__main__":
    main()
