"""Run the existing single-frame analysis across the three supplied captures."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from benchmark_single_frame import process_frame
from data_loader import check_dataset_inputs, inspect_dataset_metadata

DEFAULT_ASSIGNMENT_ROOT = Path(
    "D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup"
)
DEFAULT_OUTPUT_DIR = Path("output/cross_tier_benchmark")
DATASET_IDS = (
    "single_room/c00a170fe1",
    "single_scan_floor_only/1a8384c3f6",
    "single_scan_with_ceiling/c7d28f72c6",
)
FRAME_ID = "000000"
SEED = 42


def raw_depth_statistics(depth: np.ndarray) -> dict[str, Any]:
    values = depth.astype(np.float64, copy=False).ravel()
    finite = values[np.isfinite(values)]
    nonzero = finite[finite > 0]
    saturation = int(np.iinfo(depth.dtype).max) if np.issubdtype(depth.dtype, np.integer) else None
    saturated_count = int(np.count_nonzero(depth == saturation)) if saturation is not None else 0
    return {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "pixel_count": int(depth.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
        "zero_count": int(np.count_nonzero(depth == 0)),
        "zero_percentage": float(np.mean(depth == 0) * 100),
        "nonzero_count": int(nonzero.size),
        "nonzero_percentage": float(nonzero.size / depth.size * 100),
        "min": float(np.min(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "nonzero_percentiles": {
            str(p): float(np.percentile(nonzero, p)) if nonzero.size else None
            for p in (1, 5, 25, 50, 75, 95, 99)
        },
        "uint16_saturation_value": saturation,
        "uint16_saturation_count": saturated_count,
        "uint16_saturation_percentage": float(saturated_count / depth.size * 100),
        "units": "raw uint16 depth-image values; physical scale is unknown",
    }


def frame_inventory(dataset_path: Path, checks: dict[str, Any]) -> dict[str, Any]:
    depth_dir = dataset_path / "depth"
    confidence_dir = dataset_path / "confidence"
    depth_stems = {path.stem for path in depth_dir.glob("*.png")} if depth_dir.is_dir() else set()
    confidence_stems = {path.stem for path in confidence_dir.glob("*.png")} if confidence_dir.is_dir() else set()
    return {
        "depth_png_count": int(checks.get("depth_png_count", 0)),
        "confidence_png_count": int(checks.get("confidence_png_count", 0)),
        "matched_depth_confidence_count": int(len(depth_stems & confidence_stems)),
        "depth_without_confidence_count": int(len(depth_stems - confidence_stems)),
        "confidence_without_depth_count": int(len(confidence_stems - depth_stems)),
        "frame_zero_depth": (dataset_path / "depth" / f"{FRAME_ID}.png").is_file(),
        "frame_zero_confidence": (dataset_path / "confidence" / f"{FRAME_ID}.png").is_file(),
    }


def validate_dataset(dataset_id: str, dataset_path: Path) -> dict[str, Any]:
    checks = check_dataset_inputs(dataset_path)
    required = ("rgb.mp4", "depth", "confidence", "camera_matrix.csv")
    missing = [name for name in required if not checks.get(name, False)]
    if missing:
        raise ValueError(f"{dataset_id}: missing required inputs: {', '.join(missing)}")
    inventory = frame_inventory(dataset_path, checks)
    if not inventory["frame_zero_depth"] or not inventory["frame_zero_confidence"]:
        raise ValueError(f"{dataset_id}: missing exact frame-0 depth/confidence pair")
    metadata = inspect_dataset_metadata(dataset_path)
    depth_path = dataset_path / "depth" / f"{FRAME_ID}.png"
    confidence_path = dataset_path / "confidence" / f"{FRAME_ID}.png"
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError(f"{dataset_id}: frame-0 depth must be readable 2D uint16")
    if confidence is None or confidence.ndim != 2 or confidence.shape != depth.shape:
        raise ValueError(f"{dataset_id}: frame-0 confidence must be readable and shape-matched")
    matrix = np.loadtxt(dataset_path / "camera_matrix.csv", delimiter=",")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{dataset_id}: camera_matrix.csv must be finite 3x3")
    # inspect_dataset_metadata returns this field as a NumPy array; normalize it before JSON output.
    metadata["camera_matrix"] = matrix.tolist()
    rgb_video = metadata.get("rgb_video", {})
    if "error" in rgb_video:
        raise ValueError(f"{dataset_id}: RGB video metadata unavailable: {rgb_video['error']}")
    return {
        "dataset_identifier": dataset_id,
        "dataset_path": str(dataset_path),
        "input_checks": checks,
        "frame_inventory": inventory,
        "metadata": metadata,
        "depth_statistics": raw_depth_statistics(depth),
        "confidence": {
            "shape": list(confidence.shape),
            "dtype": str(confidence.dtype),
            "unique_values": [int(v) for v in np.unique(confidence)],
            "semantics_verified": False,
            "semantic_note": "Confidence values are reported as supplied classes; their meaning is not independently verified.",
        },
        "camera_matrix": {
            "shape": list(matrix.shape),
            "values": matrix.tolist(),
            "explicit_sensor_identification": False,
            "interpretation": "Likely RGB intrinsics from resolution/principal-point context, but not verified as RGB or depth calibration and not independently verified for depth use.",
        },
        "depth_path": str(depth_path),
        "intrinsics_path": str(dataset_path / "camera_matrix.csv"),
    }


def validate_result(result: dict[str, Any], dataset_id: str) -> None:
    required = ("valid_point_count", "xyz_extent", "plane1", "plane2", "unassigned_percentage", "runtime_seconds")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"{dataset_id}: process_frame result missing {missing}")
    if result["valid_point_count"] <= 0 or len(result["xyz_extent"]) != 3:
        raise ValueError(f"{dataset_id}: invalid point-cloud result")
    p1 = result["plane1"]["inlier_count"]
    p2 = result["plane2"]["inlier_count"]
    if p1 < 0 or p2 < 0 or p1 + p2 > result["valid_point_count"]:
        raise ValueError(f"{dataset_id}: invalid plane inlier accounting")


def run_benchmark(assignment_root: Path) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    records: list[dict[str, Any]] = []
    for dataset_id in DATASET_IDS:
        dataset_path = assignment_root / Path(dataset_id)
        base = validate_dataset(dataset_id, dataset_path)
        start = time.perf_counter()
        result = process_frame(Path(base["depth_path"]), Path(base["intrinsics_path"]))
        elapsed = time.perf_counter() - start
        validate_result(result, dataset_id)
        # The reused function stores only the capture hash; retain the full configured identifier.
        result["dataset"] = dataset_id
        result["runtime_seconds_wall_clock"] = float(elapsed)
        base["analysis"] = result
        base["limitations"] = [
            "All depth and geometry quantities are raw units; no metric conversion is applied.",
            "Same-FOV intrinsics scaling is an unverified assumption inherited from the existing pipeline.",
            "Plane labels are detection-order candidates, not floor/wall/ceiling classifications.",
            "Runtime is environment-dependent and is not an accuracy or superiority gate.",
            "No physical ground truth, confidence interval, room plan, or capture-tier superiority claim is produced.",
        ]
        records.append(base)
    if len(records) != len(DATASET_IDS):
        raise ValueError("Benchmark completeness check failed: not all datasets produced records")
    return {
        "schema": {"name": "cross_tier_benchmark", "version": "1.0"},
        "benchmark": {
            "frame_id": FRAME_ID,
            "dataset_identifiers": list(DATASET_IDS),
            "analysis_scope": "same single-frame exploratory RGB-D geometry analysis across supplied scene captures",
            "not_validated_capture_tiers": True,
            "seed": SEED,
            "ransac_iterations": 1000,
            "ransac_threshold_rule": "max(0.01 * raw Z range, 1.0 raw units)",
            "intrinsics_mode": "existing pipeline same-FOV RGB-to-depth scaling; unverified",
            "depth_validity_rule": "depth > 0",
            "units": "raw uint16 depth-derived units; physical scale unknown",
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "records": records,
        "limitations": [
            "The three supplied captures are analyzed identically but are not independently validated capture tiers.",
            "Raw point-cloud extents and plane residuals must not be interpreted as physical room measurements.",
            "Depth scale, depth intrinsics, RGB-depth extrinsics, and ground truth are not verified.",
            "This benchmark does not produce room dimensions, damage classification, or accuracy gates.",
        ],
    }


def csv_rows(report: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "dataset_identifier", "frame_id", "depth_png_count", "confidence_png_count", "rgb_frame_count",
        "rgb_width", "rgb_height", "rgb_fps", "depth_width", "depth_height", "depth_dtype",
        "depth_min", "depth_median", "depth_max", "valid_point_count", "x_extent", "y_extent", "z_extent",
        "ransac_threshold", "plane1_inlier_count", "plane1_inlier_percentage", "plane1_mean_residual",
        "plane1_median_residual", "plane2_inlier_count", "plane2_inlier_percentage", "plane2_mean_residual",
        "plane2_median_residual", "unassigned_percentage", "angular_diff_deg", "runtime_seconds",
    ]
    rows = []
    for record in report["records"]:
        metadata = record["metadata"]
        rgb = metadata["rgb_video"]
        depth = record["depth_statistics"]
        analysis = record["analysis"]
        row = {
            "dataset_identifier": record["dataset_identifier"], "frame_id": FRAME_ID,
            "depth_png_count": record["frame_inventory"]["depth_png_count"],
            "confidence_png_count": record["frame_inventory"]["confidence_png_count"],
            "rgb_frame_count": rgb.get("frame_count"), "rgb_width": rgb.get("width"), "rgb_height": rgb.get("height"),
            "rgb_fps": rgb.get("fps"), "depth_width": depth["shape"][1], "depth_height": depth["shape"][0],
            "depth_dtype": depth["dtype"], "depth_min": depth["min"], "depth_median": depth["median"], "depth_max": depth["max"],
            "valid_point_count": analysis["valid_point_count"], "x_extent": analysis["xyz_extent"][0],
            "y_extent": analysis["xyz_extent"][1], "z_extent": analysis["xyz_extent"][2], "ransac_threshold": analysis["ransac_threshold"],
            "plane1_inlier_count": analysis["plane1"]["inlier_count"], "plane1_inlier_percentage": analysis["plane1"]["inlier_percentage"],
            "plane1_mean_residual": analysis["plane1"]["mean_residual"], "plane1_median_residual": analysis["plane1"]["median_residual"],
            "plane2_inlier_count": analysis["plane2"]["inlier_count"], "plane2_inlier_percentage": analysis["plane2"]["inlier_percentage"],
            "plane2_mean_residual": analysis["plane2"]["mean_residual"], "plane2_median_residual": analysis["plane2"]["median_residual"],
            "unassigned_percentage": analysis["unassigned_percentage"], "angular_diff_deg": analysis["angular_diff_deg"],
            "runtime_seconds": analysis["runtime_seconds"],
        }
        rows.append(row)
    return fields, rows


def markdown_report(report: dict[str, Any], command: str) -> str:
    lines = [
        "# Cross-Tier Benchmark",
        "",
        "This report applies the same existing single-frame exploratory RGB-D geometry analysis to three supplied captures. The captures are reported as dataset/scene variants, not validated capture tiers, and no superiority claim is made.",
        "",
        f"- Frame: `{FRAME_ID}`",
        "- Geometry units: raw uint16 depth-derived units; physical scale unknown",
        f"- Seed: `{SEED}`; RANSAC iterations: `1000`",
        "- Threshold: `max(0.01 * raw Z range, 1.0 raw units)`",
        "",
        "## Results",
        "",
        "| Dataset | Frames | RGB | Depth | Points | XYZ extent (raw) | Plane 1 | Plane 2 | Unassigned | Runtime (s) |",
        "|---|---:|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for record in report["records"]:
        analysis = record["analysis"]
        metadata = record["metadata"]
        rgb = metadata["rgb_video"]
        ext = ", ".join(f"{value:.3f}" for value in analysis["xyz_extent"])
        lines.append(f"| `{record['dataset_identifier']}` | {record['frame_inventory']['depth_png_count']} | {rgb['width']}x{rgb['height']} | {record['depth_statistics']['shape'][1]}x{record['depth_statistics']['shape'][0]} | {analysis['valid_point_count']} | [{ext}] | {analysis['plane1']['inlier_count']} ({analysis['plane1']['inlier_percentage']:.2f}%) | {analysis['plane2']['inlier_count']} ({analysis['plane2']['inlier_percentage']:.2f}%) | {analysis['unassigned_percentage']:.2f}% | {analysis['runtime_seconds']:.3f} |")
    lines += [
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines += ["", "## Reproduction", "", "```powershell", command, "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-root", type=Path, default=DEFAULT_ASSIGNMENT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists() and next(args.output_dir.iterdir(), None) is not None:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    report = run_benchmark(args.assignment_root)
    command = f"python -u src/cross_tier_benchmark.py --assignment-root \"{args.assignment_root}\" --output-dir output/cross_tier_benchmark"
    report["reproduction_command"] = command
    fields, rows = csv_rows(report)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "cross_tier_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "cross_tier_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "cross_tier_benchmark.md").write_text(markdown_report(report, command), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "files": [str(args.output_dir / name) for name in ("cross_tier_benchmark.json", "cross_tier_benchmark.csv", "cross_tier_benchmark.md")]}, indent=2))


if __name__ == "__main__":
    main()
