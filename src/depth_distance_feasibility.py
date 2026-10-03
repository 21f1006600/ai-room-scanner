"""Assess whether the supplied depth stream supports one physical distance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_ASSIGNMENT_ROOT = Path(
    "D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup"
)
DEFAULT_DATASET = "single_scan_floor_only/1a8384c3f6"
DEFAULT_OUTPUT = Path("output/depth_distance_feasibility")
DEFAULT_FRAMES = "0,30,60,90"


REFERENCE_TERMS = ("ground truth", "measured", "tape", "laser", "calibration target", "known size", "reference distance")


def stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    flat = values.astype(np.float64, copy=False).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"count": int(flat.size), "finite_count": 0}
    return {
        "count": int(flat.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "percentiles": {
            "1": float(np.percentile(finite, 1)),
            "5": float(np.percentile(finite, 5)),
            "25": float(np.percentile(finite, 25)),
            "50": float(np.percentile(finite, 50)),
            "75": float(np.percentile(finite, 75)),
            "95": float(np.percentile(finite, 95)),
            "99": float(np.percentile(finite, 99)),
        },
    }


def parse_frame_ids(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value < 0 for value in values):
        raise ValueError("--frames must contain one or more non-negative integers")
    return values


def depth_frame_report(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read depth image: {path}")
    zero = image == 0
    finite = np.isfinite(image)
    saturation_value = int(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else None
    saturated = image == saturation_value if saturation_value is not None else np.zeros_like(image, dtype=bool)
    return {
        "path": str(path),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "statistics": stats(image),
        "zero_count": int(np.count_nonzero(zero)),
        "zero_fraction": float(np.mean(zero)),
        "nonfinite_count": int(np.count_nonzero(~finite)),
        "saturation_value": saturation_value,
        "saturation_count": int(np.count_nonzero(saturated)),
        "saturation_fraction": float(np.mean(saturated)),
        "unique_value_count": int(np.unique(image).size),
        "units": "raw image values; depth scale is not verified",
    }


def confidence_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"available": False, "path": str(path), "error": "unreadable"}
    values, counts = np.unique(image, return_counts=True)
    total = int(image.size)
    return {
        "available": True,
        "path": str(path),
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "unique_values": [int(value) for value in values],
        "counts": [int(count) for count in counts],
        "fractions": [float(count / total) for count in counts],
        "semantics_verified": False,
        "semantic_note": "Values are reported as unlabeled companion-map classes; their probability or quality meaning is not established by the supplied data.",
    }


def read_camera_matrix(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False, "path": str(path)}
    matrix = np.loadtxt(path, delimiter=",")
    return {
        "available": True,
        "path": str(path),
        "shape": list(matrix.shape),
        "values": matrix.tolist(),
        "explicit_sensor_identification": False,
        "interpretation": "The file does not identify RGB versus depth calibration. Repository context suggests it is likely RGB intrinsics, but that is a hypothesis and it is not used as depth calibration.",
        "depth_scale_verified": False,
        "depth_intrinsics_verified": False,
        "rgb_to_depth_extrinsics_available": False,
        "distortion_model_available": False,
    }


def frame_row(path: Path, frame_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row = {(key or "").strip(): (value or "").strip() for key, value in raw.items()}
            if row.get("frame", "").zfill(6) == frame_id:
                return row
    return None


def search_reference_evidence(repo_root: Path, dataset_path: Path) -> dict[str, Any]:
    inspected_paths: list[str] = []
    matches: list[dict[str, str]] = []
    candidates = [repo_root / "README.md", repo_root / "docs", dataset_path]
    for root in candidates:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in paths:
            if path.suffix.lower() not in {".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".log"}:
                continue
            inspected_paths.append(str(path))
            try:
                text = path.read_text(encoding="utf-8-sig", errors="ignore")
            except OSError:
                continue
            lowered = text.lower()
            for term in REFERENCE_TERMS:
                if term in lowered:
                    matches.append({"path": str(path), "term": term})
    return {
        "searched_scopes": [str(repo_root / "README.md"), str(repo_root / "docs"), str(dataset_path)],
        "inspected_text_file_count": len(inspected_paths),
        "matches": matches[:100],
        "independent_physical_reference_found": bool(matches),
        "interpretation": "No independent physical reference was found in the inspected text and dataset scope." if not matches else "Keyword matches require manual review; a keyword is not itself proof of an independently measured reference.",
    }


def build_report(repo_root: Path, dataset_path: Path, dataset_identifier: str, frame_ids: list[int], output_dir: Path) -> dict[str, Any]:
    depth_dir = dataset_path / "depth"
    confidence_dir = dataset_path / "confidence"
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    selected: list[dict[str, Any]] = []
    missing_frames: list[str] = []
    for frame_id in frame_ids:
        name = f"{frame_id:06d}.png"
        depth_path = depth_dir / name
        if not depth_path.is_file():
            missing_frames.append(name)
            continue
        selected.append({
            "frame_id": f"{frame_id:06d}",
            "depth": depth_frame_report(depth_path),
            "confidence": confidence_report(confidence_dir / name),
        })
    if not selected:
        raise FileNotFoundError("None of the requested depth frames exist")

    odometry_path = dataset_path / "odometry.csv"
    report = {
        "schema": {"name": "depth_distance_feasibility", "version": "1.0"},
        "dataset": {
            "identifier": dataset_identifier,
            "path": str(dataset_path),
            "requested_frame_ids": [f"{value:06d}" for value in frame_ids],
            "missing_requested_frames": missing_frames,
            "selected_frame_ids": [item["frame_id"] for item in selected],
        },
        "sampled_frames": selected,
        "available_metadata": {
            "camera_matrix": read_camera_matrix(dataset_path / "camera_matrix.csv"),
            "odometry_path": str(odometry_path),
            "odometry_frame_zero": frame_row(odometry_path, "000000"),
            "imu_available": (dataset_path / "imu.csv").is_file(),
            "rgb_video_available": (dataset_path / "rgb.mp4").is_file(),
        },
        "verified_facts": [
            "The selected depth files are uint16 PNG images containing raw integer values.",
            "Confidence files, when present, are companion PNG maps with observed integer classes; their semantics are not verified.",
            "The supplied materials do not provide a verified depth scale in physical units.",
            "The supplied materials do not provide verified depth intrinsics or RGB-to-depth extrinsics.",
        ],
        "hypotheses_not_used_as_facts": [
            "The camera_matrix.csv file may be RGB calibration based on resolution/principal-point context, but it is not explicitly identified and is not used as depth calibration.",
            "Matching frame identifiers may indicate capture indexing, but do not prove temporal synchronization or physical correspondence.",
            "Confidence class values may indicate quality categories, but they are not treated as probabilities or calibrated confidence.",
        ],
        "physical_reference_search": search_reference_evidence(repo_root, dataset_path),
        "metric_distance_verdict": {
            "defensible": False,
            "conclusion": "Metric distance recovery is not defensible from the current evidence. The experiment can characterize raw depth variation across sampled frames, but it cannot validate one physical distance without verified depth scale/calibration and an independently measured reference.",
            "smallest_missing_prerequisite": "A controlled capture containing one independently measured physical distance or known-size rigid target, together with verified depth intrinsics and depth scale.",
        },
        "room_dimension_feasibility": {
            "can_reasonably_complete_with_available_data": False,
            "reason": "The current data supports raw depth/confidence auditing only. It lacks the verified scale, depth calibration, physical reference, registered reconstruction, architectural surface identification, and ground-truth gates required for defensible room dimensions.",
            "remaining_deadline_assessment": "A room-dimension pipeline cannot be credibly completed from the available evidence alone within a remaining deadline unless new calibrated captures and ground truth are supplied; exploratory raw outputs should not be presented as metric dimensions.",
        },
        "limitations": [
            "No point clouds, plane fits, RGB-depth overlays, or room dimensions are computed by this experiment.",
            "Depth values are not converted to millimetres, centimetres, or metres.",
            "The physical-reference search covers repository README/docs text and readable text-like files under the selected dataset; absence there does not rule out an external reference not included in those materials.",
            "No accuracy or repeatability claim is made.",
        ],
        "generated_files": {
            "json": str(output_dir / "depth_distance_feasibility.json"),
            "markdown": str(output_dir / "depth_distance_feasibility.md"),
        },
    }
    return report


def markdown(report: dict[str, Any], command: str) -> str:
    verdict = report["metric_distance_verdict"]
    feasibility = report["room_dimension_feasibility"]
    lines = [
        "# Depth Distance Feasibility",
        "",
        "## Conclusion",
        "",
        f"**Metric distance recovery defensible now:** `{verdict['defensible']}`",
        "",
        verdict["conclusion"],
        "",
        f"**Smallest missing prerequisite:** {verdict['smallest_missing_prerequisite']}",
        "",
        "## Experiment",
        "",
        f"- Dataset: `{report['dataset']['identifier']}`",
        f"- Sampled frames: `{', '.join(report['dataset']['selected_frame_ids'])}`",
        "- Operation: raw depth/confidence inspection only; no back-projection or reconstruction.",
        "",
        "## Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in report["verified_facts"])
    lines += ["", "## Hypotheses deliberately not used as facts", ""]
    lines.extend(f"- {item}" for item in report["hypotheses_not_used_as_facts"])
    lines += ["", "## Physical reference search", "", f"- Independent reference found in inspected scope: `{report['physical_reference_search']['independent_physical_reference_found']}`", f"- {report['physical_reference_search']['interpretation']}", ""]
    lines += ["## Room-dimension feasibility", "", f"- Can reasonably complete from current data: `{feasibility['can_reasonably_complete_with_available_data']}`", f"- {feasibility['reason']}", f"- Deadline assessment: {feasibility['remaining_deadline_assessment']}", "", "## Reproduction", "", "```powershell", command, "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--assignment-root", type=Path, default=DEFAULT_ASSIGNMENT_ROOT)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists() and next(args.output_dir.iterdir(), None) is not None:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    frame_ids = parse_frame_ids(args.frames)
    dataset_path = args.assignment_root / args.dataset
    report = build_report(Path(__file__).resolve().parents[1], dataset_path, args.dataset, frame_ids, args.output_dir)
    command = f"python -u src/depth_distance_feasibility.py --dataset {args.dataset} --assignment-root \"{args.assignment_root}\" --frames {args.frames} --output-dir output/depth_distance_feasibility"
    report["reproduction_command"] = command
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "depth_distance_feasibility.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "depth_distance_feasibility.md").write_text(markdown(report, command), encoding="utf-8")
    print(json.dumps(report["generated_files"], indent=2))


if __name__ == "__main__":
    main()
