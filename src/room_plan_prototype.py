"""Generate a conservative 2D projection prototype from an existing frame-0 cloud."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_POINT_CLOUD = Path(
    "output/single_room_pipeline/single_scan_floor_only/1a8384c3f6/point_cloud.npy"
)
DEFAULT_GEOMETRY_RESULT = Path(
    "output/single_room_pipeline/single_scan_floor_only/1a8384c3f6/result.json"
)
DEFAULT_OUTPUT_DIR = Path("output/room_plan_prototype")

# These are the complete deterministic selection criteria. They are recorded in
# the output before any candidate is selected.
SELECTION_CRITERIA = {
    "minimum_inlier_percentage": 25.0,
    "maximum_median_residual_fraction_of_threshold": 0.25,
    "minimum_vertical_alignment": 0.90,
    "vertical_axis": [0.0, 1.0, 0.0],
    "ranking": [
        "vertical_alignment descending",
        "inlier_percentage descending",
        "median_residual ascending",
        "plane_id ascending",
    ],
}


def plane_residuals(points: np.ndarray, equation: np.ndarray) -> np.ndarray:
    normal = equation[:3]
    return np.abs(points @ normal + equation[3])


def canonical_normal(normal: np.ndarray) -> np.ndarray:
    """Choose a deterministic sign with a positive camera-down component."""
    result = normal / np.linalg.norm(normal)
    if result[1] < 0 or (result[1] == 0 and tuple(result) < (0.0, 0.0, 0.0)):
        result = -result
    return result


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic normal, first in-plane axis, and second in-plane axis."""
    n = canonical_normal(normal)
    references = np.eye(3)
    reference = references[int(np.argmin(np.abs(references @ n)))]
    u = reference - np.dot(reference, n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return n, u, v


def project_points(points: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    relative = points - origin
    return np.column_stack((relative @ u, relative @ v))


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the monotonic-chain hull, including each corner once."""
    if points.shape[0] < 3:
        return points.copy()
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) < 3:
        return np.asarray(ordered, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def run_projection_hull_validation() -> dict[str, Any]:
    """Validate basis orthogonality, projection, and hull deterministically."""
    points_3d = np.asarray(
        [[-2.0, 5.0, -1.0], [2.0, 5.0, -1.0], [2.0, 5.0, 3.0], [-2.0, 5.0, 3.0]],
        dtype=float,
    )
    normal, u, v = plane_basis(np.array([0.0, 1.0, 0.0]))
    projected = project_points(points_3d, points_3d[0], u, v)
    hull = convex_hull(projected)
    assert np.isclose(np.linalg.norm(normal), 1.0)
    assert np.isclose(np.dot(normal, u), 0.0)
    assert np.isclose(np.dot(normal, v), 0.0)
    assert np.isclose(np.dot(u, v), 0.0)
    assert hull.shape == (4, 2)
    edge_a = hull[1] - hull[0]
    edge_b = hull[2] - hull[1]
    assert np.isclose(abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]), 16.0)
    return {
        "status": "passed",
        "checks": [
            "plane basis vectors are unit length and mutually orthogonal",
            "synthetic planar points project into two dimensions",
            "monotonic-chain hull returns four corners for a rectangle",
        ],
    }


def evaluate_candidates(points: np.ndarray, geometry: dict[str, Any]) -> list[dict[str, Any]]:
    threshold = float(np.ptp(points[:, 2]) * 0.01)
    threshold = max(threshold, 1.0)
    vertical = np.asarray(SELECTION_CRITERIA["vertical_axis"], dtype=float)
    # assigned_mask tracks points already claimed by earlier candidates to avoid double-counting.
    # Initialized to all False so each candidate is evaluated against the full cloud independently.
    assigned_mask = np.zeros(len(points), dtype=bool)
    candidates: list[dict[str, Any]] = []
    for plane in geometry.get("planes", []):
        equation = np.asarray(plane["equation"], dtype=float)
        normal = equation[:3] / np.linalg.norm(equation[:3])
        alignment = float(abs(np.dot(normal, vertical)))
        residual = plane_residuals(points, equation)
        inlier_mask = (residual < threshold) & (~assigned_mask)
        inlier_count = int(np.count_nonzero(inlier_mask))
        percentage = float(100.0 * inlier_count / len(points))
        inlier_residuals = residual[inlier_mask]
        median_residual = float(np.median(inlier_residuals)) if inlier_residuals.size else math.inf
        meets = (
            percentage >= SELECTION_CRITERIA["minimum_inlier_percentage"]
            and median_residual <= threshold * SELECTION_CRITERIA["maximum_median_residual_fraction_of_threshold"]
            and alignment >= SELECTION_CRITERIA["minimum_vertical_alignment"]
        )
        candidates.append(
            {
                "plane_id": int(plane["plane_id"]),
                "equation": equation.tolist(),
                "normal": normal.tolist(),
                "vertical_alignment": alignment,
                "inlier_count_from_equation": inlier_count,
                "inlier_percentage_from_equation": percentage,
                "median_residual_from_equation": median_residual,
                "threshold_raw": threshold,
                "criteria_pass": bool(meets),
                "criteria_checks": {
                    "inlier_percentage": percentage >= SELECTION_CRITERIA["minimum_inlier_percentage"],
                    "median_residual": median_residual <= threshold * SELECTION_CRITERIA["maximum_median_residual_fraction_of_threshold"],
                    "vertical_alignment": alignment >= SELECTION_CRITERIA["minimum_vertical_alignment"],
                },
                "inlier_mask": inlier_mask,
            }
        )
    return candidates


def json_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "inlier_mask"}


def create_visualization(
    projected: np.ndarray,
    hull: np.ndarray,
    output_path: Path,
    candidate_id: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(projected[:, 0], projected[:, 1], s=2, alpha=0.25, label="Candidate-plane points")
    if len(hull) >= 3:
        closed = np.vstack((hull, hull[0]))
        ax.plot(closed[:, 0], closed[:, 1], color="tab:red", linewidth=2, label="observed_projected_hull")
    ax.set_xlabel("Projected axis U (raw / unverified units)")
    ax.set_ylabel("Projected axis V (raw / unverified units)")
    ax.set_title(f"2D geometry prototype — horizontal_plane_candidate {candidate_id}")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.text(
        0.02,
        0.01,
        "Not a room boundary; scale, calibration, and surface identity are unverified.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_result(point_cloud_path: Path, geometry_path: Path) -> dict[str, Any]:
    points = np.load(point_cloud_path)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Expected a non-empty (N, 3) point cloud, got {points.shape}")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    candidates = evaluate_candidates(points, geometry)
    eligible = [candidate for candidate in candidates if candidate["criteria_pass"]]
    eligible.sort(
        key=lambda candidate: (
            -candidate["vertical_alignment"],
            -candidate["inlier_percentage_from_equation"],
            candidate["median_residual_from_equation"],
            candidate["plane_id"],
        )
    )
    selected = eligible[0] if eligible else None
    result: dict[str, Any] = {
        "schema": {"name": "room_plan_prototype", "version": "1.0"},
        "dataset": "single_scan_floor_only/1a8384c3f6",
        "frame": "000000",
        "input_artifacts": {
            "point_cloud": str(point_cloud_path),
            "geometry_result": str(geometry_path),
        },
        "selection_criteria": SELECTION_CRITERIA,
        "candidate_evidence": [json_candidate(candidate) for candidate in candidates],
        "horizontal_plane_candidate": None,
        "observed_projected_hull": [],
        "projection": None,
        "wall_boundaries": {"status": "unavailable", "reason": "No validated wall segmentation or room-boundary evidence."},
        "room_dimensions": {"status": "unavailable", "reason": "Depth scale and architectural boundary identification are unverified."},
        "limitations": [
            "The selected label is a geometric horizontal_plane_candidate, not a floor classification.",
            "The observed_projected_hull is the convex hull of observed candidate-plane points, not a room boundary or room footprint.",
            "Point-cloud coordinates use raw depth-derived units; physical scale is unknown.",
            "The camera matrix and same-FOV RGB-to-depth scaling are not verified as depth calibration.",
            "A single frame cannot establish complete walls, openings, room dimensions, or reconstruction accuracy.",
        ],
    }
    if selected is None:
        result["selection_status"] = "unavailable"
        result["selection_reason"] = "Neither plane met all deterministic horizontal-plane candidate criteria."
        return result

    mask = selected["inlier_mask"]
    candidate_points = points[mask]
    normal, u, v = plane_basis(np.asarray(selected["equation"][:3], dtype=float))
    origin = np.mean(candidate_points, axis=0)
    projected = project_points(candidate_points, origin, u, v)
    hull = convex_hull(projected)
    if len(hull) < 3:
        result["selection_status"] = "unavailable"
        result["selection_reason"] = "Candidate passed plane criteria but did not produce a non-degenerate projected hull."
        return result

    result["selection_status"] = "selected"
    result["horizontal_plane_candidate"] = {
        **json_candidate(selected),
        "semantic_identity": "unclassified horizontal_plane_candidate",
    }
    result["projection"] = {
        "origin_camera_raw": origin.tolist(),
        "normal_camera": normal.tolist(),
        "axis_u_camera": u.tolist(),
        "axis_v_camera": v.tolist(),
        "point_count": int(len(candidate_points)),
        "units": "raw / unverified projected coordinate units",
    }
    result["observed_projected_hull"] = [
        {"u": float(point[0]), "v": float(point[1])} for point in hull
    ]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud", type=Path, default=DEFAULT_POINT_CLOUD)
    parser.add_argument("--geometry-result", type=Path, default=DEFAULT_GEOMETRY_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true", help="Run projection and hull validation only.")
    args = parser.parse_args()

    validation = run_projection_hull_validation()
    if args.self_test:
        print(json.dumps(validation, indent=2))
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = build_result(args.point_cloud, args.geometry_result)
    result["validation"] = validation
    result_path = args.output_dir / "room_plan_prototype.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result["selection_status"] == "selected":
        candidate = result["horizontal_plane_candidate"]
        projected = np.asarray(
            [[point["u"], point["v"]] for point in result["observed_projected_hull"]],
            dtype=float,
        )
        # The visualization needs all projected points, not only hull vertices.
        points = np.load(args.point_cloud)
        equation = np.asarray(candidate["equation"], dtype=float)
        mask = plane_residuals(points, equation) < candidate["threshold_raw"]
        candidate_points = points[mask]
        normal, u, v = plane_basis(equation[:3])
        origin = np.mean(candidate_points, axis=0)
        all_projected = project_points(candidate_points, origin, u, v)
        del projected
        create_visualization(
            all_projected,
            np.asarray([[point["u"], point["v"]] for point in result["observed_projected_hull"]]),
            args.output_dir / "room_plan_prototype.png",
            int(candidate["plane_id"]),
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "horizontal_plane_candidate unavailable", ha="center", va="center")
        fig.savefig(args.output_dir / "room_plan_prototype.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "json": str(result_path),
        "png": str(args.output_dir / "room_plan_prototype.png"),
        "validation": validation["status"],
        "selection_status": result["selection_status"],
        "selected_plane_id": result.get("horizontal_plane_candidate", {}).get("plane_id"),
    }, indent=2))


if __name__ == "__main__":
    main()
