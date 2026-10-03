"""2D room-plan demo: readable visualization from the existing projection prototype."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PROTOTYPE_JSON = Path("output/room_plan_prototype/room_plan_prototype.json")
DEFAULT_POINT_CLOUD = Path(
    "output/single_room_pipeline/single_scan_floor_only/1a8384c3f6/point_cloud.npy"
)
DEFAULT_GEOMETRY_RESULT = Path(
    "output/single_room_pipeline/single_scan_floor_only/1a8384c3f6/result.json"
)
DEFAULT_OUTPUT_DIR = Path("output/room_plan_demo")

# Reuse selection criteria from prototype
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
    result = normal / np.linalg.norm(normal)
    if result[1] < 0 or (result[1] == 0 and tuple(result) < (0.0, 0.0, 0.0)):
        result = -result
    return result


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def build_demo_result(
    prototype_json: Path,
    point_cloud_path: Path,
    geometry_path: Path,
) -> dict[str, Any]:
    """Build the demo result by loading prototype output and re-projecting for demo data."""
    prototype = json.loads(prototype_json.read_text(encoding="utf-8"))
    points = np.load(point_cloud_path)
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))

    candidates = evaluate_candidates(points, geometry)
    eligible = [c for c in candidates if c["criteria_pass"]]
    eligible.sort(
        key=lambda c: (
            -c["vertical_alignment"],
            -c["inlier_percentage_from_equation"],
            c["median_residual_from_equation"],
            c["plane_id"],
        )
    )
    selected = eligible[0] if eligible else None

    if selected is None:
        return {
            "schema": {"name": "room_plan_demo", "version": "1.0"},
            "dataset": "single_scan_floor_only/1a8384c3f6",
            "frame": "000000",
            "selection_status": "unavailable",
            "selection_reason": "No candidate met deterministic horizontal-plane criteria.",
            "scale_status": "UNVERIFIED - NOT PHYSICAL DIMENSIONS",
            "limitations": [
                "No horizontal plane candidate selected.",
                "Point-cloud coordinates use raw depth-derived units; physical scale is unknown.",
                "Camera matrix and same-FOV RGB-to-depth scaling are not verified.",
                "Single frame cannot establish walls, openings, or room dimensions.",
            ],
        }

    # Re-project all candidate points for demo
    mask = selected["inlier_mask"]
    candidate_points = points[mask]
    normal, u, v = plane_basis(np.asarray(selected["equation"][:3], dtype=float))
    origin = np.mean(candidate_points, axis=0)
    projected_all = project_points(candidate_points, origin, u, v)
    hull = convex_hull(projected_all)

    # Subsample for visualization if needed
    max_vis_points = 8000
    if len(projected_all) > max_vis_points:
        idx = np.random.choice(len(projected_all), max_vis_points, replace=False)
        projected_vis = projected_all[idx]
    else:
        projected_vis = projected_all

    # Compute hull bbox for scale reference
    if len(hull) >= 2:
        u_range = float(hull[:, 0].max() - hull[:, 0].min())
        v_range = float(hull[:, 1].max() - hull[:, 1].min())
        hull_extent = max(u_range, v_range)
    else:
        hull_extent = 0.0

    result: dict[str, Any] = {
        "schema": {"name": "room_plan_demo", "version": "1.0"},
        "dataset": "single_scan_floor_only/1a8384c3f6",
        "frame": "000000",
        "input_artifacts": {
            "prototype_json": str(prototype_json),
            "point_cloud": str(point_cloud_path),
            "geometry_result": str(geometry_path),
        },
        "selection_criteria": SELECTION_CRITERIA,
        "candidate_evidence": [json_candidate(c) for c in candidates],
        "horizontal_plane_candidate": {
            **json_candidate(selected),
            "semantic_identity": "unclassified horizontal_plane_candidate",
        },
        "projection": {
            "origin_camera_raw": origin.tolist(),
            "normal_camera": normal.tolist(),
            "axis_u_camera": u.tolist(),
            "axis_v_camera": v.tolist(),
            "point_count": int(len(candidate_points)),
            "units": "raw / unverified projected coordinate units",
        },
        "observed_projected_hull": [
            {"u": float(p[0]), "v": float(p[1])} for p in hull
        ],
        "hull_extent_raw": hull_extent,
        "scale_status": "UNVERIFIED - NOT PHYSICAL DIMENSIONS",
        "wall_boundaries": {"status": "unavailable", "reason": "No validated wall segmentation or room-boundary evidence."},
        "room_dimensions": {"status": "unavailable", "reason": "Depth scale and architectural boundary identification are unverified."},
        "limitations": [
            "The selected label is a geometric horizontal_plane_candidate, not a floor classification.",
            "The observed_projected_hull is the convex hull of observed candidate-plane points, not a room boundary or room footprint.",
            "Point-cloud coordinates use raw depth-derived units; physical scale is unknown.",
            "The camera matrix and same-FOV RGB-to-depth scaling are not verified as depth calibration.",
            "A single frame cannot establish complete walls, openings, room dimensions, or reconstruction accuracy.",
        ],
        "selection_status": "selected",
    }
    return result


def create_demo_visualization(
    projected_all: np.ndarray,
    hull: np.ndarray,
    projection: dict[str, Any],
    candidate: dict[str, Any],
    output_path: Path,
) -> None:
    """Create the demo visualization with info panel."""
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)

    # Main plot
    ax = fig.add_subplot(gs[0])
    ax.scatter(
        projected_all[:, 0],
        projected_all[:, 1],
        s=1.5,
        alpha=0.3,
        color="tab:blue",
        label=f"Projected points ({len(projected_all):,})",
        rasterized=True,
    )
    if len(hull) >= 3:
        closed = np.vstack((hull, hull[0]))
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="tab:red",
            linewidth=2.5,
            label="observed_projected_hull",
        )
        # Mark hull vertices
        ax.scatter(hull[:, 0], hull[:, 1], s=30, color="tab:red", zorder=5, edgecolors="white", linewidths=0.5)

    # Coordinate axes at origin
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    # Axis labels
    ax.annotate(
        "U",
        xy=(projected_all[:, 0].max() * 0.9, 0),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        color="gray",
    )
    ax.annotate(
        "V",
        xy=(0, projected_all[:, 1].max() * 0.9),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=10,
        color="gray",
    )

    ax.set_xlabel("Projected axis U (raw / unverified units)")
    ax.set_ylabel("Projected axis V (raw / unverified units)")
    ax.set_title("2D Room-Plan Demo — Horizontal Plane Candidate Projection", fontsize=12, pad=15)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)

    # Scale status banner
    ax.text(
        0.5,
        1.02,
        "UNVERIFIED - NOT PHYSICAL DIMENSIONS",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="tab:red",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="tab:red", alpha=0.9),
    )

    # Info panel
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis("off")

    lines = []
    lines.append(("GEOMETRIC EVIDENCE", "header"))
    lines.append(("", "spacer"))
    lines.append((f"Candidate: Plane {candidate['plane_id']}", "key"))
    lines.append((f"Vertical alignment: {candidate['vertical_alignment']:.4f}", "val"))
    lines.append((f"Inlier %: {candidate['inlier_percentage_from_equation']:.2f}%", "val"))
    lines.append((f"Inlier count: {candidate['inlier_count_from_equation']:,}", "val"))
    lines.append((f"Median residual: {candidate['median_residual_from_equation']:.3f}", "val"))
    lines.append(("", "spacer"))
    lines.append(("PROJECTION", "header"))
    lines.append(("", "spacer"))
    lines.append((f"Points projected: {projection['point_count']:,}", "val"))
    lines.append((f"Hull vertices: {len(hull)}", "val"))
    lines.append((f"Hull extent (max): {projection.get('hull_extent_raw', 0):.1f} raw units", "val"))
    lines.append(("", "spacer"))
    lines.append(("AVAILABLE", "header"))
    lines.append(("", "spacer"))
    lines.append(("✓ Horizontal plane candidate", "ok"))
    lines.append(("✓ 2D projection (U,V)", "ok"))
    lines.append(("✓ Observed projected hull", "ok"))
    lines.append(("✓ Deterministic selection", "ok"))
    lines.append(("", "spacer"))
    lines.append(("UNAVAILABLE", "header"))
    lines.append(("", "spacer"))
    lines.append(("✗ Wall boundaries", "na"))
    lines.append(("✗ Room dimensions", "na"))
    lines.append(("✗ Floor/ceiling identity", "na"))
    lines.append(("✗ Physical scale (meters)", "na"))
    lines.append(("✗ Door/window detection", "na"))
    lines.append(("", "spacer"))
    lines.append(("SCALE STATUS", "header"))
    lines.append(("", "spacer"))
    lines.append(("UNVERIFIED", "warn"))
    lines.append(("NOT PHYSICAL DIMENSIONS", "warn"))

    y = 0.98
    for text, style in lines:
        if style == "header":
            ax_info.text(0.02, y, text, fontsize=10, fontweight="bold", va="top", transform=ax_info.transAxes)
            y -= 0.05
        elif style == "key":
            ax_info.text(0.02, y, text, fontsize=9, fontweight="bold", va="top", transform=ax_info.transAxes)
            y -= 0.045
        elif style == "val":
            ax_info.text(0.05, y, text, fontsize=9, va="top", transform=ax_info.transAxes)
            y -= 0.038
        elif style == "ok":
            ax_info.text(0.05, y, text, fontsize=9, color="tab:green", va="top", transform=ax_info.transAxes)
            y -= 0.038
        elif style == "na":
            ax_info.text(0.05, y, text, fontsize=9, color="tab:red", va="top", transform=ax_info.transAxes)
            y -= 0.038
        elif style == "warn":
            ax_info.text(0.02, y, text, fontsize=10, fontweight="bold", color="tab:red", va="top", transform=ax_info.transAxes)
            y -= 0.045
        elif style == "spacer":
            y -= 0.015

    fig.text(
        0.01,
        0.005,
        "Source: single_scan_floor_only/1a8384c3f6 frame 0 | Same-FOV intrinsics (unverified) | Raw depth units",
        fontsize=7,
        color="gray",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_unavailable_visualization(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    ax.text(
        0.5,
        0.6,
        "horizontal_plane_candidate unavailable",
        ha="center",
        va="center",
        fontsize=14,
        color="tab:red",
    )
    ax.text(
        0.5,
        0.4,
        "No plane met all deterministic criteria.",
        ha="center",
        va="center",
        fontsize=11,
        color="gray",
    )
    ax.text(
        0.5,
        0.25,
        "UNVERIFIED - NOT PHYSICAL DIMENSIONS",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="tab:red",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-json", type=Path, default=DEFAULT_PROTOTYPE_JSON)
    parser.add_argument("--point-cloud", type=Path, default=DEFAULT_POINT_CLOUD)
    parser.add_argument("--geometry-result", type=Path, default=DEFAULT_GEOMETRY_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--self-test", action="store_true", help="Run projection/hull validation only.")
    args = parser.parse_args()

    validation = run_projection_hull_validation()
    if args.self_test:
        print(json.dumps(validation, indent=2))
        return

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    result = build_demo_result(args.prototype_json, args.point_cloud, args.geometry_result)
    result["validation"] = validation

    result_path = args.output_dir / "room_plan_demo.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result["selection_status"] == "selected":
        candidate = result["horizontal_plane_candidate"]
        projection = result["projection"]
        points = np.load(args.point_cloud)
        equation = np.asarray(candidate["equation"], dtype=float)
        mask = plane_residuals(points, equation) < candidate["threshold_raw"]
        candidate_points = points[mask]
        normal, u, v = plane_basis(equation[:3])
        origin = np.mean(candidate_points, axis=0)
        projected_all = project_points(candidate_points, origin, u, v)
        hull = np.asarray([[p["u"], p["v"]] for p in result["observed_projected_hull"]])
        # Add hull extent to projection for display
        projection_for_viz = {**projection, "hull_extent_raw": result["hull_extent_raw"]}
        create_demo_visualization(
            projected_all,
            hull,
            projection_for_viz,
            candidate,
            args.output_dir / "room_plan_demo.png",
        )
    else:
        create_unavailable_visualization(args.output_dir / "room_plan_demo.png")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "json": str(result_path),
        "png": str(args.output_dir / "room_plan_demo.png"),
        "validation": validation["status"],
        "selection_status": result["selection_status"],
        "selected_plane_id": result.get("horizontal_plane_candidate", {}).get("plane_id"),
    }, indent=2))


if __name__ == "__main__":
    main()