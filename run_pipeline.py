#!/usr/bin/env python3
"""
Unified one-command pipeline for the room-scanner project.

Processes three capture tiers through existing components:
  - single_room/c00a170fe1
  - single_scan_floor_only/1a8384c3f6
  - single_scan_with_ceiling/c7d28f72c6

Generates consolidated output under output/final_demo/ with:
  - Per-tier processing status and results
  - Cross-tier benchmark comparison
  - Room-plan prototype and demo outputs (where supported)
  - Machine-readable summary.json and human-readable summary.txt

All quantities remain in raw depth units; no physical measurements or accuracy claims.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import existing processing functions
from point_cloud import backproject_depth
from benchmark_single_frame import process_frame
from data_loader import check_dataset_inputs, inspect_dataset_metadata
from room_plan_prototype import (
    build_result as build_prototype_result,
    run_projection_hull_validation,
)
from room_plan_demo import build_demo_result
from integrated_damage_scope import run_integrated_damage_and_scope, save_integrated_result


# Configuration
ASSIGNMENT_ROOT = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
DATASET_IDS = (
    "single_room/c00a170fe1",
    "single_scan_floor_only/1a8384c3f6",
    "single_scan_with_ceiling/c7d28f72c6",
)
OUTPUT_DIR = Path("output/final_demo_run2")
FRAME_ID = "000000"


def run_single_room_pipeline(dataset_id: str, dataset_path: Path, tier_output_dir: Path) -> dict[str, Any]:
    """Run single-room pipeline for a dataset, saving point cloud and results."""
    depth_dir = dataset_path / "depth"
    intrinsics_path = dataset_path / "camera_matrix.csv"
    depth_path = depth_dir / f"{FRAME_ID}.png"

    if not depth_path.is_file():
        return {"status": "failed", "error": f"Depth image not found: {depth_path}"}
    if not intrinsics_path.is_file():
        return {"status": "failed", "error": f"Camera matrix not found: {intrinsics_path}"}

    try:
        # Back-project
        points = backproject_depth(
            depth_path,
            intrinsics_path,
            use_scaled_intrinsics=True
        )
        if points.size == 0:
            return {"status": "failed", "error": "No valid points found"}

        # Plane detection (reuse logic from benchmark_single_frame)
        result = process_frame(depth_path, intrinsics_path)

        # Save point cloud
        point_cloud_path = tier_output_dir / "point_cloud.npy"
        np.save(point_cloud_path, points)

        # Save result.json
        result_json_path = tier_output_dir / "result.json"
        result_json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "point_cloud": str(point_cloud_path),
            "result_json": str(result_json_path),
            "valid_point_count": result["valid_point_count"],
            "xyz_extent": result["xyz_extent"],
            "plane1": result["plane1"],
            "plane2": result["plane2"],
            "unassigned_percentage": result["unassigned_percentage"],
            "angular_diff_deg": result["angular_diff_deg"],
        }
    except Exception as e:
        return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}


def run_room_plan_prototype(dataset_id: str, tier_output_dir: Path, final_output_dir: Path) -> dict[str, Any]:
    """Run room-plan prototype for a dataset if it has the required inputs."""
    point_cloud_path = tier_output_dir / "point_cloud.npy"
    geometry_path = tier_output_dir / "result.json"

    if not point_cloud_path.is_file() or not geometry_path.is_file():
        return {"status": "skipped", "reason": "Required inputs not found (point_cloud.npy or result.json)"}

    try:
        # Load geometry and convert to format expected by prototype (planes list)
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        if "planes" not in geometry:
            # Convert from process_frame format (plane1, plane2) to prototype format (planes list)
            planes = []
            for key in ("plane1", "plane2"):
                if key in geometry and geometry[key].get("inlier_count", 0) > 0:
                    p = geometry[key]
                    planes.append({
                        "plane_id": 1 if key == "plane1" else 2,
                        "equation": p.get("equation", [0,0,0,0]),
                        "normal": p.get("normal", [0,0,0]),
                        "inlier_count": p.get("inlier_count", 0),
                        "inlier_percentage": p.get("inlier_percentage", 0.0),
                        "mean_residual": p.get("mean_residual", 0.0),
                        "median_residual": p.get("median_residual", 0.0),
                    })
            geometry["planes"] = planes
            # Write converted geometry for prototype
            geometry_path.write_text(json.dumps(geometry, indent=2), encoding="utf-8")

        prototype_output = final_output_dir / "room_plan_prototype" / dataset_id.replace("/", "_")
        prototype_output.mkdir(parents=True, exist_ok=True)

        result = build_prototype_result(point_cloud_path, geometry_path)
        result["validation"] = run_projection_hull_validation()

        # Save JSON
        json_path = prototype_output / "room_plan_prototype.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        # Generate visualization
        from room_plan_prototype import create_visualization, plane_residuals, plane_basis, project_points
        points = np.load(point_cloud_path)
        if result["selection_status"] == "selected":
            candidate = result["horizontal_plane_candidate"]
            equation = np.asarray(candidate["equation"], dtype=float)
            mask = plane_residuals(points, equation) < candidate["threshold_raw"]
            candidate_points = points[mask]
            normal, u, v = plane_basis(equation[:3])
            origin = np.mean(candidate_points, axis=0)
            all_projected = project_points(candidate_points, origin, u, v)
            hull = np.asarray([[p["u"], p["v"]] for p in result["observed_projected_hull"]])
            create_visualization(all_projected, hull, prototype_output / "room_plan_prototype.png", int(candidate["plane_id"]))
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.axis("off")
            ax.text(0.5, 0.5, "horizontal_plane_candidate unavailable", ha="center", va="center")
            fig.savefig(prototype_output / "room_plan_prototype.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        hpc = result.get("horizontal_plane_candidate") or {}
        return {
            "status": "success" if result["selection_status"] == "selected" else "unavailable",
            "selection_status": result["selection_status"],
            "selected_plane_id": hpc.get("plane_id"),
            "hull_vertices": len(result.get("observed_projected_hull", [])),
            "output_dir": str(prototype_output),
        }
    except Exception as e:
        return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}


def run_room_plan_demo(dataset_id: str, tier_output_dir: Path, final_output_dir: Path) -> dict[str, Any]:
    """Run room-plan demo for a dataset if it has the required inputs."""
    point_cloud_path = tier_output_dir / "point_cloud.npy"
    geometry_path = tier_output_dir / "result.json"

    if not point_cloud_path.is_file() or not geometry_path.is_file():
        return {"status": "skipped", "reason": "Required inputs not found"}

    try:
        # Ensure geometry has the expected "planes" list format
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        if "planes" not in geometry:
            planes = []
            for key in ("plane1", "plane2"):
                if key in geometry and geometry[key].get("inlier_count", 0) > 0:
                    p = geometry[key]
                    planes.append({
                        "plane_id": 1 if key == "plane1" else 2,
                        "equation": p.get("equation", [0,0,0,0]),
                        "normal": p.get("normal", [0,0,0]),
                        "inlier_count": p.get("inlier_count", 0),
                        "inlier_percentage": p.get("inlier_percentage", 0.0),
                        "mean_residual": p.get("mean_residual", 0.0),
                        "median_residual": p.get("median_residual", 0.0),
                    })
            geometry["planes"] = planes
            geometry_path.write_text(json.dumps(geometry, indent=2), encoding="utf-8")

        demo_output = final_output_dir / "room_plan_demo" / dataset_id.replace("/", "_")
        demo_output.mkdir(parents=True, exist_ok=True)

        # Use a minimal prototype-like JSON for criteria (demo re-evaluates candidates anyway)
        prototype_json_path = final_output_dir / "room_plan_prototype" / dataset_id.replace("/", "_") / "room_plan_prototype.json"
        if prototype_json_path.is_file():
            # Load existing prototype for consistency
            result = build_demo_result(prototype_json_path, point_cloud_path, geometry_path)
        else:
            # Build a minimal prototype JSON with just the criteria
            import json as _json
            minimal_proto = {
                "selection_criteria": {
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
            }
            temp_proto_path = demo_output / "_temp_proto.json"
            temp_proto_path.write_text(_json.dumps(minimal_proto), encoding="utf-8")
            result = build_demo_result(temp_proto_path, point_cloud_path, geometry_path)

        result["validation"] = run_projection_hull_validation()

        # Save JSON
        json_path = demo_output / "room_plan_demo.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        # Generate visualization
        from room_plan_demo import create_demo_visualization, create_unavailable_visualization, plane_residuals, plane_basis, project_points
        points = np.load(point_cloud_path)
        if result["selection_status"] == "selected":
            candidate = result["horizontal_plane_candidate"]
            equation = np.asarray(candidate["equation"], dtype=float)
            mask = plane_residuals(points, equation) < candidate["threshold_raw"]
            candidate_points = points[mask]
            normal, u, v = plane_basis(equation[:3])
            origin = np.mean(candidate_points, axis=0)
            all_projected = project_points(candidate_points, origin, u, v)
            hull = np.asarray([[p["u"], p["v"]] for p in result["observed_projected_hull"]])
            projection_for_viz = {**result["projection"], "hull_extent_raw": result["hull_extent_raw"]}
            create_demo_visualization(all_projected, hull, projection_for_viz, candidate, demo_output / "room_plan_demo.png")
        else:
            create_unavailable_visualization(demo_output / "room_plan_demo.png")

        hpc = result.get("horizontal_plane_candidate") or {}
        return {
            "status": "success" if result["selection_status"] == "selected" else "unavailable",
            "selection_status": result["selection_status"],
            "selected_plane_id": hpc.get("plane_id"),
            "hull_vertices": len(result.get("observed_projected_hull", [])),
            "output_dir": str(demo_output),
        }
    except Exception as e:
        return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}


def run_cross_tier_benchmark(final_output_dir: Path) -> dict[str, Any]:
    """Run cross-tier benchmark using existing cross_tier_benchmark module."""
    try:
        from cross_tier_benchmark import run_benchmark, csv_rows, markdown_report
        report = run_benchmark(ASSIGNMENT_ROOT)

        benchmark_output = final_output_dir / "cross_tier_benchmark"
        benchmark_output.mkdir(parents=True, exist_ok=True)

        (benchmark_output / "cross_tier_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        fields, rows = csv_rows(report)
        with (benchmark_output / "cross_tier_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
            import csv
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        command = f"python -u src/cross_tier_benchmark.py --assignment-root \"{ASSIGNMENT_ROOT}\" --output-dir output/cross_tier_benchmark"
        (benchmark_output / "cross_tier_benchmark.md").write_text(markdown_report(report, command), encoding="utf-8")

        return {
            "status": "success",
            "records": len(report["records"]),
            "output_dir": str(benchmark_output),
        }
    except Exception as e:
        return {"status": "failed", "error": str(e), "traceback": traceback.format_exc()}


def main() -> int:
    print("=" * 60)
    print("UNIFIED ROOM-SCANNER PIPELINE")
    print("=" * 60)
    print(f"Assignment root: {ASSIGNMENT_ROOT}")
    print(f"Datasets: {DATASET_IDS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Check output directory
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"ERROR: Output directory already exists and is not empty: {OUTPUT_DIR}")
        print("       Remove or rename it to avoid overwriting previous results.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-tier processing
    tier_results = {}
    overall_start = time.time()

    for dataset_id in DATASET_IDS:
        dataset_path = ASSIGNMENT_ROOT / dataset_id
        tier_name = dataset_id.replace("/", "_")
        tier_output_dir = OUTPUT_DIR / "tiers" / tier_name
        tier_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing {dataset_id} ---")
        tier_start = time.time()

        # 1. Single-room pipeline (point cloud + plane detection)
        print("  [1/3] Single-room pipeline...")
        pipe_result = run_single_room_pipeline(dataset_id, dataset_path, tier_output_dir)
        print(f"       Status: {pipe_result['status']}")
        if pipe_result["status"] == "success":
            print(f"       Points: {pipe_result['valid_point_count']:,}, Planes: 2, Unassigned: {pipe_result['unassigned_percentage']:.2f}%")
        else:
            print(f"       Error: {pipe_result.get('error', 'Unknown')}")

        # 2. Room-plan prototype
        print("  [2/3] Room-plan prototype...")
        prototype_result = run_room_plan_prototype(dataset_id, tier_output_dir, OUTPUT_DIR)
        print(f"       Status: {prototype_result['status']}")
        if prototype_result["status"] == "success":
            print(f"       Selected plane: {prototype_result['selected_plane_id']}, Hull vertices: {prototype_result['hull_vertices']}")

        # 3. Room-plan demo
        print("  [3/3] Room-plan demo...")
        demo_result = run_room_plan_demo(dataset_id, tier_output_dir, OUTPUT_DIR)
        print(f"       Status: {demo_result['status']}")
        if demo_result["status"] == "success":
            print(f"       Selected plane: {demo_result['selected_plane_id']}, Hull vertices: {demo_result['hull_vertices']}")

        # 4. Integrated Damage and Scope workflow (manual + image-based)
        print("  [4/4] Integrated Damage and Scope workflow...")
        prototype_json_path = OUTPUT_DIR / "room_plan_prototype" / tier_name / "room_plan_prototype.json"
        integrated_result = run_integrated_damage_and_scope(
            dataset_id, FRAME_ID, ASSIGNMENT_ROOT,
            prototype_json_path if prototype_json_path.exists() else None,
            run_image_detection=True,
        )
        integrated_scope_path = tier_output_dir / "integrated_damage_scope.json"
        save_integrated_result(integrated_result, integrated_scope_path)
        print(f"       Manual regions: {len(integrated_result.manual_regions)}, Manual scope: {len(integrated_result.manual_scope_items)}")
        print(f"       Image candidates: {len(integrated_result.image_candidates)}")

        tier_results[dataset_id] = {
            "dataset_id": dataset_id,
            "tier_name": tier_name,
            "single_room_pipeline": pipe_result,
            "room_plan_prototype": prototype_result,
            "room_plan_demo": demo_result,
            "integrated_damage_scope": integrated_result.to_dict(),
            "runtime_seconds": time.time() - tier_start,
        }

    # Cross-tier benchmark
    print("\n--- Cross-tier benchmark ---")
    bench_result = run_cross_tier_benchmark(OUTPUT_DIR)
    print(f"       Status: {bench_result['status']}")
    if bench_result["status"] == "success":
        print(f"       Records: {bench_result['records']}")

    total_runtime = time.time() - overall_start

    # Build summary
    summary = {
        "schema": {"name": "final_demo_summary", "version": "1.0"},
        "assignment_root": str(ASSIGNMENT_ROOT),
        "datasets_processed": list(DATASET_IDS),
        "total_runtime_seconds": total_runtime,
        "tiers": tier_results,
        "cross_tier_benchmark": bench_result,
        "limitations": [
            "All depth and geometry quantities are in raw uint16 depth-derived units; physical scale is unknown.",
            "Same-FOV intrinsics scaling is an unverified assumption.",
            "Plane labels are detection-order candidates, not floor/wall/ceiling classifications.",
            "Room-plan outputs are geometric projections, not architectural reconstructions.",
            "Cross-tier comparison uses identical analysis but captures are not validated tiers.",
            "No physical ground truth, confidence intervals, room dimensions, or accuracy gates are produced.",
        ],
    }

    # Write summary.json
    summary_json_path = OUTPUT_DIR / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write summary.txt
    summary_txt_path = OUTPUT_DIR / "summary.txt"
    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("FINAL DEMO SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Assignment root: {ASSIGNMENT_ROOT}\n")
        f.write(f"Total runtime: {total_runtime:.2f}s\n\n")

        f.write("PER-TIER RESULTS\n")
        f.write("-" * 60 + "\n")
        for dataset_id, tier in tier_results.items():
            f.write(f"\n{dataset_id}:\n")
            for component, result in tier.items():
                if component in ("single_room_pipeline", "room_plan_prototype", "room_plan_demo", "integrated_damage_scope"):
                    if component == "integrated_damage_scope":
                        if isinstance(result, dict):
                            manual_regions = len(result.get("manual_regions", []))
                            manual_scope = len(result.get("manual_scope_items", []))
                            image_candidates = len(result.get("image_candidates", []))
                            f.write(f"  {component}: manual_regions={manual_regions}, manual_scope={manual_scope}, image_candidates={image_candidates}\n")
                    else:
                        status = result.get("status", "unknown")
                        f.write(f"  {component}: {status}\n")
                        if result.get("status") == "success":
                            if "selected_plane_id" in result:
                                f.write(f"    Selected plane: {result['selected_plane_id']}, Hull: {result.get('hull_vertices', 0)} vertices\n")
                            if "valid_point_count" in result:
                                f.write(f"    Points: {result['valid_point_count']:,}, Unassigned: {result.get('unassigned_percentage', 0):.2f}%\n")

        f.write("\n\nCROSS-TIER BENCHMARK\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Status: {bench_result['status']}\n")
        if bench_result["status"] == "success":
            f.write(f"  Records: {bench_result['records']}\n")
            f.write(f"  Output: {bench_result['output_dir']}\n")

        f.write("\n\nOUTPUT LOCATIONS\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Root: {OUTPUT_DIR}\n")
        f.write(f"  summary.json: {summary_json_path}\n")
        f.write(f"  summary.txt: {summary_txt_path}\n")
        for dataset_id in DATASET_IDS:
            tier_name = dataset_id.replace("/", "_")
            f.write(f"  {tier_name}:\n")
            f.write(f"    Tier data: {OUTPUT_DIR / 'tiers' / tier_name}\n")
            f.write(f"    Prototype: {OUTPUT_DIR / 'room_plan_prototype' / tier_name}\n")
            f.write(f"    Demo: {OUTPUT_DIR / 'room_plan_demo' / tier_name}\n")
            f.write(f"    Integrated Damage & Scope: {OUTPUT_DIR / 'tiers' / tier_name / 'integrated_damage_scope.json'}\n")
            f.write(f"    Image Damage Candidates: {Path('output/image_damage_detection') / f'image_damage_candidates_{tier_name}_000000.json'}\n")
        f.write(f"  Benchmark: {OUTPUT_DIR / 'cross_tier_benchmark'}\n")

        f.write("\n\nLIMITATIONS\n")
        f.write("-" * 60 + "\n")
        for lim in summary["limitations"]:
            f.write(f"  - {lim}\n")

    # Final output
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Total runtime: {total_runtime:.2f}s")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary TXT:  {summary_txt_path}")
    print(f"Output root:  {OUTPUT_DIR}")

    # Check for any failures
    any_failed = False
    for tier in tier_results.values():
        for component, result in tier.items():
            if component in ("single_room_pipeline", "room_plan_prototype", "room_plan_demo", "damage_and_scope"):
                if isinstance(result, dict) and result.get("status") == "failed":
                    any_failed = True
    if bench_result.get("status") == "failed":
        any_failed = True

    if any_failed:
        print("\nWARNING: Some components failed. Check summary.json for details.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())