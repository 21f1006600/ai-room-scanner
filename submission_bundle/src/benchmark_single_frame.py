#!/usr/bin/env python3
"""
Benchmark the single-frame pipeline on the first frame of three datasets.
Reuses backprojection and plane-detection logic without modifying existing scripts.
"""

import sys
import numpy as np
import cv2
import json
import time
from pathlib import Path

# Import backproject_depth and assumed RGB resolution from point_cloud.py
try:
    from point_cloud import backproject_depth, _RGB_WIDTH, _RGB_HEIGHT
except Exception as e:
    print(f"ERROR: Could not import from point_cloud.py: {e}")
    sys.exit(1)


# -------------------------- Helper functions (copied from single_room_pipeline.py) --------------------------
def fit_plane(p1, p2, p3):
    """Return plane coefficients (a,b,c,d) for ax+by+cz+d=0, normalized normal."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal = normal / norm
    d = -np.dot(normal, p1)
    return np.array([normal[0], normal[1], normal[2], d])


def refine_plane(inlier_points):
    """Refine plane using SVD on inlier points; returns refined plane (normalized)."""
    if len(inlier_points) < 3:
        return None
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid
    cov = np.dot(centered.T, centered)
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]  # eigenvector for smallest eigenvalue
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    d = -np.dot(normal, centroid)
    return np.array([normal[0], normal[1], normal[2], d])


def plane_residuals(points_arr, plane):
    """Absolute perpendicular distances from points to plane."""
    a, b, c, d = plane
    normal = np.array([a, b, c])
    return np.abs(np.dot(points_arr, normal) + d)


def normal_angular_diff(n1, n2):
    """Angular difference between two unit normals, accounting for sign equivalence.
    Returns angle in degrees between 0 and 90."""
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    dot = abs(np.dot(n1, n2))  # absolute to ignore sign
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


def ransac_detect_plane(pts, threshold, seed=None):
    """Run RANSAC to detect a plane in pts.
    Returns (plane, inlier_mask, inlier_count) or (None, None, 0) if fails."""
    if seed is not None:
        np.random.seed(seed)
    best_inliers = None
    best_plane = None
    best_inlier_count = 0
    N = pts.shape[0]
    for it in range(1000):  # n_iterations
        idx = np.random.choice(N, size=3, replace=False)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        plane = fit_plane(p1, p2, p3)
        if plane is None:
            continue
        dists = plane_residuals(pts, plane)
        inliers = dists < threshold
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane
            best_inliers = inliers
        if inlier_count > 0.9 * N:
            break
    if best_plane is None:
        return None, None, 0
    # Refine using all inliers
    if best_inlier_count > 0:
        inlier_pts = pts[best_inliers]
        refined_plane = refine_plane(inlier_pts)
        if refined_plane is None:
            refined_plane = best_plane
    else:
        refined_plane = best_plane
    # Recompute inliers using refined plane
    all_dists = plane_residuals(pts, refined_plane)
    refined_inliers = all_dists < threshold
    refined_inlier_count = np.sum(refined_inliers)
    return refined_plane, refined_inliers, refined_inlier_count


# -------------------------- Processing function --------------------------
def process_frame(depth_path, intrinsics_path):
    """Process a single depth frame and return metrics and runtime."""
    start_time = time.time()

    # Load depth image (uint16)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image: {depth_path}")
    if depth.ndim != 2:
        raise RuntimeError(f"Depth image is not 2D (shape {depth.shape})")

    # Load camera matrix (3x3)
    try:
        K = np.loadtxt(str(intrinsics_path), delimiter=',')
    except Exception as e:
        raise RuntimeError(f"Failed to load camera matrix: {e}")
    if K.shape != (3, 3):
        raise RuntimeError(f"Camera matrix must be 3x3, got shape {K.shape}")

    # Back-project
    points = backproject_depth(
        depth_path,
        intrinsics_path,
        use_scaled_intrinsics=True
    )
    if points.size == 0:
        raise RuntimeError("No valid points found (all depth values are zero or invalid).")

    # Point cloud statistics
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]
    num_points = points.shape[0]
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())
    z_min, z_max = float(Z.min()), float(Z.max())
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    z_extent = z_max - z_min

    # Depth statistics from raw image
    valid_mask = depth > 0
    depth_vals = depth[valid_mask]
    if depth_vals.size > 0:
        depth_min = int(depth_vals.min())
        depth_median = int(np.median(depth_vals))
        depth_max = int(depth_vals.max())
    else:
        depth_min = depth_median = depth_max = 0

    # Plane detection
    z_range = z_max - z_min
    distance_threshold = 0.01 * z_range  # 1% of depth range
    if distance_threshold < 1.0:
        distance_threshold = 1.0

    # Detect first (dominant) plane
    plane1, inliers1, count1 = ransac_detect_plane(points, distance_threshold, seed=42)
    if plane1 is None:
        raise RuntimeError("Failed to detect first plane.")
    if count1 > 0:
        dists1 = plane_residuals(points[inliers1], plane1)
        mean_res1 = np.mean(dists1)
        median_res1 = np.median(dists1)
    else:
        mean_res1 = 0.0
        median_res1 = 0.0
    a1, b1, c1, d1 = plane1
    normal1 = np.array([a1, b1, c1])
    norm1 = np.linalg.norm(normal1)
    if norm1 > 0:
        normal1 = normal1 / norm1

    # Detect second plane on outliers of first
    outliers_mask = ~inliers1
    points_outliers = points[outliers_mask]
    N_out = points_outliers.shape[0]
    if N_out < 3:
        plane2, inliers2_rel, count2 = None, None, 0
    else:
        plane2, inliers2_rel, count2 = ransac_detect_plane(points_outliers, distance_threshold, seed=42)
        if plane2 is not None:
            # Convert relative indices to original indices
            inliers2 = np.zeros(num_points, dtype=bool)
            inliers2[outliers_mask] = inliers2_rel
            if count2 > 0:
                dists2 = plane_residuals(points[inliers2], plane2)
                mean_res2 = np.mean(dists2)
                median_res2 = np.median(dists2)
            else:
                mean_res2 = 0.0
                median_res2 = 0.0
            a2, b2, c2, d2 = plane2
            normal2 = np.array([a2, b2, c2])
            norm2 = np.linalg.norm(normal2)
            if norm2 > 0:
                normal2 = normal2 / norm2
        else:
            mean_res2 = 0.0
            median_res2 = 0.0
            a2 = b2 = c2 = d2 = 0.0
            normal2 = np.array([0.0, 0.0, 0.0])

    # Remaining points (outliers of both)
    if plane2 is not None and count2 > 0:
        outliers_both_mask = ~(inliers1 | inliers2)
    else:
        outliers_both_mask = ~inliers1
    count_outliers = np.sum(outliers_both_mask)

    # Angular difference between plane normals
    if plane2 is not None and count2 > 0:
        ang_diff = normal_angular_diff(normal1, normal2)
    else:
        ang_diff = None

    end_time = time.time()
    runtime = end_time - start_time

    result = {
        "dataset": depth_path.parent.parent.name,  # e.g., single_scan_floor_only/1a8384c3f6 -> we want the dataset name
        "frame": depth_path.name,
        "valid_point_count": int(num_points),
        "depth_min": depth_min,
        "depth_median": depth_median,
        "depth_max": depth_max,
        "xyz_extent": [float(x_extent), float(y_extent), float(z_extent)],
        "ransac_threshold": float(distance_threshold),
        "plane1": {
            "inlier_count": int(count1),
            "inlier_percentage": float(100.0 * count1 / num_points),
            "mean_residual": float(mean_res1),
            "median_residual": float(median_res1),
            "equation": [float(a1), float(b1), float(c1), float(d1)],
            "normal": [float(a1), float(b1), float(c1)]
        },
        "plane2": {
            "inlier_count": int(count2) if plane2 is not None else 0,
            "inlier_percentage": float(100.0 * count2 / num_points) if plane2 is not None else 0.0,
            "mean_residual": float(mean_res2) if plane2 is not None else 0.0,
            "median_residual": float(median_res2) if plane2 is not None else 0.0,
            "equation": [float(a2), float(b2), float(c2), float(d2)] if plane2 is not None else [0.0, 0.0, 0.0, 0.0],
            "normal": [float(a2), float(b2), float(c2)] if plane2 is not None else [0.0, 0.0, 0.0]
        },
        "unassigned_percentage": float(100.0 * count_outliers / num_points),
        "angular_diff_deg": float(ang_diff) if ang_diff is not None else None,
        "runtime_seconds": float(runtime)
    }
    return result


def main():
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    datasets = [
        ("single_room/c00a170fe1", base / "single_room" / "c00a170fe1"),
        ("single_scan_floor_only/1a8384c3f6", base / "single_scan_floor_only" / "1a8384c3f6"),
        ("single_scan_with_ceiling/c7d28f72c6", base / "single_scan_with_ceiling" / "c7d28f72c6")
    ]

    out_dir = Path("output/benchmark_single_frame")
    if out_dir.exists():
        if any(out_dir.iterdir()):
            print(f"ERROR: Output directory already exists and is not empty: {out_dir}")
            print("       Remove or rename it to avoid overwriting previous results.")
            sys.exit(1)
    else:
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except Exception as e:
            print(f"ERROR: Could not create output directory: {e}")
            sys.exit(1)

    results = []
    for dataset_name, dataset_path in datasets:
        depth_dir = dataset_path / "depth"
        depth_path = depth_dir / "000000.png"
        intrinsics_path = dataset_path / "camera_matrix.csv"

        if not depth_path.is_file():
            print(f"ERROR: Depth image not found: {depth_path}")
            sys.exit(1)
        if not intrinsics_path.is_file():
            print(f"ERROR: Camera matrix not found: {intrinsics_path}")
            sys.exit(1)

        try:
            result = process_frame(depth_path, intrinsics_path)
            results.append(result)
            print(f"Processed {dataset_name}: {result['valid_point_count']} points, "
                  f"runtime {result['runtime_seconds']:.2f}s")
        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            sys.exit(1)

    # Save JSON
    json_file = out_dir / "benchmark_summary.json"
    try:
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON summary to: {json_file}")
    except Exception as e:
        print(f"WARNING: Could not save JSON summary: {e}")

    # Save CSV
    csv_file = out_dir / "benchmark_summary.csv"
    try:
        # Flatten results for CSV
        import csv
        if results:
            # Determine CSV headers from the first result (excluding nested dicts for simplicity)
            # We'll select key metrics.
            headers = [
                "dataset", "frame", "valid_point_count",
                "depth_min", "depth_median", "depth_max",
                "xyz_extent_x", "xyz_extent_y", "xyz_extent_z",
                "ransac_threshold",
                "plane1_inlier_count", "plane1_inlier_percentage",
                "plane1_mean_residual", "plane1_median_residual",
                "plane2_inlier_count", "plane2_inlier_percentage",
                "plane2_mean_residual", "plane2_median_residual",
                "unassigned_percentage", "angular_diff_deg",
                "runtime_seconds"
            ]
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in results:
                    row = {
                        "dataset": r["dataset"],
                        "frame": r["frame"],
                        "valid_point_count": r["valid_point_count"],
                        "depth_min": r["depth_min"],
                        "depth_median": r["depth_median"],
                        "depth_max": r["depth_max"],
                        "xyz_extent_x": r["xyz_extent"][0],
                        "xyz_extent_y": r["xyz_extent"][1],
                        "xyz_extent_z": r["xyz_extent"][2],
                        "ransac_threshold": r["ransac_threshold"],
                        "plane1_inlier_count": r["plane1"]["inlier_count"],
                        "plane1_inlier_percentage": r["plane1"]["inlier_percentage"],
                        "plane1_mean_residual": r["plane1"]["mean_residual"],
                        "plane1_median_residual": r["plane1"]["median_residual"],
                        "plane2_inlier_count": r["plane2"]["inlier_count"],
                        "plane2_inlier_percentage": r["plane2"]["inlier_percentage"],
                        "plane2_mean_residual": r["plane2"]["mean_residual"],
                        "plane2_median_residual": r["plane2"]["median_residual"],
                        "unassigned_percentage": r["unassigned_percentage"],
                        "angular_diff_deg": r["angular_diff_deg"] if r["angular_diff_deg"] is not None else "",
                        "runtime_seconds": r["runtime_seconds"]
                    }
                    writer.writerow(row)
            print(f"Saved CSV summary to: {csv_file}")
    except Exception as e:
        print(f"WARNING: Could not save CSV summary: {e}")

    print("\nBenchmark complete.")
    print("- All quantities are in raw depth units (uint16 image values).")
    print("- The same-FOV intrinsics assumption (used to create the point cloud)")
    print("  remains unverified; thus plane orientations in real space are unknown.")
    print("- No conversion to metric units (e.g., meters) has been performed.")
    print("- The detected planes are geometric candidates only;")
    print("  they are not labeled as floor, wall, ceiling, or any specific surface.")


if __name__ == "__main__":
    main()