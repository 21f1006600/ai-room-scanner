#!/usr/bin/env python3
"""
Controlled RANSAC threshold-sensitivity experiment on a single frame.
Tests thresholds at 0.5%, 1.0%, 1.5%, and 2.0% of the depth range.
Reports plane coverage, residuals, angles, and runtime.
"""

import sys
import numpy as np
import cv2
import json
import time
import csv
from pathlib import Path

# Import backproject_depth and assumed RGB resolution from point_cloud.py
try:
    from point_cloud import backproject_depth, _RGB_WIDTH, _RGB_HEIGHT
except Exception as e:
    print(f"ERROR: Could not import from point_cloud.py: {e}")
    sys.exit(1)


# -------------------------- Helper functions (plane detection) --------------------------
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


def detect_two_planes(points, threshold, seed=42):
    """
    Detect up to two planes using RANSAC with refinement and recomputed inliers.
    Returns dict with plane metrics or None if detection fails.
    """
    if seed is not None:
        np.random.seed(seed)
    best_inliers = None
    best_plane = None
    best_inlier_count = 0
    N = points.shape[0]
    # RANSAC for first plane
    for it in range(1000):
        idx = np.random.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        plane = fit_plane(p1, p2, p3)
        if plane is None:
            continue
        dists = plane_residuals(points, plane)
        inliers = dists < threshold
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane
            best_inliers = inliers
        if inlier_count > 0.9 * N:
            break
    if best_plane is None:
        return None  # failed to find first plane
    # Refine first plane
    if best_inlier_count > 0:
        inlier_pts = points[best_inliers]
        refined_plane1 = refine_plane(inlier_pts)
        if refined_plane1 is None:
            refined_plane1 = best_plane
    else:
        refined_plane1 = best_plane
    # Recompute inliers for first plane
    all_dists1 = plane_residuals(points, refined_plane1)
    refined_inliers1 = all_dists1 < threshold
    refined_inlier_count1 = np.sum(refined_inliers1)
    if refined_inlier_count1 > 0:
        dists1_refined = all_dists1[refined_inliers1]
        mean_res1 = np.mean(dists1_refined)
        median_res1 = np.median(dists1_refined)
    else:
        mean_res1 = 0.0
        median_res1 = 0.0
    a1, b1, c1, d1 = refined_plane1
    normal1 = np.array([a1, b1, c1])
    norm1 = np.linalg.norm(normal1)
    if norm1 > 0:
        normal1 = normal1 / norm1

    # Second plane on outliers of first
    outliers_mask1 = ~refined_inliers1
    points_outlier = points[outliers_mask1]
    N_out = points_outlier.shape[0]
    if N_out < 3:
        # Not enough points for second plane
        plane2_info = {
            'inlier_count': 0,
            'inlier_percentage': 0.0,
            'mean_residual': 0.0,
            'median_residual': 0.0,
            'equation': [0.0, 0.0, 0.0, 0.0],
            'normal': [0.0, 0.0, 0.0]
        }
    else:
        # RANSAC for second plane (on points outside plane1)
        best_inliers2 = None
        best_plane2 = None
        best_inlier_count2 = 0
        for it in range(1000):
            idx = np.random.choice(N_out, size=3, replace=False)
            p1, p2, p3 = points_outlier[idx[0]], points_outlier[idx[1]], points_outlier[idx[2]]
            plane = fit_plane(p1, p2, p3)
            if plane is None:
                continue
            dists = plane_residuals(points_outlier, plane)
            inliers = dists < threshold
            inlier_count = np.sum(inliers)
            if inlier_count > best_inlier_count2:
                best_inlier_count2 = inlier_count
                best_plane2 = plane
                best_inliers2 = inliers
            if inlier_count > 0.9 * N_out:
                break
        if best_plane2 is None:
            plane2_info = {
                'inlier_count': 0,
                'inlier_percentage': 0.0,
                'mean_residual': 0.0,
                'median_residual': 0.0,
                'equation': [0.0, 0.0, 0.0, 0.0],
                'normal': [0.0, 0.0, 0.0]
            }
        else:
            # Refine second plane using its inliers (within points_outlier)
            if best_inlier_count2 > 0:
                inlier_pts2 = points_outlier[best_inliers2]
                refined_plane2 = refine_plane(inlier_pts2)
                if refined_plane2 is None:
                    refined_plane2 = best_plane2
            else:
                refined_plane2 = best_plane2
            # Recompute inliers for second plane ONLY within the outlier subset
            all_dists2_outlier = plane_residuals(points_outlier, refined_plane2)
            refined_inliers2_outlier = all_dists2_outlier < threshold
            refined_inlier_count2_outlier = np.sum(refined_inliers2_outlier)
            if refined_inlier_count2_outlier > 0:
                dists2_refined = all_dists2_outlier[refined_inliers2_outlier]
                mean_res2 = np.mean(dists2_refined)
                median_res2 = np.median(dists2_refined)
            else:
                mean_res2 = 0.0
                median_res2 = 0.0
            a2, b2, c2, d2 = refined_plane2
            normal2 = np.array([a2, b2, c2])
            norm2 = np.linalg.norm(normal2)
            if norm2 > 0:
                normal2 = normal2 / norm2
            # Map inlier mask back to original point indices
            refined_inliers2 = np.zeros(N, dtype=bool)
            refined_inliers2[outliers_mask1] = refined_inliers2_outlier
            refined_inlier_count2 = np.sum(refined_inliers2)
            plane2_info = {
                'inlier_count': int(refined_inlier_count2),
                'inlier_percentage': float(100.0 * refined_inlier_count2 / N),
                'mean_residual': float(mean_res2),
                'median_residual': float(median_res2),
                'equation': [float(a2), float(b2), float(c2), float(d2)],
                'normal': [float(a2), float(b2), float(c2)]
            }

    # Compute unassigned points (points not in either plane's refined inliers)
    if refined_inlier_count1 > 0 or refined_inlier_count2 > 0:
        assigned_mask = refined_inliers1 | refined_inliers2
    else:
        assigned_mask = np.zeros(N, dtype=bool)
    # Validation: masks must be disjoint and sum to total points
    overlap = np.sum(refined_inliers1 & refined_inliers2)
    if overlap > 0:
        raise RuntimeError(f"Plane 1 and plane 2 masks overlap: {overlap} points")
    total_assigned = np.sum(refined_inliers1) + np.sum(refined_inliers2)
    unassigned_count = np.sum(~assigned_mask)
    if total_assigned + unassigned_count != N:
        raise RuntimeError(f"Point count mismatch: {total_assigned} assigned + {unassigned_count} unassigned != {N} total")
    unassigned_percentage = float(100.0 * unassigned_count / N)

    # Angle between normals (if both planes have at least 3 inliers?)
    if refined_inlier_count1 >= 3 and refined_inlier_count2 >= 3:
        ang_diff = normal_angular_diff(normal1, normal2)
    else:
        ang_diff = None

    return {
        'plane1': {
            'inlier_count': int(np.sum(refined_inliers1)),
            'inlier_percentage': float(100.0 * np.sum(refined_inliers1) / N),
            'median_residual': float(median_res1),
            'equation': [float(a1), float(b1), float(c1), float(d1)],
            'normal': [float(a1), float(b1), float(c1)]
        },
        'plane2': plane2_info,
        'unassigned_percentage': unassigned_percentage,
        'angle_deg': float(ang_diff) if ang_diff is not None else None
    }


def main():
    # -------------------------- Configuration --------------------------
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    dataset_name = "single_scan_with_ceiling/c7d28f72c6"
    dataset_path = base / dataset_name
    frame_filename = "000000.png"

    depth_dir = dataset_path / "depth"
    depth_path = depth_dir / frame_filename
    intrinsics_path = dataset_path / "camera_matrix.csv"

    out_dir = Path("output/fix_loop_threshold_experiment_corrected")

    # -------------------------- Checks --------------------------
    if not depth_path.is_file():
        print(f"ERROR: Depth image not found: {depth_path}")
        sys.exit(1)
    if not intrinsics_path.is_file():
        print(f"ERROR: Camera matrix not found: {intrinsics_path}")
        sys.exit(1)

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

    # -------------------------- Load data --------------------------
    print(f"Loading depth frame: {depth_path}")
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"ERROR: Failed to read depth image: {depth_path}")
        sys.exit(1)
    if depth.ndim != 2:
        print(f"ERROR: Depth image is not 2D (shape {depth.shape})")
        sys.exit(1)

    try:
        K = np.loadtxt(str(intrinsics_path), delimiter=',')
    except Exception as e:
        print(f"ERROR: Failed to load camera matrix: {e}")
        sys.exit(1)
    if K.shape != (3, 3):
        print(f"ERROR: Camera matrix must be 3x3, got shape {K.shape}")
        sys.exit(1)

    # -------------------------- Back-project --------------------------
    print(f"Depth image shape: {depth.shape} (height x width)")
    print(f"Using same-FOV-scaled intrinsics (UNVERIFIED ASSUMPTION)")
    print(f"RGB resolution assumed: {_RGB_WIDTH} x {_RGB_HEIGHT}")
    print(f"Depth resolution: {depth.shape[1]} x {depth.shape[0]}")

    try:
        points = backproject_depth(
            depth_path,
            intrinsics_path,
            use_scaled_intrinsics=True
        )
    except Exception as e:
        print(f"ERROR: Back-projection failed: {e}")
        sys.exit(1)

    if points.size == 0:
        print("ERROR: No valid points found (all depth values are zero or invalid).")
        sys.exit(1)

    # Point cloud statistics
    Z = points[:, 2]
    z_min, z_max = float(Z.min()), float(Z.max())
    z_range = z_max - z_min
    print(f"\nLoaded point cloud with {points.shape[0]} points")
    print(f"Depth range (Z): [{z_min:.3f}, {z_max:.3f}] -> range = {z_range:.3f}")

    # -------------------------- Thresholds --------------------------
    fractions = [0.005, 0.01, 0.015, 0.02]  # 0.5%, 1%, 1.5%, 2%
    thresholds = [frac * z_range for frac in fractions]
    thresholds = [max(t, 1.0) for t in thresholds]  # clamp to min 1.0

    print("\nTesting thresholds:")
    for frac, th in zip(fractions, thresholds):
        print(f"  {frac*100:g}% of Z range -> {th:.3f} raw depth units")

    # -------------------------- Run experiment --------------------------
    results = []
    for frac, th in zip(fractions, thresholds):
        print(f"\n--- Threshold: {th:.3f} ({frac*100:g}% of Z range) ---")
        runs = []
        for rep in range(3):
            if rep > 0:
                print(f"  Run {rep+1}/3", end='', flush=True)
            start = time.time()
            np.random.seed(42)  # deterministic seed for each run
            result = detect_two_planes(points, th, seed=42)
            elapsed = time.time() - start
            if result is None:
                print("  -> FAILED to detect first plane.")
                sys.exit(1)
            runs.append({
                'runtime': elapsed,
                'plane1': result['plane1'],
                'plane2': result['plane2'],
                'unassigned': result['unassigned_percentage'],
                'angle': result['angle_deg']
            })
            if rep > 0:
                print(f"  -> {elapsed:.3f}s")
            else:
                print(f"  -> {elapsed:.3f}s")
        # Check consistency across runs (should be identical due to fixed seed)
        first = runs[0]
        consistent = True
        for i, r in enumerate(runs[1:], start=1):
            # Compare key metrics (allow tiny floating point differences)
            if r['plane1']['inlier_count'] != first['plane1']['inlier_count'] or \
               r['plane2']['inlier_count'] != first['plane2']['inlier_count'] or \
               abs(r['unassigned'] - first['unassigned']) > 1e-6 or \
               (r['angle'] is not None and first['angle'] is not None and abs(r['angle'] - first['angle']) > 1e-6) or \
               (r['angle'] is None and first['angle'] is not None) or (r['angle'] is not None and first['angle'] is None):
                consistent = False
                break
        if not consistent:
            print("  WARNING: Geometric metrics varied across runs despite fixed seed.")
        # Use first run's metrics (they should be same) and median runtime
        median_runtime = np.median([r['runtime'] for r in runs])
        results.append({
            'threshold_fraction': frac,
            'threshold_raw': th,
            'plane1_inlier_count': first['plane1']['inlier_count'],
            'plane1_inlier_percentage': first['plane1']['inlier_percentage'],
            'plane1_median_residual': first['plane1']['median_residual'],
            'plane2_inlier_count': first['plane2']['inlier_count'],
            'plane2_inlier_percentage': first['plane2']['inlier_percentage'],
            'plane2_median_residual': first['plane2']['median_residual'],
            'unassigned_percentage': first['unassigned'],
            'angle_deg': first['angle'],
            'median_runtime_seconds': float(median_runtime)
        })

    # -------------------------- Save results --------------------------
    # JSON
    json_file = out_dir / "threshold_sensitivity.json"
    try:
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved JSON summary to: {json_file}")
    except Exception as e:
        print(f"WARNING: Could not save JSON summary: {e}")

    # CSV
    csv_file = out_dir / "threshold_sensitivity.csv"
    try:
        if results:
            # Flatten for CSV
            headers = [
                'threshold_fraction', 'threshold_raw',
                'plane1_inlier_count', 'plane1_inlier_percentage', 'plane1_median_residual',
                'plane2_inlier_count', 'plane2_inlier_percentage', 'plane2_median_residual',
                'unassigned_percentage', 'angle_deg', 'median_runtime_seconds'
            ]
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for r in results:
                    writer.writerow(r)
            print(f"Saved CSV summary to: {csv_file}")
    except Exception as e:
        print(f"WARNING: Could not save CSV summary: {e}")

    # -------------------------- Summary --------------------------
    print("\n=== Experiment Summary ===")
    print(f"{'Threshold':>12} {'P1 Cov':>8} {'P2 Cov':>8} {'Unass':>8} {'Angle (°)':>10} {'P1 MedRes':>10} {'P2 MedRes':>10} {'Runtime (s)':>12}")
    print("-" * 92)
    for r in results:
        th_raw = r['threshold_raw']
        p1c = r['plane1_inlier_percentage']
        p2c = r['plane2_inlier_percentage']
        unc = r['unassigned_percentage']
        ang = r['angle_deg'] if r['angle_deg'] is not None else float('nan')
        r1 = r['plane1_median_residual']
        r2 = r['plane2_median_residual']
        rt = r['median_runtime_seconds']
        # Format angle: show empty if None
        ang_str = f"{ang:10.2f}" if not np.isnan(ang) else "      None"
        print(f"{th_raw:12.3f} {p1c:8.2f} {p2c:8.2f} {unc:8.2f} {ang_str} {r1:10.2f} {r2:10.2f} {rt:12.3f}")

    print("\nNotes:")
    print("- All quantities are in raw depth units (uint16 image values).")
    print("- The same-FOV intrinsics assumption (used to create the point cloud)")
    print("  remains unverified; thus plane orientations in real space are unknown.")
    print("- No conversion to metric units (e.g., meters) has been performed.")
    print("- The detected planes are geometric candidates only;")
    print("  they are not labeled as floor, wall, ceiling, or any specific surface.")
    print("- The distance threshold is varied as a fraction of the depth range.")
    print("- For each threshold, the plane detection, refinement, and inlier recomputation")
    print("  follows the same logic as in multiple_plane_experiment.py.")
    print("- Runtime is median of 3 runs with fixed random seed (42).")
    print("- Geometric metrics should be identical across runs; any variation is noted.")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()