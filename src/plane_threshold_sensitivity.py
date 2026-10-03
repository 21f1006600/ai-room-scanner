#!/usr/bin/env python3
"""
Plane threshold sensitivity experiment.
Loads the point cloud from output/single_frame_geometry/point_cloud.npy,
runs RANSAC plane detection with three different distance thresholds
(0.5%, 1%, 2% of the depth range), and reports how the fitted plane
changes with threshold.
"""

import sys
import numpy as np
from pathlib import Path

# Try to import matplotlib for potential debugging (not required)
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def main():
    # -------------------------- Configuration --------------------------
    pc_path = Path("output/single_frame_geometry/point_cloud.npy")
    # Output directory not used; we will not write any files.

    # -------------------------- Checks --------------------------
    if not pc_path.is_file():
        print(f"ERROR: Point cloud not found: {pc_path}")
        print("       Run src/single_frame_geometry.py first to generate the point cloud.")
        sys.exit(1)

    # -------------------------- Load point cloud --------------------------
    try:
        points = np.load(pc_path)
    except Exception as e:
        print(f"ERROR: Failed to load point cloud: {e}")
        sys.exit(1)

    if points.size == 0:
        print("ERROR: Loaded point cloud is empty.")
        sys.exit(1)
    if points.shape[1] != 3:
        print(f"ERROR: Expected point cloud with shape (N, 3), got {points.shape}")
        sys.exit(1)

    N = points.shape[0]
    print(f"Loaded point cloud with {N} points from {pc_path}")

    # Compute depth range for thresholds
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_range = z_max - z_min
    print(f"Depth range (Z): [{z_min:.3f}, {z_max:.3f}] -> range = {z_range:.3f}")

    # Define thresholds as fractions of z_range
    fractions = [0.005, 0.01, 0.02]  # 0.5%, 1%, 2%
    thresholds = [frac * z_range for frac in fractions]
    # Ensure a minimum threshold of 1.0 unit to avoid overly strict thresholds
    thresholds = [max(t, 1.0) for t in thresholds]

    print("\nTesting thresholds:")
    for frac, th in zip(fractions, thresholds):
        print(f"  {frac*100:g}% of Z range -> {th:.3f} raw depth units")

    # -------------------------- Helper functions --------------------------
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
        # Ensure normal is unit length (should be from eigh)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        d = -np.dot(normal, centroid)
        return np.array([normal[0], normal[1], normal[2], d])

    def plane_residuals(points, plane):
        """Absolute perpendicular distances from points to plane."""
        a, b, c, d = plane
        normal = np.array([a, b, c])
        return np.abs(np.dot(points, normal) + d)

    def normal_angular_diff(n1, n2):
        """Angular difference between two unit normals, accounting for sign equivalence.
        Returns angle in degrees between 0 and 90."""
        # Ensure unit length
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        dot = abs(np.dot(n1, n2))  # absolute to ignore sign
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        return np.degrees(angle_rad)

    # -------------------------- RANSAC settings --------------------------
    np.random.seed(42)  # deterministic seed
    n_iterations = 1000

    # Store results for each threshold
    results = []

    for th in thresholds:
        # Reset random seed for each threshold to have same random samples
        np.random.seed(42)

        best_inliers = None
        best_plane = None
        best_inlier_count = 0

        for it in range(n_iterations):
            idx = np.random.choice(N, size=3, replace=False)
            p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
            plane = fit_plane(p1, p2, p3)
            if plane is None:
                continue
            dists = plane_residuals(points, plane)
            inliers = dists < th
            inlier_count = np.sum(inliers)
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_plane = plane
                best_inliers = inliers
            # Early exit if we have a very good fit
            if inlier_count > 0.9 * N:
                break

        if best_plane is None:
            print(f"ERROR: RANSAC failed for threshold {th:.3f}")
            sys.exit(1)

        # Refine using all inliers
        if best_inlier_count > 0:
            inlier_points = points[best_inliers]
            refined_plane = refine_plane(inlier_points)
            if refined_plane is None:
                refined_plane = best_plane
        else:
            refined_plane = best_plane
            inlier_points = np.empty((0, 3))

        # Recompute distances and inliers using the refined plane and the same threshold
        all_dists_refined = plane_residuals(points, refined_plane)
        refined_inliers = all_dists_refined < th
        refined_inlier_count = np.sum(refined_inliers)

        # Compute residuals for the refined inliers
        if refined_inlier_count > 0:
            refined_inlier_dists = all_dists_refined[refined_inliers]
            mean_res = np.mean(refined_inlier_dists)
            median_res = np.median(refined_inlier_dists)
        else:
            mean_res = 0.0
            median_res = 0.0

        # Extract normal (unit)
        a, b, c, d = refined_plane
        normal = np.array([a, b, c])
        # Normalize (should already be unit)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm

        results.append({
            'threshold': th,
            'inlier_count': refined_inlier_count,
            'inlier_percent': 100.0 * refined_inlier_count / N,
            'mean_residual': mean_res,
            'median_residual': median_res,
            'normal': normal,
            'inliers_mask': refined_inliers  # keep for potential debug
        })

    # -------------------------- Compute angular differences relative to 1% threshold (index 1) --------------------------
    baseline_normal = results[1]['normal']  # 1% threshold is second in list (0.5%,1%,2%)
    for i, res in enumerate(results):
        ang_diff = normal_angular_diff(res['normal'], baseline_normal)
        res['angular_diff_deg'] = ang_diff

    # -------------------------- Print comparison table --------------------------
    print("\n=== Plane Threshold Sensitivity Results ===")
    print(f"{'Threshold':>12} {'Inliers':>10} {'%':>7} {'Mean Res':>10} {'Median Res':>12} {'Normal (nx,ny,nz)':>35} {'Ang Diff vs 1% (°)':>20}")
    print("-" * 115)
    for frac, res in zip(fractions, results):
        th = res['threshold']
        ic = res['inlier_count']
        ip = res['inlier_percent']
        mr = res['mean_residual']
        md = res['median_residual']
        nx, ny, nz = res['normal']
        ang = res['angular_diff_deg']
        print(f"{th:12.3f} {ic:10d} {ip:7.2f} {mr:10.3f} {md:12.3f} "
              f"[{nx: 7.4f},{ny: 7.4f},{nz: 7.4f}] {ang:20.2f}")

    # -------------------------- Interpretation --------------------------
    print("\nNotes:")
    print("- All quantities are in raw depth units (uint16 image values).")
    print("- The same-FOV intrinsics assumption remains unverified.")
    print("- Angular difference is computed between unit normals, treating n and -n as equivalent.")
    print("- No files were saved or overwritten; this experiment only reads the point cloud.")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()