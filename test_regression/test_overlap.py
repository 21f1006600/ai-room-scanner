#!/usr/bin/env python3
"""
Deterministic regression test for Plane 1 / Plane 2 overlap bug.
Tests the mask-mapping logic with a synthetic 11-point fixture.
"""
import numpy as np

# ----- Helper functions copied from src/fix_loop_threshold_experiment.py -----
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

# ----- Test specific logic -----
def detect_two_planes_fixed(points, threshold, seed=42):
    """
    Fixed version: second-plane inliers recomputed only on outlier subset.
    Returns dict with masks and counts, or None if detection fails.
    """
    if seed is not None:
        np.random.seed(seed)
    N = points.shape[0]
    best_inliers = None
    best_plane = None
    best_inlier_count = 0
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
        return None
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
    # Second plane on outliers of first
    outliers_mask1 = ~refined_inliers1
    points_outlier = points[outliers_mask1]
    N_out = points_outlier.shape[0]
    if N_out < 3:
        refined_inliers2 = np.zeros(N, dtype=bool)
        refined_inlier_count2 = 0
    else:
        # RANSAC for second plane (on points_outlier)
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
            refined_inliers2 = np.zeros(N, dtype=bool)
            refined_inlier_count2 = 0
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
            # Map inlier mask back to original point indices
            refined_inliers2 = np.zeros(N, dtype=bool)
            refined_inliers2[outliers_mask1] = refined_inliers2_outlier
            refined_inlier_count2 = np.sum(refined_inliers2)
    return {
        'refined_inliers1': refined_inliers1,
        'refined_inliers2': refined_inliers2,
        'refined_plane1': refined_plane1,
        'refined_plane2': refined_plane2 if N_out >= 3 else None,
        'outliers_mask1': outliers_mask1,
        'plane1_count': int(refined_inlier_count1),
        'plane2_count': int(refined_inlier_count2),
    }

def detect_two_planes_buggy(points, threshold, seed=42):
    """
    Buggy version: second-plane inliers recomputed on full point cloud.
    """
    if seed is not None:
        np.random.seed(seed)
    N = points.shape[0]
    best_inliers = None
    best_plane = None
    best_inlier_count = 0
    # RANSAC for first plane (same as fixed)
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
        return None
    # Refine first plane (same as fixed)
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
    # Second plane on outliers of first (for RANSAC selection)
    outliers_mask1 = ~refined_inliers1
    points_outlier = points[outliers_mask1]
    N_out = points_outlier.shape[0]
    if N_out < 3:
        refined_inliers2 = np.zeros(N, dtype=bool)
        refined_inlier_count2 = 0
    else:
        # RANSAC for second plane (on points_outlier) - same as fixed
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
            refined_inliers2 = np.zeros(N, dtype=bool)
            refined_inlier_count2 = 0
        else:
            # Refine second plane using its inliers (within points_outlier) - same as fixed
            if best_inlier_count2 > 0:
                inlier_pts2 = points_outlier[best_inliers2]
                refined_plane2 = refine_plane(inlier_pts2)
                if refined_plane2 is None:
                    refined_plane2 = best_plane2
            else:
                refined_plane2 = best_plane2
            # BUGGY: Recompute inliers for second plane using FULL point cloud
            all_dists2 = plane_residuals(points, refined_plane2)  # <--- BUG: uses all points
            refined_inliers2 = all_dists2 < threshold            # <--- BUG: no mapping
            refined_inlier_count2 = np.sum(refined_inliers2)
    return {
        'refined_inliers1': refined_inliers1,
        'refined_inliers2': refined_inliers2,
        'refined_plane1': refined_plane1,
        'refined_plane2': refined_plane2 if N_out >= 3 else None,
        'outliers_mask1': outliers_mask1,
        'plane1_count': int(refined_inlier_count1),
        'plane2_count': int(refined_inlier_count2),
    }

def main():
    # Build synthetic 11-point fixture
    # Plane 1 points (z=0)
    p1_pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0]
    ], dtype=np.float32)
    # Plane 2 points (z=4)
    p2_pts = np.array([
        [0.0, 0.0, 4.0],
        [1.0, 0.0, 4.0],
        [0.0, 1.0, 4.0],
        [-1.0, 0.0, 4.0],
        [0.0, -1.0, 4.0]
    ], dtype=np.float32)
    # Ambiguous point (z=2)
    amb_pt = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    points = np.vstack([p1_pts, p2_pts, amb_pt])  # shape (11,3)
    N = points.shape[0]
    print(f"Synthetic point cloud: {N} points")
    print("Indices 0-4: Plane 1 (z=0)")
    print("Indices 5-9: Plane 2 (z=4)")
    print("Index 10: Ambiguous (z=2)")
    print()

    threshold = 3.0
    seed = 42

    print(f"Running fixed version with threshold={threshold}, seed={seed}")
    result_fixed = detect_two_planes_fixed(points, threshold, seed)
    if result_fixed is None:
        print("ERROR: Fixed version failed to detect first plane")
        return
    print(f"Fixed: Plane 1 count = {result_fixed['plane1_count']}")
    print(f"Fixed: Plane 2 count = {result_fixed['plane2_count']}")
    overlap_fixed = np.sum(result_fixed['refined_inliers1'] & result_fixed['refined_inliers2'])
    print(f"Fixed: Overlap count = {overlap_fixed}")
    total_assigned_fixed = result_fixed['plane1_count'] + result_fixed['plane2_count']
    unassigned_fixed = N - np.sum(result_fixed['refined_inliers1'] | result_fixed['refined_inliers2'])
    print(f"Fixed: Total assigned (sum) = {total_assigned_fixed}")
    print(f"Fixed: Unassigned count (via union) = {unassigned_fixed}")
    print()

    print(f"Running buggy version with threshold={threshold}, seed={seed}")
    result_buggy = detect_two_planes_buggy(points, threshold, seed)
    if result_buggy is None:
        print("ERROR: Buggy version failed to detect first plane")
        return
    print(f"Buggy: Plane 1 count = {result_buggy['plane1_count']}")
    print(f"Buggy: Plane 2 count = {result_buggy['plane2_count']}")
    overlap_buggy = np.sum(result_buggy['refined_inliers1'] & result_buggy['refined_inliers2'])
    print(f"Buggy: Overlap count = {overlap_buggy}")
    total_assigned_buggy = result_buggy['plane1_count'] + result_buggy['plane2_count']
    unassigned_buggy = N - np.sum(result_buggy['refined_inliers1'] | result_buggy['refined_inliers2'])
    print(f"Buggy: Total assigned (sum) = {total_assigned_buggy}")
    print(f"Buggy: Unassigned count (via union) = {unassigned_buggy}")
    print()

    # Assertions
    print("--- Assertions ---")
    # Fixed version should have zero overlap and exact accounting
    assert overlap_fixed == 0, f"Fixed version overlap should be 0, got {overlap_fixed}"
    assert result_fixed['plane1_count'] == 6, f"Fixed plane1 count expected 6, got {result_fixed['plane1_count']}"
    assert result_fixed['plane2_count'] == 5, f"Fixed plane2 count expected 5, got {result_fixed['plane2_count']}"
    assert unassigned_fixed == 0, f"Fixed unassigned should be 0, got {unassigned_fixed}"
    # Buggy version should have overlap == 1
    assert overlap_buggy == 1, f"Buggy version overlap expected 1, got {overlap_buggy}"
    # Optionally, we can also check that buggy plane2 count is 6 (ambiguous + 5 plane2)
    assert result_buggy['plane2_count'] == 6, f"Buggy plane2 count expected 6, got {result_buggy['plane2_count']}"
    print("All assertions passed.")
    print()
    print("Test demonstrates:")
    print("- Buggy full-cloud recomputation causes overlap of 1 point (the ambiguous point).")
    print("- Corrected outlier-subset mapping yields zero overlap.")
    print("- Point counts are conserved in the fixed version.")
    print("Test completed successfully.")

if __name__ == "__main__":
    main()