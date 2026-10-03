#!/usr/bin/env python3
"""
Focused unit test of the mask-mapping logic for Plane 1 / Plane 2 overlap bug.
Uses a synthetic 11-point fixture and manually defined refined planes.
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

    # Manually defined refined planes (as would be obtained after refinement)
    # Plane 1: z = 0  => normal (0,0,1), d = 0
    refined_plane1 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    # Plane 2: z = 4  => normal (0,0,1), d = -4
    refined_plane2 = np.array([0.0, 0.0, 1.0, -4.0], dtype=np.float32)

    threshold = 3.0
    print(f"Using distance threshold = {threshold}")
    print()

    # Compute residuals to refined plane 1
    dists_to_plane1 = plane_residuals(points, refined_plane1)
    refined_inliers1 = dists_to_plane1 < threshold
    refined_inlier_count1 = np.sum(refined_inliers1)
    print(f"Plane 1 residuals: {dists_to_plane1}")
    print(f"Plane 1 inlier mask (dist < {threshold}): {refined_inliers1.astype(int)}")
    print(f"Plane 1 inlier count: {refined_inlier_count1}")
    print()

    # Outliers mask for plane 1
    outliers_mask1 = ~refined_inliers1
    points_outlier = points[outliers_mask1]
    N_out = points_outlier.shape[0]
    print(f"Outliers mask (points not in plane 1): {outliers_mask1.astype(int)}")
    print(f"Number of outliers: {N_out}")
    if N_out > 0:
        print(f"Outlier points coordinates:\n{points_outlier}")
    print()

    # ----- Fixed version: recompute plane 2 inliers ONLY within outlier subset -----
    dists2_outlier = plane_residuals(points_outlier, refined_plane2)
    refined_inliers2_outlier = dists2_outlier < threshold
    refined_inlier_count2_outlier = np.sum(refined_inliers2_outlier)
    print(f"Fixed: Residuals of outlier points to plane 2: {dists2_outlier}")
    print(f"Fixed: Inlier mask within outlier subset: {refined_inliers2_outlier.astype(int)}")
    print(f"Fixed: Inlier count within outlier subset: {refined_inlier_count2_outlier}")
    # Map back to original indices
    refined_inliers2_fixed = np.zeros(N, dtype=bool)
    refined_inliers2_fixed[outliers_mask1] = refined_inliers2_outlier
    refined_inlier_count2_fixed = np.sum(refined_inliers2_fixed)
    print(f"Fixed: Mapped back to original indices: {refined_inliers2_fixed.astype(int)}")
    print(f"Fixed: Plane 2 inlier count: {refined_inlier_count2_fixed}")
    print()

    # ----- Buggy version: recompute plane 2 inliers on FULL point cloud -----
    dists2_all = plane_residuals(points, refined_plane2)
    refined_inliers2_buggy = dists2_all < threshold
    refined_inlier_count2_buggy = np.sum(refined_inliers2_buggy)
    print(f"Buggy: Residuals of ALL points to plane 2: {dists2_all}")
    print(f"Buggy: Inlier mask (all points): {refined_inliers2_buggy.astype(int)}")
    print(f"Buggy: Plane 2 inlier count: {refined_inlier_count2_buggy}")
    print()

    # Compute overlap
    overlap_fixed = np.sum(refined_inliers1 & refined_inliers2_fixed)
    overlap_buggy = np.sum(refined_inliers1 & refined_inliers2_buggy)
    print(f"Overlap (plane1 & plane2) - Fixed: {overlap_fixed}")
    print(f"Overlap (plane1 & plane2) - Buggy: {overlap_buggy}")
    print()

    # Compute unassigned counts via union
    assigned_fixed = refined_inliers1 | refined_inliers2_fixed
    assigned_buggy = refined_inliers1 | refined_inliers2_buggy
    unassigned_fixed = N - np.sum(assigned_fixed)
    unassigned_buggy = N - np.sum(assigned_buggy)
    print(f"Assigned count (union) - Fixed: {np.sum(assigned_fixed)}")
    print(f"Assigned count (union) - Buggy: {np.sum(assigned_buggy)}")
    print(f"Unassigned count - Fixed: {unassigned_fixed}")
    print(f"Unassigned count - Buggy: {unassigned_buggy}")
    print()

    # ----- Assertions -----
    print("--- Assertions ---")
    # Fixed version should have zero overlap
    assert overlap_fixed == 0, f"Fixed version overlap should be 0, got {overlap_fixed}"
    # Fixed plane1 count should be 6 (5 plane1 points + ambiguous)
    assert refined_inlier_count1 == 6, f"Fixed plane1 count expected 6, got {refined_inlier_count1}"
    # Fixed plane2 count should be 5 (only the five plane2 points)
    assert refined_inlier_count2_fixed == 5, f"Fixed plane2 count expected 5, got {refined_inlier_count2_fixed}"
    # Fixed unassigned should be 0
    assert unassigned_fixed == 0, f"Fixed unassigned should be 0, got {unassigned_fixed}"
    # Buggy version should have overlap == 1 (the ambiguous point)
    assert overlap_buggy == 1, f"Buggy version overlap expected 1, got {overlap_buggy}"
    # Buggy plane2 count should be 6 (ambiguous + five plane2 points)
    assert refined_inlier_count2_buggy == 6, f"Buggy plane2 count expected 6, got {refined_inlier_count2_buggy}"
    # Buggy unassigned should be 0 (union still all points)
    assert unassigned_buggy == 0, f"Buggy unassigned should be 0, got {unassigned_buggy}"
    print("All assertions passed.")
    print()
    print("Test demonstrates:")
    print("- Buggy full-cloud recomputation causes overlap of 1 point (the ambiguous point).")
    print("- Corrected outlier-subset mapping yields zero overlap.")
    print("- Point counts are conserved in the fixed version.")
    print("Test completed successfully.")

if __name__ == "__main__":
    main()