#!/usr/bin/env python3
"""
Plane detection experiment using RANSAC on a single-frame point cloud.
Loads the point cloud from output/single_frame_geometry/point_cloud.npy,
fits a plane via RANSAC, and reports results in raw depth units.
"""

import sys
import numpy as np
from pathlib import Path

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def main():
    # -------------------------- Configuration --------------------------
    # Input point cloud (from previous experiment)
    pc_path = Path("output/single_frame_geometry/point_cloud.npy")
    # Output directory for this experiment
    out_dir = Path("output/plane_detection_experiment")

    # RANSAC parameters
    np.random.seed(42)  # reproducible
    n_iterations = 1000
    # Distance threshold: we'll set as 1% of the depth range (Z) of the point cloud
    # This is a heuristic; we'll compute after loading the point cloud.
    # Alternatively, we could set a fixed value (e.g., 10 units). We'll compute dynamically.
    # -------------------------- Checks --------------------------
    if not pc_path.is_file():
        print(f"ERROR: Point cloud not found: {pc_path}")
        print("       Run src/single_frame_geometry.py first to generate the point cloud.")
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

    print(f"Loaded point cloud with {points.shape[0]} points from {pc_path}")
    print(f"Point cloud dtype: {points.dtype}")

    # Compute depth range for adaptive threshold
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_range = z_max - z_min
    distance_threshold = 0.01 * z_range  # 1% of depth range
    if distance_threshold < 1.0:  # ensure a minimum threshold of 1 unit
        distance_threshold = 1.0
    print(f"Depth range (Z): [{z_min:.3f}, {z_max:.3f}] -> range = {z_range:.3f}")
    print(f"Using distance threshold = {distance_threshold:.3f} (1% of depth range, min 1.0)")

    # -------------------------- Helper: fit plane to 3 points --------------------------
    def fit_plane(p1, p2, p3):
        """Return plane coefficients (a, b, c, d) for ax+by+cz+d=0, normalized so that sqrt(a^2+b^2+c^2)=1."""
        # vectors
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return None  # points are collinear
        normal = normal / norm
        d = -np.dot(normal, p1)
        return np.array([normal[0], normal[1], normal[2], d])

    # -------------------------- Helper: distance from point to plane --------------------------
    def point_to_plane_distance(point, plane):
        """Signed distance; plane is (a,b,c,d) with normalized normal."""
        a, b, c, d = plane
        return a*point[0] + b*point[1] + c*point[2] + d

    # -------------------------- RANSAC loop --------------------------
    best_inliers = None
    best_plane = None
    best_inlier_count = 0

    N = points.shape[0]
    for it in range(n_iterations):
        # Randomly select 3 distinct points
        idx = np.random.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]
        plane = fit_plane(p1, p2, p3)
        if plane is None:
            continue
        # Compute distances for all points
        dists = np.abs(np.dot(points, plane[:3]) + plane[3])  # |ax+by+cz+d|
        inliers = dists < distance_threshold
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane
            best_inliers = inliers
        # Early exit if we have a very good fit
        if inlier_count > 0.9 * N:
            break

    if best_plane is None:
        print("ERROR: RANSAC failed to find a plane.")
        sys.exit(1)

    # -------------------------- Refine plane using all inliers (least squares) --------------------------
    if best_inlier_count > 0:
        inlier_points = points[best_inliers]
        # Fit plane via SVD: solve for normal that minimizes sum((n·p + d)^2) with ||n||=1
        centroid = np.mean(inlier_points, axis=0)
        # Center the points
        centered = inlier_points - centroid
        # Compute covariance matrix
        cov = np.dot(centered.T, centered)
        # Eigen decomposition
        evals, evecs = np.linalg.eigh(cov)
        # Normal is the eigenvector corresponding to the smallest eigenvalue
        normal = evecs[:, 0]
        # Ensure normal points roughly towards the camera? Not necessary; we just want a normal.
        # Compute d
        d = -np.dot(normal, centroid)
        refined_plane = np.array([normal[0], normal[1], normal[2], d])
        # Normalize (should already be unit norm from eigh, but ensure)
        norm = np.linalg.norm(refined_plane[:3])
        if norm > 0:
            refined_plane = refined_plane / norm
    else:
        refined_plane = best_plane
        inlier_points = np.empty((0, 3))

    # -------------------------- Compute statistics --------------------------
    a, b, c, d = refined_plane
    normal_vec = np.array([a, b, c])
    # Distance of all points to the refined plane
    all_dists = np.abs(np.dot(points, normal_vec) + d)
    inlier_mask = all_dists < distance_threshold
    inlier_count = np.sum(inlier_mask)
    outlier_count = N - inlier_count
    inlier_ratio = inlier_count / N * 100.0

    # Distances for inliers only
    if inlier_count > 0:
        inlier_dists = all_dists[inlier_mask]
        mean_inlier_dist = np.mean(inlier_dists)
        median_inlier_dist = np.median(inlier_dists)
    else:
        mean_inlier_dist = 0.0
        median_inlier_dist = 0.0

    # -------------------------- Report --------------------------
    print("\n=== Plane Detection Results (RANSAC) ===")
    print(f"Model: {a:.6f} x + {b:.6f} y + {c:.6f} z + {d:.6f} = 0")
    print(f"Normal vector: [{a:.6f}, {b:.6f}, {c:.6f}] (unit length)")
    print(f"Inliers: {inlier_count} / {N} ({inlier_ratio:.2f}%)")
    print(f"Outliers: {outlier_count} / {N} ({100.0 - inlier_ratio:.2f}%)")
    print(f"Distance threshold used: {distance_threshold:.3f} (raw depth units)")
    print(f"Mean absolute inlier distance: {mean_inlier_dist:.3f}")
    print(f"Median absolute inlier distance: {median_inlier_dist:.3f}")

    print("\nImportant notes:")
    print("- All quantities are in the raw depth units of the uint16 image.")
    print("- The plane detected is a dominant plane in the point cloud,")
    print("  but it is not necessarily the floor, wall, or any specific surface.")
    print("- The same-FOV intrinsics assumption (used to create the point cloud)")
    print("  remains unverified; thus the plane orientation in real space is unknown.")
    print("- No conversion to metric units (e.g., meters) has been performed.")

    # -------------------------- Save results (optional) --------------------------
    # Save plane coefficients to a text file
    coeff_file = out_dir / "plane_coefficients.txt"
    try:
        with open(coeff_file, 'w') as f:
            f.write(f"Plane equation: {a:.6f} x + {b:.6f} y + {c:.6f} z + {d:.6f} = 0\n")
            f.write(f"Normal vector: [{a:.6f}, {b:.6f}, {c:.6f}]\n")
            f.write(f"Inliers: {inlier_count} / {N} ({inlier_ratio:.2f}%)\n")
            f.write(f"Distance threshold: {distance_threshold:.6f}\n")
            f.write(f"Mean inlier distance: {mean_inlier_dist:.6f}\n")
            f.write(f"Median inlier distance: {median_inlier_dist:.6f}\n")
        print(f"Saved plane coefficients to: {coeff_file}")
    except Exception as e:
        print(f"WARNING: Could not save coefficients: {e}")

    # -------------------------- Visualization --------------------------
    if HAVE_MPL:
        try:
            # Subsample for clarity if too many points
            max_points = 50000
            if N > max_points:
                idx = np.random.choice(N, max_points, replace=False)
                pts_vis = points[idx]
                inlier_vis = inlier_mask[idx]
                print(f"Visualization: showing {max_points} randomly sampled points.")
            else:
                pts_vis = points
                inlier_vis = inlier_mask

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Outliers in gray
            ax.scatter(pts_vis[~inlier_vis, 0], pts_vis[~inlier_vis, 1], pts_vis[~inlier_vis, 2],
                       c='lightgray', s=1, alpha=0.5, label='Outliers')
            # Inliers in red
            ax.scatter(pts_vis[inlier_vis, 0], pts_vis[inlier_vis, 1], pts_vis[inlier_vis, 2],
                       c='red', s=1, alpha=0.8, label='Inliers')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('RANSAC Plane Detection\n'
                         f'Inliers: {inlier_count}/{N} ({inlier_ratio:.1f}%)')
            ax.legend()

            # Try to set equal aspect ratio
            max_range = np.array([
                pts_vis[:, 0].max() - pts_vis[:, 0].min(),
                pts_vis[:, 1].max() - pts_vis[:, 1].min(),
                pts_vis[:, 2].max() - pts_vis[:, 2].min()
            ]).max() / 2.0
            mid_x = (pts_vis[:, 0].max() + pts_vis[:, 0].min()) * 0.5
            mid_y = (pts_vis[:, 1].max() + pts_vis[:, 1].min()) * 0.5
            mid_z = (pts_vis[:, 2].max() + pts_vis[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            viz_file = out_dir / "plane_visualization.png"
            plt.savefig(viz_file, dpi=150)
            plt.close(fig)
            print(f"Saved visualization to: {viz_file}")
        except Exception as e:
            print(f"WARNING: Could not create visualization: {e}")
    else:
        print("INFO: Matplotlib not available; skipping visualization.")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()