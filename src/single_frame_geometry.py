#!/usr/bin/env python3
"""
Single frame geometry experiment for the Applied AI assignment.
Loads one depth frame, backprojects to a point cloud using same-FOV-scaled
intrinsics (unverified assumption), and reports basic geometric metrics.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Try to import matplotlib for visualization; if not available, we'll skip the plot.
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Import the backproject_depth function and the known RGB resolution from point_cloud.py
# We assume point_cloud.py is in the same directory.
try:
    from point_cloud import backproject_depth, _RGB_WIDTH, _RGB_HEIGHT
except Exception as e:
    print(f"ERROR: Could not import from point_cloud.py: {e}")
    sys.exit(1)


def main():
    # -------------------------- Configuration --------------------------
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    dataset_name = "single_scan_floor_only/1a8384c3f6"
    dataset_path = base / dataset_name

    depth_path = dataset_path / "depth" / "000000.png"
    intrinsics_path = dataset_path / "camera_matrix.csv"

    # Output directory
    out_dir = Path("output/single_frame_geometry")
    # -------------------------- Checks --------------------------
    if not depth_path.is_file():
        print(f"ERROR: Depth image not found: {depth_path}")
        sys.exit(1)
    if not intrinsics_path.is_file():
        print(f"ERROR: Camera matrix not found: {intrinsics_path}")
        sys.exit(1)

    # Do not overwrite existing output
    if out_dir.exists():
        # If directory exists and contains any files, refuse to overwrite.
        # We'll consider any existing content as a conflict.
        if any(out_dir.iterdir()):
            print(f"ERROR: Output directory already exists and is not empty: {out_dir}")
            print("       Remove or rename it to avoid overwriting previous results.")
            sys.exit(1)
        else:
            # Empty directory is okay; we can use it.
            pass
    else:
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except Exception as e:
            print(f"ERROR: Could not create output directory: {e}")
            sys.exit(1)

    # -------------------------- Load data --------------------------
    # Load depth image (uint16)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"ERROR: Failed to read depth image: {depth_path}")
        sys.exit(1)
    if depth.ndim != 2:
        print(f"ERROR: Depth image is not 2D (shape {depth.shape})")
        sys.exit(1)

    # Load camera matrix (3x3)
    try:
        K = np.loadtxt(str(intrinsics_path), delimiter=',')
    except Exception as e:
        print(f"ERROR: Failed to load camera matrix: {e}")
        sys.exit(1)
    if K.shape != (3, 3):
        print(f"ERROR: Camera matrix must be 3x3, got shape {K.shape}")
        sys.exit(1)

    # -------------------------- Back-project --------------------------
    print(f"Processing frame: {depth_path.name}")
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

    # -------------------------- Compute statistics --------------------------
    # Point cloud columns: X, Y, Z
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]  # This is raw depth (uint16) as returned by backproject_depth

    num_points = points.shape[0]
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())
    z_min, z_max = float(Z.min()), float(Z.max())
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    z_extent = z_max - z_min

    x_p5, x_p50, x_p95 = np.percentile(X, [5, 50, 95])
    y_p5, y_p50, y_p95 = np.percentile(Y, [5, 50, 95])
    z_p5, z_p50, z_p95 = np.percentile(Z, [5, 50, 95])

    # Also compute depth statistics directly from the depth image (should match Z stats)
    valid_mask = depth > 0
    depth_vals = depth[valid_mask]
    if depth_vals.size > 0:
        depth_min = int(depth_vals.min())
        depth_median = int(np.median(depth_vals))
        depth_max = int(depth_vals.max())
    else:
        depth_min = depth_median = depth_max = 0

    # -------------------------- Report --------------------------
    print("\n=== Results (raw depth units) ===")
    print(f"Number of valid points: {num_points}")
    print(f"X range: [{x_min:.3f}, {x_max:.3f}]  extent: {x_extent:.3f}")
    print(f"Y range: [{y_min:.3f}, {y_max:.3f}]  extent: {y_extent:.3f}")
    print(f"Z range: [{z_min:.3f}, {z_max:.3f}]  extent: {z_extent:.3f}")
    print(f"X percentiles (5th, 50th, 95th): {x_p5:.3f}, {x_p50:.3f}, {x_p95:.3f}")
    print(f"Y percentiles (5th, 50th, 95th): {y_p5:.3f}, {y_p50:.3f}, {y_p95:.3f}")
    print(f"Z percentiles (5th, 50th, 95th): {z_p5:.3f}, {z_p50:.3f}, {z_p95:.3f}")
    print(f"Depth image min/median/max: {depth_min} / {depth_median} / {depth_max}")

    print("\nNote: All coordinates are in the raw depth units of the uint16 image.")
    print("      The physical scale (e.g., meters per unit) is unknown.")
    print("      The intrinsics used are the RGB matrix scaled by the resolution ratio")
    print("      (same-field-of-view assumption), which has not been verified.")

    # -------------------------- Save point cloud --------------------------
    points_file = out_dir / "point_cloud.npy"
    try:
        np.save(points_file, points)
        print(f"Saved point cloud to: {points_file}")
    except Exception as e:
        print(f"WARNING: Could not save point cloud: {e}")

    # -------------------------- Optional visualization --------------------------
    if HAVE_MPL:
        try:
            # Create a 3D scatter plot
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            # Subsample for clarity if too many points
            max_points = 50000
            if num_points > max_points:
                idx = np.random.choice(num_points, max_points, replace=False)
                pts = points[idx]
                print(f"Visualization: showing {max_points} randomly sampled points.")
            else:
                pts = points
            sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                            s=1, alpha=0.5, c='b', marker='.')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Point Cloud (Same-FOV Scaled Intrinsics - Unverified)')
            # Attempt equal aspect ratio
            max_range = np.array([
                pts[:, 0].max() - pts[:, 0].min(),
                pts[:, 1].max() - pts[:, 1].min(),
                pts[:, 2].max() - pts[:, 2].min()
            ]).max() / 2.0
            mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
            mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
            mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            viz_file = out_dir / "point_cloud_visualization.png"
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