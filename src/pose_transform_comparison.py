import csv
import numpy as np
import os
from pathlib import Path
import sys
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def main():
    # Dataset path
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    odometry_path = base / "odometry.csv"
    rgb_path = base / "rgb.mp4"
    camera_matrix_path = base / "camera_matrix.csv"

    # Point cloud directory from multiframe_geometry_preview
    preview_dir = Path("output/multiframe_preview")
    if not preview_dir.is_dir():
        print(f"Error: Preview directory not found: {preview_dir}")
        print("Please run src/multiframe_geometry_preview.py first to generate point clouds.")
        sys.exit(1)

    # Output directory for plots (rotation-only comparison)
    output_dir = Path("output/pose_rotation_only_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load camera matrix
    try:
        K = np.loadtxt(str(camera_matrix_path), delimiter=',')
    except Exception as e:
        print(f"Error loading camera matrix: {e}")
        sys.exit(1)
    if K.shape != (3, 3):
        print(f"Error: Camera matrix must be 3x3, got shape {K.shape}")
        sys.exit(1)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Get RGB video dimensions (width, height) from first frame
    try:
        cap = cv2.VideoCapture(str(rgb_path))
        if not cap.isOpened():
            print(f"Error: Could not open RGB video: {rgb_path}")
            sys.exit(1)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame from RGB video: {rgb_path}")
            sys.exit(1)
        rgb_height, rgb_width = frame.shape[:2]
        cap.release()
    except Exception as e:
        print(f"Error reading RGB video: {e}")
        sys.exit(1)

    # Frames to process
    frame_numbers = [0, 857, 1714]
    frame_strings = [f"{n:06d}" for n in frame_numbers]

    # Load odometry data for these frames
    # We'll read the whole CSV and pick the rows we need
    frames_map = {}  # frame_str -> (timestamp, x, y, z, qx, qy, qz, qw)
    try:
        with open(odometry_path, 'r') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print("Error: Could not read CSV header")
                sys.exit(1)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            required = ['frame', 'timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            for col in required:
                if col not in reader.fieldnames:
                    print(f"Error: Missing column '{col}' in odometry CSV")
                    sys.exit(1)
            for row in reader:
                frame_id = row['frame'].strip()
                if frame_id in frame_strings:
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    qx = float(row['qx'])
                    qy = float(row['qy'])
                    qz = float(row['qz'])
                    qw = float(row['qw'])
                    frames_map[frame_id] = (float(row['timestamp']), x, y, z, qx, qy, qz, qw)
    except Exception as e:
        print(f"Error reading odometry CSV: {e}")
        sys.exit(1)

    # Check we have all three frames
    missing = [f for f in frame_strings if f not in frames_map]
    if missing:
        print(f"Error: Missing odometry data for frames: {missing}")
        sys.exit(1)

    # Function to compute rotation matrix from quaternion [x, y, z, w]
    def quat_to_rotmat(q):
        x, y, z, w = q
        # Ensure unit quaternion (should already be normalized)
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return None
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - z*w)
        r02 = 2*(x*z + y*w)
        r10 = 2*(x*y + z*w)
        r11 = 1 - 2*(x*x + z*z)
        r12 = 2*(y*z - x*w)
        r20 = 2*(x*z - y*w)
        r21 = 2*(y*z + x*w)
        r22 = 1 - 2*(x*x + y*y)
        return np.array([[r00, r01, r02],
                         [r10, r11, r12],
                         [r20, r21, r22]])

    # Store results for reporting
    results_A = []  # list of dicts for each frame
    results_B = []

    # Process each frame
    for frame_str in frame_strings:
        # Load point cloud
        pc_path = preview_dir / f"pointcloud_scaled_frame_{frame_str}.npy"
        if not pc_path.is_file():
            print(f"Error: Point cloud file not found: {pc_path}")
            sys.exit(1)
        points = np.load(pc_path)  # shape (N, 3)
        if points.shape[1] != 3:
            print(f"Error: Point cloud does not have shape (N,3): {points.shape}")
            sys.exit(1)

        # Get odometry data
        timestamp, x, y, z, qx, qy, qz, qw = frames_map[frame_str]
        translation = np.array([x, y, z])  # as row vector for addition
        q = np.array([qx, qy, qz, qw])

        # Compute rotation matrices
        R_A = quat_to_rotmat(q)          # q as camera->world
        q_conj = np.array([-qx, -qy, -qz, qw])  # conjugate
        R_B = quat_to_rotmat(q_conj)     # q^{-1} as camera->world (if q is world->camera)

        if R_A is None or R_B is None:
            print(f"Error: Failed to compute rotation matrix for frame {frame_str}")
            sys.exit(1)

        # Apply transformations (rotation-only, no translation)
        # p_world = R @ p_camera^T  -> points @ R.T
        transformed_A = points @ R_A.T
        transformed_B = points @ R_B.T

        # Compute ranges and centroid
        def compute_stats(pts):
            if pts.shape[0] == 0:
                return None, None, None
            xmin, xmax = np.min(pts[:, 0]), np.max(pts[:, 0])
            ymin, ymax = np.min(pts[:, 1]), np.max(pts[:, 1])
            zmin, zmax = np.min(pts[:, 2]), np.max(pts[:, 2])
            centroid = np.mean(pts, axis=0)
            return (xmin, xmax, ymin, ymax, zmin, zmax), centroid

        range_A, centroid_A = compute_stats(transformed_A)
        range_B, centroid_B = compute_stats(transformed_B)

        results_A.append({
            'frame': frame_str,
            'points': transformed_A,
            'range': range_A,
            'centroid': centroid_A,
            'translation': translation,
            'quat': q,
            'R': R_A
        })
        results_B.append({
            'frame': frame_str,
            'points': transformed_B,
            'range': range_B,
            'centroid': centroid_B,
            'translation': translation,
            'quat': q,
            'R': R_B
        })

        # Print per-frame info
        print(f"\n=== Frame {frame_str} ===")
        print(f"Translation: ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})")
        print(f"Quaternion: ({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})")
        if range_A:
            print(f"Transform A - XYZ range: x[{range_A[0]:.3f}, {range_A[1]:.3f}], y[{range_A[2]:.3f}, {range_A[3]:.3f}], z[{range_A[4]:.3f}, {range_A[5]:.3f}]")
            print(f"Transform A - Centroid: ({centroid_A[0]:.3f}, {centroid_A[1]:.3f}, {centroid_A[2]:.3f})")
        if range_B:
            print(f"Transform B - XYZ range: x[{range_B[0]:.3f}, {range_B[1]:.3f}], y[{range_B[2]:.3f}, {range_B[3]:.3f}], z[{range_B[4]:.3f}, {range_B[5]:.3f}]")
            print(f"Transform B - Centroid: ({centroid_B[0]:.3f}, {centroid_B[1]:.3f}, {centroid_B[2]:.3f})")

    # Generate and save plots
    colors = ['r', 'g', 'b']  # red, green, blue for the three frames
    frame_labels = ['000000', '000857', '001714']

    # Plot for Interpretation A
    fig_A = plt.figure(figsize=(10, 8))
    ax_A = fig_A.add_subplot(111, projection='3d')
    for i, res in enumerate(results_A):
        pts = res['points']
        ax_A.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors[i], label=f'Frame {frame_labels[i]}', s=1, alpha=0.5)
    ax_A.set_xlabel('X')
    ax_A.set_ylabel('Y')
    ax_A.set_zlabel('Z')
    ax_A.set_title('Transform A: p_world = R(q) @ p_camera\n(Unverified Same-FOV Assumption + Quaternion as camera->world, rotation-only)')
    ax_A.legend()
    # Try to set equal aspect ratio
    try:
        # Get the range of each axis
        all_pts_A = np.vstack([res['points'] for res in results_A])
        max_range = np.array([all_pts_A[:,0].max()-all_pts_A[:,0].min(),
                              all_pts_A[:,1].max()-all_pts_A[:,1].min(),
                              all_pts_A[:,2].max()-all_pts_A[:,2].min()]).max() / 2.0
        mid_x = (all_pts_A[:,0].max()+all_pts_A[:,0].min()) * 0.5
        mid_y = (all_pts_A[:,1].max()+all_pts_A[:,1].min()) * 0.5
        mid_z = (all_pts_A[:,2].max()+all_pts_A[:,2].min()) * 0.5
        ax_A.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_A.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_A.set_zlim(mid_z - max_range, mid_z + max_range)
    except Exception as e:
        print(f"Warning: Could not set equal aspect ratio for plot A: {e}")

    plot_A_path = output_dir / "transform_A_comparison.png"
    if plot_A_path.is_file():
        print(f"Skipped saving plot A (file already exists): {plot_A_path}")
    else:
        fig_A.savefig(plot_A_path, dpi=150)
        print(f"Saved plot A to: {plot_A_path}")
    plt.close(fig_A)

    # Plot for Interpretation B
    fig_B = plt.figure(figsize=(10, 8))
    ax_B = fig_B.add_subplot(111, projection='3d')
    for i, res in enumerate(results_B):
        pts = res['points']
        ax_B.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors[i], label=f'Frame {frame_labels[i]}', s=1, alpha=0.5)
    ax_B.set_xlabel('X')
    ax_B.set_ylabel('Y')
    ax_B.set_zlabel('Z')
    ax_B.set_title('Transform B: p_world = R(q^{-1}) @ p_camera\n(Unverified Same-FOV Assumption + Quaternion as world->camera, rotation-only)')
    ax_B.legend()
    # Try to set equal aspect ratio
    try:
        all_pts_B = np.vstack([res['points'] for res in results_B])
        max_range = np.array([all_pts_B[:,0].max()-all_pts_B[:,0].min(),
                              all_pts_B[:,1].max()-all_pts_B[:,1].min(),
                              all_pts_B[:,2].max()-all_pts_B[:,2].min()]).max() / 2.0
        mid_x = (all_pts_B[:,0].max()+all_pts_B[:,0].min()) * 0.5
        mid_y = (all_pts_B[:,1].max()+all_pts_B[:,1].min()) * 0.5
        mid_z = (all_pts_B[:,2].max()+all_pts_B[:,2].min()) * 0.5
        ax_B.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_B.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_B.set_zlim(mid_z - max_range, mid_z + max_range)
    except Exception as e:
        print(f"Warning: Could not set equal aspect ratio for plot B: {e}")

    plot_B_path = output_dir / "transform_B_comparison.png"
    if plot_B_path.is_file():
        print(f"Skipped saving plot B (file already exists): {plot_B_path}")
    else:
        fig_B.savefig(plot_B_path, dpi=150)
        print(f"Saved plot B to: {plot_B_path}")
    plt.close(fig_B)

    print("\nNOTE: Both transformations are rotation-only (no translation applied):")
    print("      - Depth camera intrinsics are approximated by scaling RGB intrinsics by depth/RGB resolution ratio (same-FOV hypothesis).")
    print("      - The quaternion convention (camera-to-world or world-to-camera) is unknown; both are tested as hypotheses.")
    print("      - Depth values are in raw units (uint16); conversion to meters requires unknown depth scale.")
    print("      - Translation values from odometry are reported separately but NOT applied, as their units and the depth scale are unknown.")
    print("      - Rotation-only comparison cannot establish correct world alignment without knowing translation and depth scale.")
    print("      - No optimization, ICP, or fusion has been applied.")
    print("      - The resulting coordinates are not claimed to be accurate or in meters.")

if __name__ == "__main__":
    main()