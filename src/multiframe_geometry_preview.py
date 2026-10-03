import cv2
import numpy as np
from pathlib import Path
import sys
import os

def main():
    # Dataset path
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    rgb_path = base / "rgb.mp4"
    camera_matrix_path = base / "camera_matrix.csv"

    # Output directory
    output_dir = Path("output/multiframe_preview")
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

    print("=== Multiframe Geometry Preview (Scaled Intrinsics - Unverified Same-FOV Assumption) ===")
    for idx, frame_str in enumerate(frame_strings):
        depth_path = base / "depth" / f"{frame_str}.png"
        # Load depth image
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Error: Could not load depth image: {depth_path}")
            sys.exit(1)
        depth_height, depth_width = depth.shape[:2]

        # Compute scaling factors (same-FOV assumption)
        scale_x = depth_width / rgb_width
        scale_y = depth_height / rgb_height
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y

        # Helper to compute point cloud from intrinsics
        def compute_point_cloud(fx, fy, cx, cy):
            # Backprojection: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = depth
            u = np.arange(depth_width)
            v = np.arange(depth_height)
            uu, vv = np.meshgrid(u, v)  # shapes (height, width)
            uu = uu.ravel()
            vv = vv.ravel()
            zz = depth.ravel()

            # Mask for valid depth (non-zero)
            valid = zz > 0
            uu = uu[valid]
            vv = vv[valid]
            zz = zz[valid]

            # Avoid division by zero (fx, fy are non-zero from calibration)
            xx = (uu - cx) * zz / fx
            yy = (vv - cy) * zz / fy
            zz = zz  # depth as is

            points = np.vstack([xx, yy, zz]).T  # shape (N, 3)
            return points

        # Compute point cloud using scaled intrinsics
        points = compute_point_cloud(fx_scaled, fy_scaled, cx_scaled, cy_scaled)

        # Report
        print(f"\n--- Frame {frame_str} ---")
        print(f"Valid point count: {points.shape[0]}")
        if points.shape[0] > 0:
            print(f"X min/max: {points[:, 0].min():.3f}, {points[:, 0].max():.3f}")
            print(f"Y min/max: {points[:, 1].min():.3f}, {points[:, 1].max():.3f}")
            print(f"Z min/max: {points[:, 2].min():.3f}, {points[:, 2].max():.3f}")
            # Center pixel coordinate
            u_center = depth_width // 2
            v_center = depth_height // 2
            z_center = depth[v_center, u_center]
            if z_center == 0:
                print(f"Center pixel (u={u_center}, v={v_center}): depth=0 -> invalid")
            else:
                x_center = (u_center - cx_scaled) * z_center / fx_scaled
                y_center = (v_center - cy_scaled) * z_center / fy_scaled
                print(f"Center pixel (u={u_center}, v={v_center}): Z={z_center} -> X={x_center:.3f}, Y={y_center:.3f}, Z={z_center:.3f}")
        else:
            print("No valid points.")

        # Save point cloud
        output_filename = output_dir / f"pointcloud_scaled_frame_{frame_str}.npy"
        if output_filename.exists():
            print(f"Skipped saving point cloud for frame {frame_str} (file already exists): {output_filename}")
        else:
            np.save(output_filename, points)
            print(f"Saved point cloud to: {output_filename}")

    print("\nNOTE: The scaled intrinsics used above are based on the unverified same-field-of-view assumption.")
    print("      They are NOT verified depth-camera calibration.")
    print("      Depth values are in raw units (uint16); conversion to meters requires unknown depth scale.")
    print("      No odometry or frame-to-frame alignment has been applied.")

if __name__ == "__main__":
    main()