import cv2
import numpy as np
from pathlib import Path
import sys

def main():
    # Dataset path
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    depth_path = base / "depth" / "000000.png"
    rgb_path = base / "rgb.mp4"
    camera_matrix_path = base / "camera_matrix.csv"

    # Load depth image (first frame)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"Error: Could not load depth image: {depth_path}")
        sys.exit(1)
    depth_height, depth_width = depth.shape[:2]

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
        # Generate pixel coordinate grids
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
        return points, zz  # return points and the depth values used (for pixel-specific lookup)

    # Compute both point clouds
    points_orig, zz_orig = compute_point_cloud(fx, fy, cx, cy)
    points_scaled, zz_scaled = compute_point_cloud(fx_scaled, fy_scaled, cx_scaled, cy_scaled)

    # Function to report statistics for a point cloud
    def report_points(title, points, depth_values_used, fx_use, fy_use, cx_use, cy_use):
        print(f"\n=== {title} ===")
        print(f"Number of valid points: {points.shape[0]}")
        if points.shape[0] > 0:
            print(f"X min/max: {points[:, 0].min():.3f}, {points[:, 0].max():.3f}")
            print(f"Y min/max: {points[:, 1].min():.3f}, {points[:, 1].max():.3f}")
            print(f"Z min/max: {points[:, 2].min():.3f}, {points[:, 2].max():.3f}")
        else:
            print("No valid points.")

        # Compute coordinates for center pixel and four corners
        # We'll compute using the same backprojection for specific pixels
        pixels = [
            ("center", (depth_width // 2, depth_height // 2)),
            ("top-left", (0, 0)),
            ("top-right", (depth_width - 1, 0)),
            ("bottom-left", (0, depth_height - 1)),
            ("bottom-right", (depth_width - 1, depth_height - 1))
        ]
        for name, (u, v) in pixels:
            z = depth[v, u]  # depth at (v, u) because array is [row, col] = [y, x]
            if z == 0:
                print(f"  {name} (u={u}, v={v}): depth=0 -> invalid")
            else:
                x = (u - cx_use) * z / fx_use
                y = (v - cy_use) * z / fy_use
                print(f"  {name} (u={u}, v={v}): Z={z} -> X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

    # Report for original intrinsics
    report_points("Original Camera Matrix (Unscaled)", points_orig, zz_orig, fx, fy, cx, cy)

    # Report for scaled intrinsics
    report_points("Scaled Intrinsics (Same-FOV Assumption)", points_scaled, zz_scaled, fx_scaled, fy_scaled, cx_scaled, cy_scaled)

    # Clear labels about assumptions
    print("\nNOTE: The scaled intrinsics above are based on the unverified same-field-of-view assumption.")
    print("      They are NOT verified depth-camera calibration.")
    print("      Depth values are in raw units (uint16); conversion to meters requires unknown depth scale.")

if __name__ == "__main__":
    main()