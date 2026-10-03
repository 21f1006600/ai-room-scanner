import cv2
import numpy as np
import os
from pathlib import Path


# Known RGB resolution from context (can be verified externally)
_RGB_WIDTH = 1920
_RGB_HEIGHT = 1440


def backproject_depth(depth_path, intrinsics_path, use_scaled_intrinsics=False):
    """
    Back-project a depth image to a point cloud using pinhole camera model.

    Args:
        depth_path: Path to depth PNG (uint16, raw values).
        intrinsics_path: Path to camera matrix CSV (3x3).
        use_scaled_intrinsics: If True, scale the intrinsics from the RGB camera
                               to the depth resolution assuming same field of view
                               (i.e., focal length and principal point scale linearly
                               with pixel count). This is a hypothesis, not verified
                               depth‑camera calibration.

    Returns:
        numpy array of shape (N, 3) with XYZ points (raw depth units).
    """
    # Load depth image
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Could not load depth image: {depth_path}")

    # Load camera matrix
    try:
        K = np.loadtxt(intrinsics_path, delimiter=',')
    except Exception as e:
        raise ValueError(f"Could not load camera matrix: {e}")

    if K.shape != (3, 3):
        raise ValueError(f"Camera matrix must be 3x3, got shape {K.shape}")

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # If using scaled intrinsics (same-FOV assumption)
    if use_scaled_intrinsics:
        h, w = depth.shape
        scale_x = w / _RGB_WIDTH
        scale_y = h / _RGB_HEIGHT
        fx = fx * scale_x
        fy = fy * scale_y
        cx = cx * scale_x
        cy = cy * scale_y

    # Image dimensions
    h, w = depth.shape

    # Generate pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)  # shapes (h, w)

    # Flatten
    uu = uu.ravel()
    vv = vv.ravel()
    zz = depth.ravel()

    # Mask for valid depth (non-zero)
    valid = zz > 0
    uu = uu[valid]
    vv = vv[valid]
    zz = zz[valid]

    # Back-project
    xx = (uu - cx) * zz / fx
    yy = (vv - cy) * zz / fy
    zz = zz  # depth as is

    points = np.vstack([xx, yy, zz]).T  # shape (N, 3)
    return points


def _test():
    """Test both raw and scaled intrinsics on the first frame."""
    # Paths relative to project root
    dataset_dir = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    depth_path = dataset_dir / "depth" / "000000.png"
    intrinsics_path = dataset_dir / "camera_matrix.csv"

    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # --- Baseline: raw intrinsics from camera_matrix.csv ---
    print("=== Using raw intrinsics (camera_matrix.csv) ===")
    points_raw = backproject_depth(depth_path, intrinsics_path, use_scaled_intrinsics=False)
        # Actually we need to retrieve the intrinsics used; we can reload or compute inside function.
    # Let's reload the matrix to display.
    K_raw = np.loadtxt(str(intrinsics_path), delimiter=',')
    fx_raw, fy_raw, cx_raw, cy_raw = K_raw[0,0], K_raw[1,1], K_raw[0,2], K_raw[1,2]
    print(f"Intrinsics used: fx={fx_raw:.2f}, fy={fy_raw:.2f}, cx={cx_raw:.2f}, cy={cy_raw:.2f}")
    print(f"Number of valid points: {points_raw.shape[0]}")
    print(f"XYZ array shape: {points_raw.shape}")
    if points_raw.size > 0:
        print(f"X min/max: {points_raw[:, 0].min():.3f}, {points_raw[:, 0].max():.3f}")
        print(f"Y min/max: {points_raw[:, 1].min():.3f}, {points_raw[:, 1].max():.3f}")
        print(f"Z min/max: {points_raw[:, 2].min():.3f}, {points_raw[:, 2].max():.3f}")
        print("First 3 points:")
        for i in range(min(3, points_raw.shape[0])):
            print(f"  {points_raw[i]}")
    else:
        print("No valid points.")
    raw_path = output_dir / "point_cloud_raw_intrinsics_000000.npy"
    if raw_path.exists():
        print(f"Skipped saving raw intrinsics point cloud (file already exists): {raw_path}")
    else:
        np.save(raw_path, points_raw)
        print(f"Saved raw intrinsics point cloud to: {raw_path}")
    print()

    # --- Scaled intrinsics (same-FOV assumption) ---
    print("=== Using scaled intrinsics (same-FOV assumption) ===")
    points_scaled = backproject_depth(depth_path, intrinsics_path, use_scaled_intrinsics=True)
    # Compute scaled intrinsics for display
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    h, w = depth_img.shape
    scale_x = w / _RGB_WIDTH
    scale_y = h / _RGB_HEIGHT
    fx_scaled = fx_raw * scale_x
    fy_scaled = fy_raw * scale_y
    cx_scaled = cx_raw * scale_x
    cy_scaled = cy_raw * scale_y
    print(f"Intrinsics used: fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, cx={cx_scaled:.2f}, cy={cy_scaled:.2f}")
    print(f"Number of valid points: {points_scaled.shape[0]}")
    print(f"XYZ array shape: {points_scaled.shape}")
    if points_scaled.size > 0:
        print(f"X min/max: {points_scaled[:, 0].min():.3f}, {points_scaled[:, 0].max():.3f}")
        print(f"Y min/max: {points_scaled[:, 1].min():.3f}, {points_scaled[:, 1].max():.3f}")
        print(f"Z min/max: {points_scaled[:, 2].min():.3f}, {points_scaled[:, 2].max():.3f}")
        print("First 3 points:")
        for i in range(min(3, points_scaled.shape[0])):
            print(f"  {points_scaled[i]}")
    else:
        print("No valid points.")
    scaled_path = output_dir / "point_cloud_scaled_intrinsics_assumption_000000.npy"
    if scaled_path.exists():
        print(f"Skipped saving scaled intrinsics point cloud (file already exists): {scaled_path}")
    else:
        np.save(scaled_path, points_scaled)
        print(f"Saved scaled intrinsics point cloud to: {scaled_path}")


if __name__ == "__main__":
    _test()