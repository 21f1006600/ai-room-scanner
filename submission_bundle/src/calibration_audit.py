import cv2
import numpy as np
from pathlib import Path
import sys

def main():
    # Paths
    base_dir = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    camera_matrix_path = base_dir / "camera_matrix.csv"
    rgb_path = base_dir / "rgb.mp4"
    depth_dir = base_dir / "depth"
    first_depth_path = depth_dir / "000000.png"

    # 1. Read camera matrix
    try:
        K = np.loadtxt(camera_matrix_path, delimiter=',')
    except Exception as e:
        print(f"Error reading camera matrix: {e}")
        sys.exit(1)

    if K.shape != (3, 3):
        print(f"Error: Camera matrix must be 3x3, got shape {K.shape}")
        sys.exit(1)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 2. Read first RGB frame dimensions
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

    # 3. Read first depth PNG dimensions
    try:
        depth_img = cv2.imread(str(first_depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"Error: Could not read depth image: {first_depth_path}")
            sys.exit(1)
        depth_height, depth_width = depth_img.shape[:2]
    except Exception as e:
        print(f"Error reading depth image: {e}")
        sys.exit(1)

    # 4. Calculate ratios
    ratio_x = depth_width / rgb_width
    ratio_y = depth_height / rgb_height

    # 5. Calculate hypothetical scaled intrinsics
    fx_scaled = fx * ratio_x
    fy_scaled = fy * ratio_y
    cx_scaled = cx * ratio_x
    cy_scaled = cy * ratio_y

    # 6. Print results
    print("=== Camera Matrix Contents ===")
    print(f"fx: {fx:.4f}")
    print(f"fy: {fy:.4f}")
    print(f"cx: {cx:.4f}")
    print(f"cy: {cy:.4f}")
    print()

    print("=== Image Dimensions ===")
    print(f"RGB: {rgb_width} x {rgb_height}")
    print(f"Depth: {depth_width} x {depth_height}")
    print(f"Width ratio (depth/rgb): {ratio_x:.4f}")
    print(f"Height ratio (depth/rgb): {ratio_y:.4f}")
    print()

    print("=== Hypothetical Scaled Intrinsics (Same-FOV Assumption) ===")
    print(f"fx_scaled: {fx_scaled:.4f}")
    print(f"fy_scaled: {fy_scaled:.4f}")
    print(f"cx_scaled: {cx_scaled:.4f}")
    print(f"cy_scaled: {cy_scaled:.4f}")
    print()

    # 7. Label as assumption
    print("NOTE: The scaled intrinsics above are based on the unverified same-field-of-view assumption.")
    print("      They are NOT verified depth-camera calibration.")
    print()

    # 8. State that matching frame counts do not prove alignment
    print("NOTE: Matching frame counts (RGB, depth, confidence, odometry) do NOT prove pixel alignment.")
    print("      They only indicate synchronized capture timing, not geometric correspondence.")

if __name__ == "__main__":
    main()