# Calibration Findings

## Evidence found
- `camera_matrix.csv` in each dataset: 3×3 floating‑point matrix treated as RGB intrinsics.
- Depth images are uint16 PNGs with size 192 × 256.
- RGB video (`rgb.mp4`) resolution is 1920 × 1440.
- Odometry (`odometry.csv`) provides timestamp, translation (x,y,z), and quaternion (qx,qy,qz,qw) – no explicit statement whether this is RGB‑camera pose or another frame.
- IMU data (`imu.csv`) present but not used in the pipeline.
- No separate depth intrinsics file, no depth‑to‑RGB extrinsics file, no depth‑scale (meters per unit) file, no coordinate‑frame definition file.

## What is missing or ambiguous
- No verified depth‑camera intrinsics; the pipeline assumes the depth camera shares the same field of view as the RGB camera and scales the RGB intrinsics by the ratio of pixel counts (hardcoded 1920 × 1440 → depth width/height).
- No depth‑to‑RGB extrinsic transformation (rotation + translation); the pipeline treats the odometry pose as if it were the depth camera’s pose.
- No explicit depth scale; depth values are raw uint16 with unknown conversion to meters.
- No metadata confirming the camera‑coordinate convention used in the back‑projection (X right, Y down, Z forward) – this is inferred from the standard pinhole formulas.

## Assumptions currently made by the pipeline
1. **Same‑FOV intrinsics scaling**: RGB intrinsics from `camera_matrix.csv` are scaled to depth resolution using factors `depth_width / 1920` and `depth_height / 1440`. This is an unverified hypothesis.
2. **Depth scale**: Depth values are used directly as the Z coordinate (in raw uint16 units); no conversion to metric units.
3. **Extrinsics**: The odometry pose (translation and quaternion) is applied to the point cloud as if it were the depth camera’s pose; any offset between depth and RGB optics is ignored.
4. **Coordinate frame**: The back‑projection follows the convention X = (u − cx)·z/fx (right), Y = (v − cy)·z/fy (down), Z = z (forward). This is assumed to match the odometry frame’s axes, but no verification exists.
5. **Timestamp synchronization**: Depth frame index is assumed to match odometry row index (with possible offsets tested in other scripts).

## Clear recommendation for the next experiment
> The dataset lacks independently verified depth intrinsics, depth-to-RGB extrinsics, and metric depth scale.
> 
> Same-FOV-scaled point clouds are exploratory visualizations, not validated 3D reconstructions.
> 
> All derived measurements remain unvalidated and in raw depth units.
> 
> Further quantitative validation requires independent calibration information or a known-size reference with corresponding depth measurements.
> 
> Do not propose same-intrinsics round-trip reprojection as a calibration test.
