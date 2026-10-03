#!/usr/bin/env python3
"""
Two-frame rotation-only diagnostic: tests quaternion conventions and
odometry index offsets by rotating point clouds from two depth frames
and comparing their alignment in a common frame (no translation applied).
"""
import sys
import numpy as np
import cv2
import json
import csv
from pathlib import Path
import time

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Import backproject_depth and assumed RGB resolution from point_cloud.py
try:
    from point_cloud import backproject_depth, _RGB_WIDTH, _RGB_HEIGHT
except Exception as e:
    print(f"ERROR: Could not import from point_cloud.py: {e}")
    sys.exit(1)


def quat_to_rotmat(q):
    """Convert quaternion [x, y, z, w] to rotation matrix (camera->world if q is that orientation)."""
    x, y, z, w = q
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


def load_odometry_row(odometry_path, frame_id):
    """Return (timestamp, x, y, z, qx, qy, qz, qw) for the given frame_id (string like '000000')."""
    with open(odometry_path, 'r') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Could not read CSV header")
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        required = ['frame', 'timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing column '{col}' in odometry CSV")
        for row in reader:
            if row['frame'].strip() == frame_id:
                return (float(row['timestamp']),
                        float(row['x']), float(row['y']), float(row['z']),
                        float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw']))
    raise ValueError(f"Frame ID {frame_id} not found in odometry CSV")


def apply_rotation_only(points, R):
    """Apply rotation-only transform: p' = R @ p (points shape Nx3)."""
    if R is None:
        return None
    return points @ R.T  # (Nx3) @ (3x3) -> (Nx3)


def compute_centroid(points):
    """Return centroid of points (3,) or None if empty."""
    if points is None or len(points) == 0:
        return None
    return np.mean(points, axis=0)


def compute_centroid_distance(p1, p2):
    """Euclidean distance between centroids."""
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(p1 - p2))


def main():
    # -------------------------- Configuration --------------------------
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    dataset_name = "single_room/c00a170fe1"
    dataset_path = base / dataset_name
    depth_dir = dataset_path / "depth"
    odometry_path = dataset_path / "odometry.csv"
    intrinsics_path = dataset_path / "camera_matrix.csv"

    # Frames to compare (as integers)
    frame_ids = [0, 1]
    frame_strings = [f"{f:06d}" for f in frame_ids]
    offsets = [-1, 0, 1]  # test odometry index offsets

    # Output directory
    out_dir = Path("output/two_frame_fusion_check")
    # -------------------------- Checks --------------------------
    if not depth_dir.is_dir():
        print(f"ERROR: Depth directory not found: {depth_dir}")
        sys.exit(1)
    if not odometry_path.is_file():
        print(f"ERROR: Odometry file not found: {odometry_path}")
        sys.exit(1)
    if not intrinsics_path.is_file():
        print(f"ERROR: Camera matrix not found: {intrinsics_path}")
        sys.exit(1)

    # Get list of depth frames to validate indices
    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        print(f"ERROR: No depth images found in {depth_dir}")
        sys.exit(1)
    max_index = len(depth_files) - 1
    for f in frame_ids:
        if f < 0 or f > max_index:
            print(f"ERROR: Frame index {f} out of range (0-{max_index})")
            sys.exit(1)

    # Do not overwrite existing output
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

    # -------------------------- Load data --------------------------
    print(f"Loading depth frames: {frame_strings}")
    depth_images = []
    point_clouds = []  # each in its own camera frame (raw depth units)
    for fs in frame_strings:
        depth_path = depth_dir / f"{fs}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"ERROR: Failed to read depth image: {depth_path}")
            sys.exit(1)
        if depth.ndim != 2:
            print(f"ERROR: Depth image is not 2D (shape {depth.shape})")
            sys.exit(1)
        depth_images.append(depth)
        # Backproject using same-FOV scaled intrinsics (unverified assumption)
        try:
            points = backproject_depth(
                depth_path,
                intrinsics_path,
                use_scaled_intrinsics=True
            )
        except Exception as e:
            print(f"ERROR: Back-projection failed for {depth_path}: {e}")
            sys.exit(1)
        if points.size == 0:
            print(f"ERROR: No valid points found in {depth_path}")
            sys.exit(1)
        point_clouds.append(points)
        print(f"  Frame {fs}: {points.shape[0]} valid points")

    # Load odometry into a map: frame_string -> (timestamp, x, y, z, qx, qy, qz, qw)
    print(f"Loading odometry from: {odometry_path}")
    odometry_map = {}
    try:
        with open(odometry_path, 'r') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("Could not read CSV header")
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            required = ['frame', 'timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            for col in required:
                if col not in reader.fieldnames:
                    raise ValueError(f"Missing column '{col}' in odometry CSV")
            for row in reader:
                frame_id = row['frame'].strip()
                odometry_map[frame_id] = (
                    float(row['timestamp']),
                    float(row['x']), float(row['y']), float(row['z']),
                    float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])
                )
    except Exception as e:
        print(f"ERROR: Failed to read odometry CSV: {e}")
        sys.exit(1)

    # Determine which offsets are valid for both frames
    valid_offsets = []
    skipped_offsets = []
    for off in offsets:
        f0_target = frame_ids[0] + off
        f1_target = frame_ids[1] + off
        f0_key = f"{f0_target:06d}"
        f1_key = f"{f1_target:06d}"
        if f0_key in odometry_map and f1_key in odometry_map:
            valid_offsets.append(off)
        else:
            skipped_offsets.append(off)
    if skipped_offsets:
        print(f"INFO: Skipping offsets {skipped_offsets} due to missing odometry data.")
    if not valid_offsets:
        print("ERROR: No valid offsets available for both frames.")
        sys.exit(1)

    # -------------------------- Process rotations --------------------------
    # We'll store results in a nested dict:
    # results[frame_idx][offset][conv] = {
    #     'odometry_frame': target_frame,
    #     'timestamp': timestamp,
    #     'quaternion': [qx, qy, qz, qw],
    #     'R_A': rotation matrix (camera->world hypothesis),
    #     'R_B': rotation matrix (world->camera hypothesis via conjugate),
    #     'points_a': rotated point cloud (frame_i points rotated with R_A),
    #     'points_b': rotated point cloud (frame_i points rotated with R_B),
    #     'centroid_a': centroid of points_a,
    #     'centroid_b': centroid of points_b,
    # }
    results = {}
    for i, fs in enumerate(frame_strings):
        results[i] = {}
        for off in valid_offsets:
            target_f = frame_ids[i] + off
            target_fs = f"{target_f:06d}"
            timestamp, x, y, z, qx, qy, qz, qw = odometry_map[target_fs]
            q = np.array([qx, qy, qz, qw])
            R_A = quat_to_rotmat(q)          # hypothesis A: q rotates camera->world
            q_conj = np.array([-qx, -qy, -qz, qw])  # conjugate
            R_B = quat_to_rotmat(q_conj)     # hypothesis B: q rotates world->camera (use inverse)
            # Apply rotations to the point cloud of frame i
            points_a = apply_rotation_only(point_clouds[i], R_A)
            points_b = apply_rotation_only(point_clouds[i], R_B)
            centroid_a = compute_centroid(points_a)
            centroid_b = compute_centroid(points_b)
            results[i][off] = {
                'odometry_frame': target_fs,
                'timestamp': timestamp,
                'quaternion': q.tolist(),
                'R_A': R_A.tolist() if R_A is not None else None,
                'R_B': R_B.tolist() if R_B is not None else None,
                'points_a': points_a,  # keep numpy array for metric computation
                'points_b': points_b,
                'centroid_a': centroid_a.tolist() if centroid_a is not None else None,
                'centroid_b': centroid_b.tolist() if centroid_b is not None else None,
            }

    # -------------------------- Compute metrics --------------------------
    # We'll compute centroid distances between the two frames for each offset and convention
    metrics = {}
    for off in valid_offsets:
        metrics[off] = {}
        # Original camera-frame points (no rotation)
        centroid_orig_0 = compute_centroid(point_clouds[0])
        centroid_orig_1 = compute_centroid(point_clouds[1])
        dist_orig = compute_centroid_distance(centroid_orig_0, centroid_orig_1)
        metrics[off]['original_camera_frame'] = {
            'centroid_0': centroid_orig_0.tolist() if centroid_orig_0 is not None else None,
            'centroid_1': centroid_orig_1.tolist() if centroid_orig_1 is not None else None,
            'centroid_distance_diagnostic': dist_orig
        }
        for conv_label, conv_key in [('camera_to_world', 'a'), ('world_to_camera', 'b')]:
            pts0 = results[0][off][f'points_{conv_key}']
            pts1 = results[1][off][f'points_{conv_key}']
            cent0 = results[0][off][f'centroid_{conv_key}']
            cent1 = results[1][off][f'centroid_{conv_key}']
            dist_cent = compute_centroid_distance(
                np.array(cent0) if cent0 is not None else None,
                np.array(cent1) if cent1 is not None else None
            )
            metrics[off][conv_label] = {
                'centroid_0': cent0,
                'centroid_1': cent1,
                'centroid_distance_diagnostic': dist_cent
            }

    # -------------------------- Save results --------------------------
    # JSON report
    report = {
        'frame_ids': frame_strings,
        'offsets_tested': valid_offsets,  # only those we actually used
        'skipped_offsets': skipped_offsets,
        'point_cloud_stats': {
            f'frame_{fs}': {
                'num_points': int(point_clouds[i].shape[0]),
                'xyz_min': point_clouds[i].min(axis=0).tolist(),
                'xyz_max': point_clouds[i].max(axis=0).tolist(),
            } for i, fs in enumerate(frame_strings)
        },
        'odometry_used': {},
        'metrics': metrics,
        'assumptions': [
            "Depth values are raw uint16 units; physical scale unknown.",
            "Same-FOV intrinsics scaling (used to back-project depth) is unverified.",
            "Odometry translation units unknown; translation NOT applied (rotation-only).",
            "Quaternion convention (camera-to-world vs world-to-camera) is unverified; both tested.",
            "Timestamp synchronization assumed via matching frame IDs; offsets tested to probe jitter/latency.",
            "No metric accuracy claims; this experiment tests internal consistency only.",
            "Centroid distance after rotation-only transformation is a limited diagnostic and does not prove surface alignment."
        ]
    }
    # Fill odometry_used per frame and offset
    for i, fs in enumerate(frame_strings):
        report['odometry_used'][fs] = {}
        for off in valid_offsets:
            target_fs = results[i][off]['odometry_frame']
            report['odometry_used'][fs][str(off)] = {
                'odometry_frame': target_fs,
                'timestamp': results[i][off]['timestamp'],
                'quaternion': results[i][off]['quaternion']
            }

    json_file = out_dir / "two_frame_fusion_report.json"
    try:
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to: {json_file}")
    except Exception as e:
        print(f"WARNING: Could not save JSON report: {e}")

    # -------------------------- Save CSV summary (optional) --------------------------
    # Flatten metrics for CSV
    csv_file = out_dir / "two_frame_fusion_summary.csv"
    try:
        if metrics:
            # Determine headers
            headers = [
                'offset', 'frame0_id', 'frame1_id',
                'orig_centroid_0_x', 'orig_centroid_0_y', 'orig_centroid_0_z',
                'orig_centroid_1_x', 'orig_centroid_1_y', 'orig_centroid_1_z',
                'orig_centroid_distance_diagnostic',
                # For each convention
                'cam2world_centroid_0_x', 'cam2world_centroid_0_y', 'cam2world_centroid_0_z',
                'cam2world_centroid_1_x', 'cam2world_centroid_1_y', 'cam2world_centroid_1_z',
                'cam2world_centroid_distance_diagnostic',
                'world2cam_centroid_0_x', 'world2cam_centroid_0_y', 'world2cam_centroid_0_z',
                'world2cam_centroid_1_x', 'world2cam_centroid_1_y', 'world2cam_centroid_1_z',
                'world2cam_centroid_distance_diagnostic'
            ]
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for off in valid_offsets:
                    row = {
                        'offset': off,
                        'frame0_id': frame_strings[0],
                        'frame1_id': frame_strings[1]
                    }
                    # Original
                    orig = metrics[off]['original_camera_frame']
                    row.update({
                        'orig_centroid_0_x': orig['centroid_0'][0] if orig['centroid_0'] is not None else '',
                        'orig_centroid_0_y': orig['centroid_0'][1] if orig['centroid_0'] is not None else '',
                        'orig_centroid_0_z': orig['centroid_0'][2] if orig['centroid_0'] is not None else '',
                        'orig_centroid_1_x': orig['centroid_1'][0] if orig['centroid_1'] is not None else '',
                        'orig_centroid_1_y': orig['centroid_1'][1] if orig['centroid_1'] is not None else '',
                        'orig_centroid_1_z': orig['centroid_1'][2] if orig['centroid_1'] is not None else '',
                        'orig_centroid_distance_diagnostic': orig['centroid_distance_diagnostic'] if orig['centroid_distance_diagnostic'] is not None else ''
                    })
                    # Camera-to-world
                    cam2w = metrics[off]['camera_to_world']
                    row.update({
                        'cam2world_centroid_0_x': cam2w['centroid_0'][0] if cam2w['centroid_0'] is not None else '',
                        'cam2world_centroid_0_y': cam2w['centroid_0'][1] if cam2w['centroid_0'] is not None else '',
                        'cam2world_centroid_0_z': cam2w['centroid_0'][2] if cam2w['centroid_0'] is not None else '',
                        'cam2world_centroid_1_x': cam2w['centroid_1'][0] if cam2w['centroid_1'] is not None else '',
                        'cam2world_centroid_1_y': cam2w['centroid_1'][1] if cam2w['centroid_1'] is not None else '',
                        'cam2world_centroid_1_z': cam2w['centroid_1'][2] if cam2w['centroid_1'] is not None else '',
                        'cam2world_centroid_distance_diagnostic': cam2w['centroid_distance_diagnostic'] if cam2w['centroid_distance_diagnostic'] is not None else ''
                    })
                    # World-to-camera
                    w2c = metrics[off]['world_to_camera']
                    row.update({
                        'world2cam_centroid_0_x': w2c['centroid_0'][0] if w2c['centroid_0'] is not None else '',
                        'world2cam_centroid_0_y': w2c['centroid_0'][1] if w2c['centroid_0'] is not None else '',
                        'world2cam_centroid_0_z': w2c['centroid_0'][2] if w2c['centroid_0'] is not None else '',
                        'world2cam_centroid_1_x': w2c['centroid_1'][0] if w2c['centroid_1'] is not None else '',
                        'world2cam_centroid_1_y': w2c['centroid_1'][1] if w2c['centroid_1'] is not None else '',
                        'world2cam_centroid_1_z': w2c['centroid_1'][2] if w2c['centroid_1'] is not None else '',
                        'world2cam_centroid_distance_diagnostic': w2c['centroid_distance_diagnostic'] if w2c['centroid_distance_diagnostic'] is not None else ''
                    })
                    writer.writerow(row)
            print(f"Saved CSV summary to: {csv_file}")
    except Exception as e:
        print(f"WARNING: Could not save CSV summary: {e}")

    # -------------------------- Visualization --------------------------
    if HAVE_MPL:
        try:
            # We'll downsample point clouds for clarity with a fixed seed for reproducibility
            max_vis_points = 20000
            np.random.seed(42)  # Fixed seed for reproducible downsampling
            def downsample(points, max_n):
                if points is None or len(points) == 0:
                    return points
                if len(points) <= max_n:
                    return points
                idx = np.random.choice(len(points), max_n, replace=False)
                return points[idx]

            # Original point clouds (camera frame)
            p0_orig_ds = downsample(point_clouds[0], max_vis_points)
            p1_orig_ds = downsample(point_clouds[1], max_vis_points)

            # We'll create a figure with subplots for each convention (A and B)
            fig = plt.figure(figsize=(12, 5))
            # Subplot 1: Convention A (camera-to-world)
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_title('Transform A: quaternion as camera->world\n(rotation-only, no translation)')
            # Original point clouds in gray
            if p0_orig_ds is not None and len(p0_orig_ds) > 0:
                ax1.scatter(p0_orig_ds[:, 0], p0_orig_ds[:, 1], p0_orig_ds[:, 2],
                            c='lightgray', s=1, alpha=0.3, label='Frame 0 original')
            if p1_orig_ds is not None and len(p1_orig_ds) > 0:
                ax1.scatter(p1_orig_ds[:, 0], p1_orig_ds[:, 1], p1_orig_ds[:, 2],
                            c='gray', s=1, alpha=0.3, label='Frame 1 original')
            # Transformed point clouds for each offset (we'll show offset 0 only for clarity)
            off_show = valid_offsets[0] if valid_offsets else 0  # use first valid offset
            for i, fs in enumerate(frame_strings):
                pts_A = results[i][off_show]['points_a']
                pts_A_ds = downsample(pts_A, max_vis_points)
                if pts_A_ds is not None and len(pts_A_ds) > 0:
                    color = 'red' if i == 0 else 'blue'
                    ax1.scatter(pts_A_ds[:, 0], pts_A_ds[:, 1], pts_A_ds[:, 2],
                                c=color, s=1, alpha=0.5,
                                label=f'Frame {fs} transformed (offset {off_show})')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.legend(loc='upper right', fontsize='small')
            # Attempt equal aspect ratio
            try:
                # Collect all points to compute range
                all_pts = []
                if p0_orig_ds is not None and len(p0_orig_ds) > 0:
                    all_pts.append(p0_orig_ds)
                if p1_orig_ds is not None and len(p1_orig_ds) > 0:
                    all_pts.append(p1_orig_ds)
                for i in range(len(frame_strings)):
                    pts = results[i][off_show]['points_a']
                    if pts is not None and len(pts) > 0:
                        all_pts.append(pts)
                if all_pts:
                    all_pts_arr = np.vstack(all_pts)
                    max_range = np.array([all_pts_arr[:,0].max()-all_pts_arr[:,0].min(),
                                          all_pts_arr[:,1].max()-all_pts_arr[:,1].min(),
                                          all_pts_arr[:,2].max()-all_pts_arr[:,2].min()]).max() / 2.0
                    mid_x = (all_pts_arr[:,0].max()+all_pts_arr[:,0].min()) * 0.5
                    mid_y = (all_pts_arr[:,1].max()+all_pts_arr[:,1].min()) * 0.5
                    mid_z = (all_pts_arr[:,2].max()+all_pts_arr[:,2].min()) * 0.5
                    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
            except Exception as e:
                print(f"Warning: Could not set equal aspect ratio for subplot A: {e}")

            # Subplot 2: Convention B (world-to-camera)
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_title('Transform B: quaternion as world->camera (conjugate)\n(rotation-only, no translation)')
            # Original point clouds in gray
            if p0_orig_ds is not None and len(p0_orig_ds) > 0:
                ax2.scatter(p0_orig_ds[:, 0], p0_orig_ds[:, 1], p0_orig_ds[:, 2],
                            c='lightgray', s=1, alpha=0.3, label='Frame 0 original')
            if p1_orig_ds is not None and len(p1_orig_ds) > 0:
                ax2.scatter(p1_orig_ds[:, 0], p1_orig_ds[:, 1], p1_orig_ds[:, 2],
                            c='gray', s=1, alpha=0.3, label='Frame 1 original')
            # Transformed point clouds for convention B, offset 0
            for i, fs in enumerate(frame_strings):
                pts_B = results[i][off_show]['points_b']
                pts_B_ds = downsample(pts_B, max_vis_points)
                if pts_B_ds is not None and len(pts_B_ds) > 0:
                    color = 'green' if i == 0 else 'yellow'
                    ax2.scatter(pts_B_ds[:, 0], pts_B_ds[:, 1], pts_B_ds[:, 2],
                                c=color, s=1, alpha=0.5,
                                label=f'Frame {fs} transformed (offset {off_show})')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend(loc='upper right', fontsize='small')
            # Equal aspect ratio
            try:
                all_pts = []
                if p0_orig_ds is not None and len(p0_orig_ds) > 0:
                    all_pts.append(p0_orig_ds)
                if p1_orig_ds is not None and len(p1_orig_ds) > 0:
                    all_pts.append(p1_orig_ds)
                for i in range(len(frame_strings)):
                    pts = results[i][off_show]['points_b']
                    if pts is not None and len(pts) > 0:
                        all_pts.append(pts)
                if all_pts:
                    all_pts_arr = np.vstack(all_pts)
                    max_range = np.array([all_pts_arr[:,0].max()-all_pts_arr[:,0].min(),
                                          all_pts_arr[:,1].max()-all_pts_arr[:,1].min(),
                                          all_pts_arr[:,2].max()-all_pts_arr[:,2].min()]).max() / 2.0
                    mid_x = (all_pts_arr[:,0].max()+all_pts_arr[:,0].min()) * 0.5
                    mid_y = (all_pts_arr[:,1].max()+all_pts_arr[:,1].min()) * 0.5
                    mid_z = (all_pts_arr[:,2].max()+all_pts_arr[:,2].min()) * 0.5
                    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
            except Exception as e:
                print(f"Warning: Could not set equal aspect ratio for subplot B: {e}")

            viz_file = out_dir / "two_frame_fusion_visualization.png"
            plt.savefig(viz_file, dpi=150)
            plt.close(fig)
            print(f"Saved visualization to: {viz_file}")
        except Exception as e:
            print(f"WARNING: Could not create visualization: {e}")
    else:
        print("INFO: Matplotlib not available; skipping visualization.")

    # -------------------------- Summary --------------------------
    print("\n=== Experiment Summary ===")
    print("Tested two depth frames (000000.png and 000001.png) from dataset:")
    print(f"  {dataset_name}")
    print("Applied rotation-only transforms using odometry poses from:")
    print(f"  Valid offsets: {valid_offsets}")
    if skipped_offsets:
        print(f"  Skipped offsets: {skipped_offsets} (missing odometry data)")
    print("Tested both quaternion interpretations:")
    print("  A: quaternion rotates point cloud from camera -> world")
    print("  B: quaternion rotates point cloud from world -> camera (conjugate)")
    print("No translation was applied due to unknown depth scale and odometry units.")
    print("Metrics: centroid distance after rotation-only transformation (limited diagnostic).")
    print("Centroid distance after rotation-only transformation is descriptive only and cannot establish alignment.")
    print(f"JSON report: {json_file}")
    if HAVE_MPL:
        print(f"Visualization: {viz_file}")
    print("\nAssumptions (see JSON for full list):")
    for assump in report['assumptions']:
        print(f"  - {assump}")
    print("\nExperiment complete.")

if __name__ == "__main__":
    main()