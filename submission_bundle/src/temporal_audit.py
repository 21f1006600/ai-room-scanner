import cv2
import numpy as np
import csv
from pathlib import Path
import sys

def main():
    base_dir = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    depth_dir = base_dir / "depth"
    conf_dir = base_dir / "confidence"
    odometry_path = base_dir / "odometry.csv"
    rgb_path = base_dir / "rgb.mp4"

    # Verify directories exist
    if not depth_dir.is_dir():
        print(f"Error: Depth directory not found: {depth_dir}")
        sys.exit(1)
    if not conf_dir.is_dir():
        print(f"Error: Confidence directory not found: {conf_dir}")
        sys.exit(1)
    if not odometry_path.is_file():
        print(f"Error: Odometry file not found: {odometry_path}")
        sys.exit(1)

    # Load odometry into dict: frame_id (string) -> row dict
    odometry_by_frame = {}
    with open(odometry_path, 'r') as f:
        reader = csv.DictReader(f)
        # Strip whitespace from fieldnames
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            frame_id = row['frame'].strip()
            odometry_by_frame[frame_id] = row

    # Get sorted list of depth frames (by filename)
    depth_files = sorted(depth_dir.glob("*.png"))
    # Extract frame IDs from filenames (without extension)
    depth_frame_ids = [f.stem for f in depth_files]
    # Verify they match confidence files
    conf_files = sorted(conf_dir.glob("*.png"))
    conf_frame_ids = [f.stem for f in conf_files]
    if depth_frame_ids != conf_frame_ids:
        print("Warning: Depth and confidence frame IDs do not match exactly.")
    else:
        print(f"Verified: Depth and confidence frame IDs match exactly. Count: {len(depth_frame_ids)}")

    # Odometry data rows count (excluding header)
    odometry_data_count = len(odometry_by_frame)
    print(f"Verified: Odometry data rows: {odometry_data_count}")

    # Check if frame IDs match between depth and odometry
    depth_set = set(depth_frame_ids)
    odom_set = set(odometry_by_frame.keys())
    if depth_set == odom_set:
        print(f"Verified: Depth frame IDs exactly match odometry frame IDs. Count: {len(depth_set)}")
    else:
        missing_in_odom = depth_set - odom_set
        missing_in_depth = odom_set - depth_set
        print(f"Inferred: Mismatch between depth and odometry frame IDs.")
        if missing_in_odom:
            print(f"  Frames in depth but not in odometry: {sorted(missing_in_odom)[:5]}")
        if missing_in_depth:
            print(f"  Frames in odometry but not in depth: {sorted(missing_in_depth)[:5]}")

    # RGB frame count using OpenCV (read-only)
    rgb_frame_count = None
    try:
        cap = cv2.VideoCapture(str(rgb_path))
        if cap.isOpened():
            rgb_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"Verified: RGB frame count (via video properties): {rgb_frame_count}")
        else:
            print(f"Inferred: Could not open RGB video to get frame count.")
    except Exception as e:
        print(f"Inferred: Error reading RGB video: {e}")

    # Determine total frame count from depth (assumed)
    total_frames = len(depth_frame_ids)
    print(f"Verified: Total frames (depth): {total_frames}")

    # Select 5 representative indices: first, 25%, 50%, 75%, last
    if total_frames == 0:
        print("Error: No frames found.")
        sys.exit(1)
    indices = [
        0,
        total_frames // 4,
        total_frames // 2,
        (3 * total_frames) // 4,
        total_frames - 1
    ]
    # Ensure uniqueness
    indices = list(dict.fromkeys(indices))
    print(f"\nSelected frame indices for inspection: {indices}")

    # For each selected index, report details
    for idx in indices:
        frame_id = f"{idx:06d}"
        print(f"\n--- Frame Index: {idx} (ID: {frame_id}) ---")
        # Depth
        depth_path = depth_dir / f"{frame_id}.png"
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"  Depth: Failed to load {depth_path}")
        else:
            shape = depth_img.shape
            dtype = depth_img.dtype
            min_val = np.min(depth_img)
            max_val = np.max(depth_img)
            median_val = np.median(depth_img)
            zero_count = np.count_nonzero(depth_img == 0)
            total_pixels = depth_img.size
            zero_pct = (zero_count / total_pixels) * 100 if total_pixels > 0 else 0
            print(f"  Depth: shape={shape}, dtype={dtype}, min={min_val}, max={max_val}, median={median_val}")
            print(f"  Depth: zero-depth pixels: {zero_count} / {total_pixels} ({zero_pct:.2f}%)")
        # Confidence
        conf_path = conf_dir / f"{frame_id}.png"
        conf_img = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        if conf_img is None:
            print(f"  Confidence: Failed to load {conf_path}")
        else:
            conf_shape = conf_img.shape
            conf_dtype = conf_img.dtype
            # Assuming confidence is single channel (grayscale)
            if conf_img.ndim == 3:
                # If it's actually RGB, convert to gray or take first channel
                conf_img = cv2.cvtColor(conf_img, cv2.COLOR_BGR2GRAY)
            unique_vals, counts = np.unique(conf_img, return_counts=True)
            print(f"  Confidence: shape={conf_shape}, dtype={conf_dtype}")
            print(f"  Confidence value counts:")
            for val, cnt in zip(unique_vals, counts):
                pct = (cnt / conf_img.size) * 100
                print(f"    Value {val}: {cnt} pixels ({pct:.2f}%)")
        # Odometry
        if frame_id in odometry_by_frame:
            row = odometry_by_frame[frame_id]
            timestamp = row['timestamp']
            # Convert to float for readability
            try:
                ts_float = float(timestamp)
                print(f"  Odometry: timestamp={ts_float:.6f}")
            except:
                print(f"  Odometry: timestamp={timestamp}")
            # Position
            try:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                print(f"  Odometry: position (x,y,z) = ({x:.6f}, {y:.6f}, {z:.6f})")
            except:
                print(f"  Odometry: position (x,y,z) = ({row['x']}, {row['y']}, {row['z']})")
        else:
            print(f"  Odometry: No entry for frame ID {frame_id}")

    # Timestamp analysis
    print("\n--- Timestamp Analysis (from odometry) ---")
    timestamps = []
    frame_ids_sorted = sorted(odometry_by_frame.keys(), key=lambda x: int(x))
    for fid in frame_ids_sorted:
        try:
            ts = float(odometry_by_frame[fid]['timestamp'])
            timestamps.append(ts)
        except:
            pass
    if len(timestamps) >= 2:
        intervals = np.diff(timestamps)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        min_interval = np.min(intervals)
        max_interval = np.max(intervals)
        print(f"Verified: Timestamp statistics (seconds):")
        print(f"  Count: {len(timestamps)}")
        print(f"  Mean interval: {avg_interval:.6f} s")
        print(f"  Std deviation: {std_interval:.6f} s")
        print(f"  Min interval: {min_interval:.6f} s")
        print(f"  Max interval: {max_interval:.6f} s")
        # Check if intervals are roughly constant (e.g., std < 1% of mean)
        if avg_interval > 0 and (std_interval / avg_interval) < 0.01:
            print(f"  Inferred: Timestamps appear regular (low jitter).")
        else:
            print(f"  Inferred: Timestamps show irregular intervals or jitter.")
        # Look for large gaps (> 2x mean)
        gaps = np.where(intervals > 2 * avg_interval)[0]
        if len(gaps) > 0:
            print(f"  Inferred: Potential missing frames or large gaps at indices: {gaps[:5]} (intervals: {intervals[gaps][:5]})")
    else:
        print(f"Inferred: Not enough timestamps to analyze intervals.")

    # Confidence semantics: examine a few frames to see what values appear
    print("\n--- Confidence Value Investigation ---")
    # Sample a few frames (e.g., first 5) to see unique confidence values
    sample_size = min(5, len(depth_frame_ids))
    sampled_ids = depth_frame_ids[:sample_size]
    all_vals = set()
    for fid in sampled_ids:
        conf_path = conf_dir / f"{fid}.png"
        conf_img = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        if conf_img is None:
            continue
        if conf_img.ndim == 3:
            conf_img = cv2.cvtColor(conf_img, cv2.COLOR_BGR2GRAY)
        vals = np.unique(conf_img)
        all_vals.update(vals)
    print(f"Verified: Confidence values observed in first {sample_size} frames: {sorted(all_vals)}")
    # Check if only 0,1,2 appear
    if all_vals.issubset({0,1,2}):
        print(f"Verified: Only values 0, 1, 2 observed in sampled frames.")
        # But we don't know semantics
        print(f"Unknown: Semantics of confidence values 0, 1, 2 (cannot be established from dataset alone).")
    else:
        print(f"Inferred: Confidence values outside {0,1,2} observed: {sorted(all_vals - {0,1,2})}")

    # Additional metadata needed for multi-frame fusion
    print("\n--- Additional Metadata Needed for Reliable Multi-Frame Fusion ---")
    print(f"VERIFIED: Requires:")
    print(f"  - Depth camera extrinsics (rotation/translation) relative to odometry frame.")
    print(f"  - Depth camera intrinsics (fx, fy, cx, cy) and distortion parameters.")
    print(f"  - Accurate time synchronization between depth, confidence, odometry, and RGB.")
    print(f"  - Scale factor for depth values (if depth is in arbitrary units).")
    print(f"  - Confidence semantics (meaning of values 0,1,2) to filter reliable measurements.")
    print(f"  - RGB camera intrinsics and extrinsics if color alignment is desired.")
    print(f"  - Lens distortion parameters for both depth and RGB cameras.")
    print(f"UNKNOWN: Without the above, any multi-frame point cloud fusion would be speculative.")
    print(f"RISK: Assuming incorrect synchronization or extrinsics will lead to misaligned point clouds and false geometry.")

if __name__ == "__main__":
    main()