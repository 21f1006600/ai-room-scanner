import csv
import numpy as np
import math
from pathlib import Path
import sys

def main():
    # Dataset path
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    odometry_path = base / "odometry.csv"

    # Read odometry data
    timestamps = []  # we may not need these but keep for completeness
    frame_ids = []   # for all rows
    quaternions_raw = []  # all rows as read
    valid_quats = []  # list of (original_index, normalized_quaternion)
    invalid_count = 0

    try:
        with open(odometry_path, 'r') as f:
            reader = csv.DictReader(f)
            # Strip whitespace from fieldnames
            if reader.fieldnames is None:
                print("Error: Could not read CSV header")
                sys.exit(1)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            # Ensure required columns exist
            required = ['frame', 'timestamp', 'qx', 'qy', 'qz', 'qw']
            for col in required:
                if col not in reader.fieldnames:
                    print(f"Error: Missing column '{col}' in odometry CSV")
                    sys.exit(1)

            row_idx = 0
            for row in reader:
                frame_ids.append(row['frame'].strip())
                timestamps.append(float(row['timestamp']))
                try:
                    qx = float(row['qx'])
                    qy = float(row['qy'])
                    qz = float(row['qz'])
                    qw = float(row['qw'])
                except ValueError:
                    # non-numeric value
                    invalid_count += 1
                    row_idx += 1
                    continue

                # Check for NaN or infinity
                if not (math.isfinite(qx) and math.isfinite(qy) and math.isfinite(qz) and math.isfinite(qw)):
                    invalid_count += 1
                    row_idx += 1
                    continue

                # Compute norm
                norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                if norm == 0.0:
                    invalid_count += 1
                    row_idx += 1
                    continue

                # Normalize
                qx_n = qx / norm
                qy_n = qy / norm
                qz_n = qz / norm
                qw_n = qw / norm
                valid_quats.append((row_idx, [qx_n, qy_n, qz_n, qw_n]))

                row_idx += 1
    except Exception as e:
        print(f"Error reading odometry CSV: {e}")
        sys.exit(1)

    total_rows = len(frame_ids)
    valid_count = len(valid_quats)

    if valid_count == 0:
        print("Error: No valid quaternion data found")
        sys.exit(1)

    # Function to multiply two quaternions: q1 * q2, where each is [x, y, z, w]
    def q_mult(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return [x, y, z, w]

    # Function to compute conjugate of quaternion [x, y, z, w]
    def q_conjugate(q):
        return [-q[0], -q[1], -q[2], q[3]]

    # Compute relative quaternions and angles only between adjacent valid rows in original sequence
    angles_deg = []  # in degrees
    angle_info = []  # each element: (angle_deg, index_of_first_quat, frame_id_first, frame_id_second)
    max_angle = -1.0
    max_pair_data = None  # tuple (angle, frame_id_first, frame_id_second)

    for j in range(len(valid_quats)-1):
        idx1, q1 = valid_quats[j]
        idx2, q2 = valid_quats[j+1]
        # Check if these two valid rows are adjacent in the original sequence
        if idx2 == idx1 + 1:
            # Relative quaternion: q_rel = q2 * q1^{-1} = q2 * conjugate(q1)
            q1_conj = q_conjugate(q1)
            q_rel = q_mult(q2, q1_conj)
            # Ensure we take the shorter path: take absolute value of w for angle calculation
            w = q_rel[3]
            # Clamp w to [-1, 1] for arccos safety
            w = max(-1.0, min(1.0, w))
            angle_rad = 2.0 * math.acos(abs(w))
            angle_deg = angle_rad * 180.0 / math.pi
            angles_deg.append(angle_deg)
            angle_info.append((angle_deg, idx1, frame_ids[idx1], frame_ids[idx2]))

            if angle_deg > max_angle:
                max_angle = angle_deg
                max_pair_data = (angle_deg, frame_ids[idx1], frame_ids[idx2])
        # else: not adjacent, skip (do not compute angle across a gap)

    if len(angles_deg) == 0:
        print("Error: No consecutive valid pairs to compute")
        sys.exit(1)

    # Convert to numpy array for statistics
    angles = np.array(angles_deg)
    num_pairs = len(angles)

    # Compute statistics
    min_angle = np.min(angles)
    max_angle = np.max(angles)  # should equal max_angle from tracking
    mean_angle = np.mean(angles)
    median_angle = np.median(angles)
    percentile_95 = np.percentile(angles, 95)

    # Get the frame pair for max angle
    if max_pair_data is not None:
        _, frame_pair_first, frame_pair_second = max_pair_data
        frame_pair = (frame_pair_first, frame_pair_second)
    else:
        frame_pair = ("unknown", "unknown")

    # First, middle, last consecutive pairs (based on the computed angles list)
    if num_pairs > 0:
        first_pair_angle = angles[0]
        first_pair_frames = (angle_info[0][2], angle_info[0][3])  # (frame_id_first, frame_id_second)
        middle_pair_angle = angles[num_pairs // 2]
        middle_pair_frames = (angle_info[num_pairs // 2][2], angle_info[num_pairs // 2][3])
        last_pair_angle = angles[-1]
        last_pair_frames = (angle_info[-1][2], angle_info[-1][3])
    else:
        first_pair_angle = middle_pair_angle = last_pair_angle = 0.0
        first_pair_frames = ("unknown", "unknown")
        middle_pair_frames = ("unknown", "unknown")
        last_pair_frames = ("unknown", "unknown")

    # Output
    print(f"=== Quaternion Motion Audit ===")
    print(f"Total rows processed: {total_rows}")
    print(f"Valid quaternion rows: {valid_count}")
    print(f"Invalid or non-numeric rows: {invalid_count}")
    print(f"Valid consecutive pairs (adjacent in original sequence): {num_pairs}")
    print()
    print(f"Angular change statistics (degrees):")
    print(f"  Minimum: {min_angle:.3f}")
    print(f"  95th percentile: {percentile_95:.3f}")
    print(f"  Median: {median_angle:.3f}")
    print(f"  Mean: {mean_angle:.3f}")
    print(f"  Maximum: {max_angle:.3f}")
    print()
    print(f"Largest angular change: {max_angle:.3f} degrees between frames {frame_pair[0]} and {frame_pair[1]}")
    print()
    print(f"Angular change for first pair (frames {first_pair_frames[0]} -> {first_pair_frames[1]}): {first_pair_angle:.3f} degrees")
    print(f"Angular change for middle pair (frames {middle_pair_frames[0]} -> {middle_pair_frames[1]}): {middle_pair_angle:.3f} degrees")
    print(f"Angular change for last pair (frames {last_pair_frames[0]} -> {last_pair_frames[1]}): {last_pair_angle:.3f} degrees")
    print()
    print(f"NOTE: This audit validates smoothness of quaternion sequence and magnitude of orientation changes.")
    print(f"      It does NOT interpret rotations as world or camera orientation.")
    print(f"      Remaining unknown: coordinate-frame convention, handedness, and whether quaternions")
    print(f"      represent camera-to-world or world-to-camera rotation.")

if __name__ == "__main__":
    main()