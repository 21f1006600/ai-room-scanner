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
    timestamps = []
    frame_ids = []
    positions = []  # each as [x, y, z]
    quaternions = []  # each as [qx, qy, qz, qw]
    valid_entries = []  # list of (original_index, frame_id, timestamp, x, y, z, qx, qy, qz, qw)
    invalid_count = 0

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

            row_idx = 0
            for row in reader:
                try:
                    frame_id = row['frame'].strip()
                    timestamp = float(row['timestamp'])
                    x = float(row['x'])
                    y = float(row['y'])
                    z = float(row['z'])
                    qx = float(row['qx'])
                    qy = float(row['qy'])
                    qz = float(row['qz'])
                    qw = float(row['qw'])
                except ValueError:
                    invalid_count += 1
                    row_idx += 1
                    continue

                # Check for NaN or infinity
                if not (math.isfinite(timestamp) and math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and
                        math.isfinite(qx) and math.isfinite(qy) and math.isfinite(qz) and math.isfinite(qw)):
                    invalid_count += 1
                    row_idx += 1
                    continue

                valid_entries.append((row_idx, frame_id, timestamp, x, y, z, qx, qy, qz, qw))
                row_idx += 1
    except Exception as e:
        print(f"Error reading odometry CSV: {e}")
        sys.exit(1)

    total_rows = row_idx  # because we incremented row_idx for each row read
    valid_count = len(valid_entries)

    if valid_count == 0:
        print("Error: No valid odometry data found")
        sys.exit(1)

    # Helper functions for quaternion operations
    def q_mult(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return [x, y, z, w]

    def q_conjugate(q):
        return [-q[0], -q[1], -q[2], q[3]]

    def quaternion_angle_deg(q1, q2):
        # Returns the smallest angle in degrees between two unit quaternions
        q1_conj = q_conjugate(q1)
        q_rel = q_mult(q2, q1_conj)
        w = q_rel[3]
        # Clamp to [-1, 1] for arccos safety
        w = max(-1.0, min(1.0, w))
        angle_rad = 2.0 * math.acos(abs(w))
        return angle_rad * 180.0 / math.pi

    # Prepare to compute consecutive differences only for adjacent valid rows in original sequence
    translation_dists = []
    time_diffs = []
    speeds = []  # in unverified units per second
    angular_changes = []
    # We'll also keep data for the 5 largest translation steps
    translation_entries = []  # each entry: (dist, time_diff, speed, angle, frame_id1, frame_id2, index1)

    for i in range(len(valid_entries)-1):
        idx1, frame1, ts1, x1, y1, z1, qx1, qy1, qz1, qw1 = valid_entries[i]
        idx2, frame2, ts2, x2, y2, z2, qx2, qy2, qz2, qw2 = valid_entries[i+1]
        # Check if these two valid rows are adjacent in the original sequence
        if idx2 == idx1 + 1:
            # Translation
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            # Time difference
            dt = ts2 - ts1
            # Speed (if dt > 0)
            if dt > 0:
                speed = dist / dt
            else:
                speed = 0.0  # or undefined, but we'll set to 0 and note
            # Angular change
            q1 = [qx1, qy1, qz1, qw1]
            q2 = [qx2, qy2, qz2, qw2]
            angle = quaternion_angle_deg(q1, q2)

            translation_dists.append(dist)
            time_diffs.append(dt)
            speeds.append(speed)
            angular_changes.append(angle)

            translation_entries.append({
                'dist': dist,
                'time_diff': dt,
                'speed': speed,
                'angle': angle,
                'frame1': frame1,
                'frame2': frame2,
                'index': idx1
            })
        # else: not adjacent, skip

    if len(translation_dists) == 0:
        print("Error: No consecutive valid pairs to compute")
        sys.exit(1)

    # Convert to numpy arrays for statistics
    trans_arr = np.array(translation_dists)
    time_arr = np.array(time_diffs)

    # Summary statistics for translation distance
    trans_min = np.min(trans_arr)
    trans_median = np.median(trans_arr)
    trans_mean = np.mean(trans_arr)
    trans_95 = np.percentile(trans_arr, 95)
    trans_max = np.max(trans_arr)

    # Summary statistics for time difference
    time_min = np.min(time_arr)
    time_median = np.median(time_arr)
    time_mean = np.mean(time_arr)
    time_95 = np.percentile(time_arr, 95)
    time_max = np.max(time_arr)

    # Find 5 largest translation steps
    # Sort translation_entries by dist descending
    sorted_entries = sorted(translation_entries, key=lambda x: x['dist'], reverse=True)
    top5 = sorted_entries[:5]

    # Output
    print(f"=== Pose Motion Audit ===")
    print(f"Total rows processed: {total_rows}")
    print(f"Valid odometry rows: {valid_count}")
    print(f"Invalid or non-finite rows: {invalid_count}")
    print(f"Valid consecutive pairs (adjacent in original sequence): {len(translation_dists)}")
    print()
    print(f"Translation distance statistics (unverified units):")
    print(f"  Minimum: {trans_min:.3f}")
    print(f"  95th percentile: {trans_95:.3f}")
    print(f"  Median: {trans_median:.3f}")
    print(f"  Mean: {trans_mean:.3f}")
    print(f"  Maximum: {trans_max:.3f}")
    print()
    print(f"Time difference statistics (seconds):")
    print(f"  Minimum: {time_min:.3f}")
    print(f"  95th percentile: {time_95:.3f}")
    print(f"  Median: {time_median:.3f}")
    print(f"  Mean: {time_mean:.3f}")
    print(f"  Maximum: {time_max:.3f}")
    print()
    print(f"Top 5 largest translation steps:")
    for i, entry in enumerate(top5):
        print(f"  {i+1}. Frames {entry['frame1']} -> {entry['frame2']}")
        print(f"     Translation distance: {entry['dist']:.3f}")
        print(f"     Time difference: {entry['time_diff']:.3f} s")
        if entry['time_diff'] > 0:
            print(f"     Speed: {entry['speed']:.3f} unverified units/s")
        else:
            print(f"     Speed: undefined (zero or negative time difference)")
        print(f"     Angular change: {entry['angle']:.3f} degrees")
        print()

    print(f"NOTE: This audit analyzes consecutive odometry readings for translation and rotation.")
    print(f"      Translation speed units are unverified (depends on unknown position units and time units).")
    print(f"      Quaternion angular change uses the shortest-angle method, accounting for q and -q equivalence.")
    print(f"      Remaining unknown: coordinate frame, handedness, position units, and whether quaternions")
    print(f"      represent camera-to-world or world-to-camera rotation.")

if __name__ == "__main__":
    main()