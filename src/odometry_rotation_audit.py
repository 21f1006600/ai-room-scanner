import csv
import math
from pathlib import Path
import sys

def main():
    odometry_path = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1/odometry.csv")
    if not odometry_path.is_file():
        print(f"Error: Odometry file not found: {odometry_path}")
        sys.exit(1)

    valid_rows = []  # each entry: (frame_id, timestamp, qx, qy, qz, qw, norm)
    invalid_count = 0

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

        for row in reader:
            frame_id = row['frame'].strip()
            timestamp = row['timestamp']
            try:
                qx = float(row['qx'])
                qy = float(row['qy'])
                qz = float(row['qz'])
                qw = float(row['qw'])
                # Check for NaN or infinity
                if math.isnan(qx) or math.isinf(qx) or math.isnan(qy) or math.isinf(qy) or math.isnan(qz) or math.isinf(qz) or math.isnan(qw) or math.isinf(qw):
                    invalid_count += 1
                    continue
                norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
                valid_rows.append((frame_id, timestamp, qx, qy, qz, qw, norm))
            except (ValueError, KeyError):
                invalid_count += 1
                continue

    total_rows = len(valid_rows) + invalid_count
    print(f"Total rows processed: {total_rows}")
    print(f"Valid quaternion rows: {len(valid_rows)}")
    print(f"Invalid or non-numeric rows: {invalid_count}")

    if len(valid_rows) == 0:
        print("No valid quaternion rows found.")
        return

    # Compute statistics on norms
    norms = [norm for (_, _, _, _, _, _, norm) in valid_rows]
    min_norm = min(norms)
    max_norm = max(norms)
    mean_norm = sum(norms) / len(norms)
    # Count outside tolerance 0.01 from 1
    outside_tol = sum(1 for n in norms if abs(n - 1.0) > 0.01)
    print(f"\nQuaternion norm statistics:")
    print(f"  Min norm: {min_norm:.6f}")
    print(f"  Max norm: {max_norm:.6f}")
    print(f"  Mean norm: {mean_norm:.6f}")
    print(f"  Rows with norm outside tolerance 0.01 of 1.0: {outside_tol} / {len(norms)}")
    if outside_tol == 0:
        print(f"  -> All quaternions appear normalized within tolerance 0.01.")
    else:
        print(f"  -> Some quaternions deviate from unit norm beyond tolerance 0.01.")

    # First, middle, last valid quaternion
    first = valid_rows[0]
    middle = valid_rows[len(valid_rows)//2]
    last = valid_rows[-1]

    def format_quat(indent, frame_id, timestamp, qx, qy, qz, qw, norm):
        return (f"{indent}Frame ID: {frame_id}\n"
                f"{indent}Timestamp: {timestamp}\n"
                f"{indent}Quaternion (qx, qy, qz, qw): ({qx:.6f}, {qy:.6f}, {qz:.6f}, {qw:.6f})\n"
                f"{indent}Norm: {norm:.6f}")

    print("\nFirst valid quaternion:")
    print(format_quat("  ", *first))
    print("\nMiddle valid quaternion:")
    print(format_quat("  ", *middle))
    print("\nLast valid quaternion:")
    print(format_quat("  ", *last))

    # Additional note about convention
    print("\nNote: This analysis does not assume any particular rotation convention or coordinate frame.")
    print("      The quaternion components are inspected only for numeric validity and norm.")

if __name__ == "__main__":
    main()