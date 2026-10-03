import csv
import numpy as np
from pathlib import Path
import sys

def main():
    # Dataset path
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    odometry_path = base / "odometry.csv"

    # Read odometry data
    frames = []
    timestamps = []
    positions = []  # each as [x, y, z]
    quaternions = []  # each as [qx, qy, qz, qw]

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

            for row in reader:
                frames.append(row['frame'].strip())
                timestamps.append(float(row['timestamp']))
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                qx = float(row['qx'])
                qy = float(row['qy'])
                qz = float(row['qz'])
                qw = float(row['qw'])
                positions.append([x, y, z])
                quaternions.append([qx, qy, qz, qw])
    except Exception as e:
        print(f"Error reading odometry CSV: {e}")
        sys.exit(1)

    if len(quaternions) == 0:
        print("Error: No odometry data found")
        sys.exit(1)

    total_rows = len(quaternions)
    print(f"Total odometry rows: {total_rows}")

    # Translation ranges
    pos_array = np.array(positions)
    x_min, x_max = np.min(pos_array[:, 0]), np.max(pos_array[:, 0])
    y_min, y_max = np.min(pos_array[:, 1]), np.max(pos_array[:, 1])
    z_min, z_max = np.min(pos_array[:, 2]), np.max(pos_array[:, 2])
    print(f"Translation ranges (unspecified units):")
    print(f"  x: [{x_min:.3f}, {x_max:.3f}]")
    print(f"  y: [{y_min:.3f}, {y_max:.3f}]")
    print(f"  z: [{z_min:.3f}, {z_max:.3f}]")
    print()

    # Function to compute rotation matrix from quaternion [x, y, z, w]
    def quat_to_rotmat(q):
        x, y, z, w = q
        # Ensure unit quaternion (should already be normalized)
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

    # Function to check orthonormality and determinant
    def check_rotmat(R):
        if R is None:
            return False, None
        # Check orthonormality: R * R^T should be identity
        RT = R.T
        should_be_identity = np.dot(R, RT)
        identity = np.eye(3, dtype=R.dtype)
        ortho_error = np.max(np.abs(should_be_identity - identity))
        # Check determinant
        det = np.linalg.det(R)
        det_error = abs(det - 1.0)
        return ortho_error < 1e-6 and det_error < 1e-6, det

    # Select first, middle, last rows
    indices = [0, total_rows // 2, total_rows - 1]
    labels = ["first", "middle", "last"]
    unit_vectors = {
        '+x': np.array([1.0, 0.0, 0.0]),
        '+y': np.array([0.0, 1.0, 0.0]),
        '+z': np.array([0.0, 0.0, 1.0])
    }

    print("=== Quaternion Convention Diagnostic ===")
    print("Testing two interpretations:")
    print("  Interpretation A: Quaternion rotates vector from camera to world.")
    print("  Interpretation B: Quaternion rotates vector from world to camera (i.e., uses conjugate).")
    print()

    for idx, label in zip(indices, labels):
        q = quaternions[idx]
        pos = positions[idx]
        print(f"--- {label} row (frame {frames[idx]}, timestamp {timestamps[idx]:.3f}) ---")
        print(f"  Translation: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        print(f"  Quaternion (qx, qy, qz, qw): ({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f})")

        # Compute rotation matrices for both interpretations
        R_A = quat_to_rotmat(q)          # q as camera->world
        q_conj = [-q[0], -q[1], -q[2], q[2]]  # Wait, conjugate is [-x, -y, -z, w]
        q_conj = [-q[0], -q[1], -q[2], q[3]]  # correct conjugate
        R_B = quat_to_rotmat(q_conj)     # q^{-1} as camera->world (if q is world->camera)

        # Check orthonormality and determinant for both
        ok_A, det_A = check_rotmat(R_A) if R_A is not None else (False, None)
        ok_B, det_B = check_rotmat(R_B) if R_B is not None else (False, None)

        print(f"  Interpretation A (q as camera->world):")
        print(f"    Rotation matrix orthonormal and det~=1: {ok_A} (det={det_A:.6f})")
        print(f"    Action on basis vectors:")
        for vec_name, vec in unit_vectors.items():
            if R_A is not None:
                res = np.dot(R_A, vec)
                print(f"      {vec_name} -> ({res[0]:.3f}, {res[1]:.3f}, {res[2]:.3f})")
            else:
                print(f"      {vec_name} -> invalid quaternion")
        print(f"  Interpretation B (q as world->camera, i.e., use q^{-1}):")
        print(f"    Rotation matrix orthonormal and det~=1: {ok_B} (det={det_B:.6f})")
        print(f"    Action on basis vectors:")
        for vec_name, vec in unit_vectors.items():
            if R_B is not None:
                res = np.dot(R_B, vec)
                print(f"      {vec_name} -> ({res[0]:.3f}, {res[1]:.3f}, {res[2]:.3f})")
            else:
                print(f"      {vec_name} -> invalid quaternion")
        print()

    print("=== Conclusion ===")
    print("The CSV alone does not establish:")
    print("  - The physical units of translation (x, y, z).")
    print("  - The coordinate-frame convention (e.g., whether x points east, north, up, etc.).")
    print("  - The handedness of the coordinate system.")
    print("  - Whether quaternions represent camera-to-world or world-to-camera rotation.")
    print("Both interpretations produce valid rotation matrices (orthonormal, det~=1),")
    print("so the data is consistent with either convention.")
    print("Additional documentation or calibration is needed to disambiguate.")

if __name__ == "__main__":
    main()