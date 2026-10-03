#!/usr/bin/env python3
"""
Single-room pipeline MVP: processes a single depth frame from a dataset,
backprojects to point cloud, detects up to two planes via RANSAC, and reports results.
"""

import sys
import numpy as np
import cv2
import json
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


def fit_plane(p1, p2, p3):
    """Return plane coefficients (a,b,c,d) for ax+by+cz+d=0, normalized normal."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal = normal / norm
    d = -np.dot(normal, p1)
    return np.array([normal[0], normal[1], normal[2], d])


def refine_plane(inlier_points):
    """Refine plane using SVD on inlier points; returns refined plane (normalized)."""
    if len(inlier_points) < 3:
        return None
    centroid = np.mean(inlier_points, axis=0)
    centered = inlier_points - centroid
    cov = np.dot(centered.T, centered)
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]  # eigenvector for smallest eigenvalue
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    d = -np.dot(normal, centroid)
    return np.array([normal[0], normal[1], normal[2], d])


def plane_residuals(points_arr, plane):
    """Absolute perpendicular distances from points to plane."""
    a, b, c, d = plane
    normal = np.array([a, b, c])
    return np.abs(np.dot(points_arr, normal) + d)


def normal_angular_diff(n1, n2):
    """Angular difference between two unit normals, accounting for sign equivalence.
    Returns angle in degrees between 0 and 90."""
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    dot = abs(np.dot(n1, n2))  # absolute to ignore sign
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


def ransac_detect_plane(pts, threshold, seed=None):
    """Run RANSAC to detect a plane in pts.
    Returns (plane, inlier_mask, inlier_count) or (None, None, 0) if fails."""
    if seed is not None:
        np.random.seed(seed)
    best_inliers = None
    best_plane = None
    best_inlier_count = 0
    N = pts.shape[0]
    for it in range(1000):  # n_iterations
        idx = np.random.choice(N, size=3, replace=False)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        plane = fit_plane(p1, p2, p3)
        if plane is None:
            continue
        dists = plane_residuals(pts, plane)
        inliers = dists < threshold
        inlier_count = np.sum(inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = plane
            best_inliers = inliers
        if inlier_count > 0.9 * N:
            break
    if best_plane is None:
        return None, None, 0
    # Refine using all inliers
    if best_inlier_count > 0:
        inlier_pts = pts[best_inliers]
        refined_plane = refine_plane(inlier_pts)
        if refined_plane is None:
            refined_plane = best_plane
    else:
        refined_plane = best_plane
    # Recompute inliers using refined plane
    all_dists = plane_residuals(pts, refined_plane)
    refined_inliers = all_dists < threshold
    refined_inlier_count = np.sum(refined_inliers)
    return refined_plane, refined_inliers, refined_inlier_count


def main():
    # -------------------------- Configuration --------------------------
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    # Default dataset: floor-only
    dataset_name = "single_scan_floor_only/1a8384c3f6"
    dataset_path = base / dataset_name

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Single-room pipeline MVP")
    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset_name,
        help=f"Dataset name relative to {base} (default: {dataset_name})",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to process (0-based, default: 0 for first frame)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/single_room_pipeline",
        help="Output directory for generated files (default: output/single_room_pipeline)",
    )
    args = parser.parse_args()

    # Update dataset path based on argument
    dataset_path = base / args.dataset
    depth_dir = dataset_path / "depth"
    intrinsics_path = dataset_path / "camera_matrix.csv"

    # Output directory
    out_dir = Path(args.output_dir)
    # -------------------------- Checks --------------------------
    if not depth_dir.is_dir():
        print(f"ERROR: Depth directory not found: {depth_dir}")
        sys.exit(1)
    if not intrinsics_path.is_file():
        print(f"ERROR: Camera matrix not found: {intrinsics_path}")
        sys.exit(1)

    # Get list of depth frames
    depth_files = sorted(depth_dir.glob("*.png"))
    if not depth_files:
        print(f"ERROR: No depth images found in {depth_dir}")
        sys.exit(1)
    if args.frame < 0 or args.frame >= len(depth_files):
        print(f"ERROR: Frame index {args.frame} out of range (0-{len(depth_files)-1})")
        sys.exit(1)

    depth_path = depth_files[args.frame]
    print(f"Selected frame: {depth_path.name} (index {args.frame} of {len(depth_files)})")

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
    # Load depth image (uint16)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"ERROR: Failed to read depth image: {depth_path}")
        sys.exit(1)
    if depth.ndim != 2:
        print(f"ERROR: Depth image is not 2D (shape {depth.shape})")
        sys.exit(1)

    # Load camera matrix (3x3)
    try:
        K = np.loadtxt(str(intrinsics_path), delimiter=',')
    except Exception as e:
        print(f"ERROR: Failed to load camera matrix: {e}")
        sys.exit(1)
    if K.shape != (3, 3):
        print(f"ERROR: Camera matrix must be 3x3, got shape {K.shape}")
        sys.exit(1)

    # -------------------------- Back-project --------------------------
    print(f"Depth image shape: {depth.shape} (height x width)")
    print(f"Using same-FOV-scaled intrinsics (UNVERIFIED ASSUMPTION)")
    print(f"RGB resolution assumed: {_RGB_WIDTH} x {_RGB_HEIGHT}")
    print(f"Depth resolution: {depth.shape[1]} x {depth.shape[0]}")

    try:
        points = backproject_depth(
            depth_path,
            intrinsics_path,
            use_scaled_intrinsics=True
        )
    except Exception as e:
        print(f"ERROR: Back-projection failed: {e}")
        sys.exit(1)

    if points.size == 0:
        print("ERROR: No valid points found (all depth values are zero or invalid).")
        sys.exit(1)

    # -------------------------- Compute statistics --------------------------
    # Point cloud columns: X, Y, Z
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]  # This is raw depth (uint16) as returned by backproject_depth

    num_points = points.shape[0]
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())
    z_min, z_max = float(Z.min()), float(Z.max())
    x_extent = x_max - x_min
    y_extent = y_max - y_min
    z_extent = z_max - z_min

    x_p5, x_p50, x_p95 = np.percentile(X, [5, 50, 95])
    y_p5, y_p50, y_p95 = np.percentile(Y, [5, 50, 95])
    z_p5, z_p50, z_p95 = np.percentile(Z, [5, 50, 95])

    # Also compute depth statistics directly from the depth image (should match Z stats)
    valid_mask = depth > 0
    depth_vals = depth[valid_mask]
    if depth_vals.size > 0:
        depth_min = int(depth_vals.min())
        depth_median = int(np.median(depth_vals))
        depth_max = int(depth_vals.max())
    else:
        depth_min = depth_median = depth_max = 0

    # -------------------------- Report point cloud statistics --------------------------
    print("\n=== Point Cloud Statistics (raw depth units) ===")
    print(f"Number of valid points: {num_points}")
    print(f"X range: [{x_min:.3f}, {x_max:.3f}]  extent: {x_extent:.3f}")
    print(f"Y range: [{y_min:.3f}, {y_max:.3f}]  extent: {y_extent:.3f}")
    print(f"Z range: [{z_min:.3f}, {z_max:.3f}]  extent: {z_extent:.3f}")
    print(f"X percentiles (5th, 50th, 95th): {x_p5:.3f}, {x_p50:.3f}, {x_p95:.3f}")
    print(f"Y percentiles (5th, 50th, 95th): {y_p5:.3f}, {y_p50:.3f}, {y_p95:.3f}")
    print(f"Z percentiles (5th, 50th, 95th): {z_p5:.3f}, {z_p50:.3f}, {z_p95:.3f}")
    print(f"Depth image min/median/max: {depth_min} / {depth_median} / {depth_max}")

    # -------------------------- Plane detection --------------------------
    # Compute depth range for threshold
    z_range = z_max - z_min
    distance_threshold = 0.01 * z_range  # 1% of depth range
    if distance_threshold < 1.0:
        distance_threshold = 1.0
    print(f"\nDepth range (Z): [{z_min:.3f}, {z_max:.3f}] -> range = {z_range:.3f}")
    print(f"Using distance threshold = {distance_threshold:.3f} (1% of depth range, min 1.0)")

    # Detect first (dominant) plane
    print("\n=== Detecting first (dominant) plane ===")
    plane1, inliers1, count1 = ransac_detect_plane(points, distance_threshold, seed=42)
    if plane1 is None:
        print("ERROR: Failed to detect first plane.")
        sys.exit(1)
    # Compute residuals for inliers of plane1
    if count1 > 0:
        dists1 = plane_residuals(points[inliers1], plane1)
        mean_res1 = np.mean(dists1)
        median_res1 = np.median(dists1)
    else:
        mean_res1 = 0.0
        median_res1 = 0.0
    # Extract normal
    a1, b1, c1, d1 = plane1
    normal1 = np.array([a1, b1, c1])
    norm1 = np.linalg.norm(normal1)
    if norm1 > 0:
        normal1 = normal1 / norm1

    print(f"First plane: {count1} / {num_points} ({100.0*count1/num_points:.2f}%) inliers")
    print(f"  Plane equation: {a1:.6f} x + {b1:.6f} y + {c1:.6f} z + {d1:.6f} = 0")
    print(f"  Normal vector: [{a1:.6f}, {b1:.6f}, {c1:.6f}] (unit length)")
    print(f"  Mean residual: {mean_res1:.3f}")
    print(f"  Median residual: {median_res1:.3f}")

    # Detect second plane on outliers of first
    outliers_mask = ~inliers1
    points_outliers = points[outliers_mask]
    N_out = points_outliers.shape[0]
    print(f"\nPoints remaining after removing first-plane inliers: {N_out}")

    if N_out < 3:
        print("Not enough points left to detect a second plane.")
        plane2, inliers2_rel, count2 = None, None, 0
    else:
        print("=== Detecting second plane in remaining points ===")
        plane2, inliers2_rel, count2 = ransac_detect_plane(points_outliers, distance_threshold, seed=42)
        if plane2 is None:
            print("Failed to detect second plane.")
        else:
            # Convert relative indices to original indices
            inliers2 = np.zeros(num_points, dtype=bool)
            inliers2[outliers_mask] = inliers2_rel
            # Compute residuals for inliers of plane2 (using original points)
            if count2 > 0:
                dists2 = plane_residuals(points[inliers2], plane2)
                mean_res2 = np.mean(dists2)
                median_res2 = np.median(dists2)
            else:
                mean_res2 = 0.0
                median_res2 = 0.0
            a2, b2, c2, d2 = plane2
            normal2 = np.array([a2, b2, c2])
            norm2 = np.linalg.norm(normal2)
            if norm2 > 0:
                normal2 = normal2 / norm2

            print(f"Second plane: {count2} / {num_points} ({100.0*count2/num_points:.2f}%) inliers")
            print(f"  Plane equation: {a2:.6f} x + {b2:.6f} y + {c2:.6f} z + {d2:.6f} = 0")
            print(f"  Normal vector: [{a2:.6f}, {b2:.6f}, {c2:.6f}] (unit length)")
            print(f"  Mean residual: {mean_res2:.3f}")
            print(f"  Median residual: {median_res2:.3f}")

    # -------------------------- Remaining points (outliers of both) --------------------------
    if plane2 is not None and count2 > 0:
        outliers_both_mask = ~(inliers1 | inliers2)
    else:
        outliers_both_mask = ~inliers1
    count_outliers = np.sum(outliers_both_mask)
    print(f"\nPoints not assigned to either plane: {count_outliers} / {num_points} ({100.0*count_outliers/num_points:.2f}%)")

    # -------------------------- Angular difference between plane normals --------------------------
    if plane2 is not None and count2 > 0:
        ang_diff = normal_angular_diff(normal1, normal2)
        print(f"\nAngular difference between plane normals: {ang_diff:.2f}°")
    else:
        ang_diff = None

    # -------------------------- Save point cloud --------------------------
    points_file = out_dir / "point_cloud.npy"
    try:
        np.save(points_file, points)
        print(f"\nSaved point cloud to: {points_file}")
    except Exception as e:
        print(f"WARNING: Could not save point cloud: {e}")

    # -------------------------- Save visualization --------------------------
    if HAVE_MPL:
        try:
            # Subsample for clarity if too many points
            max_points = 50000
            if num_points > max_points:
                idx = np.random.choice(num_points, max_points, replace=False)
                pts_vis = points[idx]
                in1_vis = inliers1[idx]
                in2_vis = inliers2[idx] if plane2 is not None and count2 > 0 else np.zeros_like(in1_vis, dtype=bool)
                out_vis = ~(in1_vis | in2_vis)
                print(f"Visualization: showing {max_points} randomly sampled points.")
            else:
                pts_vis = points
                in1_vis = inliers1
                in2_vis = inliers2 if plane2 is not None and count2 > 0 else np.zeros(num_points, dtype=bool)
                out_vis = ~(in1_vis | in2_vis)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            # Outliers (neither plane) in light gray
            if np.any(out_vis):
                ax.scatter(pts_vis[out_vis, 0], pts_vis[out_vis, 1], pts_vis[out_vis, 2],
                           c='lightgray', s=1, alpha=0.5, label='Other points')
            # First plane inliers in red
            if np.any(in1_vis):
                ax.scatter(pts_vis[in1_vis, 0], pts_vis[in1_vis, 1], pts_vis[in1_vis, 2],
                           c='red', s=1, alpha=0.8, label=f'Plane 1 ({count1} pts)')
            # Second plane inliers in blue
            if plane2 is not None and count2 > 0 and np.any(in2_vis):
                ax.scatter(pts_vis[in2_vis, 0], pts_vis[in2_vis, 1], pts_vis[in2_vis, 2],
                           c='blue', s=1, alpha=0.8, label=f'Plane 2 ({count2} pts)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            title = 'Single-Room Pipeline (RANSAC)\n'
            title += f'Frame: {depth_path.name} | Plane 1: {count1}/{num_points} ({100.0*count1/num_points:.1f}%)'
            if plane2 is not None and count2 > 0:
                title += f' | Plane 2: {count2}/{num_points} ({100.0*count2/num_points:.1f}%)'
            ax.set_title(title)
            ax.legend()

            # Attempt equal aspect ratio
            ranges = []
            for c in range(3):
                coord = pts_vis[:, c]
                ranges.append(coord.max() - coord.min())
            max_range = np.array(ranges).max() / 2.0
            mid_x = (pts_vis[:, 0].max() + pts_vis[:, 0].min()) * 0.5
            mid_y = (pts_vis[:, 1].max() + pts_vis[:, 1].min()) * 0.5
            mid_z = (pts_vis[:, 2].max() + pts_vis[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            viz_file = out_dir / "room_visualization.png"
            plt.savefig(viz_file, dpi=150)
            plt.close(fig)
            print(f"Saved visualization to: {viz_file}")
        except Exception as e:
            print(f"WARNING: Could not create visualization: {e}")
    else:
        print("INFO: Matplotlib not available; skipping visualization.")

    # -------------------------- Save result.json --------------------------
    result = {
        "frame": {
            "path": str(depth_path),
            "index": args.frame,
            "total_frames": len(depth_files)
        },
        "point_cloud": {
            "num_points": int(num_points),
            "xyz_min": [float(x_min), float(y_min), float(z_min)],
            "xyz_max": [float(x_max), float(y_max), float(z_max)],
            "xyz_extent": [float(x_extent), float(y_extent), float(z_extent)],
            "depth_range_raw": [int(depth_min), int(depth_median), int(depth_max)],
            "note": "All quantities are in raw depth units (uint16 image values)."
        },
        "planes": [],
        "metadata": {
            "coordinates": "camera-frame (X right, Y down, Z forward)",
            "depth_units": "raw uint16 values; physical scale unknown",
            "intrinsics_assumption": "same-FOV scaling from RGB to depth resolution (unverified)",
            "plane_labels": "unknown; do not interpret as floor/wall/ceiling",
            "note": "This is single-frame geometry, not a room reconstruction."
        }
    }

    # Add plane 1 info
    if plane1 is not None:
        a1, b1, c1, d1 = plane1
        result["planes"].append({
            "plane_id": 1,
            "equation": [float(a1), float(b1), float(c1), float(d1)],
            "normal": [float(a1), float(b1), float(c1)],
            "inlier_count": int(count1),
            "inlier_percentage": float(100.0*count1/num_points),
            "mean_residual": float(mean_res1),
            "median_residual": float(median_res1)
        })

    # Add plane 2 info
    if plane2 is not None and count2 > 0:
        a2, b2, c2, d2 = plane2
        result["planes"].append({
            "plane_id": 2,
            "equation": [float(a2), float(b2), float(c2), float(d2)],
            "normal": [float(a2), float(b2), float(c2)],
            "inlier_count": int(count2),
            "inlier_percentage": float(100.0*count2/num_points),
            "mean_residual": float(mean_res2),
            "median_residual": float(median_res2)
        })

    # Add angular difference if available
    if ang_diff is not None:
        result["angular_diff_deg"] = float(ang_diff)

    # Add unassigned points info
    result["unassigned_points"] = {
        "count": int(count_outliers),
        "percentage": float(100.0*count_outliers/num_points)
    }

    # Add limitations list
    result["limitations"] = [
        "Depth values are raw uint16 values; physical scale is unknown.",
        "Same-FOV intrinsics scaling is unverified.",
        "Plane labels are unknown.",
        "The distance threshold is heuristic (1% of depth range).",
        "RANSAC may fail to identify meaningful planes.",
        "This is single-frame geometry, not a complete room reconstruction.",
        "No measurement accuracy has been established against ground truth."
    ]

    # Experimental bounding box proxies (axis-aligned, raw depth units)
    result["bbox_xy_area_raw2"] = float(x_extent * y_extent)
    result["bbox_z_extent_raw"] = float(z_extent)
    result["measurement_status"] = "experimental_unvalidated"
    result["experimental_note"] = "Axis-aligned bounding box in raw depth units; not verified architectural measurement."

    result_file = out_dir / "result.json"
    try:
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved result JSON to: {result_file}")
    except Exception as e:
        print(f"WARNING: Could not save result JSON: {e}")

    print("\n=== Limitations and Notes ===")
    print("- All quantities are in raw depth units (uint16 image values).")
    print("- The same-FOV intrinsics assumption (used to create the point cloud)")
    print("  remains unverified; thus plane orientations in real space are unknown.")
    print("- No conversion to metric units (e.g., meters) has been performed.")
    print("- The detected planes are geometric candidates only;")
    print("  they are not labeled as floor, wall, ceiling, or any specific surface.")
    print("- The distance threshold (1% of depth range) is arbitrary and may affect results.")
    print("- RANSAC may fail to detect a second plane if insufficient planar structure remains.")
    print("- No files were saved or overwritten except the point cloud, visualization, and JSON in the new directory.")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()