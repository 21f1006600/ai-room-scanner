#!/usr/bin/env python3
"""
Multi-frame geometry experiment: concatenates point clouds from multiple frames
without applying odometry transformations, to assess the effect of adding
observations under the same unverified scaled-intrinsics assumption.
Saves individual and combined point clouds as .npy files.
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


def main():
    # -------------------------- Configuration --------------------------
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    dataset_name = "single_room/c00a170fe1"
    dataset_path = base / dataset_name
    depth_dir = dataset_path / "depth"
    intrinsics_path = dataset_path / "camera_matrix.csv"

    # Frames to use (as integers)
    frame_ids = [0, 30, 60, 90]
    frame_strings = [f"{f:06d}" for f in frame_ids]

    # Output directory
    out_dir = Path("output/multiframe_geometry_experiment")
    # -------------------------- Checks --------------------------
    if not depth_dir.is_dir():
        print(f"ERROR: Depth directory not found: {depth_dir}")
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

    # -------------------------- Load data and back-project --------------------------
    print(f"Loading depth frames: {frame_strings}")
    point_clouds = []  # list of (N_i, 3) arrays
    per_frame_stats = {}
    start_time = time.time()

    for fs in frame_strings:
        depth_path = depth_dir / f"{fs}.png"
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"ERROR: Failed to read depth image: {depth_path}")
            sys.exit(1)
        if depth.ndim != 2:
            print(f"ERROR: Depth image is not 2D (shape {depth.shape})")
            sys.exit(1)

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

        # Save individual point cloud as .npy
        npy_file = out_dir / f"frame_{fs}.npy"
        try:
            np.save(npy_file, points)
            print(f"Saved individual point cloud to: {npy_file}")
        except Exception as e:
            print(f"WARNING: Could not save {npy_file}: {e}")

        point_clouds.append(points)
        # Per-frame statistics
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]
        per_frame_stats[fs] = {
            "num_points": int(points.shape[0]),
            "xyz_min": [float(X.min()), float(Y.min()), float(Z.min())],
            "xyz_max": [float(X.max()), float(Y.max()), float(Z.max())],
            "xyz_extent": [float(X.max() - X.min()), float(Y.max() - Y.min()), float(Z.max() - Z.min())],
        }
        print(f"  Frame {fs}: {points.shape[0]} valid points")

    # Combine point clouds (simple concatenation, no transformation)
    combined_points = np.vstack(point_clouds) if point_clouds else np.empty((0, 3))
    combine_time = time.time()
    elapsed = combine_time - start_time

    # Save combined point cloud as .npy
    combined_npy = out_dir / "combined_cloud.npy"
    try:
        np.save(combined_npy, combined_points)
        print(f"Saved combined point cloud to: {combined_npy}")
    except Exception as e:
        print(f"WARNING: Could not save combined point cloud: {e}")

    # Combined statistics
    if combined_points.size > 0:
        Xc = combined_points[:, 0]
        Yc = combined_points[:, 1]
        Zc = combined_points[:, 2]
        combined_stats = {
            "num_points": int(combined_points.shape[0]),
            "xyz_min": [float(Xc.min()), float(Yc.min()), float(Zc.min())],
            "xyz_max": [float(Xc.max()), float(Yc.max()), float(Zc.max())],
            "xyz_extent": [float(Xc.max() - Xc.min()), float(Yc.max() - Yc.min()), float(Zc.max() - Zc.min())],
        }
        # Unique points after rounding to 2 decimal places (diagnostic for coverage/duplicates)
        rounded = np.round(combined_points, decimals=2)
        # Ensure contiguous for structured view
        rounded = np.ascontiguousarray(rounded)
        # Use structured view to find unique rows
        unique_points = np.unique(rounded.view([('', rounded.dtype)] * 3))
        num_unique = len(unique_points)
    else:
        combined_stats = {
            "num_points": 0,
            "xyz_min": [0.0, 0.0, 0.0],
            "xyz_max": [0.0, 0.0, 0.0],
            "xyz_extent": [0.0, 0.0, 0.0],
        }
        num_unique = 0

    # -------------------------- Save results --------------------------
    # JSON report
    result = {
        "frame_ids": frame_strings,
        "per_frame_stats": per_frame_stats,
        "combined_stats": combined_stats,
        "unique_points_rounded_2dp": int(num_unique),
        "runtime_seconds": float(elapsed),
        "limitations": [
            "Depth values are raw uint16 units; physical scale unknown.",
            "Same-FOV intrinsics scaling (used to back-project depth) is unverified.",
            "No odometry transformations applied (pure concatenation).",
            "No metric accuracy claims; this experiment tests effect of adding frames only.",
            "Unique point count after rounding is a duplicate/coverage diagnostic, not geometric accuracy."
        ]
    }
    json_file = out_dir / "result.json"
    try:
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved JSON report to: {json_file}")
    except Exception as e:
        print(f"WARNING: Could not save JSON report: {e}")

    # CSV summary (one consistent header)
    csv_file = out_dir / "summary.csv"
    try:
        # Prepare rows
        rows = []
        # Per-frame rows
        for fs in frame_strings:
            stats = per_frame_stats[fs]
            rows.append({
                'frame_id': fs,
                'num_points': stats['num_points'],
                'xyz_min_x': stats['xyz_min'][0],
                'xyz_min_y': stats['xyz_min'][1],
                'xyz_min_z': stats['xyz_min'][2],
                'xyz_max_x': stats['xyz_max'][0],
                'xyz_max_y': stats['xyz_max'][1],
                'xyz_max_z': stats['xyz_max'][2],
                'xyz_extent_x': stats['xyz_extent'][0],
                'xyz_extent_y': stats['xyz_extent'][1],
                'xyz_extent_z': stats['xyz_extent'][2],
                'unique_points_rounded_2dp': ''  # blank for individual frames
            })
        # Combined row
        rows.append({
            'frame_id': 'COMBINED',
            'num_points': combined_stats['num_points'],
            'xyz_min_x': combined_stats['xyz_min'][0],
            'xyz_min_y': combined_stats['xyz_min'][1],
            'xyz_min_z': combined_stats['xyz_min'][2],
            'xyz_max_x': combined_stats['xyz_max'][0],
            'xyz_max_y': combined_stats['xyz_max'][1],
            'xyz_max_z': combined_stats['xyz_max'][2],
            'xyz_extent_x': combined_stats['xyz_extent'][0],
            'xyz_extent_y': combined_stats['xyz_extent'][1],
            'xyz_extent_z': combined_stats['xyz_extent'][2],
            'unique_points_rounded_2dp': num_unique
        })
        # Write CSV
        if rows:
            fieldnames = [
                'frame_id', 'num_points',
                'xyz_min_x', 'xyz_min_y', 'xyz_min_z',
                'xyz_max_x', 'xyz_max_y', 'xyz_max_z',
                'xyz_extent_x', 'xyz_extent_y', 'xyz_extent_z',
                'unique_points_rounded_2dp'
            ]
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"Saved CSV summary to: {csv_file}")
    except Exception as e:
        print(f"WARNING: Could not save CSV summary: {e}")

    # -------------------------- Visualization --------------------------
    if HAVE_MPL:
        try:
            # We'll downsample point clouds for clarity with a fixed seed for reproducibility
            max_vis_points = 50000
            np.random.seed(42)  # Fixed seed for reproducible downsampling
            def downsample(points, max_n):
                if points is None or len(points) == 0:
                    return points
                if len(points) <= max_n:
                    return points
                idx = np.random.choice(len(points), max_n, replace=False)
                return points[idx]

            # Individual frames visualization (grid)
            n_frames = len(frame_strings)
            n_cols = 2
            n_rows = (n_frames + n_cols - 1) // n_cols
            fig_ind = plt.figure(figsize=(4 * n_cols, 4 * n_rows))
            for i, fs in enumerate(frame_strings):
                pts = point_clouds[i]
                pts_vis = downsample(pts, max_vis_points)
                ax = fig_ind.add_subplot(n_rows, n_cols, i + 1, projection='3d')
                if pts_vis is not None and len(pts_vis) > 0:
                    ax.scatter(pts_vis[:, 0], pts_vis[:, 1], pts_vis[:, 2],
                               c='blue', s=1, alpha=0.5)
                ax.set_title(f'Frame {fs}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                # Attempt equal aspect ratio
                try:
                    if pts_vis is not None and len(pts_vis) > 0:
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
                except Exception:
                    pass
            ind_viz_file = out_dir / "individual_frames.png"
            fig_ind.tight_layout()
            plt.savefig(ind_viz_file, dpi=150)
            plt.close(fig_ind)
            print(f"Saved individual frames visualization to: {ind_viz_file}")

            # Combined cloud visualization
            fig_comb = plt.figure(figsize=(8, 6))
            ax_comb = fig_comb.add_subplot(111, projection='3d')
            pts_comb_vis = downsample(combined_points, max_vis_points)
            if pts_comb_vis is not None and len(pts_comb_vis) > 0:
                ax_comb.scatter(pts_comb_vis[:, 0], pts_comb_vis[:, 1], pts_comb_vis[:, 2],
                                c='green', s=1, alpha=0.5)
            ax_comb.set_title('Combined Point Cloud (no odometry)')
            ax_comb.set_xlabel('X')
            ax_comb.set_ylabel('Y')
            ax_comb.set_zlabel('Z')
            # Attempt equal aspect ratio
            try:
                if pts_comb_vis is not None and len(pts_comb_vis) > 0:
                    ranges = []
                    for c in range(3):
                        coord = pts_comb_vis[:, c]
                        ranges.append(coord.max() - coord.min())
                    max_range = np.array(ranges).max() / 2.0
                    mid_x = (pts_comb_vis[:, 0].max() + pts_comb_vis[:, 0].min()) * 0.5
                    mid_y = (pts_comb_vis[:, 1].max() + pts_comb_vis[:, 1].min()) * 0.5
                    mid_z = (pts_comb_vis[:, 2].max() + pts_comb_vis[:, 2].min()) * 0.5
                    ax_comb.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax_comb.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax_comb.set_zlim(mid_z - max_range, mid_z + max_range)
            except Exception:
                pass
            comb_viz_file = out_dir / "combined_cloud.png"
            plt.savefig(comb_viz_file, dpi=150)
            plt.close(fig_comb)
            print(f"Saved combined cloud visualization to: {comb_viz_file}")
        except Exception as e:
            print(f"WARNING: Could not create visualization: {e}")
    else:
        print("INFO: Matplotlib not available; skipping visualization.")

    # -------------------------- Summary --------------------------
    print("\n=== Experiment Summary ===")
    print("Processed frames:", frame_strings)
    print("Per-frame point counts:", [per_frame_stats[fs]['num_points'] for fs in frame_strings])
    print("Combined point count:", combined_stats['num_points'])
    print("Unique points after rounding to 2 decimal places:", num_unique)
    print(f"Runtime: {elapsed:.3f} seconds")
    print(f"JSON report: {json_file}")
    if HAVE_MPL:
        print(f"Individual frames visualization: {ind_viz_file}")
        print(f"Combined cloud visualization: {comb_viz_file}")
    print("\nLimitations (see JSON for full list):")
    for lim in result['limitations']:
        print(f"  - {lim}")
    print("\nExperiment complete.")


if __name__ == "__main__":
    main()