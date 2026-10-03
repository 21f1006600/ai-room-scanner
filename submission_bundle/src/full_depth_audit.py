import cv2
import numpy as np
from pathlib import Path
import sys
import time

def process_dataset(dataset_path, name, sample=False, sample_size=100):
    """Process a single dataset and print summary."""
    print(f"\n=== Dataset: {name} ===")
    if not dataset_path.is_dir():
        print(f"ERROR: Dataset directory not found: {dataset_path}")
        return

    depth_dir = dataset_path / "depth"
    conf_dir = dataset_path / "confidence"

    if not depth_dir.is_dir():
        print(f"ERROR: Depth directory not found: {depth_dir}")
        return
    if not conf_dir.is_dir():
        print(f"ERROR: Confidence directory not found: {conf_dir}")
        return

    # Get all PNG files
    depth_files = sorted(depth_dir.glob("*.png"))
    conf_files = sorted(conf_dir.glob("*.png"))

    # Extract stems
    depth_stems = {f.stem: f for f in depth_files}
    conf_stems = {f.stem: f for f in conf_files}

    matched_stems = set(depth_stems.keys()) & set(conf_stems.keys())
    unmatched_depth = set(depth_stems.keys()) - matched_stems
    unmatched_conf = set(conf_stems.keys()) - matched_stems

    depth_count = len(depth_files)
    conf_count = len(conf_files)
    matched_count = len(matched_stems)

    print(f"Depth images: {depth_count}")
    print(f"Confidence images: {conf_count}")
    print(f"Matched stems: {matched_count}")
    if unmatched_depth:
        print(f"Unmatched depth stems: {len(unmatched_depth)} (examples: {sorted(list(unmatched_depth))[:5]})")
    if unmatched_conf:
        print(f"Unmatched confidence stems: {len(unmatched_conf)} (examples: {sorted(list(unmatched_conf))[:5]})")
    stems_match = (len(unmatched_depth) == 0 and len(unmatched_conf) == 0)
    print(f"Depth and confidence stems exactly match: {stems_match}")

    # Initialize accumulators
    total_pixels = 0
    total_zero_pixels = 0
    total_nonzero_pixels = 0
    overall_min = 65535  # max uint16
    overall_max = 0
    # Confidence histogram: we'll collect counts for each confidence value seen
    conf_hist = {}  # value -> count
    # Per-frame stats for depth min, max, zero-pixel percentage
    frame_min_vals = []
    frame_max_vals = []
    frame_zero_pcts = []

    # Process each matched frame
    matched_stems_list = sorted(matched_stems, key=lambda x: int(x) if x.isdigit() else x)
    if sample:
        stems_to_process = get_sampled_stems(matched_stems_list, sample_size)
        process_label = " (sampled)"
    else:
        stems_to_process = matched_stems_list
        process_label = ""
    total_to_process = len(stems_to_process)
    start_time = time.time()
    for i, stem in enumerate(stems_to_process):
        depth_path = depth_stems[stem]
        conf_path = conf_stems[stem]

        # Read depth
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"  WARNING: Failed to read depth image: {depth_path}")
            continue
        # Ensure it's 2D (should be)
        if depth.ndim != 2:
            print(f"  WARNING: Depth image is not 2D: {depth_path}")
            continue

        # Read confidence
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        if conf is None:
            print(f"  WARNING: Failed to read confidence image: {conf_path}")
            continue
        # Convert to grayscale if needed
        if conf.ndim == 3:
            conf = cv2.cvtColor(conf, cv2.COLOR_BGR2GRAY)
        # Ensure same shape as depth
        if conf.shape != depth.shape:
            print(f"  WARNING: Confidence shape {conf.shape} != depth shape {depth.shape} for {stem}")
            # We'll still process but note mismatch; for simplicity, we'll skip this frame's confidence stats
            # But we can still process depth stats.
            # To keep simple, we'll skip confidence stats for this frame but still process depth.
            # We'll set a flag.
            conf_valid = False
        else:
            conf_valid = True

        # Depth stats
        valid_mask = depth > 0
        zero_pixels = np.count_nonzero(~valid_mask)
        nonzero_pixels = np.count_nonzero(valid_mask)
        pixel_count = depth.size
        total_pixels += pixel_count
        total_zero_pixels += zero_pixels
        total_nonzero_pixels += nonzero_pixels

        if nonzero_pixels > 0:
            frame_min = np.min(depth, where=valid_mask, initial=np.iinfo(depth.dtype).max)
            frame_max = np.max(depth, where=valid_mask, initial=np.iinfo(depth.dtype).min)
            overall_min = min(overall_min, frame_min)
            overall_max = max(overall_max, frame_max)
        else:
            frame_min = 0
            frame_max = 0

        frame_min_vals.append(frame_min)
        frame_max_vals.append(frame_max)
        zero_pct = (zero_pixels / pixel_count) * 100 if pixel_count > 0 else 0.0
        frame_zero_pcts.append(zero_pct)

        # Confidence stats if valid and same shape
        if conf_valid:
            # Flatten confidence
            conf_flat = conf.ravel()
            # Update histogram
            for val in np.unique(conf_flat):
                cnt = np.count_nonzero(conf_flat == val)
                conf_hist[val] = conf_hist.get(val, 0) + cnt

        # Progress update every 500 frames (or at least every 20 if small)
        if (i + 1) % 500 == 0 or (total_to_process <= 100 and (i + 1) % 20 == 0):
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {i+1}/{total_to_process}{process_label} frames ({fps:.1f} fps)", flush=True)

        # If conf invalid, we skip confidence stats for this frame (but we already read depth)

    # After processing all frames
    print(f"\nOverall depth statistics{process_label}:")
    print(f"  Minimum nonzero depth: {overall_min if total_nonzero_pixels > 0 else 0}")
    print(f"  Maximum nonzero depth: {overall_max if total_nonzero_pixels > 0 else 0}")
    print(f"  Zero-depth pixels: {total_zero_pixels} / {total_pixels} ({100.0*total_zero_pixels/total_pixels if total_pixels>0 else 0:.2f}%)")
    print(f"  Nonzero-depth pixels: {total_nonzero_pixels} / {total_pixels} ({100.0*total_nonzero_pixels/total_pixels if total_pixels>0 else 0:.2f}%)")

    print(f"\nConfidence value distribution{process_label} (across all frames):")
    if conf_hist:
        total_conf_pixels = sum(conf_hist.values())
        for val in sorted(conf_hist.keys()):
            cnt = conf_hist[val]
            pct = 100.0 * cnt / total_conf_pixels if total_conf_pixels > 0 else 0.0
            print(f"  Value {val}: {cnt} pixels ({pct:.2f}%)")
    else:
        print("  No confidence data collected (likely due to shape mismatches or read errors).")

    print(f"\nPer-frame depth statistics{process_label} (across frames):")
    if frame_min_vals:
        print(f"  Frame depth minimum: min={np.min(frame_min_vals):.0f}, median={np.median(frame_min_vals):.0f}, 95th={np.percentile(frame_min_vals,95):.0f}, max={np.max(frame_min_vals):.0f}")
        print(f"  Frame depth maximum: min={np.min(frame_max_vals):.0f}, median={np.median(frame_max_vals):.0f}, 95th={np.percentile(frame_max_vals,95):.0f}, max={np.max(frame_max_vals):.0f}")
        print(f"  Frame zero-pixel percentage: min={np.min(frame_zero_pcts):.2f}%, median={np.median(frame_zero_pcts):.2f}%, 95th={np.percentile(frame_zero_pcts,95):.2f}%, max={np.max(frame_zero_pcts):.2f}%")
    else:
        print("  No frame depth statistics collected.")

def get_sampled_stems(stems, sample_size=100):
    """Return evenly distributed sampled stems, up to sample_size."""
    n = len(stems)
    if n <= sample_size:
        return stems
    # Use numpy to get evenly spaced indices
    indices = np.linspace(0, n-1, sample_size, dtype=int)
    return [stems[i] for i in indices]


def main():
    import sys
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    datasets = [
        ("single_room/c00a170fe1", base / "single_room" / "c00a170fe1"),
        ("single_scan_floor_only/1a8384c3f6", base / "single_scan_floor_only" / "1a8384c3f6"),
        ("single_scan_with_ceiling/c7d28f72c6", base / "single_scan_with_ceiling" / "c7d28f72c6")
    ]
    # Simple argument parsing
    sample = '--sample' in sys.argv
    selector = None
    for arg in sys.argv[1:]:
        if arg != '--sample' and not arg.startswith('--'):
            selector = arg
    # Determine which datasets to process
    if selector is None:
        to_process = datasets
    else:
        # Try to match by name exactly
        matched = [ds for ds in datasets if ds[0] == selector]
        if matched:
            to_process = matched
        else:
            # Try as index (1-based)
            try:
                idx = int(selector) - 1
                if 0 <= idx < len(datasets):
                    to_process = [datasets[idx]]
                else:
                    print(f"ERROR: Dataset index {selector} out of range.")
                    return
            except ValueError:
                print(f"ERROR: Unknown dataset '{selector}'. Available: {[d[0] for d in datasets]}")
                return

    for name, ds_path in to_process:
        process_dataset(ds_path, name, sample=sample)

if __name__ == "__main__":
    main()