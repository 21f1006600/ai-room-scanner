import cv2
import numpy as np
from pathlib import Path
import sys

def main():
    # Dataset path
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup/single_room/c00a170fe1")
    depth_dir = base / "depth"
    conf_dir = base / "confidence"

    if not depth_dir.is_dir():
        print(f"Error: Depth directory not found: {depth_dir}")
        sys.exit(1)
    if not conf_dir.is_dir():
        print(f"Error: Confidence directory not found: {conf_dir}")
        sys.exit(1)

    # Get sorted lists of PNG files
    depth_files = sorted(depth_dir.glob("*.png"))
    conf_files = sorted(conf_dir.glob("*.png"))

    # Extract stems
    depth_stems = {f.stem: f for f in depth_files}
    conf_stems = {f.stem: f for f in conf_files}

    matched_stems = set(depth_stems.keys()) & set(conf_stems.keys())
    unmatched_depth = set(depth_stems.keys()) - matched_stems
    unmatched_conf = set(conf_stems.keys()) - matched_stems

    print(f"Matched depth-confidence pairs: {len(matched_stems)}")
    print(f"Unmatched depth files: {len(unmatched_depth)}")
    if unmatched_depth:
        print(f"  Examples (up to 5): {sorted(list(unmatched_depth))[:5]}")
    print(f"Unmatched confidence files: {len(unmatched_conf)}")
    if unmatched_conf:
        print(f"  Examples (up to 5): {sorted(list(unmatched_conf))[:5]}")
    print()

    # Initialize confidence histograms (for depth>0 only)
    hist = [np.zeros(65536, dtype=np.uint64) for _ in range(3)]  # indices 0,1,2
    conf_total = [0, 0, 0]
    conf_zero = [0, 0, 0]
    conf_nonzero = [0, 0, 0]

    # For depth temporal comparison
    prev_depth = None
    prev_stem = None
    pair_median_changes = []  # list of median absolute depth change for each consecutive pair
    pair_info = []            # list of (stem1, stem2, median_change)
    total_compared_pixels = 0  # total number of pixel pairs where both depths are nonzero across all consecutive pairs
    excluded_nonconsecutive = 0  # count of pairs skipped because stems not consecutive
    excluded_no_overlap = 0      # count of consecutive pairs with no overlapping nonzero pixels

    # Process frames in order of stem (numeric)
    sorted_stems = sorted(matched_stems, key=lambda x: int(x))

    for stem in sorted_stems:
        depth_path = depth_stems[stem]
        conf_path = conf_stems[stem]

        # Load depth and confidence
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"Warning: Failed to load depth image: {depth_path}")
            continue
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        if conf is None:
            print(f"Warning: Failed to load confidence image: {conf_path}")
            continue
        # Ensure confidence is single channel
        if conf.ndim == 3:
            conf = cv2.cvtColor(conf, cv2.COLOR_BGR2GRAY)

        # Update confidence statistics (only for depth>0 pixels)
        for c_val in range(3):
            mask_c = (conf == c_val)
            total_pixels = np.count_nonzero(mask_c)
            zero_pixels = np.count_nonzero(mask_c & (depth == 0))
            nonzero_pixels = np.count_nonzero(mask_c & (depth > 0))
            conf_total[c_val] += total_pixels
            conf_zero[c_val] += zero_pixels
            conf_nonzero[c_val] += nonzero_pixels
            # Histogram of depth values where confidence==c_val and depth>0
            if nonzero_pixels > 0:
                depths = depth[mask_c & (depth > 0)]
                # Use bincount for histogram
                hist_c = np.bincount(depths, minlength=65536).astype(np.uint64)
                hist[c_val] += hist_c

        # Temporal comparison with previous frame (if consecutive in stem)
        if prev_depth is not None:
            # Check if stems are consecutive numbers
            try:
                prev_num = int(prev_stem)
                curr_num = int(stem)
                if curr_num == prev_num + 1:
                    # Convert to float to avoid unsigned integer wraparound
                    prev_depth_float = prev_depth.astype(np.float32)
                    depth_float = depth.astype(np.float32)
                    # Mask where both depths are nonzero
                    mask_both_nonzero = (prev_depth > 0) & (depth > 0)
                    if np.any(mask_both_nonzero):
                        diff = np.abs(prev_depth_float[mask_both_nonzero] - depth_float[mask_both_nonzero])
                        median_diff = np.median(diff)
                        # Defensive check: difference should not exceed 65535 for uint16 inputs
                        if median_diff > 65535.0:
                            print(f"Warning: median difference {median_diff} exceeds 65535 for pair {prev_stem}->{stem}")
                        pair_median_changes.append(median_diff)
                        pair_info.append((prev_stem, stem, median_diff))
                        total_compared_pixels += np.count_nonzero(mask_both_nonzero)
                    else:
                        # No overlapping nonzero pixels
                        excluded_no_overlap += 1
                else:
                    # Stems are not consecutive (should not happen if sorted and numeric, but just in case)
                    excluded_nonconsecutive += 1
            except ValueError:
                # Stems not numeric, skip temporal check
                excluded_nonconsecutive += 1
        # else: first frame, no previous

        # Set previous for next iteration
        prev_depth = depth
        prev_stem = stem

    # After processing all frames, compute confidence median nonzero depth from histograms
    print("=== Confidence Aggregation (across all frames) ===")
    for c_val in range(3):
        total = conf_total[c_val]
        zero = conf_zero[c_val]
        nonzero = conf_nonzero[c_val]
        print(f"Confidence value {c_val}:")
        print(f"  Total pixels: {total}")
        print(f"  Zero-depth pixels: {zero}")
        print(f"  Nonzero-depth pixels: {nonzero}")
        if nonzero > 0:
            # Compute median from histogram
            cumsum = np.cumsum(hist[c_val])
            # Find first index where cumsum >= nonzero/2.0
            half = nonzero / 2.0
            idx = np.searchsorted(cumsum, half, side='left')
            median_depth = idx
            print(f"  Median nonzero depth: {median_depth:.0f}")
        else:
            print(f"  Median nonzero depth: N/A (no nonzero pixels)")
        print()

    # Temporal comparison results
    print("=== Depth Temporal Comparison (same-pixel consecutive frames) ===")
    print(f"Number of compared pixel pairs (both depths nonzero): {total_compared_pixels}")
    if len(pair_median_changes) == 0:
        print("No consecutive frame pairs with overlapping nonzero depths found for comparison.")
    else:
        changes_arr = np.array(pair_median_changes)
        median_of_medians = np.median(changes_arr)
        p95 = np.percentile(changes_arr, 95)
        max_change = np.max(changes_arr)
        # Find pair with max change
        max_idx = np.argmax(changes_arr)
        max_pair = pair_info[max_idx]
        print(f"Median of per-pair median absolute depth change: {median_of_medians:.3f}")
        print(f"95th percentile of per-pair median absolute depth change: {p95:.3f}")
        print(f"Pair with largest median change: frames {max_pair[0]} -> {max_pair[1]} (median change = {max_change:.3f})")
    print()
    print("Frames/pairs excluded:")
    print(f"  - Non-consecutive stems: {excluded_nonconsecutive}")
    print(f"  - Consecutive stems with no overlapping nonzero pixels: {excluded_no_overlap}")
    print()
    print(f"NOTE: This compares nonzero depth values at the same pixel coordinates between consecutive frames.")
    print(f"      It does NOT prove geometric correspondence or temporal synchronization.")
    print(f"      Depth values are in raw units; confidence semantics are unknown.")
    print(f"      We do not assign meanings such as low/medium/high to confidence values.")
    print(f"      Absolute differences are computed in floating-point to avoid unsigned integer wraparound.")

if __name__ == "__main__":
    main()