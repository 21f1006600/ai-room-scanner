import cv2
import numpy as np
from pathlib import Path
import time

def main():
    base = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    dataset = base / "single_room" / "c00a170fe1"
    depth_dir = dataset / "depth"
    conf_dir = dataset / "confidence"

    if not depth_dir.is_dir() or not conf_dir.is_dir():
        print("ERROR: Dataset directories not found")
        return

    depth_files = sorted(depth_dir.glob("*.png"))
    # Take first 100 frames
    depth_files = depth_files[:100]

    # Initialize accumulators for timing
    total_read_depth = 0.0
    total_read_conf = 0.0
    total_depth_calc = 0.0
    total_conf_count = 0.0
    total_agg = 0.0

    # For confidence histogram (local)
    conf_hist = {}
    # For depth stats (we don't need to store per-frame, but we need to compute overall min/max etc.)
    overall_min = 65535
    overall_max = 0
    total_pixels = 0
    total_zero_pixels = 0
    total_nonzero_pixels = 0

    print(f"Profiling {len(depth_files)} frames from first dataset...")
    for idx, depth_path in enumerate(depth_files):
        stem = depth_path.stem
        conf_path = conf_dir / f"{stem}.png"

        # 1. Read depth
        t0 = time.perf_counter()
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        t1 = time.perf_counter()
        total_read_depth += (t1 - t0)
        if depth is None:
            print(f"  WARNING: Failed to read depth image: {depth_path}")
            continue

        # 2. Read confidence
        t0 = time.perf_counter()
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
        t1 = time.perf_counter()
        total_read_conf += (t1 - t0)
        if conf is None:
            print(f"  WARNING: Failed to read confidence image: {conf_path}")
            continue

        # Ensure conf is grayscale
        if conf.ndim == 3:
            conf = cv2.cvtColor(conf, cv2.COLOR_BGR2GRAY)

        # 3. Depth mask, zero count, min and max
        t0 = time.perf_counter()
        valid_mask = depth > 0
        zero_pixels = np.count_nonzero(~valid_mask)
        nonzero_pixels = np.count_nonzero(valid_mask)
        pixel_count = depth.size
        if nonzero_pixels > 0:
            frame_min = np.min(depth, where=valid_mask, initial=np.iinfo(depth.dtype).max)
            frame_max = np.max(depth, where=valid_mask, initial=np.iinfo(depth.dtype).min)
        else:
            frame_min = 0
            frame_max = 0
        t1 = time.perf_counter()
        total_depth_calc += (t1 - t0)

        # Update depth aggregates (for measurement)
        total_pixels += pixel_count
        total_zero_pixels += zero_pixels
        total_nonzero_pixels += nonzero_pixels
        if nonzero_pixels > 0:
            overall_min = min(overall_min, frame_min)
            overall_max = max(overall_max, frame_max)

        # 4. Confidence unique-value counting
        t0 = time.perf_counter()
        conf_flat = conf.ravel()
        vals, counts = np.unique(conf_flat, return_counts=True)
        for val, cnt in zip(vals, counts):
            conf_hist[val] = conf_hist.get(val, 0) + cnt
        t1 = time.perf_counter()
        total_conf_count += (t1 - t0)

        # 5. Updating summary lists and counters (simulated)
        t0 = time.perf_counter()
        # We don't store per-frame lists, but we simulate the appends by doing nothing (or just pass)
        # To measure the time of the operations that would be done, we can just compute the values
        # that would be appended (but we already have frame_min, frame_max, zero_pct)
        zero_pct = (zero_pixels / pixel_count) * 100 if pixel_count > 0 else 0.0
        # Simulate appending to lists (just assigning to local variables)
        _ = frame_min
        _ = frame_max
        _ = zero_pct
        t1 = time.perf_counter()
        total_agg += (t1 - t0)

        # Progress every 20 frames
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{len(depth_files)} frames", flush=True)

    # Print results
    print("\n=== Profiling Results (averages per frame) ===")
    n = len(depth_files)
    if n == 0:
        print("No frames processed.")
        return
    print(f"Total frames processed: {n}")
    print(f"1. Read depth images: {total_read_depth*1000/n:.2f} ms/frame (total {total_read_depth*1000:.1f} ms)")
    print(f"2. Read confidence images: {total_read_conf*1000/n:.2f} ms/frame (total {total_read_conf*1000:.1f} ms)")
    print(f"3. Depth mask, zero count, min/max: {total_depth_calc*1000/n:.2f} ms/frame (total {total_depth_calc*1000:.1f} ms)")
    print(f"4. Confidence unique-value counting: {total_conf_count*1000/n:.2f} ms/frame (total {total_conf_count*1000:.1f} ms)")
    print(f"5. Updating summary lists/counters: {total_agg*1000/n:.2f} ms/frame (total {total_agg*1000:.1f} ms)")
    print(f"\nOverall depth stats (from processed frames):")
    print(f"  Min nonzero depth: {overall_min if total_nonzero_pixels>0 else 0}")
    print(f"  Max nonzero depth: {overall_max if total_nonzero_pixels>0 else 0}")
    print(f"  Zero-depth pixels: {total_zero_pixels} / {total_pixels} ({100.0*total_zero_pixels/total_pixels if total_pixels>0 else 0:.2f}%)")
    print(f"  Nonzero-depth pixels: {total_nonzero_pixels} / {total_pixels} ({100.0*total_nonzero_pixels/total_pixels if total_pixels>0 else 0:.2f}%)")
    print(f"  Distinct confidence values: {len(conf_hist)}")

if __name__ == "__main__":
    main()