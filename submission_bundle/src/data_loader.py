from pathlib import Path
import cv2
import itertools
import statistics
from collections import Counter
import math
import os


def check_dataset_inputs(dataset_dir: Path) -> dict:
    """
    Check if expected inputs exist in an RGB-D dataset directory and count PNG files in depth/confidence.

    Args:
        dataset_dir: Path to the dataset directory (e.g., single_room)

    Returns:
        Dictionary with input names as keys and boolean existence as values, plus PNG counts for depth and confidence.
    """
    # Ensure we're working with a Path object
    dataset_path = Path(dataset_dir)

    # Define expected inputs with their types (file or directory)
    expected_inputs = {
        "rgb.mp4": "file",
        "depth": "dir",
        "confidence": "dir",
        "camera_matrix.csv": "file",
        "imu.csv": "file",
        "odometry.csv": "file",
    }

    # Initialize result dictionary
    existence_check = {}

    # Check each expected input
    for name, input_type in expected_inputs.items():
        full_path = dataset_path / name
        if input_type == "file":
            existence_check[name] = full_path.is_file()
        elif input_type == "dir":
            existence_check[name] = full_path.is_dir()
        else:
            # This shouldn't happen with our defined inputs
            existence_check[name] = False

    # Count PNG files in depth and confidence directories if they exist (case-insensitive, memory efficient)
    depth_dir = dataset_path / "depth"
    confidence_dir = dataset_path / "confidence"

    if existence_check.get("depth", False):
        # Count files with .png extension (case-insensitive) without loading images
        depth_png_count = sum(
            1
            for f in depth_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".png"
        )
    else:
        depth_png_count = 0

    if existence_check.get("confidence", False):
        # Count files with .png extension (case-insensitive) without loading images
        confidence_png_count = sum(
            1
            for f in confidence_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".png"
        )
    else:
        confidence_png_count = 0

    # Add the counts to the result dictionary
    existence_check["depth_png_count"] = depth_png_count
    existence_check["confidence_png_count"] = confidence_png_count

    return existence_check


def inspect_dataset_metadata(dataset_dir: Path) -> dict:
    """
    Inspect metadata of an RGB-D dataset for reconstruction pipeline.

    Args:
        dataset_dir: Path to the dataset directory (e.g., single_room)

    Returns:
        Dictionary containing metadata: video properties, camera matrix, IMU/odometry samples,
        and properties of first depth and confidence PNG files.
    """
    dataset_path = Path(dataset_dir)
    metadata = {}

    # 1. RGB video properties
    rgb_path = dataset_path / "rgb.mp4"
    if rgb_path.is_file():
        cap = cv2.VideoCapture(str(rgb_path))
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            metadata["rgb_video"] = {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration_seconds": duration
            }
            cap.release()
        else:
            metadata["rgb_video"] = {"error": "Could not open video"}
    else:
        metadata["rgb_video"] = {"error": "File not found"}

    # 2. Camera matrix
    camera_path = dataset_path / "camera_matrix.csv"
    if camera_path.is_file():
        try:
            # Try numpy first
            try:
                import numpy as np
                camera_matrix = np.loadtxt(camera_path, delimiter=',')
                # Expect 3x3
            except ImportError:
                # Fallback to csv
                import csv
                with open(camera_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                camera_matrix = [[float(val) for val in row] for row in rows]
            metadata["camera_matrix"] = camera_matrix
        except Exception as e:
            metadata["camera_matrix"] = {"error": str(e)}
    else:
        metadata["camera_matrix"] = {"error": "File not found"}

    # 3. IMU CSV
    imu_path = dataset_path / "imu.csv"
    if imu_path.is_file():
        try:
            import csv
            with open(imu_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    metadata["imu"] = {"error": "Empty file"}
                else:
                    # Read only header and up to 3 data rows
                    rows = list(itertools.islice(reader, 4))
                if rows:
                    metadata["imu"] = {
                        "column_names": rows[0],
                        "first_3_rows": rows[1:] if len(rows) > 1 else []
                    }
                else:
                    metadata["imu"] = {"error": "No data rows"}
        except Exception as e:
            metadata["imu"] = {"error": str(e)}
    else:
        metadata["imu"] = {"error": "File not found"}

    # 4. Odometry CSV
    odom_path = dataset_path / "odometry.csv"
    if odom_path.is_file():
        try:
            import csv
            with open(odom_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    metadata["odometry"] = {"error": "Empty file"}
                else:
                    # Read only header and up to 3 data rows
                    rows = list(itertools.islice(reader, 4))
                if rows:
                    metadata["odometry"] = {
                        "column_names": rows[0],
                        "first_3_rows": rows[1:] if len(rows) > 1 else []
                    }
                else:
                    metadata["odometry"] = {"error": "No data rows"}
        except Exception as e:
            metadata["odometry"] = {"error": str(e)}
    else:
        metadata["odometry"] = {"error": "File not found"}

    # 5. First depth PNG
    depth_dir = dataset_path / "depth"
    if depth_dir.is_dir():
        # Get all PNG files (case-insensitive) and sort by name for deterministic first
        png_files = []
        for f in depth_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".png":
                png_files.append(f)
        png_files.sort(key=lambda p: p.name.lower())
        if png_files:
            first_png = png_files[0]
            try:
                img = cv2.imread(str(first_png), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    metadata["first_depth_png"] = {
                        "filename": first_png.name,
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "channels": img.shape[2] if len(img.shape) == 3 else 1,
                        "dtype": str(img.dtype)
                    }
                else:
                    metadata["first_depth_png"] = {"error": "Could not read image"}
            except Exception as e:
                metadata["first_depth_png"] = {"error": str(e)}
        else:
            metadata["first_depth_png"] = {"error": "No PNG files found"}
    else:
        metadata["first_depth_png"] = {"error": "Directory not found"}

    # 6. First confidence PNG
    confidence_dir = dataset_path / "confidence"
    if confidence_dir.is_dir():
        png_files = []
        for f in confidence_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".png":
                png_files.append(f)
        png_files.sort(key=lambda p: p.name.lower())
        if png_files:
            first_png = png_files[0]
            try:
                img = cv2.imread(str(first_png), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    metadata["first_confidence_png"] = {
                        "filename": first_png.name,
                        "width": img.shape[1],
                        "height": img.shape[0],
                        "channels": img.shape[2] if len(img.shape) == 3 else 1,
                        "dtype": str(img.dtype)
                    }
                else:
                    metadata["first_confidence_png"] = {"error": "Could not read image"}
            except Exception as e:
                metadata["first_confidence_png"] = {"error": str(e)}
        else:
            metadata["first_confidence_png"] = {"error": "No PNG files found"}
    else:
        metadata["first_confidence_png"] = {"error": "Directory not found"}

    return metadata


def audit_dataset_timing(dataset_dir: Path) -> dict:
    """
    Audit timing and synchronization metadata of an RGB-D dataset.

    Args:
        dataset_dir: Path to the dataset directory (e.g., single_room)

    Returns:
        Dictionary containing:
        - RGB video frame count
        - IMU and odometry row counts, first/last timestamps, first/last frame IDs (odometry)
        - Depth and confidence PNG counts and first/last filenames
        - Timestamp differences between first and last IMU and odometry samples (if numeric)
    """
    dataset_path = Path(dataset_dir)
    audit = {}

    # Helper to safely convert to float
    def to_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # 1. RGB video frame count
    rgb_path = dataset_path / "rgb.mp4"
    if rgb_path.is_file():
        cap = cv2.VideoCapture(str(rgb_path))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            audit["rgb_video_frame_count"] = frame_count
            cap.release()
        else:
            audit["rgb_video_frame_count"] = {"error": "Could not open video"}
    else:
        audit["rgb_video_frame_count"] = {"error": "File not found"}

    # 2. IMU CSV: row count, first/last timestamps
    imu_path = dataset_path / "imu.csv"
    if imu_path.is_file():
        try:
            import csv
            with open(imu_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    audit["imu"] = {"error": "Empty file"}
                else:
                    # Determine timestamp column index
                    timestamp_col = 0
                    for i, col in enumerate(header):
                        if col.strip().lower() == "timestamp":
                            timestamp_col = i
                            break
                    row_count = 0
                    first_ts = None
                    last_ts = None
                    for row in reader:
                        row_count += 1
                        if len(row) > timestamp_col:
                            ts_val = row[timestamp_col]
                            if first_ts is None:
                                first_ts = ts_val
                            last_ts = ts_val
                    if row_count == 0:
                        audit["imu"] = {"error": "No data rows"}
                    else:
                        audit["imu"] = {
                            "row_count": row_count,
                            "first_timestamp": first_ts,
                            "last_timestamp": last_ts
                        }
        except Exception as e:
            audit["imu"] = {"error": str(e)}
    else:
        audit["imu"] = {"error": "File not found"}

    # 3. Odometry CSV: row count, first/last timestamps, first/last frame IDs
    odom_path = dataset_path / "odometry.csv"
    if odom_path.is_file():
        try:
            import csv
            with open(odom_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    audit["odometry"] = {"error": "Empty file"}
                else:
                    # Determine timestamp and frame column indices
                    timestamp_col = 0
                    frame_col = 1  # default second column
                    for i, col in enumerate(header):
                        low = col.strip().lower()
                        if low == "timestamp":
                            timestamp_col = i
                        elif low == "frame":
                            frame_col = i
                    row_count = 0
                    first_ts = None
                    last_ts = None
                    first_frame = None
                    last_frame = None
                    for row in reader:
                        row_count += 1
                        if len(row) > timestamp_col:
                            ts_val = row[timestamp_col]
                            if first_ts is None:
                                first_ts = ts_val
                            last_ts = ts_val
                        if len(row) > frame_col:
                            frm_val = row[frame_col]
                            if first_frame is None:
                                first_frame = frm_val
                            last_frame = frm_val
                    if row_count == 0:
                        audit["odometry"] = {"error": "No data rows"}
                    else:
                        audit["odometry"] = {
                            "row_count": row_count,
                            "first_timestamp": first_ts,
                            "last_timestamp": last_ts,
                            "first_frame_id": first_frame,
                            "last_frame_id": last_frame
                        }
        except Exception as e:
            audit["odometry"] = {"error": str(e)}
    else:
        audit["odometry"] = {"error": "File not found"}

    # 4. Depth and confidence PNG counts and first/last filenames
    def process_png_dir(dir_path):
        if not dir_path.is_dir():
            return {"error": "Directory not found"}
        png_files = []
        for f in dir_path.iterdir():
            if f.is_file() and f.suffix.lower() == ".png":
                png_files.append(f)
        if not png_files:
            return {"error": "No PNG files found"}
        png_files.sort(key=lambda p: p.name.lower())
        first = png_files[0].name
        last = png_files[-1].name
        return {
            "count": len(png_files),
            "first_filename": first,
            "last_filename": last
        }

    depth_dir = dataset_path / "depth"
    confidence_dir = dataset_path / "confidence"
    audit["depth_png"] = process_png_dir(depth_dir)
    audit["confidence_png"] = process_png_dir(confidence_dir)

    # 5. Timestamp differences (if both are numeric)
    imu_data = audit.get("imu", {})
    odo_data = audit.get("odometry", {})
    # Only compute if both are dicts and have timestamp keys and are not error dicts
    if isinstance(imu_data, dict) and "error" not in imu_data and isinstance(odo_data, dict) and "error" not in odo_data:
        imu_first = imu_data.get("first_timestamp")
        imu_last = imu_data.get("last_timestamp")
        odo_first = odo_data.get("first_timestamp")
        odo_last = odo_data.get("last_timestamp")
        # Try to convert to float
        imu_first_f = to_float(imu_first)
        imu_last_f = to_float(imu_last)
        odo_first_f = to_float(odo_first)
        odo_last_f = to_float(odo_last)
        audit["timestamp_difference_first"] = (
            imu_first_f - odo_first_f if imu_first_f is not None and odo_first_f is not None else None
        )
        audit["timestamp_difference_last"] = (
            imu_last_f - odo_last_f if imu_last_f is not None and odo_last_f is not None else None
        )
    else:
        audit["timestamp_difference_first"] = None
        audit["timestamp_difference_last"] = None

    return audit


def analyze_timestamp_intervals(dataset_dir: Path) -> dict:
    """
    Analyze timestamp intervals in IMU and odometry CSV files.

    Args:
        dataset_dir: Path to the dataset directory (e.g., single_room)

    Returns:
        Dictionary with keys 'imu' and 'odometry', each containing:
        - valid_timestamp_count: number of timestamps that could be parsed as float
        - min_interval: minimum difference between consecutive timestamps (or None)
        - max_interval: maximum difference between consecutive timestamps (or None)
        - mean_interval: mean of consecutive intervals (or None)
        - median_interval: median of consecutive intervals (or None)
        - non_positive_intervals: count of intervals <= 0
        For odometry only, additional fields:
        - intervals_over_1_5x_median: count of intervals greater than 1.5 times the median interval
        - intervals_over_2x_median: count of intervals greater than 2 times the median interval
        - largest_gap_index: index of the interval with the largest gap
        - largest_gap_seconds: largest interval
        - largest_gap_multiple_of_median: largest interval divided by median interval
        - largest_gap_context: object with previous/next frame IDs and timestamps, and interval seconds
        - interval_frequencies_ms: list of up to 10 most frequent interval values (in milliseconds, rounded to 3 decimal places) with counts
        If the file is missing or unreadable, returns an error string.
    """
    dataset_path = Path(dataset_dir)
    result = {}

    def parse_timestamps_from_csv(csv_path):
        """Return list of (timestamp_float, frame_str_or_none) from the CSV's timestamp column."""
        if not csv_path.is_file():
            return None, "File not found"
        try:
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    return None, "Empty file"
                # Find timestamp column
                timestamp_col = 0
                for i, col in enumerate(header):
                    if col.strip().lower() == "timestamp":
                        timestamp_col = i
                        break
                # Find frame column (only for odometry, but we'll compute for both; imu may not have frame)
                frame_col = None
                for i, col in enumerate(header):
                    if col.strip().lower() == "frame":
                        frame_col = i
                        break
                records = []  # list of (timestamp_float, frame_value_or_None)
                for row in reader:
                    if len(row) > timestamp_col:
                        ts_val = row[timestamp_col]
                        try:
                            ts = float(ts_val)
                            frame_val = row[frame_col] if frame_col is not None and len(row) > frame_col else None
                            records.append((ts, frame_val))
                        except (ValueError, TypeError):
                            # Skip non-numeric timestamps
                            pass
                return records, None
        except Exception as e:
            return None, f"Error reading file: {e}"

    def compute_interval_stats(records, is_odometry=False):
        """Given list of (timestamp, frame), return dict with interval stats and odometry-specific extras."""
        if len(records) < 2:
            base = {
                "valid_timestamp_count": len(records),
                "min_interval": None,
                "max_interval": None,
                "mean_interval": None,
                "median_interval": None,
                "non_positive_intervals": 0,
                "intervals_over_1_5x_median": None,
                "intervals_over_2x_median": None,
                "largest_gap_index": None,
                "largest_gap_seconds": None,
                "largest_gap_multiple_of_median": None,
                "largest_gap_context": None,
                "interval_frequencies_ms": None
            }
            return base if is_odometry else {
                "valid_timestamp_count": len(records),
                "min_interval": None,
                "max_interval": None,
                "mean_interval": None,
                "median_interval": None,
                "non_positive_intervals": 0
            }
        timestamps = [r[0] for r in records]
        frames = [r[1] for r in records]
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        if not intervals:
            base = {
                "valid_timestamp_count": len(records),
                "min_interval": None,
                "max_interval": None,
                "mean_interval": None,
                "median_interval": None,
                "non_positive_intervals": 0,
                "intervals_over_1_5x_median": None,
                "intervals_over_2x_median": None,
                "largest_gap_index": None,
                "largest_gap_seconds": None,
                "largest_gap_multiple_of_median": None,
                "largest_gap_context": None,
                "interval_frequencies_ms": None
            }
            return base if is_odometry else {
                "valid_timestamp_count": len(records),
                "min_interval": None,
                "max_interval": None,
                "mean_interval": None,
                "median_interval": None,
                "non_positive_intervals": 0
            }
        min_interval = min(intervals)
        max_interval = max(intervals)
        mean_interval = statistics.mean(intervals)
        median_interval = statistics.median(intervals)
        non_positive_intervals = sum(1 for iv in intervals if iv <= 0)
        # Odometry-specific extras
        over_1_5x = sum(1 for iv in intervals if iv > 1.5 * median_interval) if median_interval is not None else 0
        over_2x = sum(1 for iv in intervals if iv > 2.0 * median_interval) if median_interval is not None else 0
        max_gap = max(intervals)
        max_gap_index = intervals.index(max_gap)  # first occurrence
        multiple_of_median = (max_gap / median_interval) if median_interval != 0 else None
        # Context: previous and next records around the max gap
        prev_idx = max_gap_index
        next_idx = max_gap_index + 1
        prev_ts, prev_frame = records[prev_idx]
        next_ts, next_frame = records[next_idx]
        gap_context = {
            "previous_frame_id": prev_frame,
            "previous_timestamp": prev_ts,
            "next_frame_id": next_frame,
            "next_timestamp": next_ts,
            "interval_seconds": max_gap
        }
        # Interval frequencies in milliseconds, rounded to 3 decimal places, top 10
        if intervals:
            ms_intervals = [round(iv * 1000, 3) for iv in intervals]
            freq_counter = Counter(ms_intervals)
            # Get top 10 most common
            top10 = freq_counter.most_common(10)
            freq_list = [{"interval_ms": val, "count": cnt} for val, cnt in top10]
        else:
            freq_list = None
        odometry_result = {
            "valid_timestamp_count": len(records),
            "min_interval": min_interval,
            "max_interval": max_interval,
            "mean_interval": mean_interval,
            "median_interval": median_interval,
            "non_positive_intervals": non_positive_intervals,
            "intervals_over_1_5x_median": over_1_5x,
            "intervals_over_2x_median": over_2x,
            "largest_gap_index": max_gap_index,
            "largest_gap_seconds": max_gap,
            "largest_gap_multiple_of_median": multiple_of_median,
            "largest_gap_context": gap_context,
            "interval_frequencies_ms": freq_list
        }
        return odometry_result if is_odometry else {
            "valid_timestamp_count": len(records),
            "min_interval": min_interval,
            "max_interval": max_interval,
            "mean_interval": mean_interval,
            "median_interval": median_interval,
            "non_positive_intervals": non_positive_intervals
        }

    # Process IMU (only basic stats)
    imu_path = dataset_path / "imu.csv"
    imu_records, imu_error = parse_timestamps_from_csv(imu_path)
    if imu_error is not None:
        result["imu"] = {"error": imu_error}
    else:
        result["imu"] = compute_interval_stats(imu_records, is_odometry=False)

    # Process Odometry (with extras)
    odom_path = dataset_path / "odometry.csv"
    odo_records, odo_error = parse_timestamps_from_csv(odom_path)
    if odo_error is not None:
        result["odometry"] = {"error": odo_error}
    else:
        result["odometry"] = compute_interval_stats(odo_records, is_odometry=True)

    return result


def analyze_odometry_trajectory(dataset_dir: Path) -> dict:
    """
    Analyze the 3D position trajectory from odometry.csv.

    Args:
        dataset_dir: Path to the dataset directory (e.g., single_room)

    Returns:
        Dictionary with trajectory analysis:
        - valid_position_count: number of rows with numeric x,y,z
        - invalid_position_count: number of rows with missing or non-numeric x/y/z
        - min_x, max_x, min_y, max_y, min_z, max_z: extrema of valid positions
        - first_position: dict with timestamp, frame, x, y, z of the first valid row
        - last_position: dict with timestamp, frame, x, y, z of the last valid row
        - total_path_length: sum of Euclidean distances between consecutive valid positions
        - max_step_distance: maximum distance between any two consecutive valid positions
        - max_step_frame_ids: dict with "previous" and "current" frame IDs for the max step
        - max_step_timestamps: dict with "previous" and "current" timestamps for the max step
        If the file is missing or unreadable, returns an error string.
    """
    dataset_path = Path(dataset_dir)
    result = {}

    def parse_odometry_rows(csv_path):
        """Yield (timestamp, frame, x, y, z) as floats or None for each data row."""
        if not csv_path.is_file():
            return None, "File not found"
        try:
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    return None, "Empty file"
                # Find column indices by name (case-insensitive)
                def find_col(name):
                    for i, col in enumerate(header):
                        if col.strip().lower() == name:
                            return i
                    return None  # not found
                timestamp_col = find_col("timestamp")
                frame_col = find_col("frame")
                x_col = find_col("x")
                y_col = find_col("y")
                z_col = find_col("z")
                # If any of x,y,z columns missing, we cannot parse positions
                if x_col is None or y_col is None or z_col is None:
                    return None, "Missing x, y, or z column header"
                # timestamp and frame columns are expected but we can still proceed if missing (use None)
                rows = []
                for row in reader:
                    # Ensure row has enough columns
                    if len(row) <= max([i for i in [timestamp_col, frame_col, x_col, y_col, z_col] if i is not None], default=-1):
                        # Not enough columns, skip or treat as missing?
                        continue
                    ts_val = row[timestamp_col] if timestamp_col is not None and len(row) > timestamp_col else None
                    frm_val = row[frame_col] if frame_col is not None and len(row) > frame_col else None
                    x_val = row[x_col] if len(row) > x_col else None
                    y_val = row[y_col] if len(row) > y_col else None
                    z_val = row[z_col] if len(row) > z_col else None
                    # Try to convert to float where applicable
                    try:
                        ts = float(ts_val) if ts_val is not None else None
                    except (ValueError, TypeError):
                        ts = None
                    try:
                        x = float(x_val) if x_val is not None else None
                    except (ValueError, TypeError):
                        x = None
                    try:
                        y = float(y_val) if y_val is not None else None
                    except (ValueError, TypeError):
                        y = None
                    try:
                        z = float(z_val) if z_val is not None else None
                    except (ValueError, TypeError):
                        z = None
                    # frame we keep as string (as in CSV) or None
                    frame = frm_val if frm_val is not None else None
                    rows.append((ts, frame, x, y, z))
                return rows, None
        except Exception as e:
            return None, f"Error reading file: {e}"

    rows, error = parse_odometry_rows(dataset_path / "odometry.csv")
    if error is not None:
        result["error"] = error
        return result

    # Initialize tracking variables
    valid_count = 0
    invalid_count = 0
    min_x = max_x = min_y = max_y = min_z = max_z = None
    first_pos = None  # dict
    last_pos = None   # dict
    total_path = 0.0
    max_step = 0.0
    max_step_prev_frame = None
    max_step_prev_ts = None
    max_step_curr_frame = None
    max_step_curr_ts = None

    prev_x = prev_y = prev_z = None
    prev_ts = None
    prev_frame = None

    for ts, frame, x, y, z in rows:
        # Determine if this row has a valid position
        if x is None or y is None or z is None:
            invalid_count += 1
            continue
        # At this point x,y,z are floats (not None)
        valid_count += 1
        # Update min/max
        if min_x is None or x < min_x:
            min_x = x
        if max_x is None or x > max_x:
            max_x = x
        if min_y is None or y < min_y:
            min_y = y
        if max_y is None or y > max_y:
            max_y = y
        if min_z is None or z < min_z:
            min_z = z
        if max_z is None or z > max_z:
            max_z = z
        # Record first and last valid position
        if first_pos is None:
            first_pos = {
                "timestamp": ts,
                "frame": frame,
                "x": x,
                "y": y,
                "z": z
            }
        last_pos = {
            "timestamp": ts,
            "frame": frame,
            "x": x,
            "y": y,
            "z": z
        }
        # If we have a previous valid position, compute step
        if prev_x is not None and prev_y is not None and prev_z is not None:
            dx = x - prev_x
            dy = y - prev_y
            dz = z - prev_z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            total_path += dist
            if dist > max_step:
                max_step = dist
                max_step_prev_frame = prev_frame
                max_step_prev_ts = prev_ts
                max_step_curr_frame = frame
                max_step_curr_ts = ts
        # Set current as previous for next iteration
        prev_x, prev_y, prev_z = x, y, z
        prev_ts = ts
        prev_frame = frame

    # Prepare result
    result = {
        "valid_position_count": valid_count,
        "invalid_position_count": invalid_count,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
        "first_position": first_pos,
        "last_position": last_pos,
        "total_path_length": total_path if valid_count >= 2 else 0.0,
        "max_step_distance": max_step if valid_count >= 2 else None,
        "max_step_frame_ids": {
            "previous": max_step_prev_frame,
            "current": max_step_curr_frame
        } if valid_count >= 2 else None,
        "max_step_timestamps": {
            "previous": max_step_prev_ts,
            "current": max_step_curr_ts
        } if valid_count >= 2 else None
    }
    # If no valid positions, set extrema to None and first/last to None
    if valid_count == 0:
        result["min_x"] = result["max_x"] = result["min_y"] = result["max_y"] = result["min_z"] = result["max_z"] = None
        result["first_position"] = None
        result["last_position"] = None
        result["total_path_length"] = 0.0
        result["max_step_distance"] = None
        result["max_step_frame_ids"] = None
        result["max_step_timestamps"] = None

    return result


def plot_odometry_trajectory(dataset_dir: Path, output_path: Path | str = None) -> bool:
    """
    Plot the odometry trajectory (XY, XZ, YZ) and save to a file.

    Args:
        dataset_dir: Path to the dataset directory.
        output_path: Optional output file path. If None, defaults to
                     <dataset_dir>/trajectory.png.

    Returns:
        True if the plot was generated and saved successfully, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Matplotlib not available
        return False

    # Reuse the parsing logic from analyze_odometry_trajectory to get the list of valid positions
    dataset_path = Path(dataset_dir)
    # We'll duplicate the parsing logic here to avoid changing existing functions.
    def _parse_odometry_rows(csv_path):
        if not csv_path.is_file():
            return None, "File not found"
        try:
            import csv
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    return None, "Empty file"
                def find_col(name):
                    for i, col in enumerate(header):
                        if col.strip().lower() == name:
                            return i
                    return None
                timestamp_col = find_col("timestamp")
                frame_col = find_col("frame")
                x_col = find_col("x")
                y_col = find_col("y")
                z_col = find_col("z")
                # If any of x,y,z columns missing, we cannot parse positions
                if x_col is None or y_col is None or z_col is None:
                    return None, "Missing x, y, or z column header"
                rows = []
                for row in reader:
                    if len(row) <= max([i for i in [timestamp_col, frame_col, x_col, y_col, z_col] if i is not None], default=-1):
                        continue
                    ts_val = row[timestamp_col] if timestamp_col is not None and len(row) > timestamp_col else None
                    frm_val = row[frame_col] if frame_col is not None and len(row) > frame_col else None
                    x_val = row[x_col] if len(row) > x_col else None
                    y_val = row[y_col] if len(row) > y_col else None
                    z_val = row[z_col] if len(row) > z_col else None
                    # Try to convert to float where applicable
                    try:
                        ts = float(ts_val) if ts_val is not None else None
                    except (ValueError, TypeError):
                        ts = None
                    try:
                        x = float(x_val) if x_val is not None else None
                    except (ValueError, TypeError):
                        x = None
                    try:
                        y = float(y_val) if y_val is not None else None
                    except (ValueError, TypeError):
                        y = None
                    try:
                        z = float(z_val) if z_val is not None else None
                    except (ValueError, TypeError):
                        z = None
                    # frame we keep as string (as in CSV) or None
                    frame = frm_val if frm_val is not None else None
                    rows.append((ts, frame, x, y, z))
                return rows, None
        except Exception as e:
            return None, f"Error reading file: {e}"

    rows, error = _parse_odometry_rows(dataset_path / "odometry.csv")
    if error is not None:
        return False

    # Extract valid positions (where x, y, z are not None)
    xs, ys, zs = [], [], []
    first_valid = None
    last_valid = None
    for ts, frame, x, y, z in rows:
        if x is None or y is None or z is None:
            continue
        xs.append(x)
        ys.append(y)
        zs.append(z)
        if first_valid is None:
            first_valid = (x, y, z)
        last_valid = (x, y, z)

    if not xs:
        return False  # no valid points to plot

    # Create plots with improved layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # XY plot
    ax = axes[0]
    ax.plot(xs, ys, marker='.', linestyle='-', markersize=2, linewidth=0.5)
    if first_valid:
        ax.plot([first_valid[0]], [first_valid[1]], 'go', markersize=8, label='Start')
    if last_valid:
        ax.plot([last_valid[0]], [last_valid[1]], 'ro', markersize=8, label='End')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('XY Trajectory')
    ax.legend(loc='upper left', fontsize='small')
    ax.set_aspect('equal', adjustable='box')

    # XZ plot
    ax = axes[1]
    ax.plot(xs, zs, marker='.', linestyle='-', markersize=2, linewidth=0.5)
    if first_valid:
        ax.plot([first_valid[0]], [first_valid[2]], 'go', markersize=8, label='Start')
    if last_valid:
        ax.plot([last_valid[0]], [last_valid[2]], 'ro', markersize=8, label='End')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title('XZ Trajectory')
    ax.legend(loc='upper left', fontsize='small')
    ax.set_aspect('equal', adjustable='box')

    # YZ plot (fixed labels: horizontal=y, vertical=z)
    ax = axes[2]
    ax.plot(ys, zs, marker='.', linestyle='-', markersize=2, linewidth=0.5)
    if first_valid:
        ax.plot([first_valid[1]], [first_valid[2]], 'go', markersize=8, label='Start')
    if last_valid:
        ax.plot([last_valid[1]], [last_valid[2]], 'ro', markersize=8, label='End')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_title('YZ Trajectory')
    ax.legend(loc='upper left', fontsize='small')
    ax.set_aspect('equal', adjustable='box')

    plt.suptitle('Odometry Trajectory (raw values, units unverified)', fontsize=14)

    # Determine output path
    if output_path is None:
        output_path = os.path.join(dataset_path, "trajectory.png")
    else:
        output_path = Path(output_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception:
        plt.close(fig)
        return False


def visualize_depth_confidence(dataset_dir: Path, frame_filename: str = "000000.png", output_path: Path | str = None) -> bool:
    """
    Visualize a depth PNG and its matching confidence PNG side by side.

    Args:
        dataset_dir: Path to the dataset directory.
        frame_filename: Name of the PNG file to visualize (default: "000000.png").
        output_path: Optional output file path. If None, defaults to
                     <dataset_dir>/depth_vis.png.

    Returns:
        True if the visualization was generated and saved successfully, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Matplotlib not available
        return False

    dataset_path = Path(dataset_dir)
    depth_path = dataset_path / "depth" / frame_filename
    confidence_path = dataset_path / "confidence" / frame_filename

    # Check if files exist
    if not depth_path.is_file():
        print(f"Error: Depth file not found: {depth_path}")
        return False
    if not confidence_path.is_file():
        print(f"Error: Confidence file not found: {confidence_path}")
        return False

    # Read images with OpenCV (unchanged)
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    conf_img = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)

    if depth_img is None:
        print(f"Error: Could not read depth image: {depth_path}")
        return False
    if conf_img is None:
        print(f"Error: Could not read confidence image: {confidence_path}")
        return False

    # Get image properties
    depth_shape = depth_img.shape
    conf_shape = conf_img.shape
    depth_dtype = depth_img.dtype
    conf_dtype = conf_img.dtype
    depth_min = depth_img.min()
    depth_max = depth_img.max()
    conf_min = conf_img.min()
    conf_max = conf_img.max()
    # Unique values: handle numpy availability
    try:
        import numpy as np
        has_numpy = True
    except ImportError:
        has_numpy = False

    if has_numpy:
        depth_unique_count = len(np.unique(depth_img)) if depth_img.size < 1000000 else "too many to list"
        conf_unique_count = len(np.unique(conf_img)) if conf_img.size < 1000000 else "too many to list"
    else:
        # Without numpy, we can't efficiently compute unique counts for large images.
        # We'll provide a placeholder.
        depth_unique_count = "numpy not available"
        conf_unique_count = "numpy not available"

    # Print properties to terminal
    print(f"Depth image ({frame_filename}):")
    print(f"  Shape: {depth_shape}")
    print(f"  Dtype: {depth_dtype}")
    print(f"  Min: {depth_min}")
    print(f"  Max: {depth_max}")
    print(f"  Unique values count: {depth_unique_count}")
    print(f"Confidence image ({frame_filename}):")
    print(f"  Shape: {conf_shape}")
    print(f"  Dtype: {conf_dtype}")
    print(f"  Min: {conf_min}")
    print(f"  Max: {conf_max}")
    print(f"  Unique values count: {conf_unique_count}")

    # Create side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Depth image
    im_depth = axes[0].imshow(depth_img, cmap='viridis')
    axes[0].set_title(f'Depth: {frame_filename}')
    plt.colorbar(im_depth, ax=axes[0], label='Raw depth value')

    # Confidence image
    im_conf = axes[1].imshow(conf_img, cmap='plasma')
    axes[1].set_title(f'Confidence: {frame_filename}')
    plt.colorbar(im_conf, ax=axes[1], label='Raw confidence value')

    # Adjust layout
    plt.tight_layout()

    # Determine output path
    if output_path is None:
        output_path = os.path.join(dataset_path, "depth_vis.png")
    else:
        output_path = Path(output_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error saving visualization: {e}")
        plt.close(fig)
        return False


if __name__ == "__main__":
    # Example usage when run directly
    import sys

    if len(sys.argv) == 2:
        # Existing behavior: check dataset inputs
        dataset_path = Path(sys.argv[1])
        results = check_dataset_inputs(dataset_path)

        print(f"Checking inputs in: {dataset_path}")
        for input_name, exists in results.items():
            if input_name.endswith("_png_count"):
                print(f"  {input_name}: {exists}")
            else:
                status = "✓ FOUND" if exists else "✗ MISSING"
                print(f"  {input_name}: {status}")

    elif len(sys.argv) == 3 and sys.argv[2] == "--inspect":
        # New behavior: inspect dataset metadata
        dataset_path = Path(sys.argv[1])
        results = inspect_dataset_metadata(dataset_path)

        print(f"Inspecting metadata in: {dataset_path}")
        import json
        print(json.dumps(results, indent=2))

    elif len(sys.argv) == 3 and sys.argv[2] == "--audit":
        # New behavior: audit dataset timing
        dataset_path = Path(sys.argv[1])
        results = audit_dataset_timing(dataset_path)

        print(f"Auditing timing in: {dataset_path}")
        import json
        print(json.dumps(results, indent=2))

    elif len(sys.argv) == 3 and sys.argv[2] == "--intervals":
        # New behavior: analyze timestamp intervals
        dataset_path = Path(sys.argv[1])
        results = analyze_timestamp_intervals(dataset_path)

        print(f"Analyzing timestamp intervals in: {dataset_path}")
        import json
        print(json.dumps(results, indent=2))

    elif len(sys.argv) == 3 and sys.argv[2] == "--trajectory":
        # New behavior: analyze odometry trajectory
        dataset_path = Path(sys.argv[1])
        results = analyze_odometry_trajectory(dataset_path)

        print(f"Analyzing trajectory in: {dataset_path}")
        import json
        print(json.dumps(results, indent=2))

    elif len(sys.argv) == 3 and sys.argv[2] == "--plot-trajectory":
        # New behavior: plot odometry trajectory (default output)
        dataset_path = Path(sys.argv[1])
        success = plot_odometry_trajectory(dataset_path)
        if success:
            default_out = os.path.join(dataset_path, "trajectory.png")
            print(f"Plot saved to: {default_out}")
        else:
            print("Failed to generate plot (check if matplotlib is installed).")

    elif len(sys.argv) == 4 and sys.argv[2] == "--plot-trajectory":
        # New behavior: plot odometry trajectory with custom output path
        dataset_path = Path(sys.argv[1])
        output_path = sys.argv[3]
        success = plot_odometry_trajectory(dataset_path, output_path)
        if success:
            print(f"Plot saved to: {output_path}")
        else:
            print("Failed to generate plot (check if matplotlib is installed and output path is writable).")

    elif len(sys.argv) == 3 and sys.argv[2] == "--visualize-depth":
        # New behavior: visualize depth and confidence (default frame 000000.png)
        dataset_path = Path(sys.argv[1])
        success = visualize_depth_confidence(dataset_path)
        if success:
            default_out = os.path.join(dataset_path, "depth_vis.png")
            print(f"Visualization saved to: {default_out}")
        else:
            print("Failed to generate visualization.")

    elif len(sys.argv) == 4 and sys.argv[2] == "--visualize-depth":
        # New behavior: visualize depth and confidence with custom frame filename
        dataset_path = Path(sys.argv[1])
        frame_filename = sys.argv[3]
        success = visualize_depth_confidence(dataset_path, frame_filename)
        if success:
            default_out = os.path.join(dataset_path, "depth_vis.png")
            print(f"Visualization saved to: {default_out}")
        else:
            print("Failed to generate visualization.")

    elif len(sys.argv) == 5 and sys.argv[2] == "--visualize-depth":
        # New behavior: visualize depth and confidence with custom frame filename and output path
        dataset_path = Path(sys.argv[1])
        frame_filename = sys.argv[3]
        output_path = sys.argv[4]
        success = visualize_depth_confidence(dataset_path, frame_filename, output_path)
        if success:
            print(f"Visualization saved to: {output_path}")
        else:
            print("Failed to generate visualization.")

    else:
        print("Usage:")
        print("  python data_loader.py <dataset_directory>                 # Check expected inputs")
        print("  python data_loader.py <dataset_directory> --inspect       # Inspect metadata")
        print("  python data_loader.py <dataset_directory> --audit         # Audit timing")
        print("  python data_loader.py <dataset_directory> --intervals     # Analyze timestamp intervals")
        print("  python data_loader.py <dataset_directory> --trajectory    # Analyze odometry trajectory")
        print("  python data_loader.py <dataset_directory> --plot-trajectory [output_path]  # Plot trajectory (default: <dataset_dir>/trajectory.png)")
        print("  python data_loader.py <dataset_directory> --visualize-depth [frame_filename] [output_path]  # Visualize depth and confidence (default frame: 000000.png, default output: <dataset_dir>/depth_vis.png)")