# Unified Output Contract

This document describes the standardized JSON output format for the room-scanning prototype.

The contract aims to provide a consistent structure for exchanging results between components, while clearly indicating which fields are measured, estimated, unavailable, or not implemented.

## Structure

The unified output consists of:

- `capture_metadata`: Information about the input data capture (dataset, frame, dimensions, etc.)
- `rooms`: Array of room objects, each containing raw measurements, placeholder fields for future extensions, and processing status
- `processing_status`: Overall processing status string
- `known_limitations`: Array of known limitations applicable to the entire output

See the JSON Schema at `schemas/room_output.schema.json` for formal specification.

## Fields

### capture_metadata
- `dataset_identifier`: String identifier of the dataset (e.g., "single_scan_floor_only/1a8384c3f6")
- `selected_frame`: Integer frame index selected for processing
- `input_dimensions`: Array [height, width] in pixels of the input depth image
- `input_tier`: Optional string indicating input modality ("photo", "video", "lidar") or null if unknown
- `depth_scale_verified`: Boolean indicating whether depth-to-physical units scaling has been verified
- `camera_calibration_verified`: Boolean indicating whether camera intrinsics have been verified
- `geometric_measurements_units`: String describing units of geometric measurements (e.g., "raw depth units (uint16)")

### rooms (array)
Each room object includes:
- `room_identifier`: String identifier for the room (e.g., "room0")
- `raw_measurements`: Object containing directly computed geometric statistics:
  - `point_cloud_statistics`: Point cloud bounds, extent, and depth range
  - `planes`: Array of detected planes with equations, normals, inlier statistics, and residuals
  - `bounding_box`: Axis-aligned bounding box area in XY plane and Z extent
  - `angular_diff_deg`: Angular difference between primary plane normals (if applicable)
  - `unassigned_points`: Count and percentage of points not assigned to any plane
- `metric_measurements`: Placeholder for real-world metric measurements (currently "not_implemented")
- `walls`, `floor`, `ceiling`, `openings`: Placeholders for architectural element detection (currently "not_implemented")
- `surface_damage_regions`: Placeholder for surface damage detection (currently "not_implemented")
- `concealed_damage_flags`: Placeholder for concealed damage detection (currently "not_implemented")
- `scope_line_items`: Placeholder for renovation scope items (currently "not_implemented")
- `adjacency_to_other_rooms`: Array of room identifiers this room is adjacent to (empty if unknown)
- `processing_status`: String indicating processing status of this room (e.g., "experimental_unvalidated")
- `limitations`: Array of limitation strings specific to this room

### processing_status
Top-level processing status string (e.g., "experimental_unvalidated").

### known_limitations
Array of known limitation strings applicable to the entire output.

## Design Principles

- **Honest representation**: Fields that are not implemented are explicitly marked as "not_implemented" or null.
- **Measured vs. estimated**: Raw measurements reflect direct outputs from the algorithm; metric conversions are not applied due to unknown scale.
- **Extensibility**: The structure is intended to be extended as additional capabilities are added.
- **Validation**: The JSON Schema provides formal validation of structure and types.

## Usage

Validators can use the JSON Schema to check conformance. Example output is available at `output/demo_*/room_output.json`.