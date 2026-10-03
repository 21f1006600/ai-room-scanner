#!/usr/bin/env python3
"""
Image-based Damage Candidate Detector

Lightweight OpenCV-based detector for damage candidates in RGB frames.
Supports two candidate categories:
- WATER_DAMAGE: Discoloration/staining patches (brown/yellow regions)
- CRACK: Thin linear features (edges, lines)

All detections are CANDIDATES only - not validated damage classifications.
No semantic damage classification or accuracy claims without labeled validation data.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np


class CandidateClass(str):
    """Damage candidate classes - synthetic/manual only."""
    WATER_DAMAGE = "water_damage_candidate"
    CRACK = "crack_candidate"


@dataclass
class ImageDamageCandidate:
    """A single damage candidate detected from image features."""
    candidate_id: str
    candidate_class: str  # CandidateClass value
    # Bounding box in image coordinates [x, y, width, height]
    bbox_image: list[int]
    # Confidence score [0, 1] based on image features only
    confidence: float
    # Source frame identifier
    source_frame: str
    # Source dataset identifier
    source_dataset: str
    # Detection method details
    detection_method: str
    # Feature metrics used for detection
    feature_metrics: dict[str, float] = field(default_factory=dict)
    # Notes
    notes: str = ""
    # Annotation metadata
    annotated_by: str = "image_detector"
    annotated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageDamageResult:
    """Complete image-based damage detection result."""
    schema_version: str = "1.0"
    dataset_id: str = ""
    frame_id: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    # All candidates are IMAGE-DERIVED - not validated damage
    candidates: list[ImageDamageCandidate] = field(default_factory=list)
    # Limitations and provenance
    limitations: list[str] = field(default_factory=lambda: [
        "All candidates are IMAGE-DERIVED from visual features only - NOT validated damage detections.",
        "No labeled training data or ground truth validation was used.",
        "Water damage candidates based on color/texture heuristics; false positives likely from shadows, objects, material variation.",
        "Crack candidates based on edge/line heuristics; false positives likely from texture, grout lines, corners.",
        "No semantic damage classification or accuracy claims without labeled validation data.",
        "Physical ground truth measurements are NOT_MEASURED.",
        "Candidates require manual review and validation before any scope decisions.",
    ])
    # Status flags
    status: dict[str, str] = field(default_factory=lambda: {
        "damage_detection": "IMAGE_CANDIDATES_ONLY",
        "physical_measurements": "NOT_MEASURED",
        "ground_truth_validation": "NOT_MEASURED",
        "labeled_training_data": "NOT_USED",
        "accuracy_gates": "BLOCKED",
    })
    # Image metadata
    image_shape: list[int] = field(default_factory=list)
    depth_shape: Optional[list[int]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "frame_id": self.frame_id,
            "generated_at": self.generated_at,
            "candidates": [c.to_dict() for c in self.candidates],
            "limitations": self.limitations,
            "status": self.status,
            "image_shape": self.image_shape,
            "depth_shape": self.depth_shape,
        }

    def validate(self) -> list[str]:
        """Validate the result and return list of errors."""
        errors = []
        candidate_ids = set()
        for c in self.candidates:
            if c.candidate_id in candidate_ids:
                errors.append(f"Duplicate candidate_id: {c.candidate_id}")
            candidate_ids.add(c.candidate_id)
            if not 0 <= c.confidence <= 1:
                errors.append(f"Candidate {c.candidate_id}: confidence must be in [0, 1]")
            if len(c.bbox_image) != 4:
                errors.append(f"Candidate {c.candidate_id}: bbox_image must have 4 values [x, y, w, h]")
            x, y, w, h = c.bbox_image
            if w <= 0 or h <= 0:
                errors.append(f"Candidate {c.candidate_id}: bbox width and height must be positive")
        return errors


# Water damage detection parameters
WATER_DAMAGE_CONFIG = {
    # HSV color ranges for water stains (brown/yellow/dark regions)
    "hsv_ranges": [
        # Brown stains
        ((10, 30, 30), (25, 255, 200)),
        # Yellow/brown stains
        ((20, 30, 50), (35, 255, 255)),
        # Dark stains (low value)
        ((0, 0, 0), (180, 255, 80)),
    ],
    # Minimum contour area (pixels) - adjust based on image resolution
    "min_area": 500,
    # Maximum contour area (pixels) - filter out very large regions
    "max_area": 50000,
    # Morphological kernel size
    "morph_kernel": 5,
    # Minimum solidity (contour area / convex hull area) - filter irregular shapes
    "min_solidity": 0.3,
    # Confidence scaling factors
    "color_weight": 0.6,
    "texture_weight": 0.4,
}

# Crack detection parameters
CRACK_CONFIG = {
    # Canny edge detection thresholds
    "canny_low": 30,
    "canny_high": 100,
    # Morphological kernel for line enhancement
    "morph_kernel_length": 15,
    "morph_kernel_width": 3,
    # Hough line parameters
    "hough_rho": 1,
    "hough_theta": np.pi / 180,
    "hough_threshold": 30,
    "hough_min_line_length": 30,
    "hough_max_line_gap": 10,
    # Minimum line length for candidate (pixels)
    "min_line_length": 40,
    # Maximum line gap to merge
    "max_line_gap": 15,
    # Confidence scaling
    "length_weight": 0.5,
    "straightness_weight": 0.3,
    "contrast_weight": 0.2,
}


def detect_water_damage_candidates(
    image: np.ndarray,
    depth: Optional[np.ndarray] = None,
    confidence_map: Optional[np.ndarray] = None,
    config: Optional[dict] = None,
) -> list[ImageDamageCandidate]:
    """
    Detect water damage candidates using color and texture features.

    Args:
        image: RGB image (H, W, 3) uint8
        depth: Optional depth map (H, W) for context
        confidence_map: Optional confidence map (H, W) for context
        config: Detection configuration

    Returns:
        List of ImageDamageCandidate objects
    """
    cfg = {**WATER_DAMAGE_CONFIG, **(config or {})}
    h, w = image.shape[:2]
    candidates = []

    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create combined mask for water stain colors
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for lower, upper in cfg["hsv_ranges"]:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_kernel"], cfg["morph_kernel"]))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue

        # Filter by solidity (compactness)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < cfg["min_solidity"]:
            continue

        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(contour)

        # Compute feature metrics
        # Color metric: mean saturation in region (stains often have distinct saturation)
        roi_hsv = hsv[y:y+bh, x:x+bw]
        mean_saturation = float(np.mean(roi_hsv[:, :, 1])) / 255.0
        mean_value = float(np.mean(roi_hsv[:, :, 2])) / 255.0

        # Texture metric: local standard deviation in grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y:y+bh, x:x+bw]
        texture_std = float(np.std(roi_gray)) / 255.0

        # Depth context (if available): check if region corresponds to valid depth
        depth_support = 1.0
        if depth is not None:
            # Rescale depth to image coordinates if needed
            dh, dw = depth.shape
            if dh != h or dw != w:
                # Scale coordinates
                scale_y, scale_x = dh / h, dw / w
                dy, dx = int(y * scale_y), int(x * scale_x)
                dbh, dbw = int(bh * scale_y), int(bw * scale_x)
                depth_roi = depth[dy:dy+dbh, dx:dx+dbw]
            else:
                depth_roi = depth[y:y+bh, x:x+bw]
            valid_depth_ratio = float(np.count_nonzero(depth_roi > 0)) / max(1, depth_roi.size)
            depth_support = valid_depth_ratio

        # Confidence score (heuristic combination)
        color_score = min(1.0, mean_saturation * 2.0)  # Higher saturation = more stain-like
        texture_score = min(1.0, texture_std * 3.0)   # Moderate texture variation
        confidence = (cfg["color_weight"] * color_score +
                     cfg["texture_weight"] * texture_score) * depth_support
        confidence = float(np.clip(confidence, 0.1, 0.9))  # Cap confidence

        # Only keep candidates with reasonable confidence
        if confidence < 0.25:
            continue

        candidate = ImageDamageCandidate(
            candidate_id=f"img_water_{uuid.uuid4().hex[:8]}",
            candidate_class=CandidateClass.WATER_DAMAGE,
            bbox_image=[int(x), int(y), int(bw), int(bh)],
            confidence=confidence,
            source_frame="000000",  # Will be overridden by caller
            source_dataset="",      # Will be overridden by caller
            detection_method="color_texture_hsv_contour",
            feature_metrics={
                "contour_area": float(area),
                "solidity": solidity,
                "mean_saturation": mean_saturation,
                "mean_value": mean_value,
                "texture_std": texture_std,
                "depth_support": depth_support,
                "color_score": color_score,
                "texture_score": texture_score,
            },
            notes="IMAGE-DERIVED CANDIDATE - not validated damage detection. Based on HSV color ranges for brown/yellow stains and texture analysis.",
        )
        candidates.append(candidate)

    return candidates


def detect_crack_candidates(
    image: np.ndarray,
    depth: Optional[np.ndarray] = None,
    confidence_map: Optional[np.ndarray] = None,
    config: Optional[dict] = None,
) -> list[ImageDamageCandidate]:
    """
    Detect crack candidates using edge and line features.

    Args:
        image: RGB image (H, W, 3) uint8
        depth: Optional depth map (H, W) for context
        confidence_map: Optional confidence map (H, W) for context
        config: Detection configuration

    Returns:
        List of ImageDamageCandidate objects
    """
    cfg = {**CRACK_CONFIG, **(config or {})}
    h, w = image.shape[:2]
    candidates = []

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, cfg["canny_low"], cfg["canny_high"])

    # Morphological operation to enhance thin lines
    # Use a horizontal+vertical kernel to catch lines in multiple orientations
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg["morph_kernel_length"], 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cfg["morph_kernel_length"]))
    edges_h = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_h)
    edges_v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_v)
    edges_enhanced = cv2.bitwise_or(edges_h, edges_v)

    # Also apply a small dilation to connect nearby edge segments
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_enhanced = cv2.dilate(edges_enhanced, kernel_small, iterations=1)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges_enhanced,
        cfg["hough_rho"],
        cfg["hough_theta"],
        cfg["hough_threshold"],
        minLineLength=cfg["hough_min_line_length"],
        maxLineGap=cfg["hough_max_line_gap"],
    )

    if lines is None:
        return candidates

    # Handle different OpenCV return formats: (N, 1, 4) or (N, 4)
    if lines.ndim == 3:
        lines = lines.reshape(-1, 4)

    # Group nearby/parallel lines into crack candidates
    line_segments = []
    for line in lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length >= cfg["min_line_length"]:
            angle = np.arctan2(y2 - y1, x2 - x1)
            line_segments.append({
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "length": float(length),
                "angle": float(angle),
            })

    if not line_segments:
        return candidates

    # Cluster lines by proximity and angle similarity
    # Simple greedy clustering
    used = [False] * len(line_segments)
    clusters = []

    for i, seg in enumerate(line_segments):
        if used[i]:
            continue
        cluster = [seg]
        used[i] = True

        for j, other in enumerate(line_segments):
            if used[j]:
                continue
            # Check if lines are close and have similar angle
            dist = min(
                np.sqrt((seg["x1"] - other["x1"])**2 + (seg["y1"] - other["y1"])**2),
                np.sqrt((seg["x1"] - other["x2"])**2 + (seg["y1"] - other["y2"])**2),
                np.sqrt((seg["x2"] - other["x1"])**2 + (seg["y2"] - other["y1"])**2),
                np.sqrt((seg["x2"] - other["x2"])**2 + (seg["y2"] - other["y2"])**2),
            )
            angle_diff = abs(seg["angle"] - other["angle"])
            angle_diff = min(angle_diff, np.pi - angle_diff)  # Handle wrap-around

            if dist < cfg["max_line_gap"] and angle_diff < np.pi / 6:  # 30 degrees
                cluster.append(other)
                used[j] = True

        clusters.append(cluster)

    # Create candidate from each cluster
    for cluster in clusters:
        # Compute bounding box encompassing all lines in cluster
        all_x = [s["x1"] for s in cluster] + [s["x2"] for s in cluster]
        all_y = [s["y1"] for s in cluster] + [s["y2"] for s in cluster]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        bw = x_max - x_min
        bh = y_max - y_min

        # Expand bbox slightly for visualization
        pad = 10
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        bw = min(w - x_min, bw + 2 * pad)
        bh = min(h - y_min, bh + 2 * pad)

        # Compute feature metrics
        total_length = sum(s["length"] for s in cluster)
        avg_length = total_length / len(cluster)

        # Straightness: how aligned are the lines?
        angles = [s["angle"] for s in cluster]
        # Circular mean for angles
        sin_sum = sum(np.sin(2 * a) for a in angles)
        cos_sum = sum(np.cos(2 * a) for a in angles)
        mean_angle = 0.5 * np.arctan2(sin_sum, cos_sum)
        straightness = np.sqrt(sin_sum**2 + cos_sum**2) / len(cluster)

        # Contrast along lines
        contrast_scores = []
        for s in cluster:
            # Sample perpendicular to line
            cx = (s["x1"] + s["x2"]) / 2
            cy = (s["y1"] + s["y2"]) / 2
            angle = s["angle"]
            perp_angle = angle + np.pi / 2
            # Sample a few points perpendicular to line
            sample_len = min(20, max(bw, bh) // 2)
            for offset in np.linspace(-sample_len, sample_len, 5):
                px = int(cx + offset * np.cos(perp_angle))
                py = int(cy + offset * np.sin(perp_angle))
                if 0 <= px < w and 0 <= py < h:
                    # Compute local gradient magnitude
                    if px > 0 and px < w - 1 and py > 0 and py < h - 1:
                        # Convert to float to avoid uint8 overflow
                        gx = float(gray[py, px + 1]) - float(gray[py, px - 1])
                        gy = float(gray[py + 1, px]) - float(gray[py - 1, px])
                        grad_mag = np.sqrt(gx**2 + gy**2) / 255.0
                        contrast_scores.append(grad_mag)

        mean_contrast = float(np.mean(contrast_scores)) if contrast_scores else 0.0

        # Depth support
        depth_support = 1.0
        if depth is not None:
            dh, dw = depth.shape
            if dh != h or dw != w:
                scale_y, scale_x = dh / h, dw / w
                dy, dx = int(y_min * scale_y), int(x_min * scale_x)
                dbh, dbw = int(bh * scale_y), int(bw * scale_x)
                depth_roi = depth[dy:dy+dbh, dx:dx+dbw]
            else:
                depth_roi = depth[y_min:y_min+bh, x_min:x_min+bw]
            valid_depth_ratio = float(np.count_nonzero(depth_roi > 0)) / max(1, depth_roi.size)
            depth_support = valid_depth_ratio

        # Confidence score
        length_score = min(1.0, avg_length / 200.0)  # Longer lines = more crack-like
        straightness_score = float(straightness)
        contrast_score = min(1.0, mean_contrast * 5.0)

        confidence = (cfg["length_weight"] * length_score +
                     cfg["straightness_weight"] * straightness_score +
                     cfg["contrast_weight"] * contrast_score) * depth_support
        confidence = float(np.clip(confidence, 0.1, 0.85))

        if confidence < 0.2:
            continue

        candidate = ImageDamageCandidate(
            candidate_id=f"img_crack_{uuid.uuid4().hex[:8]}",
            candidate_class=CandidateClass.CRACK,
            bbox_image=[int(x_min), int(y_min), int(bw), int(bh)],
            confidence=confidence,
            source_frame="000000",
            source_dataset="",
            detection_method="edge_hough_clustering",
            feature_metrics={
                "num_lines": len(cluster),
                "total_length": float(total_length),
                "avg_length": float(avg_length),
                "straightness": float(straightness),
                "mean_contrast": mean_contrast,
                "depth_support": depth_support,
                "length_score": length_score,
                "straightness_score": straightness_score,
                "contrast_score": contrast_score,
            },
            notes="IMAGE-DERIVED CANDIDATE - not validated damage detection. Based on edge detection, Hough lines, and clustering.",
        )
        candidates.append(candidate)

    return candidates


def create_visualization(
    image: np.ndarray,
    candidates: list[ImageDamageCandidate],
    output_path: Path,
    depth: Optional[np.ndarray] = None,
) -> None:
    """
    Create annotated visualization of candidates on the image.

    Args:
        image: RGB image
        candidates: List of detected candidates
        output_path: Output PNG path
        depth: Optional depth map for overlay
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    # Color map for candidate classes
    colors = {
        CandidateClass.WATER_DAMAGE: (0, 165, 255),   # Orange
        CandidateClass.CRACK: (0, 255, 255),          # Yellow
    }

    for c in candidates:
        x, y, bw, bh = c.bbox_image
        color = colors.get(c.candidate_class, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

        # Draw label background
        label = f"{c.candidate_class}: {c.confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x, y - label_h - 6), (x + label_w + 4, y), color, -1)

        # Draw label text
        cv2.putText(vis, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw candidate ID
        id_label = c.candidate_id[:12]
        cv2.putText(vis, id_label, (x + 2, y + bh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Add legend
    legend_y = 30
    for cls_name, color in colors.items():
        cv2.rectangle(vis, (10, legend_y - 15), (30, legend_y + 5), color, -1)
        cv2.putText(vis, cls_name, (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis, cls_name, (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        legend_y += 25

    # Add status banner
    banner_text = "IMAGE-DERIVED CANDIDATES ONLY - NOT VALIDATED DAMAGE"
    cv2.rectangle(vis, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, banner_text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

    # Add frame info
    info_text = f"Frame: {candidates[0].source_frame if candidates else 'N/A'} | Dataset: {candidates[0].source_dataset if candidates else 'N/A'} | Candidates: {len(candidates)}"
    cv2.putText(vis, info_text, (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(str(output_path), vis)


def load_depth_confidence(
    dataset_dir: Path,
    frame_id: str = "000000",
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load depth and confidence maps for a frame."""
    depth_path = dataset_dir / "depth" / f"{frame_id}.png"
    conf_path = dataset_dir / "confidence" / f"{frame_id}.png"

    depth = None
    conf = None

    if depth_path.exists():
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if conf_path.exists():
        conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)

    return depth, conf


def extract_rgb_frame(
    video_path: Path,
    frame_index: int = 0,
) -> Optional[np.ndarray]:
    """Extract a specific frame from RGB video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    return None


def run_image_damage_detection(
    dataset_id: str,
    frame_id: str,
    dataset_root: Path,
    output_dir: Path,
) -> ImageDamageResult:
    """
    Run image-based damage detection for a dataset frame.

    Args:
        dataset_id: Dataset identifier (e.g., "single_scan_floor_only/1a8384c3f6")
        frame_id: Frame ID (e.g., "000000")
        dataset_root: Root directory containing dataset folders
        output_dir: Output directory for results

    Returns:
        ImageDamageResult with candidates
    """
    dataset_path = dataset_root / dataset_id
    rgb_video = dataset_path / "rgb.mp4"

    # Extract RGB frame
    frame_idx = int(frame_id)
    image = extract_rgb_frame(rgb_video, frame_idx)

    if image is None:
        # Fallback: try to load from already extracted frames
        extracted_frame = Path("output") / f"rgb_frame_{frame_id}_{dataset_id.replace('/', '_')}.png"
        if extracted_frame.exists():
            image = cv2.imread(str(extracted_frame))

    if image is None:
        raise ValueError(f"Could not extract/load RGB frame for {dataset_id} frame {frame_id}")

    # Load depth and confidence for context
    depth, confidence_map = load_depth_confidence(dataset_path, frame_id)

    # Resize depth/confidence to match image if needed (for visualization only)
    h, w = image.shape[:2]
    if depth is not None and (depth.shape[0] != h or depth.shape[1] != w):
        depth_vis = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        depth_vis = depth

    # Run detectors
    water_candidates = detect_water_damage_candidates(image, depth, confidence_map)
    crack_candidates = detect_crack_candidates(image, depth, confidence_map)

    # Update metadata for all candidates
    all_candidates = water_candidates + crack_candidates
    for c in all_candidates:
        c.source_frame = frame_id
        c.source_dataset = dataset_id

    # Create result
    result = ImageDamageResult(
        dataset_id=dataset_id,
        frame_id=frame_id,
        candidates=all_candidates,
        image_shape=[h, w, 3],
        depth_shape=list(depth.shape) if depth is not None else None,
    )

    # Validate
    errors = result.validate()
    if errors:
        result.limitations.extend([f"VALIDATION_WARNING: {e}" for e in errors])

    return result


def save_result(result: ImageDamageResult, output_dir: Path) -> tuple[Path, Path]:
    """Save JSON result and PNG visualization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / f"image_damage_candidates_{result.dataset_id.replace('/', '_')}_{result.frame_id}.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    # Save visualization (need to reload image)
    dataset_root = Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup")
    image = extract_rgb_frame(dataset_root / result.dataset_id / "rgb.mp4", int(result.frame_id))
    if image is None:
        extracted_frame = Path("output") / f"rgb_frame_{result.frame_id}_{result.dataset_id.replace('/', '_')}.png"
        if extracted_frame.exists():
            image = cv2.imread(str(extracted_frame))

    if image is not None:
        png_path = output_dir / f"image_damage_candidates_{result.dataset_id.replace('/', '_')}_{result.frame_id}.png"
        create_visualization(image, result.candidates, png_path)
    else:
        png_path = None

    return json_path, png_path


def main():
    parser = argparse.ArgumentParser(description="Image-based Damage Candidate Detector")
    parser.add_argument("--dataset-id", help="Dataset identifier (e.g., single_scan_floor_only/1a8384c3f6)")
    parser.add_argument("--frame-id", default="000000", help="Frame ID")
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/image_damage_detection"))
    parser.add_argument("--self-test", action="store_true", help="Run synthetic test only")
    args = parser.parse_args()

    if args.self_test:
        # Run on synthetic test image
        print("Running self-test with synthetic image...")
        # Create synthetic test image with known features
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[:] = (180, 180, 180)  # Gray background

        # Add a brown stain (water damage simulation)
        cv2.ellipse(test_img, (200, 200), (80, 60), 0, 0, 360, (60, 100, 150), -1)  # Brownish
        cv2.ellipse(test_img, (200, 200), (70, 50), 0, 0, 360, (50, 90, 140), -1)

        # Add a crack (dark line)
        cv2.line(test_img, (400, 100), (500, 300), (30, 30, 30), 3)
        cv2.line(test_img, (405, 105), (505, 305), (40, 40, 40), 2)

        # Add some noise/texture
        noise = np.random.randint(0, 30, test_img.shape, dtype=np.uint8)
        test_img = cv2.add(test_img, noise)

        # Save test image
        test_path = Path("output/test_synthetic_damage.png")
        test_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(test_path), test_img)

        # Run detection
        water_cands = detect_water_damage_candidates(test_img)
        crack_cands = detect_crack_candidates(test_img)

        result = ImageDamageResult(
            dataset_id="synthetic_test",
            frame_id="000000",
            candidates=water_cands + crack_cands,
            image_shape=[480, 640, 3],
        )

        # Save
        out_dir = Path("output/image_damage_detection_test")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "image_damage_candidates_synthetic_test_000000.json"
        json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

        png_path = out_dir / "image_damage_candidates_synthetic_test_000000.png"
        create_visualization(test_img, result.candidates, png_path)

        print(f"Self-test complete. Candidates: {len(result.candidates)}")
        print(f"  Water damage: {len(water_cands)}")
        print(f"  Cracks: {len(crack_cands)}")
        print(f"JSON: {json_path}")
        print(f"PNG: {png_path}")
        print(json.dumps(result.to_dict(), indent=2))
        return

    # Run actual detection
    print(f"Running image damage detection on {args.dataset_id} frame {args.frame_id}")
    result = run_image_damage_detection(
        args.dataset_id,
        args.frame_id,
        args.dataset_root,
        args.output_dir,
    )

    json_path, png_path = save_result(result, args.output_dir)

    print(f"Detection complete. Candidates: {len(result.candidates)}")
    water_count = sum(1 for c in result.candidates if c.candidate_class == CandidateClass.WATER_DAMAGE)
    crack_count = sum(1 for c in result.candidates if c.candidate_class == CandidateClass.CRACK)
    print(f"  Water damage candidates: {water_count}")
    print(f"  Crack candidates: {crack_count}")
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")

    # Print summary
    for c in result.candidates:
        print(f"  {c.candidate_id}: {c.candidate_class} @ {c.bbox_image} conf={c.confidence:.3f}")


if __name__ == "__main__":
    main()