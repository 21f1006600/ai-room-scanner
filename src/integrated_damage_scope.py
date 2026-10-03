#!/usr/bin/env python3
"""
Integration of Image-Based Damage Candidates with Damage & Scope Workflow

Combines image-derived candidates (from image_damage_detector) with
manual/synthetic annotations (from damage_and_scope), clearly distinguishing sources.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from damage_and_scope import (
    DamageAndScopeResult,
    DamageRegion,
    DamageClass,
    SurfaceType,
    ConcealedDamageFlag,
    ScopeLineItem,
    run_damage_and_scope,
    create_synthetic_annotations,
    load_annotations,
)
from image_damage_detector import (
    ImageDamageResult,
    ImageDamageCandidate,
    CandidateClass,
    run_image_damage_detection,
)


@dataclass
class IntegratedDamageResult:
    """Complete integrated damage assessment combining all sources."""
    schema_version: str = "1.0"
    dataset_id: str = ""
    frame_id: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Manual/synthetic annotations (existing workflow)
    manual_regions: list[DamageRegion] = field(default_factory=list)
    manual_scope_items: list[ScopeLineItem] = field(default_factory=list)

    # Image-derived candidates (new workflow)
    image_candidates: list[ImageDamageCandidate] = field(default_factory=list)

    # Limitations and provenance
    limitations: list[str] = field(default_factory=lambda: [
        "Manual regions: SYNTHETIC/MANUAL annotations only - no automated detection performed.",
        "Image candidates: IMAGE-DERIVED from visual features only - NOT validated damage detections.",
        "No labeled training data or ground truth validation was used for image candidates.",
        "Water damage candidates based on color/texture heuristics; false positives likely.",
        "Crack candidates based on edge/line heuristics; false positives likely.",
        "No semantic damage classification or accuracy claims without labeled validation data.",
        "Physical ground truth measurements are NOT_MEASURED.",
        "All candidates require manual review and validation before any scope decisions.",
    ])

    # Status flags
    status: dict[str, str] = field(default_factory=lambda: {
        "manual_damage_detection": "SYNTHETIC_ANNOTATION_ONLY",
        "image_damage_detection": "IMAGE_CANDIDATES_ONLY",
        "physical_measurements": "NOT_MEASURED",
        "ground_truth_validation": "NOT_MEASURED",
        "labeled_training_data": "NOT_USED",
        "accuracy_gates": "BLOCKED",
    })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "frame_id": self.frame_id,
            "generated_at": self.generated_at,
            "manual_regions": [r.to_dict() for r in self.manual_regions],
            "manual_scope_items": [i.to_dict() for i in self.manual_scope_items],
            "image_candidates": [c.to_dict() for c in self.image_candidates],
            "limitations": self.limitations,
            "status": self.status,
        }

    def validate(self) -> list[str]:
        """Validate the integrated result."""
        errors = []

        # Validate manual regions
        region_ids = set()
        for r in self.manual_regions:
            if r.region_id in region_ids:
                errors.append(f"Duplicate manual region_id: {r.region_id}")
            region_ids.add(r.region_id)
            if not 0 <= r.confidence <= 1:
                errors.append(f"Manual region {r.region_id}: confidence must be in [0, 1]")

        # Validate manual scope items
        for item in self.manual_scope_items:
            if item.damage_region_id not in region_ids:
                errors.append(f"Manual scope item {item.line_item_id}: references unknown region_id {item.damage_region_id}")

        # Validate image candidates
        candidate_ids = set()
        for c in self.image_candidates:
            if c.candidate_id in candidate_ids:
                errors.append(f"Duplicate image candidate_id: {c.candidate_id}")
            candidate_ids.add(c.candidate_id)
            if not 0 <= c.confidence <= 1:
                errors.append(f"Image candidate {c.candidate_id}: confidence must be in [0, 1]")
            if len(c.bbox_image) != 4:
                errors.append(f"Image candidate {c.candidate_id}: bbox_image must have 4 values [x, y, w, h]")

        return errors


def map_candidate_class_to_damage_class(candidate_class: str) -> DamageClass:
    """Map image candidate class to damage class."""
    if candidate_class == CandidateClass.WATER_DAMAGE:
        return DamageClass.WATER_DAMAGE
    elif candidate_class == CandidateClass.CRACK:
        return DamageClass.CRACK
    return DamageClass.WATER_DAMAGE  # default


def create_damage_region_from_candidate(
    candidate: ImageDamageCandidate,
    surface_type: SurfaceType = SurfaceType.UNKNOWN,
    surface_id: Optional[str] = None,
) -> DamageRegion:
    """
    Create a DamageRegion from an image candidate.
    Clearly marked as image-derived.
    """
    # Map image candidate class to damage class
    damage_class = map_candidate_class_to_damage_class(candidate.candidate_class)

    # Convert image bbox to approximate camera coordinates
    # This is a rough approximation - image coords (x,y) -> camera coords
    # For now, use placeholder values with notes
    x, y, w, h = candidate.bbox_image

    # Create region with image-derived metadata
    region = DamageRegion(
        region_id=f"img_{candidate.candidate_id}",
        damage_class=damage_class,
        center_camera=[float(x + w/2), float(-y - h/2), 800.0],  # Rough placeholder
        extent_camera=[float(w), float(h), 5.0],
        confidence=candidate.confidence,
        surface_id=surface_id,
        surface_type=surface_type,
        concealed_flag=ConcealedDamageFlag.NO_EVIDENCE,
        concealed_rationale="",
        notes=f"IMAGE-DERIVED from {candidate.detection_method}: {candidate.notes}",
        annotated_by="image_detector",
        annotated_at=candidate.annotated_at,
        source_dataset=candidate.source_dataset,
        source_frame=candidate.source_frame,
    )
    return region


def create_scope_item_from_region(
    region: DamageRegion,
    line_item_id: str,
    trade: str = "general",
) -> ScopeLineItem:
    """Create a scope line item from a damage region."""
    # Determine trade based on damage class and surface
    if region.damage_class == DamageClass.WATER_DAMAGE:
        trade = "flooring" if region.surface_type == SurfaceType.FLOOR else "drywall"
    elif region.damage_class == DamageClass.CRACK:
        trade = "drywall"

    return ScopeLineItem(
        line_item_id=line_item_id,
        damage_region_id=region.region_id,
        trade=trade,
        description=f"Investigate and repair {region.damage_class.value.replace('_', ' ')} candidate from image detection",
        quantity=1.0,
        unit="m2" if region.damage_class == DamageClass.WATER_DAMAGE else "linear_m",
        surface_id=region.surface_id,
        surface_type=region.surface_type,
        estimated_cost_usd=None,  # No cost estimation for unvalidated candidates
        priority="medium",
        notes="IMAGE-DERIVED SCOPE ITEM - candidate requires validation before scoping",
    )


def run_integrated_damage_and_scope(
    dataset_id: str,
    frame_id: str,
    dataset_root: Path,
    room_plan_prototype_path: Optional[Path] = None,
    run_image_detection: bool = True,
) -> IntegratedDamageResult:
    """
    Run integrated damage and scope assessment combining:
    1. Manual/synthetic annotations (existing)
    2. Image-derived candidates (new)

    Args:
        dataset_id: Dataset identifier
        frame_id: Frame ID
        dataset_root: Root directory containing datasets
        room_plan_prototype_path: Optional path to room plan prototype for surface linking
        run_image_detection: Whether to run image-based detection

    Returns:
        IntegratedDamageResult with both manual and image-derived items
    """
    result = IntegratedDamageResult(
        dataset_id=dataset_id,
        frame_id=frame_id,
    )

    # 1. Run existing manual/synthetic workflow
    manual_result = run_damage_and_scope(dataset_id, frame_id, room_plan_prototype_path)
    result.manual_regions = manual_result.damage_regions
    result.manual_scope_items = manual_result.scope_line_items

    # 2. Run image-based detection if requested
    if run_image_detection:
        try:
            image_result = run_image_damage_detection(
                dataset_id, frame_id, dataset_root, Path("output/image_damage_detection")
            )
            result.image_candidates = image_result.candidates
        except Exception as e:
            result.limitations.append(f"Image detection failed: {e}")

    # Validate
    errors = result.validate()
    if errors:
        result.limitations.extend([f"VALIDATION_WARNING: {e}" for e in errors])

    return result


def save_integrated_result(
    result: IntegratedDamageResult,
    output_path: Path,
) -> Path:
    """Save integrated result to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return output_path


def load_integrated_result(json_path: Path) -> IntegratedDamageResult:
    """Load integrated result from JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    result = IntegratedDamageResult(
        schema_version=data.get("schema_version", "1.0"),
        dataset_id=data.get("dataset_id", ""),
        frame_id=data.get("frame_id", ""),
        generated_at=data.get("generated_at", datetime.utcnow().isoformat() + "Z"),
        limitations=data.get("limitations", []),
        status=data.get("status", {}),
    )

    # Load manual regions
    for r in data.get("manual_regions", []):
        result.manual_regions.append(DamageRegion(
            region_id=r["region_id"],
            damage_class=DamageClass(r["damage_class"]),
            center_camera=r["center_camera"],
            extent_camera=r["extent_camera"],
            confidence=r["confidence"],
            surface_id=r.get("surface_id"),
            surface_type=SurfaceType(r.get("surface_type", "unknown")),
            concealed_flag=ConcealedDamageFlag(r.get("concealed_flag", "no_evidence")),
            concealed_rationale=r.get("concealed_rationale", ""),
            notes=r.get("notes", ""),
            annotated_by=r.get("annotated_by", "manual"),
            annotated_at=r.get("annotated_at", datetime.utcnow().isoformat() + "Z"),
            source_dataset=r.get("source_dataset", ""),
            source_frame=r.get("source_frame", ""),
        ))

    # Load manual scope items
    for item in data.get("manual_scope_items", []):
        result.manual_scope_items.append(ScopeLineItem(
            line_item_id=item["line_item_id"],
            damage_region_id=item["damage_region_id"],
            trade=item["trade"],
            description=item["description"],
            quantity=item["quantity"],
            unit=item["unit"],
            surface_id=item.get("surface_id"),
            surface_type=SurfaceType(item.get("surface_type", "unknown")),
            estimated_cost_usd=item.get("estimated_cost_usd"),
            priority=item.get("priority", "medium"),
            notes=item.get("notes", ""),
        ))

    # Load image candidates
    for c in data.get("image_candidates", []):
        result.image_candidates.append(ImageDamageCandidate(
            candidate_id=c["candidate_id"],
            candidate_class=c["candidate_class"],
            bbox_image=c["bbox_image"],
            confidence=c["confidence"],
            source_frame=c["source_frame"],
            source_dataset=c["source_dataset"],
            detection_method=c["detection_method"],
            feature_metrics=c.get("feature_metrics", {}),
            notes=c.get("notes", ""),
            annotated_by=c.get("annotated_by", "image_detector"),
            annotated_at=c.get("annotated_at", datetime.utcnow().isoformat() + "Z"),
        ))

    return result


def create_integrated_visualization(
    image_path: Path,
    result: IntegratedDamageResult,
    output_path: Path,
) -> None:
    """Create visualization showing both manual and image-derived items."""
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        return

    vis = image.copy()
    h, w = vis.shape[:2]

    # Draw manual regions (green)
    for region in result.manual_regions:
        # For manual regions, we don't have image coords, so draw a placeholder
        pass

    # Draw image candidates (orange for water, yellow for crack)
    colors = {
        CandidateClass.WATER_DAMAGE: (0, 165, 255),  # Orange
        CandidateClass.CRACK: (0, 255, 255),         # Yellow
    }

    for candidate in result.image_candidates:
        x, y, bw, bh = candidate.bbox_image
        color = colors.get(candidate.candidate_class, (255, 255, 255))

        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

        label = f"{candidate.candidate_class}: {candidate.confidence:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x, y - label_h - 6), (x + label_w + 4, y), color, -1)
        cv2.putText(vis, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Add legend
    legend_y = 30
    cv2.rectangle(vis, (10, legend_y - 15), (30, legend_y + 5), (0, 255, 0), -1)
    cv2.putText(vis, "MANUAL (synthetic)", (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    legend_y += 25

    for cls_name, color in colors.items():
        cv2.rectangle(vis, (10, legend_y - 15), (30, legend_y + 5), color, -1)
        cv2.putText(vis, f"IMAGE: {cls_name}", (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        legend_y += 25

    # Status banner
    banner = "INTEGRATED: MANUAL (green) + IMAGE-DERIVED (orange/yellow) - NOT VALIDATED"
    cv2.rectangle(vis, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, banner, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imwrite(str(output_path), vis)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Integrated Damage & Scope with Image Detection")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier")
    parser.add_argument("--frame-id", default="000000", help="Frame ID")
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/brynz-applied-AI/Assignment - YC Startup-20261001T152812Z-1-001/Assignment - YC Startup"))
    parser.add_argument("--room-plan-prototype", type=Path, help="Path to room plan prototype JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--no-image-detection", action="store_true", help="Skip image-based detection")
    parser.add_argument("--create-visualization", action="store_true", help="Create visualization PNG")
    args = parser.parse_args()

    result = run_integrated_damage_and_scope(
        args.dataset_id,
        args.frame_id,
        args.dataset_root,
        args.room_plan_prototype,
        run_image_detection=not args.no_image_detection,
    )

    save_integrated_result(result, args.output)
    print(f"Integrated result saved to {args.output}")
    print(f"Manual regions: {len(result.manual_regions)}, Manual scope: {len(result.manual_scope_items)}")
    print(f"Image candidates: {len(result.image_candidates)}")

    errors = result.validate()
    if errors:
        print("Validation warnings:", errors)

    if args.create_visualization:
        # Try to load the RGB frame for visualization
        from image_damage_detector import extract_rgb_frame
        dataset_path = args.dataset_root / args.dataset_id
        rgb_video = dataset_path / "rgb.mp4"
        image = extract_rgb_frame(rgb_video, int(args.frame_id))

        if image is not None:
            vis_path = args.output.with_suffix(".png")
            create_integrated_visualization(Path("temp_frame.png"), result, vis_path)
            # Need to save frame first
            cv2.imwrite("temp_frame.png", image)
            create_integrated_visualization(Path("temp_frame.png"), result, vis_path)
            print(f"Visualization saved to {vis_path}")


if __name__ == "__main__":
    main()