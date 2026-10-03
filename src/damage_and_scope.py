#!/usr/bin/env python3
"""
Damage and Scope Workflow for Room Scanner

Provides:
- Manual damage annotation workflow with schema validation
- Two damage classes: WATER_DAMAGE, CRACK
- Region coordinates, class, extent, confidence, surface association
- Explicit concealed-damage rules
- Scope line items linked to surfaces
- JSON output and validation

All annotations are MANUAL/SYNTHETIC - no automated damage detection is performed.
Physical ground truth is NOT_MEASURED.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from enum import Enum


class DamageClass(str, Enum):
    """Damage classification - synthetic/manual only."""
    WATER_DAMAGE = "water_damage"
    CRACK = "crack"


class SurfaceType(str, Enum):
    """Surface type for scope line items."""
    WALL = "wall"
    CEILING = "ceiling"
    FLOOR = "floor"
    DOOR = "door"
    WINDOW = "window"
    TRIM = "trim"
    UNKNOWN = "unknown"


class ConcealedDamageFlag(str, Enum):
    """Concealed damage assessment flags with explicit rules."""
    NO_EVIDENCE = "no_evidence"  # No sensor data suggests concealed damage
    POSSIBLE = "possible"  # Adjacent visible damage + moisture-prone construction
    LIKELY = "likely"  # Visible damage on both sides of assembly OR active moisture reading
    CONFIRMED = "confirmed"  # Invasive inspection or destructive testing confirmed


@dataclass
class DamageRegion:
    """A single damage region annotation (manual/synthetic)."""
    region_id: str
    damage_class: DamageClass
    # 3D coordinates in camera-frame raw depth units (x, y, z)
    center_camera: list[float]
    # Extent in camera-frame raw depth units (width, height, depth)
    extent_camera: list[float]
    # Confidence in this annotation [0, 1] - manual assessment
    confidence: float
    # Associated surface ID from room plan (if available)
    surface_id: Optional[str] = None
    # Surface type for scope
    surface_type: SurfaceType = SurfaceType.UNKNOWN
    # Concealed damage assessment
    concealed_flag: ConcealedDamageFlag = ConcealedDamageFlag.NO_EVIDENCE
    # Concealed damage rationale (required if not NO_EVIDENCE)
    concealed_rationale: str = ""
    # Notes
    notes: str = ""
    # Annotation metadata
    annotated_by: str = "manual"
    annotated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    # Source data reference
    source_dataset: str = ""
    source_frame: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["damage_class"] = self.damage_class.value
        d["surface_type"] = self.surface_type.value
        d["concealed_flag"] = self.concealed_flag.value
        return d


@dataclass
class ScopeLineItem:
    """Scope of work line item linked to a damage region."""
    line_item_id: str
    damage_region_id: str
    # Trade/category
    trade: str
    # Description of work
    description: str
    # Quantity and unit
    quantity: float
    unit: str
    # Surface reference
    surface_id: Optional[str] = None
    surface_type: SurfaceType = SurfaceType.UNKNOWN
    # Cost estimation (optional, synthetic)
    estimated_cost_usd: Optional[float] = None
    # Priority
    priority: Literal["high", "medium", "low"] = "medium"
    # Notes
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["surface_type"] = self.surface_type.value
        return d


@dataclass
class DamageAndScopeResult:
    """Complete damage and scope assessment result."""
    schema_version: str = "1.0"
    dataset_id: str = ""
    frame_id: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    # All regions are MANUAL/SYNTHETIC annotations
    damage_regions: list[DamageRegion] = field(default_factory=list)
    scope_line_items: list[ScopeLineItem] = field(default_factory=list)
    # Limitations and provenance
    limitations: list[str] = field(default_factory=lambda: [
        "All damage regions are MANUAL/SYNTHETIC annotations - no automated detection performed.",
        "Physical ground truth measurements are NOT_MEASURED.",
        "Surface associations are inferred from room-plan prototype where available.",
        "Concealed damage flags use explicit rules but lack invasive verification.",
        "Scope line items are synthetic templates for demonstration only.",
        "No consumer scan app comparison or accuracy gates are evaluated.",
    ])
    # Status flags
    status: dict[str, str] = field(default_factory=lambda: {
        "damage_detection": "MANUAL_ANNOTATION_ONLY",
        "physical_measurements": "NOT_MEASURED",
        "ground_truth_validation": "NOT_MEASURED",
        "repeat_capture_comparison": "NOT_MEASURED",
        "consumer_app_comparison": "NOT_MEASURED",
        "accuracy_gates": "BLOCKED",
    })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "frame_id": self.frame_id,
            "generated_at": self.generated_at,
            "damage_regions": [r.to_dict() for r in self.damage_regions],
            "scope_line_items": [i.to_dict() for i in self.scope_line_items],
            "limitations": self.limitations,
            "status": self.status,
        }

    def validate(self) -> list[str]:
        """Validate the result and return list of errors."""
        errors = []
        region_ids = set()
        for r in self.damage_regions:
            if r.region_id in region_ids:
                errors.append(f"Duplicate region_id: {r.region_id}")
            region_ids.add(r.region_id)
            if not 0 <= r.confidence <= 1:
                errors.append(f"Region {r.region_id}: confidence must be in [0, 1]")
            if r.concealed_flag != ConcealedDamageFlag.NO_EVIDENCE and not r.concealed_rationale:
                errors.append(f"Region {r.region_id}: concealed_rationale required when flag != NO_EVIDENCE")

        for item in self.scope_line_items:
            if item.damage_region_id not in region_ids:
                errors.append(f"Scope item {item.line_item_id}: references unknown region_id {item.damage_region_id}")
        return errors


# Concealed damage rules (explicit, documented)
CONCEALED_DAMAGE_RULES = {
    "NO_EVIDENCE": "Default. No sensor data (depth, confidence, RGB) suggests concealed damage.",
    "POSSIBLE": "Triggered when: (a) visible damage region on one side of a wall assembly AND "
                "construction type is moisture-prone (drywall, plaster), OR "
                "(b) confidence map shows low-confidence region adjacent to visible damage.",
    "LIKELY": "Triggered when: (a) visible damage regions on BOTH sides of same wall assembly, OR "
              "(b) active moisture sensor reading (not available in this dataset), OR "
              "(c) depth anomaly pattern consistent with subsurface delamination.",
    "CONFIRMED": "Requires invasive inspection or destructive testing confirmation. Not achievable from remote sensing alone.",
}

# Damage class definitions
DAMAGE_CLASS_DEFINITIONS = {
    DamageClass.WATER_DAMAGE: {
        "description": "Discoloration, staining, efflorescence, or material degradation from moisture intrusion",
        "typical_extent_cm2": "100-5000",
        "concealed_risk": "HIGH - moisture migrates through assemblies",
    },
    DamageClass.CRACK: {
        "description": "Linear fracture in surface material (drywall, plaster, concrete, tile)",
        "typical_extent_cm2": "10-1000 (linear cm)",
        "concealed_risk": "MEDIUM - may indicate structural movement",
    },
}


def create_synthetic_annotations(dataset_id: str, frame_id: str, room_plan_hull: Optional[list[dict]] = None) -> DamageAndScopeResult:
    """
    Create synthetic damage annotations for demonstration.
    Clearly marked as synthetic - NOT real detected damage.
    """
    result = DamageAndScopeResult(
        dataset_id=dataset_id,
        frame_id=frame_id,
    )

    # Example synthetic annotations for floor-only dataset
    if "floor_only" in dataset_id:
        # Water damage on floor (horizontal plane candidate)
        r1 = DamageRegion(
            region_id="synth_water_001",
            damage_class=DamageClass.WATER_DAMAGE,
            center_camera=[100.0, -200.0, 900.0],
            extent_camera=[50.0, 40.0, 5.0],
            confidence=0.85,
            surface_id="horizontal_plane_candidate_2",
            surface_type=SurfaceType.FLOOR,
            concealed_flag=ConcealedDamageFlag.POSSIBLE,
            concealed_rationale="Visible staining on floor surface; subfloor moisture migration possible per CONCEALED_DAMAGE_RULES['POSSIBLE'] rule (a)",
            notes="SYNTHETIC ANNOTATION - not detected from sensor data",
            source_dataset=dataset_id,
            source_frame=frame_id,
        )
        # Crack on wall (inferred vertical)
        r2 = DamageRegion(
            region_id="synth_crack_001",
            damage_class=DamageClass.CRACK,
            center_camera=[-300.0, -150.0, 850.0],
            extent_camera=[5.0, 120.0, 3.0],
            confidence=0.70,
            surface_id=None,
            surface_type=SurfaceType.WALL,
            concealed_flag=ConcealedDamageFlag.NO_EVIDENCE,
            concealed_rationale="",
            notes="SYNTHETIC ANNOTATION - not detected from sensor data",
            source_dataset=dataset_id,
            source_frame=frame_id,
        )
        result.damage_regions.extend([r1, r2])

        # Scope line items
        result.scope_line_items.extend([
            ScopeLineItem(
                line_item_id="scope_001",
                damage_region_id="synth_water_001",
                trade="flooring",
                description="Remove and replace water-damaged flooring section; inspect subfloor for moisture",
                quantity=2.0,
                unit="m2",
                surface_id="horizontal_plane_candidate_2",
                surface_type=SurfaceType.FLOOR,
                estimated_cost_usd=450.0,
                priority="high",
                notes="SYNTHETIC SCOPE ITEM - template only",
            ),
            ScopeLineItem(
                line_item_id="scope_002",
                damage_region_id="synth_crack_001",
                trade="drywall",
                description="Repair drywall crack; investigate structural cause",
                quantity=1.2,
                unit="linear_m",
                surface_type=SurfaceType.WALL,
                estimated_cost_usd=200.0,
                priority="medium",
                notes="SYNTHETIC SCOPE ITEM - template only",
            ),
        ])

    # For other datasets, add placeholder annotations
    elif "single_room" in dataset_id:
        r = DamageRegion(
            region_id="synth_water_002",
            damage_class=DamageClass.WATER_DAMAGE,
            center_camera=[0.0, -100.0, 800.0],
            extent_camera=[30.0, 30.0, 5.0],
            confidence=0.60,
            surface_type=SurfaceType.FLOOR,
            concealed_flag=ConcealedDamageFlag.NO_EVIDENCE,
            notes="SYNTHETIC ANNOTATION - placeholder for single_room dataset",
            source_dataset=dataset_id,
            source_frame=frame_id,
        )
        result.damage_regions.append(r)
        result.scope_line_items.append(ScopeLineItem(
            line_item_id="scope_003",
            damage_region_id="synth_water_002",
            trade="flooring",
            description="Investigate and repair potential water damage",
            quantity=1.0,
            unit="m2",
            surface_type=SurfaceType.FLOOR,
            estimated_cost_usd=300.0,
            priority="medium",
            notes="SYNTHETIC SCOPE ITEM - placeholder",
        ))

    elif "with_ceiling" in dataset_id:
        r = DamageRegion(
            region_id="synth_crack_002",
            damage_class=DamageClass.CRACK,
            center_camera=[200.0, -300.0, 700.0],
            extent_camera=[3.0, 80.0, 2.0],
            confidence=0.65,
            surface_type=SurfaceType.CEILING,
            concealed_flag=ConcealedDamageFlag.NO_EVIDENCE,
            notes="SYNTHETIC ANNOTATION - placeholder for ceiling dataset",
            source_dataset=dataset_id,
            source_frame=frame_id,
        )
        result.damage_regions.append(r)
        result.scope_line_items.append(ScopeLineItem(
            line_item_id="scope_004",
            damage_region_id="synth_crack_002",
            trade="drywall",
            description="Repair ceiling crack",
            quantity=0.8,
            unit="linear_m",
            surface_type=SurfaceType.CEILING,
            estimated_cost_usd=150.0,
            priority="medium",
            notes="SYNTHETIC SCOPE ITEM - placeholder",
        ))

    return result


def save_annotation_template(output_path: Path) -> None:
    """Save a blank annotation template for manual entry."""
    template = {
        "schema_version": "1.0",
        "dataset_id": "ENTER_DATASET_ID",
        "frame_id": "000000",
        "damage_regions": [
            {
                "region_id": "manual_001",
                "damage_class": "water_damage",
                "center_camera": [0.0, 0.0, 0.0],
                "extent_camera": [0.0, 0.0, 0.0],
                "confidence": 0.0,
                "surface_id": None,
                "surface_type": "wall",
                "concealed_flag": "no_evidence",
                "concealed_rationale": "",
                "notes": "Enter manual annotation here",
                "annotated_by": "manual",
                "annotated_at": datetime.utcnow().isoformat() + "Z",
                "source_dataset": "ENTER_DATASET_ID",
                "source_frame": "000000",
            }
        ],
        "scope_line_items": [
            {
                "line_item_id": "scope_001",
                "damage_region_id": "manual_001",
                "trade": "drywall",
                "description": "Enter scope description",
                "quantity": 0.0,
                "unit": "m2",
                "surface_id": None,
                "surface_type": "wall",
                "estimated_cost_usd": None,
                "priority": "medium",
                "notes": "",
            }
        ],
        "limitations": [
            "All damage regions are MANUAL annotations.",
            "Physical ground truth is NOT_MEASURED.",
        ],
        "status": {
            "damage_detection": "MANUAL_ANNOTATION_ONLY",
            "physical_measurements": "NOT_MEASURED",
            "ground_truth_validation": "NOT_MEASURED",
            "repeat_capture_comparison": "NOT_MEASURED",
            "consumer_app_comparison": "NOT_MEASURED",
            "accuracy_gates": "BLOCKED",
        },
    }
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")


def load_annotations(json_path: Path) -> DamageAndScopeResult:
    """Load annotations from JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    result = DamageAndScopeResult(
        schema_version=data.get("schema_version", "1.0"),
        dataset_id=data.get("dataset_id", ""),
        frame_id=data.get("frame_id", ""),
        generated_at=data.get("generated_at", datetime.utcnow().isoformat() + "Z"),
        limitations=data.get("limitations", []),
        status=data.get("status", {}),
    )
    for r in data.get("damage_regions", []):
        result.damage_regions.append(DamageRegion(
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
    for item in data.get("scope_line_items", []):
        result.scope_line_items.append(ScopeLineItem(
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
    return result


def run_damage_and_scope(dataset_id: str, frame_id: str, room_plan_prototype_path: Optional[Path] = None) -> DamageAndScopeResult:
    """Main entry point: create synthetic annotations for a dataset."""
    hull = None
    if room_plan_prototype_path and room_plan_prototype_path.exists():
        try:
            proto = json.loads(room_plan_prototype_path.read_text(encoding="utf-8"))
            hull = proto.get("observed_projected_hull")
        except Exception:
            pass

    result = create_synthetic_annotations(dataset_id, frame_id, hull)
    errors = result.validate()
    if errors:
        result.limitations.extend([f"VALIDATION_WARNING: {e}" for e in errors])
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Damage and Scope Workflow")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier (e.g., single_scan_floor_only/1a8384c3f6)")
    parser.add_argument("--frame-id", default="000000", help="Frame ID")
    parser.add_argument("--room-plan-prototype", type=Path, help="Path to room_plan_prototype.json for surface linking")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--create-template", action="store_true", help="Create blank annotation template instead")
    args = parser.parse_args()

    if args.create_template:
        save_annotation_template(args.output)
        print(f"Template saved to {args.output}")
        return

    result = run_damage_and_scope(args.dataset_id, args.frame_id, args.room_plan_prototype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"Damage and scope result saved to {args.output}")
    print(f"Regions: {len(result.damage_regions)}, Scope items: {len(result.scope_line_items)}")
    errors = result.validate()
    if errors:
        print("Validation warnings:", errors)


if __name__ == "__main__":
    main()