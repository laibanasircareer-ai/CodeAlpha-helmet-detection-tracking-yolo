"""
Central configuration for the helmet detection + tracking pipeline.
No class IDs are hardcoded here - they are resolved dynamically at
runtime from each model's own `model.names` mapping (see detector.py).
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paths:
    coco_model: str = "models/yolo11n.pt"
    helmet_model: str = "models/best.pt"          # <- your Roboflow-trained weights go here
    test_image_dir: str = "project_data/test_images"
    output_dir: str = "project_data/outputs"


@dataclass
class DetectionConfig:
    person_conf_threshold: float = 0.4
    motorcycle_conf_threshold: float = 0.4
    helmet_conf_threshold: float = 0.35


@dataclass
class AssociationConfig:
    # Fraction of a person's bounding box (from the top) treated as the "head region"
    head_region_fraction: float = 0.30
    # Minimum IoU between a helmet box and the head region to count as "wearing a helmet"
    helmet_iou_threshold: float = 0.10
    # Max normalized distance (relative to frame diagonal) to associate a rider with a motorcycle
    max_rider_motorcycle_distance_ratio: float = 0.15


@dataclass
class TrackingConfig:
    tracker_yaml: str = "bytetrack.yaml"   # built into ultralytics
    persist: bool = True


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


CONFIG = Config()

# Ensure output dir exists
Path(CONFIG.paths.output_dir).mkdir(parents=True, exist_ok=True)
