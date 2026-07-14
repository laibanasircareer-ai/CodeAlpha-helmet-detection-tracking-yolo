"""
Detection layer only. No tracking logic lives here (that's tracker.py) and
no rider/helmet association logic lives here (that's association.py).

This module is deliberately dumb: give it a frame, it gives you back raw
boxes with class names and confidences, for both the COCO model (person,
motorcycle) and the helmet model (helmet / no-helmet).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config import CONFIG


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_name: str
    confidence: float


class HelmetModelNotLoaded(Exception):
    """Raised when helmet inference is requested but no helmet model is available."""


def _resolve_class_id(model: YOLO, aliases: list[str]) -> int | None:
    """
    Look up a class id by trying each alias in turn against model.names.
    Returns None if no alias matches — callers must handle this explicitly
    rather than assuming the class exists (never hardcode/assume class ids).
    """
    name_to_id = {v: k for k, v in model.names.items()}
    for alias in aliases:
        if alias in name_to_id:
            return name_to_id[alias]
    return None


class Detector:
    def __init__(self, load_helmet_model: bool = True):
        self.coco_model = YOLO(CONFIG.paths.coco_model)
        self.person_id = _resolve_class_id(self.coco_model, ["person"])
        self.motorcycle_id = _resolve_class_id(self.coco_model, ["motorcycle"])

        if self.person_id is None or self.motorcycle_id is None:
            raise RuntimeError(
                "COCO model is missing 'person' or 'motorcycle' class — "
                f"available classes: {list(self.coco_model.names.values())}"
            )

        self.helmet_model: YOLO | None = None
        self.helmet_id: int | None = None
        self.no_helmet_id: int | None = None

        if load_helmet_model:
            self._load_helmet_model()

    def _load_helmet_model(self) -> None:
        import os

        if not os.path.exists(CONFIG.paths.helmet_model):
            print(
                f"[detector] WARNING: helmet model not found at "
                f"'{CONFIG.paths.helmet_model}'. Helmet detection will be skipped "
                f"until you place your downloaded weights there."
            )
            return

        model = YOLO(CONFIG.paths.helmet_model)

        helmet_aliases = ["helmet", "Helmet", "with helmet", "Helmet_Detection"]
        no_helmet_aliases = [
            "no helmet",
            "no-helmet",
            "No helmet",
            "No-Helmet",
            "without helmet",
        ]

        helmet_id = _resolve_class_id(model, helmet_aliases)
        no_helmet_id = _resolve_class_id(model, no_helmet_aliases)

        if helmet_id is None:
            print(
                "[detector] WARNING: could not find a 'helmet' class in the "
                f"loaded helmet model. Available classes: {list(model.names.values())}. "
                "Add the exact class name to HELMET aliases in detector.py if it "
                "uses different wording."
            )
            return

        self.helmet_model = model
        self.helmet_id = helmet_id
        self.no_helmet_id = no_helmet_id  # may legitimately be None for some datasets

        print(
            f"[detector] Helmet model loaded OK. "
            f"'helmet' -> id {helmet_id}, "
            f"'no helmet' -> id {no_helmet_id if no_helmet_id is not None else 'N/A (not in this dataset)'}"
        )

    @property
    def helmet_model_ready(self) -> bool:
        return self.helmet_model is not None

    def detect_persons_and_motorcycles(self, frame: np.ndarray) -> list[Detection]:
        results = self.coco_model.predict(
            frame,
            classes=[self.person_id, self.motorcycle_id],
            conf=min(
                CONFIG.detection.person_conf_threshold,
                CONFIG.detection.motorcycle_conf_threshold,
            ),
            verbose=False,
        )
        return self._to_detections(results[0], self.coco_model.names)

    def detect_helmets(self, frame: np.ndarray) -> list[Detection]:
        if not self.helmet_model_ready:
            raise HelmetModelNotLoaded(
                "Helmet model was not loaded. Place your Roboflow weights at "
                f"'{CONFIG.paths.helmet_model}' and restart."
            )
        results = self.helmet_model.predict(
            frame, conf=CONFIG.detection.helmet_conf_threshold, verbose=False
        )
        return self._to_detections(results[0], self.helmet_model.names)

    @staticmethod
    def _to_detections(result, names: dict[int, str]) -> list[Detection]:
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(box=tuple(xyxy), class_name=names[cls_id], confidence=conf)
            )
        return detections
