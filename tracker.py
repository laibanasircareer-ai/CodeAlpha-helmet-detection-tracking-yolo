"""
Tracking layer only. Wraps ultralytics' built-in ByteTrack integration
(model.track(...)) so that persons and motorcycles keep consistent IDs
across frames. No detection thresholds or association logic here.

We track using the SAME coco model as detection, because ultralytics'
.track() call both detects and assigns/updates IDs in one pass — running
a second, separate detection call on top of it would be redundant and
risks the two calls disagreeing with each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config import CONFIG


@dataclass
class TrackedObject:
    track_id: int
    box: tuple[float, float, float, float]
    class_name: str
    confidence: float


class Tracker:
    def __init__(self, coco_model: YOLO, person_id: int, motorcycle_id: int):
        self.model = coco_model
        self.person_id = person_id
        self.motorcycle_id = motorcycle_id

    def update(self, frame: np.ndarray) -> list[TrackedObject]:
        results = self.model.track(
            frame,
            classes=[self.person_id, self.motorcycle_id],
            conf=min(
                CONFIG.detection.person_conf_threshold,
                CONFIG.detection.motorcycle_conf_threshold,
            ),
            persist=CONFIG.tracking.persist,
            tracker=CONFIG.tracking.tracker_yaml,
            verbose=False,
        )

        tracked: list[TrackedObject] = []
        result = results[0]

        if result.boxes.id is None:
            # No tracks yet (e.g. first frame, or nothing detected) — this is
            # normal, not an error.
            return tracked

        ids = result.boxes.id.int().tolist()
        for box, track_id in zip(result.boxes, ids):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = tuple(box.xyxy[0].tolist())
            tracked.append(
                TrackedObject(
                    track_id=track_id,
                    box=xyxy,
                    class_name=self.model.names[cls_id],
                    confidence=conf,
                )
            )
        return tracked

    def persons_only(self, tracked: list[TrackedObject]) -> dict[int, tuple[float, float, float, float]]:
        return {t.track_id: t.box for t in tracked if t.class_name == "person"}

    def motorcycles_only(self, tracked: list[TrackedObject]) -> list[tuple[float, float, float, float]]:
        return [t.box for t in tracked if t.class_name == "motorcycle"]
