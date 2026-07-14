"""
Video input/output only. No detection, tracking, or drawing logic here.
"""
from __future__ import annotations

import time
from typing import Iterator

import cv2
import numpy as np


class VideoSource:
    def __init__(self, source: str | int = 0):
        """source: 0 (default webcam), another int for a different camera
        index, or a file path / RTSP URL string."""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self.cap.release()


class VideoSink:
    """Optional — only used if the user wants to save annotated output."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()


class FPSCounter:
    """Rolling FPS estimate based on real wall-clock time between frames,
    not just frames-processed-per-second-of-video, so it reflects actual
    pipeline performance on this machine."""

    def __init__(self, smoothing: float = 0.9):
        self._last_time: float | None = None
        self._fps: float = 0.0
        self._smoothing = smoothing

    def tick(self) -> float:
        now = time.time()
        if self._last_time is not None:
            instant_fps = 1.0 / max(now - self._last_time, 1e-6)
            self._fps = (
                self._smoothing * self._fps + (1 - self._smoothing) * instant_fps
                if self._fps > 0
                else instant_fps
            )
        self._last_time = now
        return self._fps
