"""
Main entry point. Wires together (in order):
  video.py       -> read frames
  tracker.py     -> track persons & motorcycles (ByteTrack, via YOLO.track)
  detector.py    -> run the helmet model on the same frame
  association.py -> decide Helmet / No Helmet per tracked rider
  utils.py       -> draw results

Run:
  python3 main.py                      # webcam
  python3 main.py --source path.mp4    # video file
  python3 main.py --source path.mp4 --save output.mp4
"""
from __future__ import annotations

import argparse
import math

import cv2

from association import associate_helmets_to_riders, link_riders_to_motorcycles
from config import CONFIG
from detector import Detector, HelmetModelNotLoaded
from tracker import Tracker
from utils import (
    COLOR_HELMET,
    COLOR_MOTORCYCLE,
    COLOR_NO_HELMET,
    draw_box,
    draw_fps,
)
from video import FPSCounter, VideoSink, VideoSource


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Helmet detection + rider tracking")
    parser.add_argument(
        "--source",
        default=0,
        help="Video source: webcam index (default 0) or path to a video file",
    )
    parser.add_argument(
        "--save", default=None, help="Optional path to save annotated output video"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't open a preview window (useful when running headless/over SSH)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # argparse gives webcam index as a string by default; convert if numeric
    source = int(args.source) if str(args.source).isdigit() else args.source

    detector = Detector(load_helmet_model=True)
    tracker = Tracker(detector.coco_model, detector.person_id, detector.motorcycle_id)

    if not detector.helmet_model_ready:
        print(
            "\n[main] Running WITHOUT helmet detection — every rider will show "
            "as 'No Helmet' until you place your Roboflow weights at "
            f"'{CONFIG.paths.helmet_model}'.\n"
        )

    video_source = VideoSource(source)
    frame_diagonal = math.hypot(video_source.width, video_source.height)

    sink = None
    if args.save:
        sink = VideoSink(args.save, video_source.width, video_source.height, video_source.fps)

    fps_counter = FPSCounter()

    try:
        for frame in video_source.frames():
            tracked = tracker.update(frame)
            persons = tracker.persons_only(tracked)
            motorcycles = tracker.motorcycles_only(tracked)

            if detector.helmet_model_ready:
                try:
                    helmet_dets = detector.detect_helmets(frame)
                except HelmetModelNotLoaded:
                    helmet_dets = []
                # Resolved once at startup in detector.py — never re-guessed here
                helmet_class_name = detector.helmet_model.names[detector.helmet_id]
                rider_results = associate_helmets_to_riders(
                    persons, helmet_dets, helmet_class_name
                )
            else:
                # No helmet model available yet — every rider is "unknown/no helmet"
                from association import RiderResult

                rider_results = {
                    tid: RiderResult(
                        track_id=tid,
                        person_box=box,
                        has_helmet=False,
                        helmet_confidence=None,
                        nearest_motorcycle_box=None,
                    )
                    for tid, box in persons.items()
                }

            link_riders_to_motorcycles(rider_results, motorcycles, frame_diagonal)

            # --- draw ---
            for moto_box in motorcycles:
                draw_box(frame, moto_box, "Motorcycle", COLOR_MOTORCYCLE)

            for result in rider_results.values():
                label = f"ID {result.track_id}: {'Helmet' if result.has_helmet else 'No Helmet'}"
                color = COLOR_HELMET if result.has_helmet else COLOR_NO_HELMET
                draw_box(frame, result.person_box, label, color)

            fps = fps_counter.tick()
            draw_fps(frame, fps)

            if sink:
                sink.write(frame)

            if not args.no_display:
                cv2.imshow("Helmet Detection + Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        video_source.release()
        if sink:
            sink.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
