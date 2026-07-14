# Helmet Detection + Rider Tracking

Real-time detection of motorcycles/riders (COCO YOLO11) + tracking (ByteTrack)
+ helmet classification (your Roboflow model), with per-rider "Helmet" /
"No Helmet" labels that persist across frames via track ID.

## Project structure
```
config.py       - all thresholds/paths/settings, single source of truth
utils.py        - geometry (IoU, head region) + drawing, no model code
detector.py     - loads both models, runs per-frame inference
tracker.py      - ByteTrack wrapper, keeps rider IDs consistent
association.py  - pure geometry: matches helmets to riders' head regions
video.py        - webcam/file input, optional output writer, FPS counter
main.py         - wires everything together, the file you actually run
test_model.py   - Stage 1 sanity check for the COCO model (already passing)
download_helmet_model.py - one-time script to pull your helmet weights
requirements.txt
models/yolo11n.pt        - already downloaded, COCO-pretrained (person, motorcycle)
models/best.pt            - YOUR helmet weights go here (not included)
```

## Setup (run these on your own machine)

```bash
pip install -r requirements.txt
```

### 1. Get the helmet model
```bash
export ROBOFLOW_API_KEY="your_key_here"
python3 download_helmet_model.py
```
This tries the "Seatbelt Helmet Detection" project by default. If it has no
hosted trained checkpoint, the script tells you exactly how to train one in
~5 minutes with `model.train(...)`, or you can point `--workspace/--project`
at a different Universe project that shows a "Model" badge.

Once done, confirm `models/best.pt` exists.

### 2. Verify the helmet model before running the full pipeline
```bash
python3 test_model.py
```
Extend this if you want (it currently checks the COCO model; add the same
`verify_model()` + `run_sample_inference()` calls for `models/best.pt` using
one of your own helmet photos — this is the one validation step I couldn't
do myself, since my sandbox can't reach Roboflow).

### 3. Run it
```bash
python3 main.py                        # webcam
python3 main.py --source clip.mp4      # video file
python3 main.py --source clip.mp4 --save output.mp4   # save annotated output
python3 main.py --no-display           # headless / over SSH
```
Press `q` to quit the preview window.

## What's already verified (done in the build sandbox)
- COCO model loads, `person`→id 0 and `motorcycle`→id 3 resolved dynamically
- Real inference confirmed on a sample image (4 people detected, 0.86-0.94 conf)
- ByteTrack assigns and holds consistent track IDs
- Association logic unit-tested on synthetic boxes (correct helmet-to-rider
  matching, correct "no helmet" fallback, correct motorcycle linking)
- Full pipeline wiring smoke-tested end-to-end on a static image

## What's NOT verified yet (do this first)
- Real-world accuracy of the helmet model itself — untested by me because
  Roboflow isn't reachable from the build sandbox. Test it on a few of your
  own images before trusting it on live video.

## Tuning
All thresholds live in `config.py`:
- `detection.helmet_conf_threshold` — raise if you get false positives, lower if it misses helmets
- `association.head_region_fraction` — how much of the person box counts as "head" (default top 30%)
- `association.helmet_iou_threshold` — how much overlap counts as a match (default 0.10, fairly lenient)
