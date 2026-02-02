# Crowd People Detector

Python project for detecting people in a video and drawing bounding boxes with class name and confidence.

This repository is a solution template for the test assignment: **detect people in `crowd.mp4`, draw detections, save annotated video, and provide analysis & improvement steps.**

## Features

- Reads a video file using OpenCV.
- Runs **Ultralytics YOLO** inference on each frame.
- Filters detections to class **person**.
- Draws **thin** bounding boxes + label `person <conf>` to avoid heavy occlusion.
- Saves annotated video to MP4 (`mp4v` codec for cross-platform compatibility).
- Prints basic runtime stats (processed frames, effective FPS, avg people/frame).

## Requirements

- Python **3.8+** (tested target: 3.8.10)
- OS: Linux / macOS / Windows

> **Note about Torch:** `ultralytics` depends on `torch`. On some systems you may need to install a matching `torch` wheel first.
> If `pip install -r requirements.txt` fails on your machine, install PyTorch following the official instructions for your OS/CUDA,
> then re-run the requirements installation.

## Installation

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

## Run

Put your input video near the project root (or provide a path).

```bash
python main.py --input crowd.mp4 --output output_annotated.mp4
```

Optional flags:

- `--model yolov8n.pt` (default) or `yolov8s.pt` / `yolov8m.pt` for better quality
- `--conf 0.25` confidence threshold
- `--device cpu` or `--device 0` for GPU
- `--show` to preview frames (press `q` to stop)
- `--max-frames 300` to test on a subset

## Output

The resulting file is an MP4 video with detected people outlined and labeled.

## Quality analysis & improvement ideas (high-level)

1. **Use a larger model** (`yolov8s.pt`/`yolov8m.pt`) or a crowd-focused detector.
2. **Tune thresholds** (`--conf`, `--iou`) for the crowd density and camera viewpoint.
3. **Track across frames** (ByteTrack/DeepSORT) to stabilize detections and reduce flicker.
4. **Improve resolution**: upscaling or tiling/patch inference for far-away small people.
5. **Domain adaptation**: fine-tune on a small labeled subset from `crowd.mp4`.
6. **Post-processing**: geometric priors (min/max box size by perspective), NMS variants, soft-NMS.

See `REPORT.md` for a more structured discussion.
