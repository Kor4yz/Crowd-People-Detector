# Report: detection quality and improvement steps

## What the current baseline does

- Runs Ultralytics YOLO on each frame and keeps only **person** detections.
- Draws thin bounding boxes and a small label (`person <conf>`).

This is a strong baseline because YOLO models pre-trained on COCO usually detect people well,
but crowd videos often expose typical failure modes.

## Expected failure modes in a crowded scene

1. **Occlusions & overlaps**
   - People partially hidden by others often produce missed detections.
   - Dense overlaps can lead to NMS suppressing true positives.

2. **Small objects**
   - Far-away people can be very small in pixels, falling below the detector’s effective scale.

3. **Motion blur / compression artifacts**
   - Fast motion, low light, or strong compression can reduce confidence and increase misses.

4. **False positives**
   - Posters, mannequins, reflections, and similar patterns can be detected as people.

## Practical steps to improve quality (in order)

1. **Pick a stronger model**
   - Try `yolov8s.pt` or `yolov8m.pt` first. Larger models usually improve recall on small/occluded people.
   - If you must stay CPU-only, balance quality vs speed (e.g., `yolov8s` may be slower).

2. **Threshold tuning**
   - For crowds, lowering `--conf` can increase recall but may introduce false positives.
   - Adjust `--iou` to reduce suppression of nearby people (sometimes slightly higher IoU helps keep more boxes).

3. **Increase effective resolution**
   - If the video is low-res, consider:
     - running inference on an upscaled frame (simple resize),
     - or tiling the frame into overlapping patches (better for very small people).
   - Tiling is more compute-heavy but can dramatically increase recall for small objects.

4. **Temporal consistency via tracking**
   - Add a tracker (e.g., ByteTrack/DeepSORT) on top of detections.
   - Benefits:
     - smoother boxes (less flicker),
     - ability to filter short-lived false positives,
     - better perceived quality even if raw detector quality is unchanged.

5. **Light fine-tuning**
   - Label ~200–500 frames (or crops) from the target video and fine-tune the detector.
   - This helps adapt to camera angle, clothing distribution, lighting, and background.

6. **Smart post-processing**
   - Apply perspective-aware box size constraints (if the camera is fixed).
   - Consider soft-NMS or weighted boxes fusion in dense scenes.

## How to evaluate improvements

- Sample and manually review a fixed set of frames (e.g., every N-th frame).
- Track:
  - missed people (false negatives),
  - ghost detections (false positives),
  - stability over time (flicker).
- If time allows, annotate a small test set and compute mAP/precision/recall.
