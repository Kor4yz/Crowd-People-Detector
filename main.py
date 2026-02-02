#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2

from crowd_detector.detector import PeopleDetector
from crowd_detector.video_io import VideoReader, VideoWriter
from crowd_detector.visualization import draw_detections


def build_arg_parser() -> argparse.ArgumentParser:
    #Create CLI argument parser
    parser = argparse.ArgumentParser(
        description="Detect people on a video and save an annotated copy."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video file (e.g., crowd.mp4).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_annotated.mp4",
        help="Path to output annotated video.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="yolov8n.pt",
        help=(
            "Ultralytics YOLO model name or path (e.g., yolov8n.pt, yolov8s.pt, "
            "or /path/to/weights.pt). If not present locally, Ultralytics will try "
            "to download it."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference: 'cpu' or e.g. '0' for CUDA GPU.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process at most N frames (0 = process full video).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window (press 'q' to stop).",
    )
    parser.add_argument("--merge-iou", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument("--tile", type=int, default=1, help="1=off, 2 or 3 for tiled inference")
    parser.add_argument("--overlap", type=float, default=0.15, help="Tile overlap, e.g. 0.10..0.25")

    return parser


def main() -> int:
    #Точка входа в программу
    parser = build_arg_parser()
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"ERROR: входное видео не найдено: {in_path}", file=sys.stderr)
        return 2

    detector = PeopleDetector(
        model_name_or_path=args.model,  # <-- ОБЯЗАТЕЛЬНО
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        max_det=args.max_det,
        tile=args.tile,
        overlap=args.overlap,
        merge_iou=args.merge_iou,
    )

    reader = VideoReader(in_path)
    writer = VideoWriter(out_path, reader.fps, reader.frame_size)

    total_frames = 0
    total_people = 0
    t0 = perf_counter()

    try:
        for frame in reader:
            total_frames += 1
            dets = detector.predict(frame)
            total_people += len(dets)

            annotated = draw_detections(frame, dets)

            writer.write(annotated)

            if args.show:
                cv2.imshow("detections", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and total_frames >= args.max_frames:
                break
    finally:
        reader.close()
        writer.close()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = perf_counter() - t0
    fps = total_frames / elapsed if elapsed > 0 else 0.0
    avg_people = total_people / total_frames if total_frames else 0.0

    print("Done.")
    print(f"Frames processed: {total_frames}")
    print(f"Avg people/frame: {avg_people:.2f}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Effective FPS: {fps:.2f}")
    print(f"Output saved to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


