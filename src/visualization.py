"""
Для отрисовки обнаружений на фреймах
"""

from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from crowd_detector.detector import Detection


def _clamp_box(xyxy: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def draw_detections(frame_bgr, dets):
    h, w = frame_bgr.shape[:2]
    img = frame_bgr.copy()

    for det in dets:
        if hasattr(det, "xyxy"):
            xyxy = det.xyxy
            conf = getattr(det, "conf", None)
        else:
            if len(det) >= 4:
                xyxy = det[:4]
                conf = det[4] if len(det) >= 5 else None
            else:
                continue

        x1, y1, x2, y2 = _clamp_box(xyxy, w, h)

        # Тонкий прямоугольник, чтобы не закрывать сцену слишком сильно
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"person {conf:.2f}" if conf is not None else "person"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        ty = y1 - 6
        if ty - th - baseline < 0:
            ty = y1 + th + baseline + 6

        x2_bg = min(w - 1, x1 + tw + 6)
        y1_bg = max(0, ty - th - baseline - 3)
        y2_bg = min(h - 1, ty + baseline + 3)

        cv2.rectangle(img, (x1, y1_bg), (x2_bg, y2_bg), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 3, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return img
