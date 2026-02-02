"""
Для ввода/вывода видео.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


class VideoReader:

    def __init__(self, path: Path) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {path}")

        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size: Tuple[int, int] = (width, height)

    def __iter__(self) -> Iterator[np.ndarray]:
        return self

    def __next__(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise StopIteration
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoWriter:

    def __init__(self, path: Path, fps: float, frame_size: Tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        # Выбираем mp4v в качестве кроссплатформенного по умолчанию.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(path),
            fourcc,
            float(fps),
            (int(frame_size[0]), int(frame_size[1])),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Не удалось открыть программу записи видео для: {path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        """Запишите кадр в выходное видео"""
        self._writer.write(frame_bgr)

    def close(self) -> None:
        """Освободите ресурсы для записи"""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
