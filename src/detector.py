"""
Оболочка модели, используемая для определения личности.

В реализации используются модели Ultralytics YOLO и фильтруются обнаруженные данные для отнесения к классу "личность".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Detection:
    #Единичный результат обнаружения
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]


class PeopleDetector:
    """
        Детектор людей на базе Ultralytics YOLO.
        - Фильтрует обнаружения по классу "персона".
        - Устанавливает пороговые значения доверия/долговой расписки с помощью конструктора.
    """

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        imgsz: int = 1280,
        max_det: int = 2000,
        merge_iou: float = 0.30,
        tile: int = 1,
        overlap: float = 0.15,
    ) -> None:
        """
            Инициализация детектора и загрузка модели

            Аргументы:
                    model_name_or_path: имя модели или путь к файловой системе
                    conf: порог достоверности
                    iou: порог ввода-вывода для NMS
                    устройство: строка индекса процессора или устройства CUDA (например, "0")
                """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required. Install dependencies with:\n"
                "  pip install -r requirements.txt"
            ) from exc

        self._conf = float(conf)
        self._iou = float(iou)
        self._device = device
        from ultralytics import YOLO
        self._model = YOLO(model_name_or_path)
        self._imgsz = imgsz
        self._max_det = max_det
        self._merge_iou = merge_iou
        self._tile = tile
        self._overlap = overlap

        # Имена классов моделей кэширования
        self._names = getattr(self._model, "names", None) or {}
        self._person_ids = {
            cls_id for cls_id, name in self._names.items() if str(name).lower() == "person"
        }
        # Запасной вариант для обычного отображения
        if not self._person_ids:
            self._person_ids = {0}

    def _predict_single_roi(
            self,
            frame_bgr: np.ndarray,
            roi: Tuple[int, int, int, int],
            conf_override: float,
    ) -> List[Tuple[int, int, int, int, float]]:
        x1, y1, x2, y2 = roi
        crop = frame_bgr[y1:y2, x1:x2]

        results = self._model.predict(
            source=crop,
            conf=float(conf_override),
            iou=self._iou,
            imgsz=self._imgsz,
            max_det=self._max_det,
            device=self._device,
            classes=list(self._person_ids),
            verbose=False,
        )

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        boxes = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()

        # сместить в координаты полного кадра
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1

        out = []
        for (bx1, by1, bx2, by2), c in zip(boxes, confs):
            out.append((int(bx1), int(by1), int(bx2), int(by2), float(c)))
        return out

    def _predict_tiled_roi(
            self,
            frame_bgr: np.ndarray,
            roi: Tuple[int, int, int, int],
            conf_override: float,
    ) -> List[Tuple[int, int, int, int, float]]:
        x1, y1, x2, y2 = roi
        crop = frame_bgr[y1:y2, x1:x2]

        # временно переопределим conf
        old_conf = self._conf
        self._conf = float(conf_override)

        dets = self._predict_tiled(crop)

        self._conf = old_conf

        # сместить dets в координаты полного кадра
        out = []
        for bx1, by1, bx2, by2, c in dets:
            out.append((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, c))
        return out

    def _nms_xyxy(
            self,
            boxes: np.ndarray,  # (N,4) xyxy
            scores: np.ndarray,  # (N,)
            iou_thr: float
    ) -> np.ndarray:
        """Возвращает индексы сохраненных ящиков после NMS."""
        if len(boxes) == 0:
            return np.array([], dtype=int)

        x1 = boxes[:, 0].astype(float)
        y1 = boxes[:, 1].astype(float)
        x2 = boxes[:, 2].astype(float)
        y2 = boxes[:, 3].astype(float)

        areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1.0)
            h = np.maximum(0.0, yy2 - yy1 + 1.0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)

            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=int)

    def _stitch_vertical_person_parts(
            self,
            dets: List[Tuple[int, int, int, int, float]],
            x_overlap_thr: float = 0.60,
            max_gap_px: int = 35,
            width_ratio_thr: float = 2.2,
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Склеивает боксы одного человека, если модель дала два бокса (верх и низ).
        У таких боксов часто маленький IoU, но:
          - сильный overlap по X
          - небольшой вертикальный разрыв
        """
        if not dets:
            return dets

        # сортируем по уверенности, чтобы "сильные" поглощали "слабые фрагменты"
        dets = sorted(dets, key=lambda x: x[4], reverse=True)
        used = [False] * len(dets)
        out = []

        def x_overlap_ratio(a, b) -> float:
            ax1, _, ax2, _, _ = a
            bx1, _, bx2, _, _ = b
            inter = max(0, min(ax2, bx2) - max(ax1, bx1))
            minw = max(1, min(ax2 - ax1, bx2 - bx1))
            return inter / minw

        for i, a in enumerate(dets):
            if used[i]:
                continue

            ax1, ay1, ax2, ay2, ac = a
            merged = [ax1, ay1, ax2, ay2, ac]
            used[i] = True

            changed = True
            while changed:
                changed = False
                for j, b in enumerate(dets):
                    if used[j]:
                        continue

                    bx1, by1, bx2, by2, bc = b

                    # overlap по X
                    xr = x_overlap_ratio((merged[0], merged[1], merged[2], merged[3], merged[4]), b)
                    if xr < x_overlap_thr:
                        continue

                    # сходство ширины (чтобы не склеить двух разных людей рядом)
                    mw = max(1, merged[2] - merged[0])
                    bw = max(1, bx2 - bx1)
                    if max(mw, bw) / min(mw, bw) > width_ratio_thr:
                        continue

                    # вертикальный разрыв (если боксы "столбиком")
                    top = min(merged[1], by1)
                    bottom = max(merged[3], by2)

                    # gap между низом верхнего и верхом нижнего
                    gap = 0
                    if by1 > merged[3]:
                        gap = by1 - merged[3]
                    elif merged[1] > by2:
                        gap = merged[1] - by2

                    if gap > max_gap_px:
                        continue

                    # склеиваем
                    merged[0] = min(merged[0], bx1)
                    merged[1] = min(merged[1], by1)
                    merged[2] = max(merged[2], bx2)
                    merged[3] = max(merged[3], by2)
                    merged[4] = max(merged[4], bc)
                    used[j] = True
                    changed = True

            out.append((int(merged[0]), int(merged[1]), int(merged[2]), int(merged[3]), float(merged[4])))

        return out

    def _dedupe_by_center(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            k: float = 0.35,  # насколько близки центры
            size_ratio_thr: float = 0.50,  # насколько похожи размеры
    ) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=int)

        order = scores.argsort()[::-1]
        keep = []
        suppressed = np.zeros(len(boxes), dtype=bool)

        w = (boxes[:, 2] - boxes[:, 0]).clip(min=1).astype(float)
        h = (boxes[:, 3] - boxes[:, 1]).clip(min=1).astype(float)
        diag = np.sqrt(w * w + h * h)

        cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

        area = w * h

        for i in order:
            if suppressed[i]:
                continue
            keep.append(i)

            # расстояние между центрами в пикселях
            dx = cx - cx[i]
            dy = cy - cy[i]
            dist = np.sqrt(dx * dx + dy * dy)

            # порог по дистанции: относительно диагонали большего бокса
            dist_thr = k * np.maximum(diag, diag[i])

            # похожесть размеров
            size_ratio = np.minimum(area, area[i]) / np.maximum(area, area[i])

            dup = (dist < dist_thr) & (size_ratio > size_ratio_thr)
            suppressed |= dup

        return np.array(keep, dtype=int)

    def _predict_tiled(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        h, w = frame_bgr.shape[:2]
        t = int(self._tile)
        overlap = float(self._overlap)

        # шаг тайла с overlap
        tile_w = int(np.ceil(w / t))
        tile_h = int(np.ceil(h / t))
        step_x = int(tile_w * (1.0 - overlap))
        step_y = int(tile_h * (1.0 - overlap))
        step_x = max(1, step_x)
        step_y = max(1, step_y)

        all_boxes = []
        all_scores = []

        y0 = 0
        while y0 < h:
            x0 = 0
            y1 = min(h, y0 + tile_h)
            while x0 < w:
                x1 = min(w, x0 + tile_w)
                crop = frame_bgr[y0:y1, x0:x1]

                results = self._model.predict(
                    source=crop,
                    conf=self._conf,
                    iou=self._iou,
                    imgsz=self._imgsz,
                    max_det=self._max_det,
                    device=self._device,
                    classes=list(self._person_ids),
                    verbose=False,
                )

                r0 = results[0]
                if r0.boxes is not None and len(r0.boxes) > 0:
                    boxes = r0.boxes.xyxy.cpu().numpy()
                    confs = r0.boxes.conf.cpu().numpy()

                    # перенос координат в глобальные

                    boxes[:, [0, 2]] += x0
                    boxes[:, [1, 3]] += y0

                    mx = int(tile_w * overlap * 0.5)
                    my = int(tile_h * overlap * 0.5)

                    core_x1 = x0 + mx
                    core_y1 = y0 + my
                    core_x2 = x1 - mx
                    core_y2 = y1 - my

                    if core_x2 <= core_x1:
                        core_x1, core_x2 = x0, x1
                    if core_y2 <= core_y1:
                        core_y1, core_y2 = y0, y1

                    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0

                    mask = (cx >= core_x1) & (cx < core_x2) & (cy >= core_y1) & (cy < core_y2)

                    boxes = boxes[mask]
                    confs = confs[mask]

                    if len(boxes) > 0:
                        all_boxes.append(boxes)
                        all_scores.append(confs)

                x0 += step_x
            y0 += step_y

        if not all_boxes:
            return []

        boxes = np.vstack(all_boxes)
        scores = np.concatenate(all_scores)

        # Склейка дублей между тайлами
        keep = self._nms_xyxy(boxes, scores, iou_thr=self._merge_iou)
        keep2 = self._dedupe_by_center(boxes[keep], scores[keep], k=0.40, size_ratio_thr=0.45)
        keep = keep[keep2]

        dets = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            c = scores[i]
            dets.append((int(x1), int(y1), int(x2), int(y2), float(c)))

        dets = self._stitch_vertical_person_parts(dets)

        filtered = []
        for x1, y1, x2, y2, c in dets:
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            ar = bh / bw

            # базовые мягкие ограничения
            if bh < 8 or bw < 4:
                continue
            if ar < 1.0 or ar > 6.0:
                continue

            # внизу (витрины/картины) требуем больше уверенность и минимальную высоту
            if y2 > int(h * 0.55):
                if c < 0.55:
                    continue
                if bh < 40:
                    continue

            filtered.append((x1, y1, x2, y2, c))

        return filtered

    def predict(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        h, w = frame_bgr.shape[:2]

        # Граница зон
        y_split = int(h * 0.55)

        # Настройки зон
        conf_far = float(self._conf)
        conf_near = max(0.60, float(self._conf))


        dets_bottom = self._predict_single_roi(
            frame_bgr,
            roi=(0, y_split, w, h),
            conf_override=conf_near,
        )

        # Жёсткий фильтр только для низа
        bottom_filtered = []
        for x1, y1, x2, y2, c in dets_bottom:
            bh = y2 - y1
            bw = x2 - x1
            if bh < 45:
                continue
            if c < conf_near:
                continue
            ar = bh / max(1, bw)
            if ar < 1.1 or ar > 6.0:
                continue
            bottom_filtered.append((x1, y1, x2, y2, c))
        dets_bottom = bottom_filtered

        if self._tile and self._tile > 1:
            dets_top = self._predict_tiled_roi(
                frame_bgr,
                roi=(0, 0, w, y_split),
                conf_override=conf_far,
            )
        else:
            dets_top = self._predict_single_roi(
                frame_bgr,
                roi=(0, 0, w, y_split),
                conf_override=conf_far,
            )

        top_filtered = []
        for x1, y1, x2, y2, c in dets_top:
            bh = y2 - y1
            bw = x2 - x1
            if bh < 8 or bw < 4:
                continue
            ar = bh / max(1, bw)
            if ar < 0.9 or ar > 7.0:
                continue
            top_filtered.append((x1, y1, x2, y2, c))
        dets_top = top_filtered

        dets = dets_top + dets_bottom

        if not dets:
            return dets

        boxes = np.array([d[:4] for d in dets], dtype=float)
        scores = np.array([d[4] for d in dets], dtype=float)

        keep = self._nms_xyxy(boxes, scores, iou_thr=float(self._merge_iou))

        out = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            c = scores[i]
            out.append((int(x1), int(y1), int(x2), int(y2), float(c)))

        return out

    def _predict_single(self, frame_bgr: np.ndarray, conf_override: float | None = None) -> List[
        Tuple[int, int, int, int, float]]:
        conf = self._conf if conf_override is None else float(conf_override)

        results = self._model.predict(
            source=frame_bgr,
            conf=conf,
            iou=self._iou,
            imgsz=self._imgsz,
            max_det=self._max_det,
            device=self._device,
            classes=list(self._person_ids),
            verbose=False,
        )

        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []

        boxes = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()

        dets = []
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            dets.append((int(x1), int(y1), int(x2), int(y2), float(c)))
        return dets

    def _final_dedupe(
            self,
            dets: List[Tuple[int, int, int, int, float]],
            iou_thr: float = 0.25,
            k: float = 0.45,
            size_ratio_thr: float = 0.45,
    ) -> List[Tuple[int, int, int, int, float]]:
        if not dets:
            return dets

        boxes = np.array([d[:4] for d in dets], dtype=float)
        scores = np.array([d[4] for d in dets], dtype=float)

        keep = self._nms_xyxy(boxes, scores, iou_thr=iou_thr)
        boxes = boxes[keep]
        scores = scores[keep]

        keep2 = self._dedupe_by_center(boxes, scores, k=k, size_ratio_thr=size_ratio_thr)
        boxes = boxes[keep2]
        scores = scores[keep2]

        out = []
        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = b
            out.append((int(x1), int(y1), int(x2), int(y2), float(s)))
        return out

