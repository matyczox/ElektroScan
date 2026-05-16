"""Shared geometry helpers for detector candidates."""

from __future__ import annotations

import numpy as np


def _bbox_metrics(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> tuple[int, float, float, float]:
    """Return intersection area, IoU, IoM and normalized center distance."""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    union_area = area_a + area_b - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    iom = inter_area / min(area_a, area_b)

    center_a = (ax + aw / 2.0, ay + ah / 2.0)
    center_b = (bx + bw / 2.0, by + bh / 2.0)
    center_distance = float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    ref_distance = max(1.0, min(np.hypot(aw, ah), np.hypot(bw, bh)))
    normalized_center_distance = center_distance / ref_distance

    return inter_area, iou, iom, normalized_center_distance


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the geometric center of a bbox."""

    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _center_inside_box(
    center: tuple[float, float],
    box: tuple[int, int, int, int],
    margin_ratio: float = 0.05,
) -> bool:
    """Check whether a center point lies inside a bbox with a small safety margin."""

    x, y = center
    bx, by, bw, bh = box
    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio
    return (bx - pad_x) <= x <= (bx + bw + pad_x) and (by - pad_y) <= y <= (by + bh + pad_y)


def _axis_overlap_fraction(
    start_a: int,
    length_a: int,
    start_b: int,
    length_b: int,
) -> float:
    overlap = max(0, min(start_a + length_a, start_b + length_b) - max(start_a, start_b))
    return overlap / max(1, min(length_a, length_b))


def _axis_gap(
    start_a: int,
    length_a: int,
    start_b: int,
    length_b: int,
) -> int:
    return max(0, max(start_a, start_b) - min(start_a + length_a, start_b + length_b))
