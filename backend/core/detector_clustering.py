"""Candidate prefiltering and clustering helpers."""

from __future__ import annotations

import cv2
import numpy as np

from core.detector_config import (
    CLUSTER_CENTER_DISTANCE_RATIO,
    CLUSTER_IOM_THRESHOLD,
    CLUSTER_IOU_THRESHOLD,
    COLOR_HUE_TOLERANCE,
    CROSS_COLOR_CENTER_DISTANCE_RATIO,
    CROSS_COLOR_CLUSTER_IOM_THRESHOLD,
    CROSS_COLOR_CLUSTER_IOU_THRESHOLD,
    PREFILTER_NMS_IOU_THRESHOLD,
    PREFILTER_NMS_MIN_CANDIDATES,
    PROMOTED_PARENT_MIN_AREA_RATIO,
    PROMOTED_PARENT_MIN_VERIFICATION,
    PROMOTED_PARENT_OVERRIDE_MARGIN,
    RAW_PREFILTER_CENTER_DISTANCE_RATIO,
    RAW_PREFILTER_IOM_THRESHOLD,
    RAW_PREFILTER_IOU_THRESHOLD,
    RAW_PREFILTER_MIN_CANDIDATES,
)
from core.detector_models import CandidateHit
from core.detector_masks import _hue_distance


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


def _should_cluster(hit_a: CandidateHit, hit_b: CandidateHit) -> bool:
    """Decide whether two candidates describe the same physical object."""

    inter_area, iou, iom, center_distance = _bbox_metrics(hit_a.bbox, hit_b.bbox)
    if inter_area <= 0:
        return False

    center_a = _box_center(hit_a.bbox)
    center_b = _box_center(hit_b.bbox)
    centers_nested = _center_inside_box(center_a, hit_b.bbox) or _center_inside_box(center_b, hit_a.bbox)

    if (
        hit_a.dominant_hsv is not None
        and hit_b.dominant_hsv is not None
        and _hue_distance(hit_a.dominant_hsv[0], hit_b.dominant_hsv[0]) > (COLOR_HUE_TOLERANCE + 6)
    ):
        return (
            centers_nested
            and iou >= CROSS_COLOR_CLUSTER_IOU_THRESHOLD
            and iom >= CROSS_COLOR_CLUSTER_IOM_THRESHOLD
            and center_distance <= CROSS_COLOR_CENTER_DISTANCE_RATIO
        )

    if iou >= CLUSTER_IOU_THRESHOLD and center_distance <= CLUSTER_CENTER_DISTANCE_RATIO:
        return True

    if iom >= CLUSTER_IOM_THRESHOLD and centers_nested and center_distance <= (CLUSTER_CENTER_DISTANCE_RATIO * 1.15):
        return True

    return False


def _prefilter_candidates(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Use a conservative NMS only when the candidate set becomes very large."""

    if len(candidates) < PREFILTER_NMS_MIN_CANDIDATES:
        return candidates

    boxes = [list(hit.bbox) for hit in candidates]
    scores = [float(hit.verification_score or hit.match_score) for hit in candidates]

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.0,
        nms_threshold=PREFILTER_NMS_IOU_THRESHOLD,
    )
    if len(indices) == 0:
        return candidates

    keep = set(indices.flatten().tolist())
    return [candidate for idx, candidate in enumerate(candidates) if idx in keep]


def _raw_candidates_overlap_strongly(left: CandidateHit, right: CandidateHit) -> bool:
    """Detect duplicate raw peaks for the same template before expensive validation."""

    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False

    if iou >= RAW_PREFILTER_IOU_THRESHOLD:
        return True

    return (
        iom >= RAW_PREFILTER_IOM_THRESHOLD
        and center_distance <= RAW_PREFILTER_CENTER_DISTANCE_RATIO
    )


def _prefilter_raw_template_hits(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Drop near-identical raw candidates only inside the same template family member."""

    if len(candidates) < RAW_PREFILTER_MIN_CANDIDATES:
        return candidates

    grouped: dict[int, list[CandidateHit]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.template_id, []).append(candidate)

    filtered: list[CandidateHit] = []
    for template_hits in grouped.values():
        kept: list[CandidateHit] = []
        for candidate in sorted(template_hits, key=lambda hit: hit.match_score, reverse=True):
            if any(_raw_candidates_overlap_strongly(candidate, existing) for existing in kept):
                continue
            kept.append(candidate)
        filtered.extend(kept)

    filtered.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.match_score))
    return filtered


def _candidate_rank_key(hit: CandidateHit) -> tuple[float, float, float, int]:
    """Return the default winner ranking inside a cluster."""

    return (
        float(hit.verification_score),
        float(hit.color_similarity),
        float(hit.match_score),
        1 if hit.source == "pdf_text" else 0,
    )


def _select_cluster_winner(
    group_hits: list[CandidateHit],
    parent_ids_by_child: dict[int, set[int]],
) -> CandidateHit:
    """Pick one winner per cluster, preferring promoted fuller symbols over simpler cores."""

    base_winner = max(group_hits, key=_candidate_rank_key)
    override_candidates: list[CandidateHit] = []

    for hit in group_hits:
        child_id = hit.promoted_from_template_id
        if child_id is None:
            continue
        if hit.template_id not in parent_ids_by_child.get(child_id, set()):
            continue
        if hit.verification_score < PROMOTED_PARENT_MIN_VERIFICATION:
            continue

        child_hits = [candidate for candidate in group_hits if candidate.template_id == child_id]
        if not child_hits:
            continue

        best_child = max(child_hits, key=_candidate_rank_key)
        child_area = max(1, best_child.bbox[2] * best_child.bbox[3])
        parent_area = max(1, hit.bbox[2] * hit.bbox[3])
        if parent_area < child_area * PROMOTED_PARENT_MIN_AREA_RATIO:
            continue

        if hit.verification_score + PROMOTED_PARENT_OVERRIDE_MARGIN < best_child.verification_score:
            continue

        override_candidates.append(hit)

    if override_candidates:
        return max(override_candidates, key=_candidate_rank_key)

    return base_winner


def _cluster_candidates(
    candidates: list[CandidateHit],
    parent_ids_by_child: dict[int, set[int]] | None = None,
) -> list[CandidateHit]:
    """Cluster class-agnostic overlaps and keep one winner per physical place."""

    if not candidates:
        return []

    parent_ids_by_child = parent_ids_by_child or {}

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _should_cluster(candidates[left], candidates[right]):
                union(left, right)

    groups: dict[int, list[CandidateHit]] = {}
    for idx, candidate in enumerate(candidates):
        groups.setdefault(find(idx), []).append(candidate)

    winners: list[CandidateHit] = []
    for group_hits in groups.values():
        winners.append(_select_cluster_winner(group_hits, parent_ids_by_child))

    winners.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.verification_score))
    return winners
