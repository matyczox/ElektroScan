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
    GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP,
    GRAY_FULLER_SYMBOL_MIN_AREA_RATIO,
    GRAY_FULLER_SYMBOL_MIN_COVERAGE,
    GRAY_FULLER_SYMBOL_MIN_PURITY,
    PREFILTER_NMS_IOU_THRESHOLD,
    PREFILTER_NMS_MIN_CANDIDATES,
    PROMOTED_PARENT_MIN_AREA_RATIO,
    PROMOTED_PARENT_MIN_VERIFICATION,
    PROMOTED_PARENT_OVERRIDE_MARGIN,
    RAW_PREFILTER_CENTER_DISTANCE_RATIO,
    RAW_PREFILTER_IOM_THRESHOLD,
    RAW_PREFILTER_IOU_THRESHOLD,
    RAW_PREFILTER_MIN_CANDIDATES,
    TEXT_LABEL_FULLER_AREA_RATIO,
    TEXT_LABEL_FULLER_MAX_CONTENT_DROP,
    TEXT_LABEL_FULLER_MAX_VERIFICATION_DROP,
)
from core.detector_masks import _hue_distance
from core.detector_models import CandidateHit


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
    centers_nested = _center_inside_box(center_a, hit_b.bbox) or _center_inside_box(
        center_b, hit_a.bbox
    )

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

    if (
        iom >= CLUSTER_IOM_THRESHOLD
        and centers_nested
        and center_distance <= (CLUSTER_CENTER_DISTANCE_RATIO * 1.15)
    ):
        return True

    return False


def _prefilter_candidates(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Use a conservative NMS only when the candidate set becomes very large."""

    if len(candidates) < PREFILTER_NMS_MIN_CANDIDATES:
        return candidates

    boxes = [list(hit.bbox) for hit in candidates]
    scores = [
        float(
            hit.content_score
            if hit.is_text_label and hit.content_score > 0.0
            else hit.verification_score or hit.match_score
        )
        for hit in candidates
    ]

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

    if left.is_text_label or right.is_text_label:
        return False

    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False

    if iou >= RAW_PREFILTER_IOU_THRESHOLD:
        return True

    return (
        iom >= RAW_PREFILTER_IOM_THRESHOLD
        and center_distance <= RAW_PREFILTER_CENTER_DISTANCE_RATIO
    )


def _should_keep_gray_scale_alternative(
    candidate: CandidateHit,
    existing: CandidateHit,
    kept: list[CandidateHit],
) -> bool:
    """Keep a nearby gray scale variant until full geometry validation decides."""

    if candidate.dominant_hsv is not None or existing.dominant_hsv is not None:
        return False
    if candidate.is_text_label or existing.is_text_label:
        return False
    if candidate.template_id != existing.template_id:
        return False
    if abs(float(candidate.scale) - float(existing.scale)) < 0.09:
        return False

    cx, cy = _box_center(candidate.bbox)
    candidate_diag = max(1.0, float(np.hypot(candidate.bbox[2], candidate.bbox[3])))
    nearby_alternatives = 0
    for other in kept:
        if other.template_id != candidate.template_id:
            continue
        ox, oy = _box_center(other.bbox)
        if float(np.hypot(cx - ox, cy - oy)) <= candidate_diag * 0.25:
            nearby_alternatives += 1

    return nearby_alternatives < 3


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
            overlapping_existing = next(
                (
                    existing
                    for existing in kept
                    if _raw_candidates_overlap_strongly(candidate, existing)
                ),
                None,
            )
            if overlapping_existing is not None and not _should_keep_gray_scale_alternative(
                candidate,
                overlapping_existing,
                kept,
            ):
                continue
            kept.append(candidate)
        filtered.extend(kept)

    filtered.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.match_score))
    return filtered


def _suppress_same_template_ghosts(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Remove weak interior ghosts surrounded by stronger hits of the same template."""

    grouped: dict[int, list[CandidateHit]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.template_id, []).append(candidate)

    suppressed: set[int] = set()
    index_by_identity = {id(candidate): idx for idx, candidate in enumerate(candidates)}

    for template_hits in grouped.values():
        if len(template_hits) < 4:
            continue

        for candidate in template_hits:
            if candidate.context_purity >= 0.34:
                continue

            cx, cy = _box_center(candidate.bbox)
            candidate_diag = max(1.0, float(np.hypot(candidate.bbox[2], candidate.bbox[3])))
            stronger: list[CandidateHit] = []
            overlapping_stronger = 0

            for other in template_hits:
                if other is candidate:
                    continue
                if other.verification_score < candidate.verification_score + 0.08:
                    continue

                ox, oy = _box_center(other.bbox)
                if float(np.hypot(cx - ox, cy - oy)) > candidate_diag * 1.65:
                    continue

                inter_area, _, _, _ = _bbox_metrics(candidate.bbox, other.bbox)
                if inter_area > 0:
                    overlapping_stronger += 1
                stronger.append(other)

            if len(stronger) < 3 or overlapping_stronger < 2:
                continue

            min_x = min(hit.bbox[0] for hit in stronger)
            min_y = min(hit.bbox[1] for hit in stronger)
            max_x = max(hit.bbox[0] + hit.bbox[2] for hit in stronger)
            max_y = max(hit.bbox[1] + hit.bbox[3] for hit in stronger)
            union_box = (min_x, min_y, max_x - min_x, max_y - min_y)
            if _center_inside_box((cx, cy), union_box, margin_ratio=0.08):
                suppressed.add(index_by_identity[id(candidate)])

    if not suppressed:
        return candidates

    return [candidate for idx, candidate in enumerate(candidates) if idx not in suppressed]


def _candidate_rank_key(hit: CandidateHit) -> tuple[float, float, float, int]:
    """Return the default winner ranking inside a cluster."""

    if hit.is_text_label:
        return (
            float(hit.content_score),
            float(hit.verification_score),
            float(hit.match_score),
            1 if hit.source == "pdf_text" else 0,
        )

    return (
        float(hit.verification_score),
        float(hit.color_similarity),
        float(hit.match_score),
        1 if hit.source == "pdf_text" else 0,
    )


def _maybe_prefer_fuller_text_label(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """Prefer a fuller framed label over a smaller partial text overlap."""

    if not base_winner.is_text_label:
        return base_winner

    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    contenders: list[CandidateHit] = []

    for hit in group_hits:
        if hit is base_winner or not hit.is_text_label:
            continue

        area = max(1, hit.bbox[2] * hit.bbox[3])
        if area < base_area * TEXT_LABEL_FULLER_AREA_RATIO:
            continue

        if hit.content_score + TEXT_LABEL_FULLER_MAX_CONTENT_DROP < base_winner.content_score:
            continue

        if (
            hit.verification_score + TEXT_LABEL_FULLER_MAX_VERIFICATION_DROP
            < base_winner.verification_score
        ):
            continue

        contenders.append(hit)

    if not contenders:
        return base_winner

    return max(
        contenders + [base_winner],
        key=lambda hit: (
            max(1, hit.bbox[2] * hit.bbox[3]),
            float(hit.content_score),
            float(hit.verification_score),
            float(hit.match_score),
        ),
    )


def _maybe_prefer_fuller_gray_symbol(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """Prefer a fuller gray symbol over a smaller core when both overlap."""

    if base_winner.dominant_hsv is not None or base_winner.is_text_label:
        return base_winner

    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    contenders: list[CandidateHit] = []
    for hit in group_hits:
        if hit is base_winner or hit.dominant_hsv is not None or hit.is_text_label:
            continue

        area = max(1, hit.bbox[2] * hit.bbox[3])
        if area < base_area * GRAY_FULLER_SYMBOL_MIN_AREA_RATIO:
            continue
        if hit.coverage < GRAY_FULLER_SYMBOL_MIN_COVERAGE:
            continue
        if hit.purity < GRAY_FULLER_SYMBOL_MIN_PURITY:
            continue
        if (
            hit.verification_score + GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP
            < base_winner.verification_score
        ):
            continue

        contenders.append(hit)

    if not contenders:
        return base_winner

    return max(
        contenders + [base_winner],
        key=lambda hit: (
            max(1, hit.bbox[2] * hit.bbox[3]),
            float(hit.verification_score),
            float(hit.match_score),
        ),
    )


def _select_cluster_winner(
    group_hits: list[CandidateHit],
    parent_ids_by_child: dict[int, set[int]],
) -> CandidateHit:
    """Pick one winner per cluster, preferring promoted fuller symbols over simpler cores."""

    base_winner = max(group_hits, key=_candidate_rank_key)
    base_winner = _maybe_prefer_fuller_text_label(group_hits, base_winner)
    base_winner = _maybe_prefer_fuller_gray_symbol(group_hits, base_winner)
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

    candidates = _suppress_same_template_ghosts(candidates)
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
