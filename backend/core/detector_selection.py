"""Shared local candidate selection helpers.

The gray detector can produce several plausible interpretations for the same
ink. Keep the local arbitration rules in one place so clustering, rescue and
final gray de-duplication agree on what wins.
"""

from __future__ import annotations

import math

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
    PROMOTED_PARENT_MIN_AREA_RATIO,
    PROMOTED_PARENT_MIN_VERIFICATION,
    PROMOTED_PARENT_OVERRIDE_MARGIN,
)
from core.detector_masks import _hue_distance
from core.detector_models import CandidateHit


def _bbox_metrics(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> tuple[int, float, float, float]:
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
    center_distance = float(math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    ref_distance = max(1.0, min(math.hypot(aw, ah), math.hypot(bw, bh)))
    normalized_center_distance = center_distance / ref_distance

    return inter_area, iou, iom, normalized_center_distance


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, width, height = box
    return x + width / 2.0, y + height / 2.0


def _center_inside_box(
    center: tuple[float, float],
    box: tuple[int, int, int, int],
    margin_ratio: float = 0.05,
) -> bool:
    x, y = center
    bx, by, bw, bh = box
    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio
    return (bx - pad_x) <= x <= (bx + bw + pad_x) and (by - pad_y) <= y <= (
        by + bh + pad_y
    )


def _hit_area(hit: CandidateHit) -> int:
    return max(1, hit.bbox[2] * hit.bbox[3])


def _is_gray(hit: CandidateHit) -> bool:
    return hit.dominant_hsv is None


def _same_color_place(left: CandidateHit, right: CandidateHit) -> bool:
    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False

    left_center = _box_center(left.bbox)
    right_center = _box_center(right.bbox)
    centers_nested = _center_inside_box(left_center, right.bbox) or _center_inside_box(
        right_center,
        left.bbox,
    )

    if (
        left.dominant_hsv is not None
        and right.dominant_hsv is not None
        and _hue_distance(left.dominant_hsv[0], right.dominant_hsv[0])
        > (COLOR_HUE_TOLERANCE + 6)
    ):
        return (
            centers_nested
            and iou >= CROSS_COLOR_CLUSTER_IOU_THRESHOLD
            and iom >= CROSS_COLOR_CLUSTER_IOM_THRESHOLD
            and center_distance <= CROSS_COLOR_CENTER_DISTANCE_RATIO
        )

    if iou >= CLUSTER_IOU_THRESHOLD and center_distance <= CLUSTER_CENTER_DISTANCE_RATIO:
        return True

    return (
        iom >= CLUSTER_IOM_THRESHOLD
        and centers_nested
        and center_distance <= (CLUSTER_CENTER_DISTANCE_RATIO * 1.15)
    )


def _same_gray_place(left: CandidateHit, right: CandidateHit) -> bool:
    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False

    left_center = _box_center(left.bbox)
    right_center = _box_center(right.bbox)
    centers_nested = _center_inside_box(left_center, right.bbox) or _center_inside_box(
        right_center,
        left.bbox,
    )

    if iou >= CLUSTER_IOU_THRESHOLD and center_distance <= CLUSTER_CENTER_DISTANCE_RATIO:
        return True

    if (
        iom >= CLUSTER_IOM_THRESHOLD
        and centers_nested
        and center_distance <= (CLUSTER_CENTER_DISTANCE_RATIO * 1.15)
    ):
        return True

    area_ratio = max(_hit_area(left), _hit_area(right)) / max(1, min(_hit_area(left), _hit_area(right)))
    axis_offset = min(
        abs(left_center[0] - right_center[0]) / max(1.0, min(left.bbox[2], right.bbox[2])),
        abs(left_center[1] - right_center[1]) / max(1.0, min(left.bbox[3], right.bbox[3])),
    )
    if (
        left.template_id == right.template_id
        and area_ratio <= 1.70
        and center_distance <= 0.42
        and axis_offset <= 0.13
        and (iom >= 0.35 or iou >= 0.20)
    ):
        return True

    edge_overlap_same_place = (
        area_ratio >= 3.00
        and iom >= 0.38
        and iou >= 0.08
    )
    return edge_overlap_same_place


def same_physical_place(
    left: CandidateHit,
    right: CandidateHit,
    mode: str = "gray",
) -> bool:
    """Return whether two candidates compete for the same local ink."""

    if mode == "gray" and _is_gray(left) and _is_gray(right):
        return _same_gray_place(left, right)
    return _same_color_place(left, right)


def candidate_quality_key(
    hit: CandidateHit,
    mode: str = "gray",
) -> tuple[float, ...]:
    """Return the shared local quality rank for candidate arbitration."""

    not_mirrored = 0.0 if hit.mirrored else 1.0
    pdf_bonus = 1.0 if mode == "gray" and hit.source == "pdf_text" else 0.0

    if hit.is_text_label:
        return (
            float(hit.content_score),
            float(hit.verification_score),
            float(hit.match_score),
            pdf_bonus,
            float(hit.coverage),
            float(hit.purity),
            float(hit.context_purity),
            not_mirrored,
        )

    if mode == "gray" and _is_gray(hit):
        area_bonus = min(0.08, math.log1p(float(_hit_area(hit))) * 0.006)
        return (
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
            float(hit.context_purity),
            area_bonus,
            not_mirrored,
        )

    return (
        float(hit.verification_score),
        float(hit.color_similarity),
        float(hit.match_score),
        pdf_bonus,
    )


def _is_promoted_parent_of(
    winner: CandidateHit,
    loser: CandidateHit,
    parent_ids_by_child: dict[int, set[int]] | None,
) -> bool:
    if parent_ids_by_child is None:
        return False
    child_id = loser.template_id
    if winner.template_id not in parent_ids_by_child.get(child_id, set()):
        return False
    if winner.promoted_from_template_id not in {None, child_id}:
        return False
    if winner.verification_score < PROMOTED_PARENT_MIN_VERIFICATION:
        return False
    if _hit_area(winner) < _hit_area(loser) * PROMOTED_PARENT_MIN_AREA_RATIO:
        return False
    return winner.verification_score + PROMOTED_PARENT_OVERRIDE_MARGIN >= loser.verification_score


def _fuller_gray_dominates(winner: CandidateHit, loser: CandidateHit) -> bool:
    if not (_is_gray(winner) and _is_gray(loser)):
        return False
    if winner.is_text_label or loser.is_text_label:
        return False
    if _hit_area(winner) < _hit_area(loser) * GRAY_FULLER_SYMBOL_MIN_AREA_RATIO:
        return False
    if winner.coverage < GRAY_FULLER_SYMBOL_MIN_COVERAGE:
        return False
    if winner.purity < GRAY_FULLER_SYMBOL_MIN_PURITY:
        return False
    return winner.verification_score + GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP >= loser.verification_score


def _edge_gray_dominates(winner: CandidateHit, loser: CandidateHit) -> bool:
    if not (_is_gray(winner) and _is_gray(loser)):
        return False
    if winner.is_text_label != loser.is_text_label:
        return False
    if _hit_area(winner) < _hit_area(loser) * 3.00:
        return False
    _inter, iou, iom, _center_distance = _bbox_metrics(winner.bbox, loser.bbox)
    if iou < 0.08 or iom < 0.38:
        return False
    return (
        winner.verification_score >= loser.verification_score + 0.09
        and winner.match_score >= loser.match_score + 0.05
    )


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


def _color_fuller_label_dominates(winner: CandidateHit, loser: CandidateHit) -> bool:
    """Let a complete color label beat a high-match partial local label."""

    if winner.dominant_hsv is None or loser.dominant_hsv is None:
        return False
    if winner.source == "pdf_text" or loser.source == "pdf_text":
        return False
    if _hue_distance(winner.dominant_hsv[0], loser.dominant_hsv[0]) > (
        COLOR_HUE_TOLERANCE + 6
    ):
        return False
    if not (winner.is_text_label or loser.is_text_label):
        return False

    winner_area = _hit_area(winner)
    loser_area = _hit_area(loser)
    if winner_area < loser_area * 1.08 or winner_area > loser_area * 1.75:
        return False

    inter_area, iou, iom, center_distance = _bbox_metrics(winner.bbox, loser.bbox)
    y_overlap = _axis_overlap_fraction(winner.bbox[1], winner.bbox[3], loser.bbox[1], loser.bbox[3])
    x_gap = _axis_gap(winner.bbox[0], winner.bbox[2], loser.bbox[0], loser.bbox[2])
    same_row_edge = (
        y_overlap >= 0.72
        and x_gap <= max(12, int(min(winner.bbox[2], loser.bbox[2]) * 0.38))
    )
    same_local_ink = (
        inter_area > 0
        and (iom >= 0.42 or iou >= 0.15 or center_distance <= 0.85)
    )
    if not (same_local_ink or same_row_edge):
        return False

    if (
        winner.match_score < 0.40
        or winner.coverage < 0.58
        or winner.purity < 0.50
        or winner.context_purity < 0.16
        or winner.color_similarity < 0.90
    ):
        return False

    # Preserve a genuinely strong smaller symbol unless the larger candidate
    # is nearly as credible and covers extra same-color ink.
    loser_very_strong = (
        loser.match_score >= 0.84
        and loser.verification_score >= 0.78
        and loser.coverage >= 0.86
        and loser.purity >= 0.86
    )
    if loser_very_strong and not same_row_edge:
        return False

    return (
        winner.match_score + 0.46 >= loser.match_score
        and winner.verification_score + 0.36 >= loser.verification_score
        and winner.coverage + 0.18 >= loser.coverage
        and winner.purity + 0.35 >= loser.purity
        and winner.context_purity + 0.14 >= loser.context_purity
    )


def local_dominates(
    winner: CandidateHit,
    loser: CandidateHit,
    mode: str = "gray",
    parent_ids_by_child: dict[int, set[int]] | None = None,
) -> bool:
    """Return true when winner should suppress loser in the same local place."""

    if winner is loser:
        return False
    if not same_physical_place(winner, loser, mode=mode):
        return False

    if winner.template_id == loser.template_id:
        return candidate_quality_key(winner, mode=mode) >= candidate_quality_key(loser, mode=mode)

    if _is_promoted_parent_of(winner, loser, parent_ids_by_child):
        return True

    if mode == "gray" and _fuller_gray_dominates(winner, loser):
        return True

    if mode == "gray" and _edge_gray_dominates(winner, loser):
        return True

    if mode != "gray" and _color_fuller_label_dominates(winner, loser):
        return True

    winner_key = candidate_quality_key(winner, mode=mode)
    loser_key = candidate_quality_key(loser, mode=mode)
    if winner_key <= loser_key:
        return False

    if (
        mode == "gray"
        and _is_gray(winner)
        and _is_gray(loser)
        and not winner.is_text_label
        and not loser.is_text_label
    ):
        area_ratio = max(_hit_area(winner), _hit_area(loser)) / max(
            1,
            min(_hit_area(winner), _hit_area(loser)),
        )
        if area_ratio <= 1.25:
            return (
                winner.verification_score >= loser.verification_score + 0.065
                and winner.match_score >= loser.match_score + 0.09
                and winner.coverage >= loser.coverage + 0.08
                and winner.purity >= loser.purity + 0.025
            )

    return (
        winner.verification_score >= loser.verification_score + 0.065
        and winner.match_score >= loser.match_score + 0.09
        and winner.coverage >= loser.coverage + 0.08
    )


def _stable_rank_key(hit: CandidateHit, mode: str) -> tuple[float, ...]:
    x, y, width, height = hit.bbox
    return (
        *candidate_quality_key(hit, mode=mode),
        -float(y),
        -float(x),
        -float(width * height),
        -float(hit.template_id),
    )


def select_local_winners(
    candidates: list[CandidateHit],
    mode: str = "gray",
    parent_ids_by_child: dict[int, set[int]] | None = None,
) -> list[CandidateHit]:
    """Select deterministic local winners without transitive bridge clustering."""

    if len(candidates) < 2:
        return list(candidates)

    selected: list[CandidateHit] = []
    for candidate in sorted(candidates, key=lambda hit: _stable_rank_key(hit, mode), reverse=True):
        if any(
            local_dominates(existing, candidate, mode=mode, parent_ids_by_child=parent_ids_by_child)
            for existing in selected
        ):
            continue
        selected = [
            existing
            for existing in selected
            if not local_dominates(candidate, existing, mode=mode, parent_ids_by_child=parent_ids_by_child)
        ]
        selected.append(candidate)

    selected.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.verification_score))
    return selected
