"""Candidate prefiltering and clustering helpers."""

from __future__ import annotations

import cv2
import numpy as np

from core.detector_candidate_selection import (
    _candidate_rank_key,
    _color_template_score,
    _is_color_label_like_shape,
    _is_strong_color_satellite_candidate,
    _is_strong_full_gray_text_label,
    _is_strong_tiny_gray_candidate,
    _select_cluster_winner,
)
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
    RAW_PREFILTER_CENTER_DISTANCE_RATIO,
    RAW_PREFILTER_IOM_THRESHOLD,
    RAW_PREFILTER_IOU_THRESHOLD,
    RAW_PREFILTER_LOCAL_CENTER_DISTANCE_RATIO,
    RAW_PREFILTER_LOCAL_IOM_THRESHOLD,
    RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES,
    RAW_PREFILTER_MIN_CANDIDATES,
)
from core.detector_geometry import (
    _axis_gap,
    _axis_overlap_fraction,
    _bbox_metrics,
    _box_center,
    _center_inside_box,
)
from core.detector_masks import _hue_distance
from core.detector_models import CandidateHit
from core.detector_raw_prefilter import (
    _dedupe_raw_template_hits_before_validation,
    _prefilter_candidates,
    _prefilter_raw_template_hits,
)


def _is_separate_vertical_color_stack(hit_a: CandidateHit, hit_b: CandidateHit) -> bool:
    """Keep compact same-hue symbols separate when they form a vertical stack."""

    if hit_a.dominant_hsv is None or hit_b.dominant_hsv is None:
        return False
    if hit_a.is_text_label or hit_b.is_text_label:
        return False
    if _hue_distance(hit_a.dominant_hsv[0], hit_b.dominant_hsv[0]) > (
        COLOR_HUE_TOLERANCE + 6
    ):
        return False

    area_a = max(1, hit_a.bbox[2] * hit_a.bbox[3])
    area_b = max(1, hit_b.bbox[2] * hit_b.bbox[3])
    if max(area_a, area_b) > 4_500:
        return False

    aspect_a = max(hit_a.bbox[2], hit_a.bbox[3]) / max(1, min(hit_a.bbox[2], hit_a.bbox[3]))
    aspect_b = max(hit_b.bbox[2], hit_b.bbox[3]) / max(1, min(hit_b.bbox[2], hit_b.bbox[3]))
    if max(aspect_a, aspect_b) > 1.75:
        return False

    inter_area, _iou, iom, _center_distance = _bbox_metrics(hit_a.bbox, hit_b.bbox)
    if inter_area <= 0 or iom >= 0.58:
        return False

    x_overlap = _axis_overlap_fraction(
        hit_a.bbox[0],
        hit_a.bbox[2],
        hit_b.bbox[0],
        hit_b.bbox[2],
    )
    y_overlap = _axis_overlap_fraction(
        hit_a.bbox[1],
        hit_a.bbox[3],
        hit_b.bbox[1],
        hit_b.bbox[3],
    )
    if x_overlap < 0.40 or y_overlap >= 0.48:
        return False

    _ax, ay, _aw, ah = hit_a.bbox
    _bx, by, _bw, bh = hit_b.bbox
    center_gap_y = abs((ay + ah / 2.0) - (by + bh / 2.0))
    return center_gap_y >= min(ah, bh) * 0.58
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

    if _is_separate_vertical_color_stack(hit_a, hit_b):
        return False

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
            if _is_strong_full_gray_text_label(candidate):
                continue
            candidate_area = max(1, candidate.bbox[2] * candidate.bbox[3])
            candidate_aspect = max(
                float(candidate.bbox[2]) / max(1.0, float(candidate.bbox[3])),
                float(candidate.bbox[3]) / max(1.0, float(candidate.bbox[2])),
            )
            strong_compact_color_candidate = (
                candidate.source.startswith("template")
                and 700 <= candidate_area <= 2_200
                and candidate_aspect <= 1.85
                and candidate.match_score >= 0.62
                and candidate.verification_score >= 0.62
                and candidate.coverage >= 0.70
                and candidate.purity >= 0.72
            )
            if strong_compact_color_candidate:
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


def _labelish_aspect(hit: CandidateHit) -> bool:
    aspect = max(
        float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
        float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
    )
    return 1.15 <= aspect <= 3.35


def _loose_labelish_aspect(hit: CandidateHit) -> bool:
    aspect = max(
        float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
        float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
    )
    return 1.05 <= aspect <= 3.45


def _is_weaker_adjacent_color_fragment(
    candidate: CandidateHit,
    stronger: CandidateHit,
) -> bool:
    if candidate.dominant_hsv is None or stronger.dominant_hsv is None:
        return False
    if candidate.source == "pdf_text" or stronger.source == "pdf_text":
        return False
    if candidate.template_id == stronger.template_id:
        return False
    if _hue_distance(candidate.dominant_hsv[0], stronger.dominant_hsv[0]) > (
        COLOR_HUE_TOLERANCE + 6
    ):
        return False
    candidate_is_strong = candidate.verification_score > 0.70 or candidate.match_score > 0.76
    stronger_has_quality_margin = (
        stronger.verification_score >= candidate.verification_score + 0.06
        and stronger.match_score + 0.12 >= candidate.match_score
    )

    candidate_area = max(1, candidate.bbox[2] * candidate.bbox[3])
    stronger_area = max(1, stronger.bbox[2] * stronger.bbox[3])
    if candidate_area > stronger_area * 1.15:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(candidate.bbox, stronger.bbox)
    cx, cy = _box_center(candidate.bbox)
    sx, sy = _box_center(stronger.bbox)
    x_overlap = _axis_overlap_fraction(
        candidate.bbox[0],
        candidate.bbox[2],
        stronger.bbox[0],
        stronger.bbox[2],
    )
    y_overlap = _axis_overlap_fraction(
        candidate.bbox[1],
        candidate.bbox[3],
        stronger.bbox[1],
        stronger.bbox[3],
    )
    x_distance = abs(cx - sx) / max(1.0, min(candidate.bbox[2], stronger.bbox[2]))
    y_distance = abs(cy - sy) / max(1.0, min(candidate.bbox[3], stronger.bbox[3]))
    x_gap = _axis_gap(candidate.bbox[0], candidate.bbox[2], stronger.bbox[0], stronger.bbox[2])
    same_row_edge_fragment = (
        _loose_labelish_aspect(candidate)
        and _loose_labelish_aspect(stronger)
        and
        stronger_area >= candidate_area * 1.20
        and stronger.match_score >= 0.40
        and stronger.coverage >= 0.58
        and stronger.purity >= 0.50
        and stronger.context_purity >= 0.16
        and y_overlap >= 0.78
        and x_gap <= max(8, int(min(candidate.bbox[2], stronger.bbox[2]) * 0.24))
        and candidate.match_score <= stronger.match_score + 0.34
        and candidate.verification_score <= stronger.verification_score + 0.32
    )
    if same_row_edge_fragment and candidate_is_strong and inter_area <= 0:
        return False
    separate_stacked_labels = (
        _labelish_aspect(candidate)
        and _labelish_aspect(stronger)
        and x_overlap >= 0.60
        and y_overlap < 0.45
    )
    if separate_stacked_labels:
        return False

    return (
        same_row_edge_fragment
        or (
            not candidate_is_strong
            and
            stronger_has_quality_margin
            and (
                (inter_area > 0 and iom >= 0.20 and center_distance <= 0.90)
                or (x_overlap >= 0.60 and y_distance <= 0.90)
                or (y_overlap >= 0.60 and x_distance <= 0.90)
            )
        )
    )


def _color_fragment_suppression_reason(
    candidate: CandidateHit,
    stronger: CandidateHit,
) -> str | None:
    if candidate.dominant_hsv is None or stronger.dominant_hsv is None:
        return None
    if candidate.source == "pdf_text" or stronger.source == "pdf_text":
        return None
    if _hue_distance(candidate.dominant_hsv[0], stronger.dominant_hsv[0]) > (
        COLOR_HUE_TOLERANCE + 6
    ):
        return None

    candidate_area = max(1, candidate.bbox[2] * candidate.bbox[3])
    stronger_area = max(1, stronger.bbox[2] * stronger.bbox[3])
    inter_area, _iou, iom, center_distance = _bbox_metrics(candidate.bbox, stronger.bbox)
    candidate_score = _color_template_score(candidate)
    stronger_score = _color_template_score(stronger)
    x_overlap = _axis_overlap_fraction(
        candidate.bbox[0],
        candidate.bbox[2],
        stronger.bbox[0],
        stronger.bbox[2],
    )
    y_overlap = _axis_overlap_fraction(
        candidate.bbox[1],
        candidate.bbox[3],
        stronger.bbox[1],
        stronger.bbox[3],
    )

    protected_adjacent_text_label = (
        candidate.source == "template"
        and (candidate.is_text_label or _is_color_label_like_shape(candidate))
        and inter_area > 0
        and not _overlaps_as_same_object(candidate, stronger)
        and 800 <= candidate_area <= 2_000
        and candidate.match_score >= 0.56
        and candidate.verification_score >= 0.56
        and candidate.coverage >= 0.84
        and candidate.purity >= 0.62
        and (iom <= 0.45 or x_overlap < 0.88 or y_overlap < 0.88)
    )
    if protected_adjacent_text_label:
        return None

    candidate_aspect = max(
        float(candidate.bbox[2]) / max(1.0, float(candidate.bbox[3])),
        float(candidate.bbox[3]) / max(1.0, float(candidate.bbox[2])),
    )
    protected_compact_color_symbol = (
        candidate.source == "template"
        and not candidate.is_text_label
        and 1_500 <= candidate_area <= 3_300
        and candidate_aspect <= 1.65
        and stronger_area >= candidate_area * 1.25
        and candidate.match_score >= 0.62
        and candidate.verification_score >= 0.62
        and candidate.coverage >= 0.78
        and candidate.purity >= 0.74
        and (iom <= 0.48 or x_overlap < 0.70 or y_overlap < 0.70)
    )
    if protected_compact_color_symbol:
        return None

    same_template_parent_label_pair = (
        (
            candidate.source.startswith("template_parent_search_")
            or stronger.source.startswith("template_parent_search_")
        )
        and (candidate.is_text_label or _is_color_label_like_shape(candidate))
        and (stronger.is_text_label or _is_color_label_like_shape(stronger))
    )
    if _is_separate_vertical_color_stack(candidate, stronger):
        return None

    same_template_duplicate = (
        candidate.template_id == stronger.template_id
        and 0.70 <= candidate_area / max(1, stronger_area) <= 1.35
        and iom >= 0.28
        and (
            not same_template_parent_label_pair
            or (x_overlap >= 0.70 and y_overlap >= 0.70)
        )
        and center_distance <= 0.60
        and candidate_score + 0.005 < stronger_score
    )
    if same_template_duplicate:
        return "color_same_template_duplicate"

    if candidate.source != "template_color_recovery" and _is_strong_color_satellite_candidate(
        candidate
    ):
        return None

    same_template_recovery_fragment = (
        candidate.source == "template_color_recovery"
        and stronger.source != "template_color_recovery"
        and candidate.template_id == stronger.template_id
        and candidate_area <= stronger_area * 1.45
        and (
            (inter_area > 0 and iom >= 0.18)
            or (x_overlap >= 0.55 and y_overlap >= 0.55)
            or center_distance <= 0.95
        )
        and candidate.match_score <= stronger.match_score + 0.18
        and candidate.verification_score <= stronger.verification_score + 0.20
    )
    if same_template_recovery_fragment:
        return "color_recovery_same_template_bridge"

    if _is_weaker_adjacent_color_fragment(candidate, stronger):
        if candidate.source == "template_color_recovery":
            return "color_recovery_adjacent_fragment"
        return "color_local_fragment"

    weak_overlapping_color_fragment = (
        candidate.template_id != stronger.template_id
        and candidate_area <= stronger_area * 2.35
        and candidate_area >= stronger_area * 0.55
        and inter_area > 0
        and (iom >= 0.30 or center_distance <= 0.92 or x_overlap >= 0.55 or y_overlap >= 0.55)
        and stronger.coverage >= 0.62
        and stronger.purity >= 0.60
        and stronger.verification_score + 0.02 >= candidate.verification_score
        and stronger.match_score + 0.05 >= candidate.match_score
        and (
            (
                candidate.match_score < 0.50
                and stronger.match_score >= candidate.match_score + 0.08
            )
            or (
                candidate.coverage < 0.70
                and stronger.context_purity + 0.10 >= candidate.context_purity
            )
            or (
                candidate.verification_score < 0.60
                and stronger.match_score >= candidate.match_score + 0.10
            )
        )
    )
    if weak_overlapping_color_fragment:
        return (
            "color_recovery_adjacent_fragment"
            if candidate.source == "template_color_recovery"
            else "color_local_fragment"
        )

    weak_template_fragment = (
        candidate.source == "template"
        and candidate_area <= stronger_area * 1.60
        and candidate.match_score < 0.56
        and candidate.verification_score < 0.60
        and candidate.coverage < 0.80
        and candidate.context_purity < 0.50
        and stronger.verification_score >= candidate.verification_score + 0.07
        and stronger.coverage >= 0.65
        and (
            (inter_area > 0 and iom >= 0.18 and center_distance <= 0.95)
            or (x_overlap >= 0.55 and y_overlap >= 0.55)
        )
    )
    if weak_template_fragment:
        return "color_local_fragment"

    return None


def _suppress_color_local_fragments(
    candidates: list[CandidateHit],
) -> tuple[list[CandidateHit], list[CandidateHit], dict[int, str]]:
    """Drop weak color fragments adjacent to a stronger local same-hue symbol."""

    suppressed: set[int] = set()
    reasons: dict[int, str] = {}
    for idx, candidate in enumerate(candidates):
        if candidate.dominant_hsv is None:
            continue
        for other_idx, other in enumerate(candidates):
            if idx == other_idx:
                continue
            reason = _color_fragment_suppression_reason(candidate, other)
            if reason:
                suppressed.add(idx)
                reasons[id(candidate)] = reason
                break

    if not suppressed:
        return candidates, [], {}

    return (
        [candidate for idx, candidate in enumerate(candidates) if idx not in suppressed],
        [candidate for idx, candidate in enumerate(candidates) if idx in suppressed],
        reasons,
    )


def _overlaps_as_same_object(left: CandidateHit, right: CandidateHit) -> bool:
    """Use direct geometry, not transitive cluster links, for satellite pruning."""

    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False

    if iou >= CLUSTER_IOU_THRESHOLD:
        return True

    left_center = _box_center(left.bbox)
    right_center = _box_center(right.bbox)
    centers_nested = _center_inside_box(left_center, right.bbox) or _center_inside_box(
        right_center,
        left.bbox,
    )
    return (
        iom >= max(0.35, CLUSTER_IOM_THRESHOLD * 0.70)
        and centers_nested
        and center_distance <= (CLUSTER_CENTER_DISTANCE_RATIO * 1.20)
    )


def _is_gray_satellite_candidate(hit: CandidateHit) -> bool:
    """Keep tiny validated gray marks from being swallowed by bridge clusters."""

    return _is_strong_tiny_gray_candidate(hit)


def _is_strong_medium_color_symbol_candidate(hit: CandidateHit) -> bool:
    """Keep a full adjacent color symbol that is larger than the compact-symbol guard."""

    if hit.dominant_hsv is None or hit.source == "pdf_text" or hit.is_text_label:
        return False
    area = max(1, hit.bbox[2] * hit.bbox[3])
    aspect = max(
        float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
        float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
    )
    if area < 2_200 or area > 3_600 or aspect > 1.55:
        return False
    return (
        hit.match_score >= 0.62
        and hit.verification_score >= 0.62
        and hit.coverage >= 0.70
        and hit.purity >= 0.70
        and hit.context_purity >= 0.45
    )


def _is_strong_color_text_label_satellite_candidate(hit: CandidateHit) -> bool:
    """Keep validated color text labels from disappearing through transitive bridges."""

    if hit.dominant_hsv is None or not hit.is_text_label or hit.source == "pdf_text":
        return False
    area = max(1, hit.bbox[2] * hit.bbox[3])
    aspect = max(
        float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
        float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
    )
    if area < 1_200 or area > 4_200 or aspect > 3.40:
        return False
    return (
        hit.match_score >= 0.45
        and hit.verification_score >= 0.67
        and hit.coverage >= 0.68
        and hit.purity >= 0.80
        and hit.context_purity >= 0.54
        and hit.content_score >= 0.70
    )


def _is_adjacent_color_text_label_satellite_candidate(
    hit: CandidateHit,
    winner: CandidateHit,
) -> bool:
    """Keep a separate validated label that only partially touches a fuller neighbor."""

    if (
        hit.dominant_hsv is None
        or winner.dominant_hsv is None
        or not (hit.is_text_label or _is_color_label_like_shape(hit))
        or hit.source != "template"
        or hit.source == "pdf_text"
    ):
        return False
    if _hue_distance(hit.dominant_hsv[0], winner.dominant_hsv[0]) > (
        COLOR_HUE_TOLERANCE + 6
    ):
        return False
    area = max(1, hit.bbox[2] * hit.bbox[3])
    aspect = max(
        float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
        float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
    )
    if area < 850 or area > 1_800 or aspect > 1.55:
        return False
    if (
        hit.match_score < 0.56
        or hit.verification_score < 0.56
        or hit.coverage < 0.78
        or hit.purity < 0.62
    ):
        return False
    inter_area, iou, iom, center_distance = _bbox_metrics(hit.bbox, winner.bbox)
    if inter_area <= 0:
        return False
    if _overlaps_as_same_object(hit, winner):
        return False
    return iom <= 0.36 or iou <= 0.24 or center_distance >= 0.42


def _is_color_parent_label_satellite_candidate(hit: CandidateHit) -> bool:
    """Keep promoted fuller color labels that are separate but bridged by local cores."""

    if (
        hit.dominant_hsv is None
        or not hit.source.startswith("template_parent_search_")
        or hit.source == "pdf_text"
    ):
        return False
    area = max(1, hit.bbox[2] * hit.bbox[3])
    aspect = max(
        float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
        float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
    )
    if area < 1_150 or area > 2_200 or aspect > 1.85:
        return False
    return (
        hit.match_score >= 0.64
        and hit.verification_score >= 0.56
        and hit.coverage >= 0.70
        and hit.purity >= 0.58
        and hit.context_purity >= 0.18
    )


def _is_satellite_candidate(hit: CandidateHit) -> bool:
    return (
        _is_gray_satellite_candidate(hit)
        or _is_strong_color_satellite_candidate(hit)
        or _is_strong_medium_color_symbol_candidate(hit)
        or _is_strong_color_text_label_satellite_candidate(hit)
        or _is_color_parent_label_satellite_candidate(hit)
    )


def _satellite_rank_key(hit: CandidateHit) -> tuple[float, ...]:
    if _is_strong_color_satellite_candidate(hit):
        return (
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
            float(hit.context_purity),
        )
    if _is_strong_medium_color_symbol_candidate(hit):
        return (
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
            float(hit.context_purity),
        )
    if _is_strong_color_text_label_satellite_candidate(hit):
        return (
            float(hit.coverage),
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.content_score),
        )
    return _candidate_rank_key(hit)


def _select_cluster_satellites(
    group_hits: list[CandidateHit],
    winner: CandidateHit,
) -> list[CandidateHit]:
    """Return strong compact hits connected only through a bridge candidate."""

    satellites: list[CandidateHit] = []
    for hit in sorted(group_hits, key=_satellite_rank_key, reverse=True):
        if hit is winner:
            continue
        if not (
            _is_satellite_candidate(hit)
            or _is_adjacent_color_text_label_satellite_candidate(hit, winner)
        ):
            continue
        if _overlaps_as_same_object(hit, winner):
            continue
        if any(_overlaps_as_same_object(hit, selected) for selected in satellites):
            continue
        satellites.append(hit)

    strong_compact_candidates: list[CandidateHit] = []
    for hit in group_hits:
        area = max(1, hit.bbox[2] * hit.bbox[3])
        aspect = max(
            float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
            float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
        )
        compact_color_hit = (
            hit.dominant_hsv is not None
            and hit.source != "pdf_text"
            and 700 <= area <= 2_200
            and aspect <= 1.85
            and hit.match_score >= 0.62
            and hit.verification_score >= 0.62
            and hit.coverage >= 0.70
            and hit.purity >= 0.72
        )
        if hit is winner or not compact_color_hit:
            continue
        strong_compact_candidates.append(hit)

    for hit in sorted(
        strong_compact_candidates,
        key=_satellite_rank_key,
        reverse=True,
    ):
        if hit in satellites:
            continue
        if _overlaps_as_same_object(hit, winner) and not (
            hit.template_id == winner.template_id
            and _satellite_rank_key(hit) > _satellite_rank_key(winner)
        ):
            continue

        replaced = False
        blocked = False
        for index, selected in enumerate(satellites):
            if not _overlaps_as_same_object(hit, selected):
                continue
            if (
                hit.template_id == selected.template_id
                and _satellite_rank_key(hit) > _satellite_rank_key(selected)
            ):
                satellites[index] = hit
                replaced = True
                break
            blocked = True
            break
        if replaced or blocked:
            continue
        satellites.append(hit)

    return satellites


def _cluster_candidates(
    candidates: list[CandidateHit],
    parent_ids_by_child: dict[int, set[int]] | None = None,
    mode: str = "color",
    prefer_direct_color_family_parent: bool = True,
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
        winner = _select_cluster_winner(
            group_hits,
            parent_ids_by_child,
            prefer_direct_color_family_parent=prefer_direct_color_family_parent,
        )
        winners.append(winner)
        winners.extend(_select_cluster_satellites(group_hits, winner))

    winners.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.verification_score))
    return winners
