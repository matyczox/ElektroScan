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
    GRAY_TINY_GEOMETRY_MAX_TEMPLATE_PIXELS,
    GRAY_TINY_GEOMETRY_MIN_CONTEXT,
    GRAY_TINY_GEOMETRY_MIN_COVERAGE,
    GRAY_TINY_GEOMETRY_MIN_PURITY,
    GRAY_TINY_GEOMETRY_MIN_VERIFICATION,
    PREFILTER_NMS_IOU_THRESHOLD,
    PREFILTER_NMS_MIN_CANDIDATES,
    PROMOTED_PARENT_MIN_AREA_RATIO,
    PROMOTED_PARENT_MIN_VERIFICATION,
    PROMOTED_PARENT_OVERRIDE_MARGIN,
    RAW_PREFILTER_CENTER_DISTANCE_RATIO,
    RAW_PREFILTER_IOM_THRESHOLD,
    RAW_PREFILTER_IOU_THRESHOLD,
    RAW_PREFILTER_LOCAL_CENTER_DISTANCE_RATIO,
    RAW_PREFILTER_LOCAL_IOM_THRESHOLD,
    RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES,
    RAW_PREFILTER_MIN_CANDIDATES,
    TEXT_LABEL_FULLER_AREA_RATIO,
    TEXT_LABEL_FULLER_MAX_CONTENT_DROP,
    TEXT_LABEL_FULLER_MAX_VERIFICATION_DROP,
)
from core.detector_masks import _hue_distance
from core.detector_models import CandidateHit
from core.detector_selection import candidate_quality_key


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


def _raw_candidates_are_local_alternatives(left: CandidateHit, right: CandidateHit) -> bool:
    """Return true for same-template gray alternatives in one local place."""

    if left.dominant_hsv is not None or right.dominant_hsv is not None:
        return False
    if left.template_id != right.template_id:
        return False
    if left.source != right.source:
        return False
    if left.roi_strategy != right.roi_strategy:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    return (
        inter_area > 0
        and iom >= RAW_PREFILTER_LOCAL_IOM_THRESHOLD
        and center_distance <= RAW_PREFILTER_LOCAL_CENTER_DISTANCE_RATIO
    )


def _limit_same_template_raw_local_alternatives(
    candidates: list[CandidateHit],
) -> list[CandidateHit]:
    """Cap duplicate same-template raw alternatives before full validation.

    This is intentionally not a class competition.  It only limits many raw
    variants from the same template/source/ROI strategy in the same local
    place, while keeping several alternatives so validation can still choose
    the scale/rotation that carries real evidence.
    """

    if len(candidates) < RAW_PREFILTER_MIN_CANDIDATES:
        return candidates
    if RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES <= 0:
        return candidates

    grouped: dict[int, list[CandidateHit]] = {}
    passthrough: list[CandidateHit] = []
    for candidate in candidates:
        if candidate.dominant_hsv is not None:
            passthrough.append(candidate)
            continue
        grouped.setdefault(candidate.template_id, []).append(candidate)

    grid_cell_px = 32
    filtered: list[CandidateHit] = list(passthrough)
    for template_hits in grouped.values():
        kept: list[CandidateHit] = []
        kept_grid: dict[tuple[str, str, int, int], list[CandidateHit]] = {}
        for candidate in sorted(template_hits, key=lambda hit: hit.match_score, reverse=True):
            _x, _y, w, h = candidate.bbox
            cx, cy = _box_center(candidate.bbox)
            cell_x = int(cx // grid_cell_px)
            cell_y = int(cy // grid_cell_px)
            radius_px = (
                max(1.0, float(np.hypot(w, h)))
                * RAW_PREFILTER_LOCAL_CENTER_DISTANCE_RATIO
            )
            cell_radius = int(np.ceil(radius_px / grid_cell_px)) + 1
            key_prefix = (str(candidate.source), str(candidate.roi_strategy))

            local_alternatives = 0
            for dx in range(-cell_radius, cell_radius + 1):
                if local_alternatives >= RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES:
                    break
                for dy in range(-cell_radius, cell_radius + 1):
                    bucket = kept_grid.get(
                        (key_prefix[0], key_prefix[1], cell_x + dx, cell_y + dy)
                    )
                    if not bucket:
                        continue
                    for existing in bucket:
                        if _raw_candidates_are_local_alternatives(candidate, existing):
                            local_alternatives += 1
                            if local_alternatives >= RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES:
                                break
                    if local_alternatives >= RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES:
                        break
            if local_alternatives >= RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES:
                continue
            kept.append(candidate)
            kept_grid.setdefault(
                (key_prefix[0], key_prefix[1], cell_x, cell_y),
                [],
            ).append(candidate)
        filtered.extend(kept)

    filtered.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.match_score))
    return filtered


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


def _dedupe_raw_template_hits_before_validation(
    candidates: list[CandidateHit],
) -> list[CandidateHit]:
    """Run conservative same-template raw de-duplication before validation."""

    return _limit_same_template_raw_local_alternatives(candidates)


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


def _candidate_rank_key(hit: CandidateHit) -> tuple[float, ...]:
    """Return the default winner ranking inside a cluster."""

    return candidate_quality_key(hit, mode="color")


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


def _color_template_score(hit: CandidateHit) -> float:
    area_bonus = 0.0
    if (hit.is_text_label or _is_color_label_like_shape(hit)) and hit.source != "pdf_text":
        area_bonus = min(0.18, np.log1p(float(max(1, hit.bbox[2] * hit.bbox[3]))) * 0.018)
    return (
        float(hit.verification_score)
        + float(hit.match_score)
        + 0.12 * float(hit.content_score)
        + 0.06 * float(hit.coverage)
        + 0.06 * float(hit.purity)
        + area_bonus
    )


def _is_color_label_like_shape(hit: CandidateHit) -> bool:
    if hit.dominant_hsv is None or hit.source == "pdf_text":
        return False
    area = max(1, hit.bbox[2] * hit.bbox[3])
    aspect = max(
        float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
        float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
    )
    return 900 <= area <= 3_800 and 1.15 <= aspect <= 3.35


def _maybe_prefer_tighter_color_template(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """Prefer a tighter real color template over a broader local overlap."""

    if base_winner.dominant_hsv is None:
        return base_winner

    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    base_aspect = max(
        float(base_winner.bbox[2]) / max(1.0, float(base_winner.bbox[3])),
        float(base_winner.bbox[3]) / max(1.0, float(base_winner.bbox[2])),
    )
    base_score = _color_template_score(base_winner)
    contenders: list[CandidateHit] = []
    compact_symbol_contenders: list[CandidateHit] = []

    for hit in group_hits:
        if hit.dominant_hsv is None or hit.source == "pdf_text":
            continue
        if _hue_distance(hit.dominant_hsv[0], base_winner.dominant_hsv[0]) > (
            COLOR_HUE_TOLERANCE + 6
        ):
            continue

        hit_area = max(1, hit.bbox[2] * hit.bbox[3])
        hit_aspect = max(
            float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
            float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
        )
        base_aspect = max(
            float(base_winner.bbox[2]) / max(1.0, float(base_winner.bbox[3])),
            float(base_winner.bbox[3]) / max(1.0, float(base_winner.bbox[2])),
        )
        if (
            base_winner.is_text_label
            and hit.is_text_label
            and hit_area < base_area * 0.90
        ):
            continue
        if hit_area > base_area * 0.90:
            continue

        inter_area, iou, iom, center_distance = _bbox_metrics(hit.bbox, base_winner.bbox)
        stronger_tighter_label = (
            _is_color_label_like_shape(hit)
            and _is_color_label_like_shape(base_winner)
            and hit_area < base_area
            and inter_area > 0
            and (iom >= 0.24 or center_distance <= 0.65)
            and hit.match_score >= base_winner.match_score + 0.16
            and hit.verification_score >= base_winner.verification_score + 0.02
            and hit.coverage >= 0.70
            and hit.purity >= 0.75
        )
        compact_symbol_over_text_label = (
            base_winner.is_text_label
            and not hit.is_text_label
            and hit_area <= base_area * 0.78
            and inter_area > 0
            and (iom >= 0.20 or center_distance <= 0.75)
            and hit.match_score >= base_winner.match_score + 0.08
            and hit.verification_score >= 0.60
            and hit.coverage >= 0.70
            and hit.purity >= 0.74
            and hit.context_purity >= 0.32
        )
        compact_square_over_elongated_symbol = (
            850 <= hit_area <= 1_900
            and hit_aspect <= 1.28
            and base_aspect >= 1.45
            and hit_area <= base_area * 0.78
            and inter_area > 0
            and (iom >= 0.55 or iou >= 0.24 or center_distance <= 0.70)
            and hit.verification_score + 0.08 >= base_winner.verification_score
            and hit.match_score + 0.18 >= base_winner.match_score
            and hit.coverage >= 0.58
            and hit.purity >= 0.68
            and hit.context_purity + 0.12 >= base_winner.context_purity
            and base_winner.coverage < 0.78
        )
        if hit is not base_winner and not (
            inter_area > 0
            and (iom >= 0.45 or iou >= 0.20 or center_distance <= 0.35)
        ) and not (
            stronger_tighter_label
            or compact_symbol_over_text_label
            or compact_square_over_elongated_symbol
        ):
            continue

        if hit.verification_score < 0.56 or hit.coverage < 0.58 or hit.purity < 0.62:
            continue

        if compact_symbol_over_text_label or compact_square_over_elongated_symbol:
            compact_symbol_contenders.append(hit)
            contenders.append(hit)
            continue

        hit_score = _color_template_score(hit)
        if not stronger_tighter_label and hit_score <= base_score + 0.05:
            continue
        if hit.match_score + hit.verification_score <= (
            base_winner.match_score + base_winner.verification_score
        ):
            continue

        contenders.append(hit)

    if not contenders:
        return base_winner

    if compact_symbol_contenders:
        return max(
            compact_symbol_contenders,
            key=lambda hit: (
                float(hit.match_score),
                float(hit.verification_score),
                float(hit.coverage),
                float(hit.purity),
            ),
        )

    return max(
        contenders + [base_winner],
        key=lambda hit: (
            _color_template_score(hit),
            -float(max(1, hit.bbox[2] * hit.bbox[3])),
            float(hit.color_similarity),
            0.0 if hit.mirrored else 1.0,
        ),
    )


def _maybe_prefer_fuller_color_candidate(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """Let a fuller same-color symbol survive over a smaller local core."""

    if base_winner.dominant_hsv is None or base_winner.source == "pdf_text":
        return base_winner

    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    base_aspect = max(
        float(base_winner.bbox[2]) / max(1.0, float(base_winner.bbox[3])),
        float(base_winner.bbox[3]) / max(1.0, float(base_winner.bbox[2])),
    )
    base_score = _color_template_score(base_winner)
    contenders: list[CandidateHit] = []

    for hit in group_hits:
        if hit is base_winner or hit.dominant_hsv is None or hit.source == "pdf_text":
            continue
        if _hue_distance(hit.dominant_hsv[0], base_winner.dominant_hsv[0]) > (
            COLOR_HUE_TOLERANCE + 6
        ):
            continue

        hit_area = max(1, hit.bbox[2] * hit.bbox[3])
        label_like_competition = (
            hit.is_text_label
            or base_winner.is_text_label
            or (_is_color_label_like_shape(hit) and _is_color_label_like_shape(base_winner))
        )
        min_area_ratio = 1.06 if label_like_competition else 1.10
        max_area_ratio = 1.55 if label_like_competition else 2.20
        if hit_area < base_area * min_area_ratio or hit_area > base_area * max_area_ratio:
            continue

        inter_area, iou, iom, center_distance = _bbox_metrics(hit.bbox, base_winner.bbox)
        hit_aspect = max(
            float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
            float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
        )
        base_is_good_compact_symbol = (
            850 <= base_area <= 1_900
            and base_aspect <= 1.12
            and base_winner.coverage >= 0.58
            and base_winner.purity >= 0.68
            and base_winner.verification_score >= 0.56
        )
        hit_is_elongated_attachment = (
            hit_aspect >= 1.45
            and hit_area <= base_area * 2.25
            and inter_area > 0
            and (iom >= 0.55 or iou >= 0.24 or center_distance <= 0.70)
            and hit.coverage < 0.78
            and hit.verification_score <= base_winner.verification_score + 0.08
        )
        if base_is_good_compact_symbol and hit_is_elongated_attachment:
            continue
        if label_like_competition:
            adds_own_ink_area = hit_area - inter_area
            if not (
                inter_area > 0
                and (iom >= 0.48 or iou >= 0.18 or center_distance <= 0.70)
                and center_distance <= 0.82
                and adds_own_ink_area >= max(90, int(hit_area * 0.12))
            ):
                continue
        else:
            if not (
                inter_area > 0
                and (iom >= 0.55 or iou >= 0.25)
                and center_distance <= 0.55
                and (inter_area / hit_area) <= 0.92
            ):
                continue

        fuller_color_symbol = (
            not label_like_competition
            and hit.match_score >= 0.68
            and hit.verification_score >= 0.62
            and hit.coverage >= 0.70
            and hit.purity >= 0.75
            and hit_area <= base_area * 1.80
        )
        fuller_socket_parent = (
            not label_like_competition
            and hit_area >= base_area * 1.20
            and hit_area <= base_area * 1.90
            and inter_area > 0
            and (iom >= 0.58 or iou >= 0.28 or center_distance <= 0.58)
            and hit.match_score + 0.14 >= base_winner.match_score
            and hit.verification_score + 0.10 >= base_winner.verification_score
            and hit.coverage >= 0.62
            and hit.purity >= 0.66
            and hit.context_purity >= 0.28
        )

        if label_like_competition:
            if hit.match_score < 0.50 or hit.coverage < 0.60 or hit.purity < 0.55:
                continue
            if hit.context_purity < 0.16:
                continue
            hit_is_parent_recovery = (
                hit.source.startswith("template_parent_search_")
                or hit.source.startswith("template_promoted_")
            )
            hit_is_fuller_label = (
                hit_area >= base_area * 1.10
                and hit.coverage >= 0.62
                and hit.purity >= 0.58
                and hit.context_purity + 0.10 >= base_winner.context_purity
                and hit.match_score + 0.36 >= base_winner.match_score
            )
            hit_is_fuller_parent_label = (
                hit_is_parent_recovery
                and hit_area >= base_area * 1.22
                and hit_area <= base_area * 1.70
                and inter_area > 0
                and (iom >= 0.55 or iou >= 0.22 or center_distance <= 0.72)
                and hit.match_score >= 0.68
                and hit.verification_score >= 0.62
                and hit.coverage >= 0.66
                and hit.purity >= 0.78
                and hit.context_purity >= 0.28
                and hit.match_score + 0.09 >= base_winner.match_score
                and hit.verification_score + 0.14 >= base_winner.verification_score
            )
            hit_is_fuller_label = hit_is_fuller_label or hit_is_fuller_parent_label
            base_is_strong_local_symbol = (
                _is_color_label_like_shape(base_winner)
                and base_winner.match_score >= 0.68
                and base_winner.verification_score >= 0.64
                and base_winner.coverage >= 0.74
                and base_winner.purity >= 0.78
            )
            if (
                base_is_strong_local_symbol
                and not hit_is_parent_recovery
                and hit.match_score + 0.12 < base_winner.match_score
                and hit.verification_score <= base_winner.verification_score + 0.02
            ):
                continue
            base_is_very_strong_label = (
                _is_color_label_like_shape(base_winner)
                and base_winner.match_score >= 0.82
                and base_winner.verification_score >= 0.78
                and base_winner.coverage >= 0.80
                and not hit_is_fuller_label
            )
            if base_is_very_strong_label:
                continue
            hit_score = _color_template_score(hit)
            if hit_score + (0.62 if hit_is_fuller_label else 0.36) < base_score:
                continue
            if (
                not hit_is_fuller_label
                and
                hit.match_score
                + hit.verification_score
                + 0.12 * hit.coverage
                + 0.16
                < (
                    base_winner.match_score
                    + base_winner.verification_score
                    + 0.12 * base_winner.coverage
                )
            ):
                continue
        else:
            if hit.match_score < 0.40 or hit.coverage < 0.60 or hit.purity < 0.50:
                continue
            if hit.context_purity < 0.16:
                continue
            hit_is_parent_recovery = (
                hit.source.startswith("template_parent_search_")
                or hit.source.startswith("template_promoted_")
            )
            base_is_strong_compact_symbol = (
                not base_winner.is_text_label
                and base_winner.match_score >= 0.70
                and base_winner.verification_score >= 0.66
                and base_winner.coverage >= 0.72
                and base_winner.purity >= 0.80
            )
            hit_has_full_parent_evidence = (
                hit.match_score >= 0.74
                and hit.verification_score >= 0.68
                and hit.coverage >= 0.70
                and hit.purity >= 0.76
            ) or (
                hit.match_score >= 0.70
                and hit.verification_score >= 0.62
                and hit.coverage >= 0.62
                and hit.purity >= 0.80
                and hit.context_purity >= 0.28
            )
            if (
                hit_is_parent_recovery
                and base_is_strong_compact_symbol
                and not hit_has_full_parent_evidence
                and (
                    hit.coverage + 0.08 < base_winner.coverage
                    or hit.purity + 0.12 < base_winner.purity
                )
            ):
                continue
            if not (fuller_color_symbol or fuller_socket_parent):
                if hit.match_score + 0.02 < base_winner.match_score:
                    continue
                if _color_template_score(hit) + 0.04 < base_score:
                    continue

        contenders.append(hit)

    if not contenders:
        return base_winner

    return max(
        contenders,
        key=lambda hit: (
            float(max(1, hit.bbox[2] * hit.bbox[3])),
            _color_template_score(hit),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
        ),
    )


def _maybe_prefer_coverage_color_text_label(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """For same-label color variants, prefer the full ink box over a text-only core."""

    if (
        base_winner.dominant_hsv is None
        or not base_winner.is_text_label
        or base_winner.source == "pdf_text"
    ):
        return base_winner

    contenders: list[CandidateHit] = []
    for hit in group_hits:
        if (
            hit is base_winner
            or hit.template_id != base_winner.template_id
            or hit.dominant_hsv is None
            or not hit.is_text_label
            or hit.source == "pdf_text"
        ):
            continue
        if _hue_distance(hit.dominant_hsv[0], base_winner.dominant_hsv[0]) > (
            COLOR_HUE_TOLERANCE + 6
        ):
            continue

        inter_area, iou, iom, center_distance = _bbox_metrics(hit.bbox, base_winner.bbox)
        if inter_area <= 0 or not (iom >= 0.40 or iou >= 0.18 or center_distance <= 0.68):
            continue
        if hit.coverage < base_winner.coverage + 0.12:
            continue
        if hit.verification_score + 0.11 < base_winner.verification_score:
            continue
        if hit.match_score + 0.08 < base_winner.match_score:
            continue
        if hit.content_score < 0.68 or hit.purity < 0.78 or hit.context_purity < 0.50:
            continue
        contenders.append(hit)

    if not contenders:
        return base_winner

    return max(
        contenders + [base_winner],
        key=lambda hit: (
            float(hit.coverage),
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.content_score),
        ),
    )


def _maybe_prefer_stronger_same_template_color_variant(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """Prefer the best verified compact same-template color variant in a local group."""

    if (
        base_winner.dominant_hsv is None
        or base_winner.source == "pdf_text"
    ):
        return base_winner

    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    contenders: list[CandidateHit] = []
    for hit in group_hits:
        if (
            hit.template_id != base_winner.template_id
            or hit.dominant_hsv is None
            or hit.source == "pdf_text"
        ):
            continue
        if _hue_distance(hit.dominant_hsv[0], base_winner.dominant_hsv[0]) > (
            COLOR_HUE_TOLERANCE + 6
        ):
            continue
        hit_area = max(1, hit.bbox[2] * hit.bbox[3])
        if not (base_area * 0.70 <= hit_area <= base_area * 1.40):
            continue
        inter_area, _iou, _iom, center_distance = _bbox_metrics(hit.bbox, base_winner.bbox)
        if hit is not base_winner and inter_area <= 0 and center_distance > 1.10:
            continue
        if not _is_strong_color_satellite_candidate(hit):
            continue
        contenders.append(hit)

    if not contenders:
        return base_winner

    return max(
        contenders,
        key=lambda hit: (
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
        ),
    )


def _maybe_prefer_direct_color_family_parent(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
    parent_ids_by_child: dict[int, set[int]],
) -> CandidateHit:
    """Prefer a validated fuller same-color family parent over its local core."""

    if (
        base_winner.dominant_hsv is None
        or base_winner.source == "pdf_text"
        or base_winner.is_text_label
    ):
        return base_winner

    parent_ids = parent_ids_by_child.get(base_winner.template_id, set())
    if not parent_ids:
        return base_winner

    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    contenders: list[CandidateHit] = []

    for hit in group_hits:
        if (
            hit is base_winner
            or hit.template_id not in parent_ids
            or hit.dominant_hsv is None
            or hit.source == "pdf_text"
            or hit.is_text_label
        ):
            continue
        if _hue_distance(hit.dominant_hsv[0], base_winner.dominant_hsv[0]) > (
            COLOR_HUE_TOLERANCE + 6
        ):
            continue

        hit_area = max(1, hit.bbox[2] * hit.bbox[3])
        if hit_area < base_area * 1.08 or hit_area > base_area * 2.35:
            continue

        inter_area, iou, iom, center_distance = _bbox_metrics(hit.bbox, base_winner.bbox)
        if inter_area <= 0:
            continue

        own_area = hit_area - inter_area
        if own_area < max(80, int(hit_area * 0.16)):
            continue
        if not (iom >= 0.78 or iou >= 0.46 or center_distance <= 0.46):
            continue

        if (
            hit.match_score < 0.54
            or hit.verification_score < 0.56
            or hit.coverage < 0.58
            or hit.purity < 0.64
            or hit.context_purity < 0.36
        ):
            continue

        base_is_strong_compact = (
            not base_winner.is_text_label
            and base_winner.match_score >= 0.64
            and base_winner.verification_score >= 0.64
            and base_winner.coverage >= 0.66
            and base_winner.purity >= 0.78
            and base_winner.context_purity >= 0.36
        )
        if base_is_strong_compact and (
            hit.verification_score + 0.08 < base_winner.verification_score
            or hit.coverage < 0.62
            or hit.purity < 0.72
        ):
            continue

        if hit.verification_score + 0.10 < base_winner.verification_score:
            continue
        if hit.match_score + 0.18 < base_winner.match_score:
            continue

        contenders.append(hit)

    if not contenders:
        return base_winner

    return max(
        contenders + [base_winner],
        key=lambda hit: (
            float(max(1, hit.bbox[2] * hit.bbox[3])),
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
        ),
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

    same_template_duplicate = (
        candidate.template_id == stronger.template_id
        and 0.70 <= candidate_area / max(1, stronger_area) <= 1.35
        and iom >= 0.28
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


def _is_strong_full_gray_text_label(hit: CandidateHit) -> bool:
    if hit.dominant_hsv is not None or not hit.is_text_label:
        return False
    if hit.scale < 0.85:
        return False
    area = max(1, hit.bbox[2] * hit.bbox[3])
    aspect = max(hit.bbox[2] / max(1, hit.bbox[3]), hit.bbox[3] / max(1, hit.bbox[2]))
    if area < 1000 or aspect < 1.35:
        return False
    if hit.coverage < 0.88 or hit.purity < 0.34:
        return False
    if hit.content_score < 0.70 or hit.verification_score < 0.66:
        return False
    if hit.source == "template":
        return hit.match_score >= 0.62
    if hit.source == "template_content":
        return hit.match_score >= 0.70 and hit.content_score >= 0.82
    return False


def _maybe_prefer_full_gray_text_label(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """Keep complete gray labels from losing to partial local fragments."""

    contenders = [hit for hit in group_hits if _is_strong_full_gray_text_label(hit)]
    if not contenders:
        return base_winner

    best_full_label = max(
        contenders,
        key=lambda hit: (
            float(hit.content_score),
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
            float(max(1, hit.bbox[2] * hit.bbox[3])),
            0.0 if hit.mirrored else 1.0,
        ),
    )
    if base_winner is best_full_label:
        return base_winner

    if _is_strong_full_gray_text_label(base_winner):
        return base_winner

    best_area = max(1, best_full_label.bbox[2] * best_full_label.bbox[3])
    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    base_is_low_purity_content_fragment = (
        base_winner.source == "template_content"
        and base_winner.purity <= 0.35
        and base_winner.context_purity <= 0.18
        and best_area >= base_area * 1.45
        and best_full_label.purity >= base_winner.purity + 0.30
    )
    if base_is_low_purity_content_fragment:
        return best_full_label

    return base_winner


def _is_strong_tiny_gray_candidate(hit: CandidateHit) -> bool:
    """Identify small gray symbols that already validate on their own geometry."""

    if hit.dominant_hsv is not None or hit.is_text_label:
        return False
    if hit.pixel_count > GRAY_TINY_GEOMETRY_MAX_TEMPLATE_PIXELS:
        return False
    if hit.verification_score < GRAY_TINY_GEOMETRY_MIN_VERIFICATION:
        return False
    if hit.coverage < GRAY_TINY_GEOMETRY_MIN_COVERAGE:
        return False
    if hit.purity < GRAY_TINY_GEOMETRY_MIN_PURITY:
        return False
    if hit.context_purity < GRAY_TINY_GEOMETRY_MIN_CONTEXT:
        return False
    return True


def _maybe_prefer_fuller_gray_symbol(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """Prefer a fuller gray symbol over a smaller core when both overlap."""

    if base_winner.dominant_hsv is not None or base_winner.is_text_label:
        return base_winner

    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    base_is_strong_tiny_gray = _is_strong_tiny_gray_candidate(base_winner)
    contenders: list[CandidateHit] = []
    for hit in group_hits:
        if hit is base_winner or hit.dominant_hsv is not None or hit.is_text_label:
            continue

        if (
            base_is_strong_tiny_gray
            and hit.template_id != base_winner.template_id
            and hit.verification_score <= base_winner.verification_score + 0.02
        ):
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
    prefer_direct_color_family_parent: bool = True,
) -> CandidateHit:
    """Pick one winner per cluster, preferring promoted fuller symbols over simpler cores."""

    base_winner = max(group_hits, key=_candidate_rank_key)
    base_winner = _maybe_prefer_tighter_color_template(group_hits, base_winner)
    base_winner = _maybe_prefer_fuller_color_candidate(group_hits, base_winner)
    base_winner = _maybe_prefer_coverage_color_text_label(group_hits, base_winner)
    base_winner = _maybe_prefer_stronger_same_template_color_variant(group_hits, base_winner)
    if prefer_direct_color_family_parent:
        base_winner = _maybe_prefer_direct_color_family_parent(
            group_hits,
            base_winner,
            parent_ids_by_child,
        )
    base_winner = _maybe_prefer_full_gray_text_label(group_hits, base_winner)
    base_winner = _maybe_prefer_fuller_text_label(group_hits, base_winner)
    base_winner = _maybe_prefer_fuller_gray_symbol(group_hits, base_winner)
    base_winner = _maybe_prefer_stronger_same_template_color_variant(group_hits, base_winner)
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
        if hit.dominant_hsv is not None and best_child.dominant_hsv is not None:
            child_is_strong_compact = (
                not best_child.is_text_label
                and best_child.match_score >= 0.62
                and best_child.verification_score >= 0.62
                and best_child.coverage >= 0.70
                and best_child.purity >= 0.72
            )
            parent_has_full_symbol_evidence = (
                hit.match_score >= 0.74
                and hit.verification_score >= 0.68
                and hit.coverage >= 0.70
                and hit.purity >= 0.76
            ) or (
                hit.match_score >= 0.70
                and hit.verification_score >= 0.62
                and hit.coverage >= 0.62
                and hit.purity >= 0.80
                and hit.context_purity >= 0.28
            )
            if (
                child_is_strong_compact
                and not parent_has_full_symbol_evidence
                and (
                    hit.coverage + 0.06 < best_child.coverage
                    or hit.purity + 0.10 < best_child.purity
                    or hit.context_purity + 0.04 < best_child.context_purity
                )
            ):
                continue

        override_candidates.append(hit)

    if override_candidates:
        return max(override_candidates, key=_candidate_rank_key)

    return base_winner


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


def _is_strong_color_satellite_candidate(hit: CandidateHit) -> bool:
    """Keep compact color symbols from being swallowed by nearby text-label bridges."""

    if hit.dominant_hsv is None or hit.source == "pdf_text":
        return False
    area = max(1, hit.bbox[2] * hit.bbox[3])
    aspect = max(
        float(hit.bbox[2]) / max(1.0, float(hit.bbox[3])),
        float(hit.bbox[3]) / max(1.0, float(hit.bbox[2])),
    )
    if area < 700 or area > 2_200 or aspect > 1.85:
        return False
    strong_direct_hit = (
        hit.match_score >= 0.62
        and hit.verification_score >= 0.62
        and hit.coverage >= 0.70
        and hit.purity >= 0.72
    )
    strong_verified_full_symbol = (
        hit.match_score >= 0.44
        and hit.verification_score >= 0.64
        and hit.coverage >= 0.68
        and hit.purity >= 0.74
        and hit.context_purity >= 0.40
    )
    return strong_direct_hit or strong_verified_full_symbol


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


def _is_satellite_candidate(hit: CandidateHit) -> bool:
    return (
        _is_gray_satellite_candidate(hit)
        or _is_strong_color_satellite_candidate(hit)
        or _is_strong_color_text_label_satellite_candidate(hit)
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

    strongest_compact_by_template: dict[int, CandidateHit] = {}
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
        existing = strongest_compact_by_template.get(hit.template_id)
        if existing is None or _satellite_rank_key(hit) > _satellite_rank_key(existing):
            strongest_compact_by_template[hit.template_id] = hit

    for hit in sorted(
        strongest_compact_by_template.values(),
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
