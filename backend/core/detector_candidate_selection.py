"""Candidate winner-selection policy for detector clustering."""

from __future__ import annotations

import numpy as np

from core.detector_config import (
    COLOR_HUE_TOLERANCE,
    GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP,
    GRAY_FULLER_SYMBOL_MIN_AREA_RATIO,
    GRAY_FULLER_SYMBOL_MIN_COVERAGE,
    GRAY_FULLER_SYMBOL_MIN_PURITY,
    GRAY_TINY_GEOMETRY_MAX_TEMPLATE_PIXELS,
    GRAY_TINY_GEOMETRY_MIN_CONTEXT,
    GRAY_TINY_GEOMETRY_MIN_COVERAGE,
    GRAY_TINY_GEOMETRY_MIN_PURITY,
    GRAY_TINY_GEOMETRY_MIN_VERIFICATION,
    PROMOTED_PARENT_MIN_AREA_RATIO,
    PROMOTED_PARENT_MIN_VERIFICATION,
    PROMOTED_PARENT_OVERRIDE_MARGIN,
    TEXT_LABEL_FULLER_AREA_RATIO,
    TEXT_LABEL_FULLER_MAX_CONTENT_DROP,
    TEXT_LABEL_FULLER_MAX_VERIFICATION_DROP,
)
from core.detector_geometry import _axis_overlap_fraction, _bbox_metrics
from core.detector_masks import _hue_distance
from core.detector_models import CandidateHit
from core.detector_selection import candidate_quality_key


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
                and hit.match_score >= 0.64
                and hit.verification_score >= 0.56
                and hit.coverage >= 0.66
                and hit.purity >= 0.58
                and hit.context_purity >= 0.18
                and hit.match_score + 0.20 >= base_winner.match_score
                and hit.verification_score + 0.18 >= base_winner.verification_score
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


def _maybe_prefer_stronger_same_place_color_label(
    group_hits: list[CandidateHit],
    base_winner: CandidateHit,
) -> CandidateHit:
    """Prefer the stronger full color label over a weaker same-place family sibling."""

    if base_winner.dominant_hsv is None or base_winner.source == "pdf_text":
        return base_winner
    if not (base_winner.is_text_label or _is_color_label_like_shape(base_winner)):
        return base_winner

    base_area = max(1, base_winner.bbox[2] * base_winner.bbox[3])
    base_score = _color_template_score(base_winner)
    contenders: list[CandidateHit] = []

    for hit in group_hits:
        if (
            hit is base_winner
            or hit.dominant_hsv is None
            or hit.source == "pdf_text"
            or not (hit.is_text_label or _is_color_label_like_shape(hit))
        ):
            continue
        if _hue_distance(hit.dominant_hsv[0], base_winner.dominant_hsv[0]) > (
            COLOR_HUE_TOLERANCE + 6
        ):
            continue

        hit_area = max(1, hit.bbox[2] * hit.bbox[3])
        if not (base_area * 0.70 <= hit_area <= base_area * 1.35):
            continue

        inter_area, iou, iom, center_distance = _bbox_metrics(hit.bbox, base_winner.bbox)
        x_overlap = _axis_overlap_fraction(
            hit.bbox[0],
            hit.bbox[2],
            base_winner.bbox[0],
            base_winner.bbox[2],
        )
        y_overlap = _axis_overlap_fraction(
            hit.bbox[1],
            hit.bbox[3],
            base_winner.bbox[1],
            base_winner.bbox[3],
        )
        if not (
            inter_area > 0
            and (iou >= 0.30 or iom >= 0.52 or (x_overlap >= 0.72 and y_overlap >= 0.72))
            and center_distance <= 0.72
        ):
            continue
        base_is_fuller_parent_label = (
            base_winner.source.startswith("template_parent_search_")
            and base_area >= hit_area * 1.15
            and base_winner.match_score >= 0.64
            and base_winner.verification_score >= 0.56
            and base_winner.coverage >= 0.62
            and base_winner.purity >= 0.58
        )
        hit_is_parent_label = hit.source.startswith("template_parent_search_")
        if base_is_fuller_parent_label and not hit_is_parent_label:
            continue

        if hit.is_text_label and base_winner.is_text_label:
            hit_is_smaller_fragment = hit_area < base_area * 0.92
            hit_loses_text_payload = hit.content_score + 0.12 < base_winner.content_score
            hit_loses_local_context = (
                hit.context_purity + 0.10 < base_winner.context_purity
            )
            if hit_is_smaller_fragment and hit_loses_text_payload and hit_loses_local_context:
                continue

        if hit.match_score < 0.62 or hit.verification_score < 0.56:
            continue
        if hit.coverage < 0.62 or hit.purity < 0.58 or hit.context_purity < 0.18:
            continue
        if _color_template_score(hit) < base_score + 0.08:
            continue

        contenders.append(hit)

    if not contenders:
        return base_winner

    return max(
        contenders + [base_winner],
        key=lambda hit: (
            _color_template_score(hit),
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
            hit.verification_score + 0.12 < base_winner.verification_score
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
    base_winner = _maybe_prefer_stronger_same_place_color_label(group_hits, base_winner)
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
    base_winner = _maybe_prefer_stronger_same_place_color_label(group_hits, base_winner)
    if prefer_direct_color_family_parent:
        base_winner = _maybe_prefer_direct_color_family_parent(
            group_hits,
            base_winner,
            parent_ids_by_child,
        )
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

