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
from core.detector_color_candidate_selection import (
    _color_template_score,
    _is_color_label_like_shape,
    _is_strong_color_satellite_candidate,
    _maybe_prefer_coverage_color_text_label,
    _maybe_prefer_direct_color_family_parent,
    _maybe_prefer_fuller_color_candidate,
    _maybe_prefer_stronger_same_place_color_label,
    _maybe_prefer_stronger_same_template_color_variant,
    _maybe_prefer_tighter_color_template,
)
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
