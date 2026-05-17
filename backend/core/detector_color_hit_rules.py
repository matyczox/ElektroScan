"""Color-specific single-hit validation rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.detector_config import (
    COLOR_CONTENT_LABEL_MIN_CONTEXT,
    COLOR_CONTENT_LABEL_MIN_COVERAGE,
    COLOR_CONTENT_LABEL_MIN_MATCH,
    COLOR_CONTENT_LABEL_MIN_PURITY,
    COLOR_ELONGATED_STROKE_MAX_AREA,
    COLOR_ELONGATED_STROKE_MIN_ASPECT,
    COLOR_ELONGATED_STROKE_MIN_CONTEXT,
    COLOR_ELONGATED_STROKE_MIN_MATCH,
    COLOR_ELONGATED_STROKE_MIN_PURITY,
    COLOR_MIN_HUE_SIMILARITY,
    COLOR_RECOVERY_MIN_COLOR_SIMILARITY,
    COLOR_RECOVERY_MIN_CONTEXT,
    COLOR_RECOVERY_MIN_COVERAGE,
    COLOR_RECOVERY_MIN_PURITY,
    COLOR_TEXT_LABEL_MIN_CONTEXT,
    COLOR_TEXT_LABEL_MIN_COVERAGE,
    COLOR_TEXT_LABEL_MIN_MATCH,
    COLOR_TEXT_LABEL_MIN_PURITY,
    COLOR_TEXT_LABEL_WEAK_MATCH_MIN_CONTEXT,
)
from core.detector_models import CandidateHit
from core.detector_mask_builders import _roi_color_similarity


@dataclass(slots=True)
class ColorHitRuleState:
    """Computed color validation state needed by later gray/common checks."""

    early_color_similarity: float = 1.0
    strong_color_partial_geometry: bool = False
    color_recovery_geometry: bool = False
    color_full_label_geometry: bool = False
    color_wavy_low_match_geometry: bool = False
    reject_reason: str | None = None


def _reject(
    reason: str,
    *,
    early_color_similarity: float,
    strong_color_partial_geometry: bool = False,
    color_recovery_geometry: bool = False,
    color_full_label_geometry: bool = False,
    color_wavy_low_match_geometry: bool = False,
) -> ColorHitRuleState:
    return ColorHitRuleState(
        early_color_similarity=early_color_similarity,
        strong_color_partial_geometry=strong_color_partial_geometry,
        color_recovery_geometry=color_recovery_geometry,
        color_full_label_geometry=color_full_label_geometry,
        color_wavy_low_match_geometry=color_wavy_low_match_geometry,
        reject_reason=reason,
    )


def evaluate_color_hit_rules(
    *,
    hit: CandidateHit,
    plan_image: np.ndarray,
    plan_mask: np.ndarray,
    plan_hsv: np.ndarray | None,
    coverage: float,
    purity: float,
    context_purity: float,
    hit_area: int,
    hit_aspect: float,
    major_span_ratio: float,
    minor_span_ratio: float,
    path_variance_ratio: float,
) -> ColorHitRuleState:
    early_color_similarity = 1.0
    if hit.dominant_hsv is not None:
        early_color_similarity = _roi_color_similarity(
            plan_image,
            plan_mask,
            hit.bbox,
            hit.dominant_hsv,
            plan_hsv,
        )

    strong_color_partial_geometry = (
        hit.dominant_hsv is not None
        and early_color_similarity >= 0.90
        and hit.match_score >= 0.60
        and coverage >= 0.56
        and purity >= 0.75
        and context_purity >= 0.28
    )
    small_color_recovery_fragment = (
        hit.dominant_hsv is not None
        and hit.source == "template_color_recovery"
        and hit_area < 1_000
        and hit.match_score < 0.45
        and coverage < 0.70
    )
    if small_color_recovery_fragment:
        return _reject(
            "color_recovery_tiny_fragment",
            early_color_similarity=early_color_similarity,
            strong_color_partial_geometry=strong_color_partial_geometry,
        )

    color_recovery_geometry = (
        hit.dominant_hsv is not None
        and hit.source == "template_color_recovery"
        and early_color_similarity >= COLOR_RECOVERY_MIN_COLOR_SIMILARITY
        and hit.scale >= 0.90
        and 900 <= hit_area <= 3_800
        and hit_aspect <= 3.20
        and (
            (
                hit.match_score >= 0.52
                and coverage >= max(0.70, COLOR_RECOVERY_MIN_COVERAGE)
                and purity >= max(0.60, COLOR_RECOVERY_MIN_PURITY)
                and context_purity >= max(0.18, COLOR_RECOVERY_MIN_CONTEXT)
                and hit_area >= 1_100
            )
            or (
                hit.match_score >= 0.46
                and coverage >= 0.74
                and purity >= 0.66
                and context_purity >= 0.30
                and hit_area >= 900
            )
        )
    )
    if (
        hit.dominant_hsv is not None
        and hit.source == "template_color_recovery"
        and not color_recovery_geometry
    ):
        return _reject(
            "color_recovery_isolated_fragment",
            early_color_similarity=early_color_similarity,
            strong_color_partial_geometry=strong_color_partial_geometry,
            color_recovery_geometry=color_recovery_geometry,
        )

    color_full_label_source = (
        hit.source in {"template", "template_color_recovery"}
        or hit.source == "roi_inspector"
        or hit.source.startswith("template_parent_search_")
        or hit.source.startswith("template_promoted_")
    )
    color_full_label_geometry = (
        hit.dominant_hsv is not None
        and color_full_label_source
        and early_color_similarity >= 0.90
        and hit.scale >= 0.90
        and 900 <= hit_area <= 3_800
        and hit_aspect <= 3.20
        and (
            (
                hit.match_score >= 0.50
                and coverage >= 0.60
                and purity >= 0.58
                and context_purity >= 0.18
            )
            or (
                hit.source == "template_color_recovery"
                and color_recovery_geometry
            )
            or (
                hit.source != "template_color_recovery"
                and
                hit.match_score >= 0.40
                and coverage >= 0.58
                and purity >= 0.50
                and context_purity >= 0.16
                and hit_area >= 1_250
                and hit_aspect <= 2.35
            )
        )
    )
    color_full_text_label_geometry = (
        plan_hsv is not None
        and hit.is_text_label
        and color_full_label_source
        and hit.source != "pdf_text"
        and hit.scale >= 0.90
        and 900 <= hit_area <= 3_800
        and hit_aspect <= 3.20
        and hit.match_score >= 0.50
        and coverage >= 0.70
        and purity >= 0.75
        and context_purity >= 0.45
    )
    color_full_label_geometry = color_full_label_geometry or color_full_text_label_geometry
    color_low_content_text_fragment = False
    if hit.dominant_hsv is not None and hit.is_text_label and hit.content_pixel_count > 0:
        content_ink_ratio = hit.content_pixel_count / max(1, hit.pixel_count)
        low_content_symbol_label = content_ink_ratio <= 0.18
        sideways_weak_fragment = hit.rotation % 180 == 90 and hit.match_score < 0.60
        weak_local_fragment = (
            hit.match_score < 0.50
            and coverage < 0.78
            and context_purity < 0.45
        )
        color_low_content_text_fragment = low_content_symbol_label and (
            sideways_weak_fragment or weak_local_fragment
        )
    if color_low_content_text_fragment:
        return _reject(
            "color_text_fragment",
            early_color_similarity=early_color_similarity,
            strong_color_partial_geometry=strong_color_partial_geometry,
            color_recovery_geometry=color_recovery_geometry,
            color_full_label_geometry=color_full_label_geometry,
        )

    if (
        hit.dominant_hsv is not None
        and hit.source != "pdf_text"
        and early_color_similarity < COLOR_MIN_HUE_SIMILARITY
    ):
        return _reject(
            "color_similarity",
            early_color_similarity=early_color_similarity,
            strong_color_partial_geometry=strong_color_partial_geometry,
            color_recovery_geometry=color_recovery_geometry,
            color_full_label_geometry=color_full_label_geometry,
        )

    color_straight_stroke_fragment = (
        hit.dominant_hsv is not None
        and not hit.is_text_label
        and hit.source
        in {"template", "template_near_threshold", "template_color_recovery", "roi_inspector"}
        and hit_area <= int(COLOR_ELONGATED_STROKE_MAX_AREA)
        and hit_aspect >= float(COLOR_ELONGATED_STROKE_MIN_ASPECT)
        and major_span_ratio >= 0.52
        and minor_span_ratio <= 0.30
        and hit.match_score < 0.72
    )
    if color_straight_stroke_fragment:
        return _reject(
            "color_straight_stroke_fragment",
            early_color_similarity=early_color_similarity,
            strong_color_partial_geometry=strong_color_partial_geometry,
            color_recovery_geometry=color_recovery_geometry,
            color_full_label_geometry=color_full_label_geometry,
        )

    color_flat_elongated_fragment = (
        hit.dominant_hsv is not None
        and not hit.is_text_label
        and hit.source
        in {"template", "template_near_threshold", "template_color_recovery", "roi_inspector"}
        and hit_area <= int(COLOR_ELONGATED_STROKE_MAX_AREA)
        and hit_aspect >= float(COLOR_ELONGATED_STROKE_MIN_ASPECT)
        and major_span_ratio >= 0.88
        and minor_span_ratio <= 0.62
        and path_variance_ratio <= 0.035
        and hit.match_score < 0.70
    )
    if color_flat_elongated_fragment:
        return _reject(
            "color_flat_elongated_fragment",
            early_color_similarity=early_color_similarity,
            strong_color_partial_geometry=strong_color_partial_geometry,
            color_recovery_geometry=color_recovery_geometry,
            color_full_label_geometry=color_full_label_geometry,
        )

    if hit.dominant_hsv is not None and hit.is_text_label and hit.source == "template_content":
        if (
            hit.match_score < COLOR_CONTENT_LABEL_MIN_MATCH
            or coverage < COLOR_CONTENT_LABEL_MIN_COVERAGE
            or purity < COLOR_CONTENT_LABEL_MIN_PURITY
            or context_purity < COLOR_CONTENT_LABEL_MIN_CONTEXT
        ):
            return _reject(
                "color_content_fragment",
                early_color_similarity=early_color_similarity,
                strong_color_partial_geometry=strong_color_partial_geometry,
                color_recovery_geometry=color_recovery_geometry,
                color_full_label_geometry=color_full_label_geometry,
            )

    if (
        hit.dominant_hsv is not None
        and hit.is_text_label
        and hit.source in {"template", "template_near_threshold"}
    ):
        if (
            coverage < COLOR_TEXT_LABEL_MIN_COVERAGE
            or purity < COLOR_TEXT_LABEL_MIN_PURITY
            or context_purity < COLOR_TEXT_LABEL_MIN_CONTEXT
        ):
            return _reject(
                "color_text_geometry",
                early_color_similarity=early_color_similarity,
                strong_color_partial_geometry=strong_color_partial_geometry,
                color_recovery_geometry=color_recovery_geometry,
                color_full_label_geometry=color_full_label_geometry,
            )
        if (
            hit.match_score < COLOR_TEXT_LABEL_MIN_MATCH
            and context_purity < COLOR_TEXT_LABEL_WEAK_MATCH_MIN_CONTEXT
        ):
            return _reject(
                "color_text_weak_match",
                early_color_similarity=early_color_similarity,
                strong_color_partial_geometry=strong_color_partial_geometry,
                color_recovery_geometry=color_recovery_geometry,
                color_full_label_geometry=color_full_label_geometry,
            )

    color_wavy_low_match_geometry = (
        hit.dominant_hsv is not None
        and not hit.is_text_label
        and hit.source == "roi_inspector"
        and hit_area <= int(COLOR_ELONGATED_STROKE_MAX_AREA)
        and hit_aspect >= float(COLOR_ELONGATED_STROKE_MIN_ASPECT)
        and minor_span_ratio > 0.38
        and hit.match_score >= 0.45
        and coverage >= 0.80
        and purity >= 0.52
        and context_purity >= 0.16
    )

    color_elongated_stroke_fragment = (
        hit.dominant_hsv is not None
        and not hit.is_text_label
        and hit.source in {"template", "template_near_threshold"}
        and hit_area <= int(COLOR_ELONGATED_STROKE_MAX_AREA)
        and hit_aspect >= float(COLOR_ELONGATED_STROKE_MIN_ASPECT)
        and not color_wavy_low_match_geometry
        and (
            hit.match_score < float(COLOR_ELONGATED_STROKE_MIN_MATCH)
            or purity < float(COLOR_ELONGATED_STROKE_MIN_PURITY)
            or (
                context_purity < float(COLOR_ELONGATED_STROKE_MIN_CONTEXT)
                and hit.match_score < 0.68
            )
        )
    )
    if color_elongated_stroke_fragment:
        return _reject(
            "color_elongated_stroke_fragment",
            early_color_similarity=early_color_similarity,
            strong_color_partial_geometry=strong_color_partial_geometry,
            color_recovery_geometry=color_recovery_geometry,
            color_full_label_geometry=color_full_label_geometry,
            color_wavy_low_match_geometry=color_wavy_low_match_geometry,
        )

    return ColorHitRuleState(
        early_color_similarity=early_color_similarity,
        strong_color_partial_geometry=strong_color_partial_geometry,
        color_recovery_geometry=color_recovery_geometry,
        color_full_label_geometry=color_full_label_geometry,
        color_wavy_low_match_geometry=color_wavy_low_match_geometry,
    )
