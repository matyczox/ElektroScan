"""Candidate validation policy and validation-time mask cache."""

from __future__ import annotations

import cv2
import numpy as np

from core.detector_config import (
    COLOR_HUE_REJECTION_THRESHOLD,
    COLOR_HUE_TOLERANCE,
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
    COLOR_NEAR_THRESHOLD_RECOVERY_MIN_MATCH,
    COLOR_RECOVERY_MIN_COLOR_SIMILARITY,
    COLOR_RECOVERY_MIN_CONTEXT,
    COLOR_RECOVERY_MIN_COVERAGE,
    COLOR_RECOVERY_MIN_PURITY,
    COLOR_SAT_TOLERANCE,
    COLOR_TEXT_LABEL_MIN_CONTEXT,
    COLOR_TEXT_LABEL_MIN_COVERAGE,
    COLOR_TEXT_LABEL_MIN_MATCH,
    COLOR_TEXT_LABEL_MIN_PURITY,
    COLOR_TEXT_LABEL_WEAK_MATCH_MIN_CONTEXT,
    COLOR_VAL_TOLERANCE,
    CONTEXT_MARGIN_RATIO,
    DILATE_KERNEL,
    GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT,
    GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE,
    GRAY_COMPLEX_GEOMETRY_MIN_PURITY,
    GRAY_DISRUPTED_LABEL_MIN_ASPECT,
    GRAY_DISRUPTED_LABEL_MIN_COVERAGE,
    GRAY_DISRUPTED_LABEL_MIN_MATCH,
    GRAY_DISRUPTED_LABEL_MIN_PURITY,
    GRAY_DISRUPTED_LABEL_MIN_SCALE,
    GRAY_DIAGONAL_CONTENT_LABEL_MIN_CONTEXT,
    GRAY_DIAGONAL_CONTENT_LABEL_MIN_COVERAGE,
    GRAY_DIAGONAL_CONTENT_LABEL_MIN_MATCH,
    GRAY_DIAGONAL_CONTENT_LABEL_MIN_PURITY,
    GRAY_DIAGONAL_TEXT_LABEL_MAX_ASPECT,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_AREA,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_CONTEXT,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_COVERAGE,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_MATCH,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_PURITY,
    GRAY_INTERRUPTED_LABEL_DARK_MIN_CONTEXT,
    GRAY_INTERRUPTED_LABEL_DARK_MIN_COVERAGE,
    GRAY_INTERRUPTED_LABEL_DARK_MIN_MATCH,
    GRAY_INTERRUPTED_LABEL_DARK_MIN_PURITY,
    GRAY_INTERRUPTED_CONTENT_LABEL_DARK_MIN_CONTEXT,
    GRAY_INTERRUPTED_CONTENT_LABEL_DARK_MIN_COVERAGE,
    GRAY_INTERRUPTED_CONTENT_LABEL_DARK_MIN_MATCH,
    GRAY_INTERRUPTED_CONTENT_LABEL_DARK_MIN_PURITY,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ASPECT,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_TEMPLATE_AREA,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_CONTEXT,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_COVERAGE,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_MATCH,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_PURITY,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA,
    GRAY_ANGLED_INK_MIN_CONTEXT,
    GRAY_ANGLED_INK_MIN_COVERAGE,
    GRAY_ANGLED_INK_MIN_MATCH,
    GRAY_ANGLED_INK_MIN_PURITY,
    GRAY_ANGLED_INK_MIN_SCALE,
    GRAY_ANGLED_INK_STRONG_MIN_CONTEXT,
    GRAY_ANGLED_INK_STRONG_MIN_MATCH,
    GRAY_COHERENT_INK_MIN_CONTEXT,
    GRAY_COHERENT_INK_MIN_COVERAGE,
    GRAY_COHERENT_INK_ELONGATED_ASPECT,
    GRAY_COHERENT_INK_ELONGATED_MIN_COVERAGE,
    GRAY_COHERENT_INK_MIN_MATCH,
    GRAY_COHERENT_INK_MIN_PURITY,
    GRAY_COHERENT_INK_MIN_SCALE,
    GRAY_DARK_EVIDENCE_MIN_COVERAGE,
    GRAY_DARK_EVIDENCE_MIN_PIXELS,
    GRAY_LARGE_SCALE_PARTIAL_MAX_COVERAGE,
    GRAY_LARGE_SCALE_PARTIAL_MIN_CONTEXT,
    GRAY_LARGE_SCALE_PARTIAL_MIN_PURITY,
    GRAY_LARGE_SCALE_PARTIAL_MIN_SCALE,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_CONTEXT,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_COVERAGE,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_MATCH,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_PURITY,
    GRAY_LINE_CROSSED_LABEL_MIN_CONTEXT,
    GRAY_LINE_CROSSED_LABEL_MIN_COVERAGE,
    GRAY_LINE_CROSSED_LABEL_MIN_MATCH,
    GRAY_LINE_CROSSED_LABEL_MIN_PURITY,
    GRAY_LINE_CROSSED_LABEL_MIN_SCALE,
    GRAY_MID_GEOMETRY_MIN_CONTEXT,
    GRAY_MID_GEOMETRY_MIN_COVERAGE,
    GRAY_MID_GEOMETRY_MIN_MATCH,
    GRAY_MID_GEOMETRY_MIN_PURITY,
    GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_CONTEXT,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_COVERAGE,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_MATCH,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_PURITY,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA,
    GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS,
    GRAY_RAW_SCAN_THRESHOLD,
    GRAY_RECT_FRAME_MAX_CENTER_DENSITY,
    GRAY_RECT_FRAME_MAX_DENSITY,
    GRAY_RECT_FRAME_MIN_ASPECT,
    GRAY_RECT_FRAME_MIN_DENSITY,
    GRAY_RECT_FRAME_LONG_EDGE_MIN_RUN,
    GRAY_RECT_FRAME_SHORT_EDGE_STRONG_RUN,
    GRAY_RECT_FRAME_SHORT_EDGE_WEAK_RUN,
    GRAY_RECT_FRAME_STRONG_EDGE_COVERAGE,
    GRAY_RECT_FRAME_WEAK_EDGE_COVERAGE,
    GRAY_SMALL_SCALE_COMPACT_MAX_ASPECT,
    GRAY_SMALL_SCALE_COMPACT_MIN_COVERAGE,
    GRAY_SMALL_SCALE_ELONGATED_ASPECT,
    GRAY_SMALL_SCALE_ELONGATED_MAX_DENSITY,
    GRAY_SMALL_SCALE_ELONGATED_MIN_CONTEXT,
    GRAY_SMALL_SCALE_ELONGATED_MIN_COVERAGE,
    GRAY_SMALL_SCALE_ELONGATED_MIN_PURITY,
    GRAY_SMALL_SCALE_HIGH_PURITY_MAX_CONTEXT,
    GRAY_SMALL_SCALE_HIGH_PURITY_MAX_COVERAGE,
    GRAY_SMALL_SCALE_HIGH_PURITY_MAX_SCALE,
    GRAY_SMALL_SCALE_HIGH_PURITY_MIN_PURITY,
    GRAY_SMALL_SCALE_MIN_COVERAGE,
    GRAY_SMALL_SCALE_SUSPICIOUS_PURITY,
    GRAY_SMALL_SCALE_THRESHOLD,
    GRAY_SPARSE_TINY_FRAGMENT_MAX_DENSITY,
    GRAY_SPARSE_TINY_FRAGMENT_MAX_DIMENSION,
    GRAY_SPARSE_TINY_FRAGMENT_MAX_SCALE,
    GRAY_SPARSE_TINY_FRAGMENT_MIN_ASPECT,
    GRAY_TINY_FRAGMENT_MAX_CONTEXT,
    GRAY_TINY_FRAGMENT_MAX_DIMENSION,
    GRAY_TINY_FRAGMENT_MAX_SCALE,
    GRAY_STRONG_GEOMETRY_MIN_COVERAGE,
    GRAY_STRONG_GEOMETRY_MIN_MATCH,
    GRAY_STRONG_RESCUE_MIN_PURITY,
    GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
    GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    HSV_LOWER,
    HSV_UPPER,
    LABEL_CONTENT_MAX_RATIO,
    LABEL_CONTENT_MIN_PIXELS,
    LABEL_CONTENT_MIN_RATIO,
    LABEL_CONTENT_MIN_SCORE,
    LABEL_CONTENT_SCORE_WEIGHT,
    LABEL_FULL_WIDTH_CONTENT_MIN_SCORE,
    LABEL_LINE_MIN_RATIO,
    LABEL_TEMPLATE_MIN_ASPECT_RATIO,
    LABEL_TEMPLATE_MIN_WIDTH,
    LOCAL_MAX_KERNEL_RATIO,
    LOW_MATCH_STRICT_THRESHOLD,
    MAX_CENTROID_OFFSET_RATIO,
    MIN_CONTEXT_PURITY,
    MIN_COVERAGE_RATIO,
    MIN_PURITY_RATIO,
    MIN_VERIFICATION_SCORE,
    NOISY_PARTIAL_CONTEXT_THRESHOLD,
    NOISY_PARTIAL_COVERAGE_THRESHOLD,
    NOISY_PARTIAL_PURITY_THRESHOLD,
    ROI_COMPONENT_DILATE_PIXELS,
    ROI_FULL_SCAN_AREA_RATIO,
    ROI_MAX_COMPONENTS,
    ROI_MERGE_GAP_PIXELS,
    ROI_MIN_COMPONENT_PIXELS,
    ROI_PADDING_RATIO,
)
from core.detector_models import CandidateHit
from core.detector_color_hit_rules import evaluate_color_hit_rules
from core.detector_validation_cache import (
    ValidationMaskCache,
    _context_purity,
    _gray_rect_frame_evidence_ok,
    _is_gray_rect_frame_candidate,
)
from core.detector_mask_builders import (
    _build_search_rois,
    _cached_dilated_mask,
    _color_mask_for_template,
    _dominant_hsv_color,
    _find_local_maxima,
    _hsv_mask,
    _hue_distance,
    _ink_mask,
    _roi_color_similarity,
    _suppress_long_strokes,
)
from core.detector_shape_metrics import (
    _extract_label_content_mask,
    _foreground_path_variance_ratio,
    _foreground_span_ratios,
    _label_content_score,
    _mask_bbox,
    _mask_centroid,
    _roi_mask,
    _thickness_normalized_mask,
    _tight_mask_crop,
)


def _validate_template_hit(
    hit: CandidateHit,
    plan_mask: np.ndarray,
    plan_image: np.ndarray,
    reasons: dict[str, int] | None = None,
    plan_hsv: np.ndarray | None = None,
    evidence_mask: np.ndarray | None = None,
    relaxed_evidence_mask: np.ndarray | None = None,
    validation_cache: ValidationMaskCache | None = None,
) -> bool:
    """Validate a candidate by foreground overlap, purity and hue consistency.

    When ``reasons`` is provided, the first failed check name is incremented in it
    so callers can build an aggregate rejection histogram without re-running checks.
    """

    def _record(reason: str) -> None:
        if reasons is not None:
            reasons[reason] = reasons.get(reason, 0) + 1

    if hit.transformed_mask is None:
        return True

    roi = _roi_mask(plan_mask, hit.bbox)
    if roi is None or roi.shape != hit.transformed_mask.shape:
        _record("roi_shape")
        return False

    roi_foreground = (
        validation_cache.foreground_count(plan_mask, hit.bbox)
        if validation_cache is not None
        else int(cv2.countNonZero(roi))
    )
    if roi_foreground == 0 or hit.pixel_count <= 0:
        _record("empty_roi")
        return False

    intersection_mask = cv2.bitwise_and(roi, hit.transformed_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    coverage = intersection / hit.pixel_count
    purity = intersection / roi_foreground

    if coverage < MIN_COVERAGE_RATIO:
        _record("coverage")
        return False
    if purity < MIN_PURITY_RATIO:
        _record("purity")
        return False

    if (
        context_purity := _context_purity(
            plan_mask,
            hit.bbox,
            intersection_mask,
            explained_pixels=intersection,
            validation_cache=validation_cache,
        )
    ) <= 0.0:
        _record("context_purity")
        return False

    # Keep diagnostics useful for rejected hits as well.  These fields are
    # overwritten with the final rounded values at the end for accepted hits.
    hit.coverage = round(coverage, 4)
    hit.purity = round(purity, 4)
    hit.context_purity = round(context_purity, 4)
    hit_area = max(1, hit.bbox[2] * hit.bbox[3])
    hit_aspect = max(hit.bbox[2] / max(1, hit.bbox[3]), hit.bbox[3] / max(1, hit.bbox[2]))
    span_x_ratio, span_y_ratio = _foreground_span_ratios(roi)
    minor_span_ratio = span_y_ratio if hit.bbox[2] >= hit.bbox[3] else span_x_ratio
    major_span_ratio = span_x_ratio if hit.bbox[2] >= hit.bbox[3] else span_y_ratio
    path_variance_ratio = _foreground_path_variance_ratio(roi)
    interrupted_label_recovery_seed = (
        hit.dominant_hsv is None
        and hit.source == "template_interrupted_recovery"
        and hit.match_score >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_MATCH
        and GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA
        <= hit_area
        <= GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_TEMPLATE_AREA
        and hit_aspect <= GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ASPECT
        and coverage >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_COVERAGE
        and purity >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_PURITY
        and context_purity >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_CONTEXT
    )
    color_state = evaluate_color_hit_rules(
        hit=hit,
        plan_image=plan_image,
        plan_mask=plan_mask,
        plan_hsv=plan_hsv,
        coverage=coverage,
        purity=purity,
        context_purity=context_purity,
        hit_area=hit_area,
        hit_aspect=hit_aspect,
        major_span_ratio=major_span_ratio,
        minor_span_ratio=minor_span_ratio,
        path_variance_ratio=path_variance_ratio,
    )
    early_color_similarity = color_state.early_color_similarity
    strong_color_partial_geometry = color_state.strong_color_partial_geometry
    color_recovery_geometry = color_state.color_recovery_geometry
    color_full_label_geometry = color_state.color_full_label_geometry
    color_wavy_low_match_geometry = color_state.color_wavy_low_match_geometry
    if color_state.reject_reason is not None:
        _record(color_state.reject_reason)
        return False

    if (
        context_purity < NOISY_PARTIAL_CONTEXT_THRESHOLD
        and coverage < NOISY_PARTIAL_COVERAGE_THRESHOLD
        and purity < NOISY_PARTIAL_PURITY_THRESHOLD
        and not strong_color_partial_geometry
        and not color_recovery_geometry
        and not color_wavy_low_match_geometry
    ):
        _record("noisy_partial")
        return False

    is_gray_rect_frame = _is_gray_rect_frame_candidate(hit)
    effective_evidence_mask = (
        relaxed_evidence_mask
        if is_gray_rect_frame and relaxed_evidence_mask is not None
        else evidence_mask
    )
    if is_gray_rect_frame:
        frame_roi = roi
        if effective_evidence_mask is not None:
            evidence_roi = _roi_mask(effective_evidence_mask, hit.bbox)
            if evidence_roi is not None and evidence_roi.shape == hit.transformed_mask.shape:
                frame_roi = evidence_roi
        if not _gray_rect_frame_evidence_ok(frame_roi, hit.transformed_mask):
            _record("gray_rect_frame_evidence")
            return False

    gray_evidence_failed = False
    if hit.dominant_hsv is None and effective_evidence_mask is not None:
        evidence_roi = _roi_mask(effective_evidence_mask, hit.bbox)
        if evidence_roi is None or evidence_roi.shape != hit.transformed_mask.shape:
            gray_evidence_failed = True
        else:
            evidence_intersection = int(
                cv2.countNonZero(cv2.bitwise_and(evidence_roi, hit.transformed_mask))
            )
            evidence_coverage = evidence_intersection / max(1, hit.pixel_count)
            gray_evidence_failed = (
                evidence_intersection < GRAY_DARK_EVIDENCE_MIN_PIXELS
                or evidence_coverage < GRAY_DARK_EVIDENCE_MIN_COVERAGE
            )

    diagonal_content_label_seed = False
    if hit.dominant_hsv is None:
        template_area = max(1, hit.bbox[2] * hit.bbox[3])
        template_density = hit.pixel_count / template_area
        diagonal_content_label_seed = (
            hit.is_text_label
            and hit.source == "template_content"
            and hit.rotation % 90 != 0
            and hit.match_score >= GRAY_DIAGONAL_CONTENT_LABEL_MIN_MATCH
            and coverage >= GRAY_DIAGONAL_CONTENT_LABEL_MIN_COVERAGE
            and purity >= GRAY_DIAGONAL_CONTENT_LABEL_MIN_PURITY
            and context_purity >= GRAY_DIAGONAL_CONTENT_LABEL_MIN_CONTEXT
        )
        normalized_roi = _thickness_normalized_mask(roi)
        normalized_template = (
            validation_cache.normalized_mask(hit.transformed_mask)
            if validation_cache is not None
            else _thickness_normalized_mask(hit.transformed_mask)
        )
        normalized_intersection = int(
            cv2.countNonZero(cv2.bitwise_and(normalized_roi, normalized_template))
        )
        normalized_template_pixels = (
            validation_cache.normalized_pixel_count(hit.transformed_mask)
            if validation_cache is not None
            else max(1, int(cv2.countNonZero(normalized_template)))
        )
        normalized_roi_pixels = max(1, int(cv2.countNonZero(normalized_roi)))
        normalized_coverage = normalized_intersection / normalized_template_pixels
        normalized_purity = normalized_intersection / normalized_roi_pixels

        # In gray/black PDFs every text glyph and wall line shares the same
        # ink mask. Sparse outline templates are especially prone to matching
        # random text strokes after rotation, so require a much fuller shape
        # agreement before accepting them.
        if template_density < 0.18:
            if max(coverage, normalized_coverage) < 0.62:
                _record("gray_coverage")
                return False
            if (
                max(purity, normalized_purity) < 0.30
                and context_purity < 0.55
                and not diagonal_content_label_seed
            ):
                _record("gray_purity")
                return False
            if hit.match_score < 0.68 and max(coverage, normalized_coverage) < 0.74:
                _record("gray_low_match")
                return False
        elif purity < 0.18 and context_purity < 0.45:
            _record("gray_purity")
            return False

    # Small-scale anomaly: a sparse template (e.g. 01 rectangle at 0.5×) that
    # latches onto an isolated stroke fragment shows partial coverage AND
    # anomalously high purity (very little foreign ink in the ROI). Real gray
    # plan detections sit in dense ink and have purity ~0.5-0.7. Reject only
    # this combination to avoid killing valid imperfect-coverage hits.
    if (
        hit.dominant_hsv is None
        and hit.scale <= GRAY_SMALL_SCALE_THRESHOLD
        and coverage < GRAY_SMALL_SCALE_MIN_COVERAGE
        and purity > GRAY_SMALL_SCALE_SUSPICIOUS_PURITY
    ):
        _record("gray_small_scale_anomaly")
        return False

    is_sparse_elongated = False
    strong_gray_elongated_geometry = False
    if hit.dominant_hsv is None and hit.scale <= GRAY_SMALL_SCALE_THRESHOLD:
        template_area = hit_area
        template_density = hit.pixel_count / template_area
        aspect = hit_aspect
        is_sparse_elongated = (
            template_density <= GRAY_SMALL_SCALE_ELONGATED_MAX_DENSITY
            and aspect >= GRAY_SMALL_SCALE_ELONGATED_ASPECT
        )
        if is_sparse_elongated:
            strong_gray_elongated_geometry = (
                coverage >= GRAY_SMALL_SCALE_ELONGATED_MIN_COVERAGE
                and purity >= GRAY_SMALL_SCALE_ELONGATED_MIN_PURITY
                and context_purity >= GRAY_SMALL_SCALE_ELONGATED_MIN_CONTEXT
            )
        if is_sparse_elongated and coverage < GRAY_SMALL_SCALE_ELONGATED_MIN_COVERAGE:
            _record("gray_small_scale_elongated_coverage")
            return False
        if (
            is_sparse_elongated
            and hit.match_score < GRAY_STRONG_GEOMETRY_MIN_MATCH
            and not interrupted_label_recovery_seed
        ):
            _record("gray_elongated_low_match")
            return False
        if (
            aspect <= GRAY_SMALL_SCALE_COMPACT_MAX_ASPECT
            and coverage < GRAY_SMALL_SCALE_COMPACT_MIN_COVERAGE
        ):
            _record("gray_small_scale_compact_coverage")
            return False

    if (
        hit.dominant_hsv is None
        and hit.scale <= GRAY_SMALL_SCALE_HIGH_PURITY_MAX_SCALE
        and coverage < GRAY_SMALL_SCALE_HIGH_PURITY_MAX_COVERAGE
        and purity > GRAY_SMALL_SCALE_HIGH_PURITY_MIN_PURITY
        and context_purity < GRAY_SMALL_SCALE_HIGH_PURITY_MAX_CONTEXT
    ):
        _record("gray_small_scale_high_purity_partial")
        return False

    near_threshold_geometry_seed = (
        hit.dominant_hsv is None
        and hit.source == "template_near_threshold"
        and hit.match_score >= GRAY_NEAR_THRESHOLD_RECOVERY_MIN_MATCH
        and coverage >= GRAY_NEAR_THRESHOLD_RECOVERY_MIN_COVERAGE
        and purity >= GRAY_NEAR_THRESHOLD_RECOVERY_MIN_PURITY
        and context_purity >= GRAY_NEAR_THRESHOLD_RECOVERY_MIN_CONTEXT
    )
    line_crossed_near_threshold_common = (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.source == "template_near_threshold"
        and hit.scale >= GRAY_LINE_CROSSED_LABEL_MIN_SCALE
        and GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA
        <= hit_area
        <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA
        and hit_aspect <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT
    )
    line_crossed_near_threshold_label_geometry = (
        line_crossed_near_threshold_common
        and (
            (
                hit.match_score >= GRAY_LINE_CROSSED_LABEL_MIN_MATCH
                and coverage >= GRAY_LINE_CROSSED_LABEL_MIN_COVERAGE
                and purity >= GRAY_LINE_CROSSED_LABEL_MIN_PURITY
                and context_purity >= GRAY_LINE_CROSSED_LABEL_MIN_CONTEXT
            )
            or (
                hit.match_score >= GRAY_LINE_CROSSED_LABEL_ALT_MIN_MATCH
                and coverage >= GRAY_LINE_CROSSED_LABEL_ALT_MIN_COVERAGE
                and purity >= GRAY_LINE_CROSSED_LABEL_ALT_MIN_PURITY
                and context_purity >= GRAY_LINE_CROSSED_LABEL_ALT_MIN_CONTEXT
            )
        )
    )
    line_interrupted_dark_label_geometry = (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.source == "template"
        and hit.scale >= GRAY_DISRUPTED_LABEL_MIN_SCALE
        and hit_aspect >= GRAY_DISRUPTED_LABEL_MIN_ASPECT
        and hit_area >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA
        and hit.match_score >= GRAY_INTERRUPTED_LABEL_DARK_MIN_MATCH
        and coverage >= GRAY_INTERRUPTED_LABEL_DARK_MIN_COVERAGE
        and purity >= GRAY_INTERRUPTED_LABEL_DARK_MIN_PURITY
        and context_purity >= GRAY_INTERRUPTED_LABEL_DARK_MIN_CONTEXT
    )
    line_interrupted_content_label_strict = (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.source == "template_content"
        and hit.scale >= GRAY_DISRUPTED_LABEL_MIN_SCALE
        and hit_area >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA
        and hit.match_score >= GRAY_INTERRUPTED_CONTENT_LABEL_DARK_MIN_MATCH
        and coverage >= GRAY_INTERRUPTED_CONTENT_LABEL_DARK_MIN_COVERAGE
        and purity >= GRAY_INTERRUPTED_CONTENT_LABEL_DARK_MIN_PURITY
        and context_purity >= GRAY_INTERRUPTED_CONTENT_LABEL_DARK_MIN_CONTEXT
    )
    line_interrupted_content_label_relaxed = (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.source == "template_content"
        and hit.scale >= 0.85
        and GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA
        <= hit_area
        <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA
        and hit_aspect <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT
        and hit.match_score >= 0.69
        and coverage >= 0.84
        and purity >= 0.28
        and context_purity >= 0.16
    )
    line_interrupted_content_label_geometry = (
        line_interrupted_content_label_strict
        or line_interrupted_content_label_relaxed
    )

    if (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and not near_threshold_geometry_seed
        and not interrupted_label_recovery_seed
        and hit.scale <= GRAY_TINY_FRAGMENT_MAX_SCALE
        and max(hit.bbox[2], hit.bbox[3]) <= GRAY_TINY_FRAGMENT_MAX_DIMENSION
        and context_purity < GRAY_TINY_FRAGMENT_MAX_CONTEXT
    ):
        _record("gray_tiny_fragment")
        return False

    if (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and not near_threshold_geometry_seed
        and not interrupted_label_recovery_seed
        and hit.scale <= GRAY_SPARSE_TINY_FRAGMENT_MAX_SCALE
        and max(hit.bbox[2], hit.bbox[3]) <= GRAY_SPARSE_TINY_FRAGMENT_MAX_DIMENSION
        and max(hit.bbox[2], hit.bbox[3]) / max(1, min(hit.bbox[2], hit.bbox[3]))
        >= GRAY_SPARSE_TINY_FRAGMENT_MIN_ASPECT
        and template_density <= GRAY_SPARSE_TINY_FRAGMENT_MAX_DENSITY
        and context_purity < GRAY_TINY_FRAGMENT_MAX_CONTEXT
    ):
        _record("gray_sparse_tiny_fragment")
        return False

    if (
        hit.dominant_hsv is None
        and hit.scale >= GRAY_LARGE_SCALE_PARTIAL_MIN_SCALE
        and coverage < GRAY_LARGE_SCALE_PARTIAL_MAX_COVERAGE
        and purity > GRAY_LARGE_SCALE_PARTIAL_MIN_PURITY
        and context_purity > GRAY_LARGE_SCALE_PARTIAL_MIN_CONTEXT
    ):
        _record("gray_large_scale_partial")
        return False

    template_centroid = (
        validation_cache.centroid(hit.transformed_mask)
        if validation_cache is not None
        else _mask_centroid(hit.transformed_mask)
    )
    intersection_centroid = _mask_centroid(intersection_mask)
    if template_centroid is None or intersection_centroid is None:
        _record("centroid")
        return False

    centroid_offset = float(
        np.hypot(
            template_centroid[0] - intersection_centroid[0],
            template_centroid[1] - intersection_centroid[1],
        )
    )
    bbox_diagonal = max(1.0, float(np.hypot(hit.bbox[2], hit.bbox[3])))
    centroid_offset_ratio = centroid_offset / bbox_diagonal
    if centroid_offset_ratio > MAX_CENTROID_OFFSET_RATIO:
        _record("centroid_offset")
        return False

    coherent_ink_geometry = (
        hit.dominant_hsv is None
        and hit.scale >= GRAY_COHERENT_INK_MIN_SCALE
        and hit.match_score >= GRAY_COHERENT_INK_MIN_MATCH
        and coverage >= GRAY_COHERENT_INK_MIN_COVERAGE
        and purity >= GRAY_COHERENT_INK_MIN_PURITY
        and context_purity >= GRAY_COHERENT_INK_MIN_CONTEXT
        and (
            hit_aspect < GRAY_COHERENT_INK_ELONGATED_ASPECT
            or coverage >= GRAY_COHERENT_INK_ELONGATED_MIN_COVERAGE
        )
    )
    angled_ink_geometry = (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and hit.rotation % 90 != 0
        and hit.scale >= GRAY_ANGLED_INK_MIN_SCALE
        and hit.match_score >= GRAY_ANGLED_INK_MIN_MATCH
        and coverage >= GRAY_ANGLED_INK_MIN_COVERAGE
        and purity >= GRAY_ANGLED_INK_MIN_PURITY
        and (
            context_purity >= GRAY_ANGLED_INK_MIN_CONTEXT
            or (
                hit.match_score >= GRAY_ANGLED_INK_STRONG_MIN_MATCH
                and context_purity >= GRAY_ANGLED_INK_STRONG_MIN_CONTEXT
            )
        )
    )
    diagonal_text_label_geometry = (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.rotation % 90 != 0
        and hit_area >= GRAY_DIAGONAL_TEXT_LABEL_MIN_AREA
        and hit_aspect <= GRAY_DIAGONAL_TEXT_LABEL_MAX_ASPECT
        and hit.match_score >= GRAY_DIAGONAL_TEXT_LABEL_MIN_MATCH
        and coverage >= GRAY_DIAGONAL_TEXT_LABEL_MIN_COVERAGE
        and purity >= GRAY_DIAGONAL_TEXT_LABEL_MIN_PURITY
        and context_purity >= GRAY_DIAGONAL_TEXT_LABEL_MIN_CONTEXT
    )
    diagonal_template_text_candidate = (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.source in {"template", "template_near_threshold"}
        and hit.rotation % 90 != 0
        and hit_area >= GRAY_DIAGONAL_TEXT_LABEL_MIN_AREA
        and hit_aspect <= GRAY_DIAGONAL_TEXT_LABEL_MAX_ASPECT
    )
    diagonal_template_text_fragment = (
        diagonal_template_text_candidate
        and not (
            (
                hit.match_score >= 0.66
                and coverage >= 0.88
                and purity >= 0.72
                and context_purity >= 0.34
            )
            or (
                hit.match_score >= 0.62
                and coverage >= 0.90
                and purity >= 0.82
                and context_purity >= 0.45
            )
            or (
                hit.scale >= 1.0
                and hit.match_score >= 0.60
                and coverage >= 0.94
                and purity >= 0.50
                and context_purity >= 0.26
            )
        )
    )
    if diagonal_template_text_fragment:
        _record("gray_diagonal_text_fragment")
        return False
    near_threshold_recovery_geometry = (
        near_threshold_geometry_seed
        and GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA
        <= hit.bbox[2] * hit.bbox[3]
        <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA
        and hit_aspect <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT
    )
    disrupted_label_geometry = (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.source == "template"
        and hit.scale >= GRAY_DISRUPTED_LABEL_MIN_SCALE
        and hit_aspect >= GRAY_DISRUPTED_LABEL_MIN_ASPECT
        and hit.match_score >= GRAY_DISRUPTED_LABEL_MIN_MATCH
        and coverage >= GRAY_DISRUPTED_LABEL_MIN_COVERAGE
        and purity >= GRAY_DISRUPTED_LABEL_MIN_PURITY
    )
    strong_gray_geometry = (
        hit.dominant_hsv is None
        and hit.match_score >= GRAY_STRONG_GEOMETRY_MIN_MATCH
        and coverage >= GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        and purity >= GRAY_STRONG_RESCUE_MIN_PURITY
    ) or (
        strong_gray_elongated_geometry
        and hit.match_score >= GRAY_STRONG_GEOMETRY_MIN_MATCH
    ) or (
        hit.dominant_hsv is None
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_RAW_SCAN_THRESHOLD
        and coverage >= GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE
        and purity >= GRAY_COMPLEX_GEOMETRY_MIN_PURITY
        and context_purity >= GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT
    ) or (
        hit.dominant_hsv is None
        and hit.pixel_count >= GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_MID_GEOMETRY_MIN_MATCH
        and coverage >= GRAY_MID_GEOMETRY_MIN_COVERAGE
        and purity >= GRAY_MID_GEOMETRY_MIN_PURITY
        and context_purity >= GRAY_MID_GEOMETRY_MIN_CONTEXT
    ) or coherent_ink_geometry or angled_ink_geometry or diagonal_text_label_geometry or diagonal_content_label_seed or near_threshold_recovery_geometry or line_crossed_near_threshold_label_geometry or interrupted_label_recovery_seed or disrupted_label_geometry or line_interrupted_dark_label_geometry or line_interrupted_content_label_geometry
    if gray_evidence_failed and not strong_gray_geometry:
        _record("gray_dark_evidence")
        return False
    if (
        hit.match_score < LOW_MATCH_STRICT_THRESHOLD
        and context_purity < MIN_CONTEXT_PURITY
        and not strong_gray_geometry
        and not color_recovery_geometry
        and not color_full_label_geometry
        and not color_wavy_low_match_geometry
    ):
        _record("low_match_strict")
        return False

    color_similarity = early_color_similarity
    if color_similarity <= 0.0:
        _record("color_similarity")
        return False

    verification_score = (
        0.45 * hit.match_score + 0.20 * coverage + 0.15 * purity + 0.20 * context_purity
    )

    content_score = 0.0
    if hit.is_text_label:
        content_score = _label_content_score(
            roi,
            hit.content_mask,
            hit.content_pixel_count,
            validation_cache=validation_cache,
        )
        content_threshold = LABEL_CONTENT_MIN_SCORE
        if hit.content_bbox is not None:
            content_width_ratio = hit.content_bbox[2] / max(1, hit.bbox[2])
            if (
                content_width_ratio >= 0.80
                and hit.source == "template"
                and hit.match_score >= 0.66
                and coverage >= 0.64
                and purity >= 0.74
            ):
                content_threshold = LABEL_FULL_WIDTH_CONTENT_MIN_SCORE
        if line_crossed_near_threshold_label_geometry:
            content_threshold = min(content_threshold, 0.62)
        if hit.dominant_hsv is not None and color_full_label_geometry:
            content_threshold = min(content_threshold, 0.40)

        if content_score < content_threshold:
            _record("content_score")
            return False
        verification_score = (
            1.0 - LABEL_CONTENT_SCORE_WEIGHT
        ) * verification_score + LABEL_CONTENT_SCORE_WEIGHT * content_score

    if (
        verification_score < MIN_VERIFICATION_SCORE
        and not color_recovery_geometry
        and not color_full_label_geometry
    ):
        _record("verification")
        return False

    hit.coverage = round(coverage, 4)
    hit.purity = round(purity, 4)
    hit.context_purity = round(context_purity, 4)
    hit.color_similarity = round(color_similarity, 4)
    hit.verification_score = round(verification_score, 4)
    hit.content_score = round(content_score, 4)
    return True
