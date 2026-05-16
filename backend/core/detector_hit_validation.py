"""Candidate validation policy and validation-time mask cache."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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
    _context_purity,
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


@dataclass(slots=True)
class ValidationMaskCache:
    """Per-validation-run cache for template-mask derived metrics."""

    normalized_masks: dict[int, np.ndarray]
    normalized_pixels: dict[int, int]
    centroids: dict[int, tuple[float, float] | None]
    content_bboxes: dict[int, tuple[int, int, int, int] | None]
    foreground_integrals: dict[int, np.ndarray]

    @classmethod
    def build(
        cls,
        hits: list[CandidateHit],
        plan_masks: Iterable[np.ndarray] = (),
    ) -> "ValidationMaskCache":
        transformed_masks: dict[int, np.ndarray] = {}
        content_masks: dict[int, np.ndarray] = {}
        for hit in hits:
            if hit.transformed_mask is not None:
                transformed_masks[id(hit.transformed_mask)] = hit.transformed_mask
            if hit.content_mask is not None:
                content_masks[id(hit.content_mask)] = hit.content_mask

        foreground_integrals: dict[int, np.ndarray] = {}
        for mask in plan_masks:
            if not isinstance(mask, np.ndarray):
                continue
            mask_id = id(mask)
            if mask_id in foreground_integrals:
                continue
            foreground_integrals[mask_id] = cv2.integral(
                (mask > 0).astype(np.uint8, copy=False),
                sdepth=cv2.CV_32S,
            )

        normalized_masks: dict[int, np.ndarray] = {}
        normalized_pixels: dict[int, int] = {}
        centroids: dict[int, tuple[float, float] | None] = {}
        for mask_id, mask in transformed_masks.items():
            normalized = _thickness_normalized_mask(mask)
            normalized_masks[mask_id] = normalized
            normalized_pixels[mask_id] = max(1, int(cv2.countNonZero(normalized)))
            centroids[mask_id] = _mask_centroid(mask)

        return cls(
            normalized_masks=normalized_masks,
            normalized_pixels=normalized_pixels,
            centroids=centroids,
            content_bboxes={
                mask_id: _mask_bbox(mask)
                for mask_id, mask in content_masks.items()
            },
            foreground_integrals=foreground_integrals,
        )

    def normalized_mask(self, mask: np.ndarray) -> np.ndarray:
        cached = self.normalized_masks.get(id(mask))
        if cached is not None:
            return cached
        return _thickness_normalized_mask(mask)

    def normalized_pixel_count(self, mask: np.ndarray) -> int:
        cached = self.normalized_pixels.get(id(mask))
        if cached is not None:
            return cached
        return max(1, int(cv2.countNonZero(self.normalized_mask(mask))))

    def centroid(self, mask: np.ndarray) -> tuple[float, float] | None:
        cached = self.centroids.get(id(mask))
        if id(mask) in self.centroids:
            return cached
        return _mask_centroid(mask)

    def content_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        cached = self.content_bboxes.get(id(mask))
        if id(mask) in self.content_bboxes:
            return cached
        return _mask_bbox(mask)

    def foreground_count(
        self,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> int:
        integral = self.foreground_integrals.get(id(mask))
        if integral is None:
            roi = _roi_mask(mask, bbox)
            return 0 if roi is None else int(cv2.countNonZero(roi))

        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
        return int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])



def _is_gray_rect_frame_candidate(hit: CandidateHit) -> bool:
    if hit.transformed_mask is None:
        return False

    width, height = hit.bbox[2], hit.bbox[3]
    aspect = max(width / max(1, height), height / max(1, width))
    density = hit.pixel_count / max(1, width * height)
    return (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and aspect >= GRAY_RECT_FRAME_MIN_ASPECT
        and GRAY_RECT_FRAME_MIN_DENSITY <= density <= GRAY_RECT_FRAME_MAX_DENSITY
    )


def _gray_rect_frame_evidence_ok(roi: np.ndarray, template_mask: np.ndarray) -> bool:
    """Check that a hollow frame has real ink on its perimeter and an empty middle."""

    height, width = template_mask.shape[:2]
    band = max(2, min(height, width) // 5)
    intersection = cv2.bitwise_and(roi, template_mask)
    edge_slices = (
        (slice(0, band), slice(None)),
        (slice(height - band, height), slice(None)),
        (slice(None), slice(0, band)),
        (slice(None), slice(width - band, width)),
    )

    edge_coverages: list[float] = []
    for edge_slice in edge_slices:
        template_pixels = cv2.countNonZero(template_mask[edge_slice])
        intersection_pixels = cv2.countNonZero(intersection[edge_slice])
        edge_coverages.append(intersection_pixels / max(1, template_pixels))

    inner = roi[band : height - band, band : width - band]
    center_density = cv2.countNonZero(inner) / max(1, inner.size) if inner.size else 0.0

    def max_run(values: np.ndarray) -> int:
        best = 0
        current = 0
        for value in values.astype(bool, copy=False).tolist():
            if value:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    top_run = max_run((intersection[:band, :] > 0).any(axis=0)) / max(1, width)
    bottom_run = max_run((intersection[height - band : height, :] > 0).any(axis=0)) / max(1, width)
    left_run = max_run((intersection[:, :band] > 0).any(axis=1)) / max(1, height)
    right_run = max_run((intersection[:, width - band : width] > 0).any(axis=1)) / max(1, height)

    if width >= height:
        long_edges = (top_run, bottom_run)
        short_edges = (left_run, right_run)
    else:
        long_edges = (left_run, right_run)
        short_edges = (top_run, bottom_run)

    continuous_frame = (
        min(long_edges) >= GRAY_RECT_FRAME_LONG_EDGE_MIN_RUN
        and max(short_edges) >= GRAY_RECT_FRAME_SHORT_EDGE_STRONG_RUN
        and min(short_edges) >= GRAY_RECT_FRAME_SHORT_EDGE_WEAK_RUN
    )

    return (
        sum(score >= GRAY_RECT_FRAME_STRONG_EDGE_COVERAGE for score in edge_coverages) >= 3
        and all(score >= GRAY_RECT_FRAME_WEAK_EDGE_COVERAGE for score in edge_coverages)
        and center_density <= GRAY_RECT_FRAME_MAX_CENTER_DENSITY
        and continuous_frame
    )


def _context_purity(
    plan_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    intersection_mask: np.ndarray,
    *,
    explained_pixels: int | None = None,
    validation_cache: ValidationMaskCache | None = None,
) -> float:
    """Measure how much local foreground around the hit is explained by the template."""

    x, y, w, h = bbox
    margin = max(3, int(round(max(w, h) * CONTEXT_MARGIN_RATIO)))

    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(plan_mask.shape[1], x + w + margin)
    y1 = min(plan_mask.shape[0], y + h + margin)

    context_bbox = (x0, y0, x1 - x0, y1 - y0)
    if context_bbox[2] <= 0 or context_bbox[3] <= 0:
        return 0.0

    context_foreground = (
        validation_cache.foreground_count(plan_mask, context_bbox)
        if validation_cache is not None
        else int(cv2.countNonZero(plan_mask[y0:y1, x0:x1]))
    )
    if context_foreground == 0:
        return 0.0

    if explained_pixels is None:
        explained_pixels = int(cv2.countNonZero(intersection_mask))
    return explained_pixels / context_foreground


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
        _record("color_recovery_tiny_fragment")
        return False
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
        _record("color_recovery_isolated_fragment")
        return False
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
        _record("color_text_fragment")
        return False

    if (
        hit.dominant_hsv is not None
        and hit.source != "pdf_text"
        and early_color_similarity < COLOR_MIN_HUE_SIMILARITY
    ):
        _record("color_similarity")
        return False

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
        _record("color_straight_stroke_fragment")
        return False

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
        _record("color_flat_elongated_fragment")
        return False

    if hit.dominant_hsv is not None and hit.is_text_label and hit.source == "template_content":
        if (
            hit.match_score < COLOR_CONTENT_LABEL_MIN_MATCH
            or coverage < COLOR_CONTENT_LABEL_MIN_COVERAGE
            or purity < COLOR_CONTENT_LABEL_MIN_PURITY
            or context_purity < COLOR_CONTENT_LABEL_MIN_CONTEXT
        ):
            _record("color_content_fragment")
            return False

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
            _record("color_text_geometry")
            return False
        if (
            hit.match_score < COLOR_TEXT_LABEL_MIN_MATCH
            and context_purity < COLOR_TEXT_LABEL_WEAK_MATCH_MIN_CONTEXT
        ):
            _record("color_text_weak_match")
            return False

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
        _record("color_elongated_stroke_fragment")
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
