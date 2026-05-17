"""Gray detector mask, threshold and scan policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.detector_config import (
    GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT,
    GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE,
    GRAY_COMPLEX_GEOMETRY_MIN_PURITY,
    GRAY_DARK_EVIDENCE_THRESHOLD,
    GRAY_DARK_INK_THRESHOLD,
    GRAY_DARK_ZONE_THRESHOLD,
    GRAY_DIAGONAL_TEXT_LABEL_MAX_ASPECT,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_AREA,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_CONTEXT,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_COVERAGE,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_MATCH,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_PURITY,
    GRAY_ELONGATED_SCAN_MAX_TEMPLATE_PIXELS,
    GRAY_ELONGATED_SCAN_THRESHOLD,
    GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP,
    GRAY_FULLER_SYMBOL_MIN_AREA_RATIO,
    GRAY_FULLER_SYMBOL_MIN_COVERAGE,
    GRAY_FULLER_SYMBOL_MIN_PURITY,
    GRAY_LEGEND_DARK_MARGIN,
    GRAY_LEGEND_EVIDENCE_MARGIN,
    GRAY_LEGEND_INK_PERCENTILE,
    GRAY_LABEL_GEOMETRY_MIN_CONTEXT,
    GRAY_LABEL_GEOMETRY_MIN_COVERAGE,
    GRAY_LABEL_GEOMETRY_MIN_MATCH,
    GRAY_LABEL_GEOMETRY_MIN_PURITY,
    GRAY_LABEL_GEOMETRY_MIN_VERIFICATION,
    GRAY_LEGEND_MIN_EVIDENCE_THRESHOLD,
    GRAY_LEGEND_MIN_ZONE_THRESHOLD,
    GRAY_LEGEND_ZONE_MARGIN,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ASPECT,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_TEMPLATE_AREA,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_CONTEXT,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_COVERAGE,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_MATCH,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_PURITY,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA,
    GRAY_MID_GEOMETRY_MIN_CONTEXT,
    GRAY_MID_GEOMETRY_MIN_COVERAGE,
    GRAY_MID_GEOMETRY_MIN_MATCH,
    GRAY_MID_GEOMETRY_MIN_PURITY,
    GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS,
    GRAY_MID_GEOMETRY_MIN_VERIFICATION,
    GRAY_LINE_CROSSED_LABEL_MIN_CONTEXT,
    GRAY_LINE_CROSSED_LABEL_MIN_COVERAGE,
    GRAY_LINE_CROSSED_LABEL_MIN_MATCH,
    GRAY_LINE_CROSSED_LABEL_MIN_PURITY,
    GRAY_LINE_CROSSED_LABEL_MIN_SCALE,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_CONTEXT,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_COVERAGE,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_MATCH,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_PURITY,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_CONTEXT,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_COVERAGE,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_MATCH,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_PURITY,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA,
    GRAY_RAW_MAX_HITS_PER_TEMPLATE,
    GRAY_RAW_MAX_HITS_PER_VARIANT,
    GRAY_RAW_MAX_TOTAL_HITS,
    GRAY_RAW_MIN_HITS_PER_TEMPLATE,
    GRAY_RAW_SCAN_MIN_TEMPLATE_AREA,
    GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS,
    GRAY_RAW_SCAN_THRESHOLD,
    GRAY_RECT_FRAME_MAX_CENTER_DENSITY,
    GRAY_RECT_FRAME_MAX_DENSITY,
    GRAY_RECT_FRAME_MAX_RAW_SCAN_SCALE,
    GRAY_RECT_FRAME_MERGE_CENTER_DISTANCE,
    GRAY_RECT_FRAME_MERGE_IOM,
    GRAY_RECT_FRAME_MERGE_MAX_SCALE_DELTA,
    GRAY_RECT_FRAME_MIN_ASPECT,
    GRAY_RECT_FRAME_MIN_DENSITY,
    GRAY_RECT_FRAME_MIN_RAW_SCAN_SCALE,
    GRAY_RECT_FRAME_STRONG_EDGE_COVERAGE,
    GRAY_RECT_FRAME_WEAK_EDGE_COVERAGE,
    GRAY_SEARCH_COMPONENT_PADDING_RATIO,
    GRAY_SEARCH_COMPONENT_DILATE_ITERATIONS,
    GRAY_SEARCH_CONNECTED_FAST_COMPONENTS_MAX,
    GRAY_SEARCH_CONNECTED_FAST_MAX_TILE_ROIS,
    GRAY_SEARCH_FAST_MAX_TILE_ROIS,
    GRAY_SEARCH_FAST_TILE_SIZE,
    GRAY_SEARCH_LARGE_TEXT_MAX_TILE_ROIS,
    GRAY_SEARCH_LARGE_TEXT_MIN_TEMPLATE_AREA,
    GRAY_SEARCH_LARGE_TEXT_TILE_SIZE,
    GRAY_SEARCH_MAX_ROIS,
    GRAY_SEARCH_MAX_TILE_ROIS,
    GRAY_SEARCH_ROI_CONTAINMENT_THRESHOLD,
    GRAY_SEARCH_ROI_OVERLAP_THRESHOLD,
    GRAY_SEARCH_SAFE_ELONGATED_ASPECT,
    GRAY_SEARCH_TILE_MIN_FOREGROUND,
    GRAY_SEARCH_TILE_PADDING,
    GRAY_SEARCH_TILE_SIZE,
    GRAY_SPATIAL_FAIR_PEAKS_PER_ROI,
    GRAY_STRICT_SCAN_THRESHOLD,
    GRAY_STRONG_GEOMETRY_MIN_COVERAGE,
    GRAY_STRONG_GEOMETRY_MIN_MATCH,
    GRAY_STRONG_RESCUE_MIN_PURITY,
    GRAY_STRONG_RESCUE_MIN_VERIFICATION,
    GRAY_STRONG_TRACE_MAX_ITEMS,
    GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
    GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    GRAY_TINY_GEOMETRY_MAX_SCALE,
    GRAY_TINY_GEOMETRY_MAX_TEMPLATE_PIXELS,
    GRAY_TINY_GEOMETRY_MIN_CONTEXT,
    GRAY_TINY_GEOMETRY_MIN_COVERAGE,
    GRAY_TINY_GEOMETRY_MIN_MATCH,
    GRAY_TINY_GEOMETRY_MIN_PURITY,
    GRAY_TINY_GEOMETRY_MIN_VERIFICATION,
    GRAY_WEAK_LABEL_MAX_CONTEXT,
    GRAY_WEAK_LABEL_MAX_PURITY,
)
from core.detector_masks import _ink_mask, _suppress_long_strokes
from core.detector_models import TemplateInfo

class GrayScanMasks:
    scan_masks_by_template: dict[int, np.ndarray]
    scan_mask_kinds_by_template: dict[int, str]
    zone_mask: np.ndarray
    evidence_mask: np.ndarray
    dark_threshold: int
    zone_threshold: int
    evidence_threshold: int
    raw_ink_pixels: int
    suppressed_pixels: int
    dark_ink_pixels: int
    zone_ink_pixels: int
    evidence_ink_pixels: int
    dark_suppressed_pixels: int
    zone_suppressed_pixels: int

    @property
    def suppressed_ratio(self) -> float:
        return round(self.suppressed_pixels / max(1, self.raw_ink_pixels), 3)

    @property
    def dark_suppressed_ratio(self) -> float:
        return round(self.dark_suppressed_pixels / max(1, self.dark_ink_pixels), 3)

    @property
    def zone_suppressed_ratio(self) -> float:
        return round(self.zone_suppressed_pixels / max(1, self.zone_ink_pixels), 3)


class GrayLegendThresholds:
    dark: int
    zone: int
    evidence: int
    anchor: int | None


def _clamp_roi(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int] | None:
    image_h, image_w = image_shape[:2]
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_w, x + w)
    y2 = min(image_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def gray_template_area(template: TemplateInfo) -> int:
    height, width = template.mask.shape[:2]
    return int(width * height)


def gray_template_pixels(template: TemplateInfo) -> int:
    return int(getattr(template, "pixel_count", 0) or cv2.countNonZero(template.mask))


def use_large_text_tile_rois(template: TemplateInfo) -> bool:
    """Use smaller coarse tiles for large gray text labels while preserving coverage."""

    return (
        template.is_text_label
        and gray_template_area(template) >= GRAY_SEARCH_LARGE_TEXT_MIN_TEMPLATE_AREA
    )


def _gray_template_aspect(template: TemplateInfo) -> float:
    height, width = template.mask.shape[:2]
    return max(width / max(1, height), height / max(1, width))


def gray_tile_roi_strategy(template: TemplateInfo) -> tuple[str, int, int]:
    """Return the coarse tile strategy for gray ROI coverage.

    Compact symbols can use smaller tiles with a larger coverage budget. Slender
    symbols are more sensitive to tile boundaries and isolated line fragments, so
    they keep the old safety tiles as a fallback.
    """

    if use_large_text_tile_rois(template):
        return (
            "large_text_fast",
            int(GRAY_SEARCH_LARGE_TEXT_TILE_SIZE),
            int(GRAY_SEARCH_LARGE_TEXT_MAX_TILE_ROIS),
        )

    if (not template.is_text_label) and _gray_template_aspect(template) >= float(
        GRAY_SEARCH_SAFE_ELONGATED_ASPECT
    ):
        return (
            "safe_elongated",
            int(GRAY_SEARCH_TILE_SIZE),
            int(GRAY_SEARCH_MAX_TILE_ROIS),
        )

    return (
        "fast_compact",
        int(GRAY_SEARCH_FAST_TILE_SIZE),
        int(GRAY_SEARCH_FAST_MAX_TILE_ROIS),
    )


def adapt_gray_tile_roi_strategy_for_plan(
    strategy: str,
    tile_size: int,
    max_tile_rois: int,
    component_index: GraySearchComponentIndex | None,
) -> tuple[str, int, int]:
    """Tighten compact-tile coverage only on highly connected gray plans.

    A low component count with substantial foreground usually means symbols are
    attached to large connected linework.  In that topology, the broad compact
    tile budget creates very large merged ROIs; a smaller spatial budget cuts
    matchTemplate pixels while component ROIs still provide local coverage.
    Plans with many isolated components keep the wider safe budget.
    """

    if (
        strategy == "fast_compact"
        and component_index is not None
        and component_index.components <= GRAY_SEARCH_CONNECTED_FAST_COMPONENTS_MAX
    ):
        return (
            "fast_compact_connected",
            int(tile_size),
            min(int(max_tile_rois), int(GRAY_SEARCH_CONNECTED_FAST_MAX_TILE_ROIS)),
        )
    return strategy, tile_size, max_tile_rois


def _edge_band_densities(mask: np.ndarray) -> tuple[float, float, float, float, float]:
    height, width = mask.shape[:2]
    band = max(2, min(height, width) // 5)

    def density(region: np.ndarray) -> float:
        return cv2.countNonZero(region) / max(1, region.size)

    inner = mask[band : height - band, band : width - band]
    center_density = density(inner) if inner.size else 0.0
    return (
        density(mask[:band, :]),
        density(mask[height - band :, :]),
        density(mask[:, :band]),
        density(mask[:, width - band :]),
        center_density,
    )


def is_gray_rect_frame_template(template: TemplateInfo) -> bool:
    """Detect hollow elongated frame templates whose signal is long strokes."""

    height, width = template.mask.shape[:2]
    aspect = max(width / max(1, height), height / max(1, width))
    density = gray_template_pixels(template) / max(1, width * height)
    if aspect < GRAY_RECT_FRAME_MIN_ASPECT:
        return False
    if density < GRAY_RECT_FRAME_MIN_DENSITY or density > GRAY_RECT_FRAME_MAX_DENSITY:
        return False

    edge_scores = _edge_band_densities(template.mask)
    edge_coverages = edge_scores[:4]
    center_density = edge_scores[4]
    return (
        sum(score >= GRAY_RECT_FRAME_STRONG_EDGE_COVERAGE for score in edge_coverages) >= 3
        and all(score >= GRAY_RECT_FRAME_WEAK_EDGE_COVERAGE for score in edge_coverages)
        and center_density <= GRAY_RECT_FRAME_MAX_CENTER_DENSITY
    )


def use_relaxed_gray_scan_threshold(template: TemplateInfo) -> bool:
    """Relax gray scan threshold only for genuinely large framed templates."""

    return (
        gray_template_area(template) >= GRAY_RAW_SCAN_MIN_TEMPLATE_AREA
        or gray_template_pixels(template) >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
    )


def use_raw_gray_scan_mask(template: TemplateInfo) -> bool:
    """Use dark raw ink for large/complex shapes whose frame is the signal."""

    return (
        use_relaxed_gray_scan_threshold(template)
        or is_gray_rect_frame_template(template)
        or gray_template_pixels(template) >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
    )


def should_scan_gray_variant(
    template: TemplateInfo,
    variant_scale: float,
    scan_mask_kind: str,
) -> bool:
    """Keep raw-frame scans focused on real-size frame variants."""

    if scan_mask_kind == "zone_raw" and is_gray_rect_frame_template(template):
        return (
            variant_scale >= GRAY_RECT_FRAME_MIN_RAW_SCAN_SCALE
            and variant_scale <= GRAY_RECT_FRAME_MAX_RAW_SCAN_SCALE
        )
    return True


def use_lenient_gray_elongated_scan_threshold(template: TemplateInfo) -> bool:
    """Allow low scan threshold only for sparse elongated symbols like 04."""

    height, width = template.mask.shape[:2]
    aspect = max(width / max(1, height), height / max(1, width))
    return aspect >= 2.0 and gray_template_pixels(template) <= GRAY_ELONGATED_SCAN_MAX_TEMPLATE_PIXELS


def gray_scan_threshold(template: TemplateInfo, base_threshold: float) -> float:
    """Return profile-specific threshold for gray template scanning."""

    if use_relaxed_gray_scan_threshold(template):
        return max(base_threshold, GRAY_RAW_SCAN_THRESHOLD)
    if use_lenient_gray_elongated_scan_threshold(template):
        return GRAY_ELONGATED_SCAN_THRESHOLD
    return max(base_threshold, GRAY_STRICT_SCAN_THRESHOLD)


def use_gray_spatial_fair_peaks(template: TemplateInfo) -> bool:
    # Gray elongated symbols generate many plausible peaks on text/linework.
    # A global "first N peaks then stop" can skip later ROIs entirely, while
    # the ROI inspector still shows a clean PASS there.  Keep a small fair
    # quota per ROI so every dark-symbol island gets a chance before budgeting.
    return use_relaxed_gray_scan_threshold(template) or use_lenient_gray_elongated_scan_threshold(
        template
    )


def gray_spatial_fair_peaks_per_roi() -> int:
    return GRAY_SPATIAL_FAIR_PEAKS_PER_ROI


def _clamp_threshold(value: float, *, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(value))))


def _estimate_gray_legend_thresholds(templates: list[TemplateInfo]) -> GrayLegendThresholds:
    """Calibrate gray ink thresholds from black/gray legend template pixels."""

    samples: list[np.ndarray] = []
    for template in templates:
        if template.dominant_hsv is not None or template.image_bgr.size == 0:
            continue

        gray = cv2.cvtColor(template.image_bgr, cv2.COLOR_BGR2GRAY)
        ink = _ink_mask(template.image_bgr, dilate=False, threshold=238)
        values = gray[ink > 0]
        if values.size < 8:
            continue

        # Only the dark body of legend symbols should define "black".
        # Pale antialiasing and legend crop noise stay out of the calibration.
        core_values = values[values <= GRAY_DARK_ZONE_THRESHOLD]
        if core_values.size >= 8:
            samples.append(core_values.astype(np.uint8, copy=False))

    if not samples:
        return GrayLegendThresholds(
            dark=int(GRAY_DARK_INK_THRESHOLD),
            zone=int(GRAY_DARK_ZONE_THRESHOLD),
            evidence=int(GRAY_DARK_EVIDENCE_THRESHOLD),
            anchor=None,
        )

    all_values = np.concatenate(samples)
    percentile = max(1.0, min(99.0, float(GRAY_LEGEND_INK_PERCENTILE)))
    anchor = int(round(float(np.percentile(all_values, percentile))))
    zone = _clamp_threshold(
        anchor + GRAY_LEGEND_ZONE_MARGIN,
        minimum=int(GRAY_LEGEND_MIN_ZONE_THRESHOLD),
        maximum=int(GRAY_DARK_ZONE_THRESHOLD),
    )
    evidence = _clamp_threshold(
        anchor + GRAY_LEGEND_EVIDENCE_MARGIN,
        minimum=int(GRAY_LEGEND_MIN_EVIDENCE_THRESHOLD),
        maximum=min(int(GRAY_DARK_EVIDENCE_THRESHOLD), max(1, zone - 1)),
    )
    dark = _clamp_threshold(
        anchor + GRAY_LEGEND_DARK_MARGIN,
        minimum=max(zone + 1, evidence + 1),
        maximum=int(GRAY_DARK_INK_THRESHOLD),
    )

    return GrayLegendThresholds(dark=dark, zone=zone, evidence=evidence, anchor=anchor)


def build_gray_scan_masks(
    *,
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    plan_masks_by_template: dict[int, np.ndarray],
    exclude_rects: list[tuple[int, int, int, int]],
    raw_dilated: np.ndarray,
) -> GrayScanMasks:
    """Build strict dark gray scan masks while keeping raw masks for validation."""

    thresholds = _estimate_gray_legend_thresholds(templates)
    raw_ink_pixels = int(cv2.countNonZero(raw_dilated))
    dark_base = _ink_mask(
        plan_image,
        dilate=False,
        threshold=thresholds.dark,
    )
    zone_base = _ink_mask(
        plan_image,
        dilate=False,
        threshold=thresholds.zone,
    )
    evidence_base = _ink_mask(
        plan_image,
        dilate=False,
        threshold=thresholds.evidence,
    )
    for ex, ey, ew, eh in exclude_rects:
        cv2.rectangle(dark_base, (ex, ey), (ex + ew, ey + eh), 0, -1)
        cv2.rectangle(zone_base, (ex, ey), (ex + ew, ey + eh), 0, -1)
        cv2.rectangle(evidence_base, (ex, ey), (ex + ew, ey + eh), 0, -1)

    dark_dilated = cv2.dilate(dark_base, np.ones((3, 3), np.uint8), iterations=1)
    zone_dilated = cv2.dilate(zone_base, np.ones((3, 3), np.uint8), iterations=1)
    dark_ink_pixels = int(cv2.countNonZero(dark_dilated))
    zone_ink_pixels = int(cv2.countNonZero(zone_dilated))
    evidence_ink_pixels = int(cv2.countNonZero(evidence_base))
    suppressed = _suppress_long_strokes(
        raw_dilated,
        GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
        GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    )
    dark_suppressed = _suppress_long_strokes(
        dark_dilated,
        GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
        GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    )
    zone_suppressed = _suppress_long_strokes(
        zone_dilated,
        GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
        GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    )

    scan_masks_by_template: dict[int, np.ndarray] = {}
    scan_mask_kinds_by_template: dict[int, str] = {}
    for template_id in plan_masks_by_template:
        template = templates[template_id]
        if use_raw_gray_scan_mask(template):
            scan_masks_by_template[template_id] = zone_dilated
            scan_mask_kinds_by_template[template_id] = "zone_raw"
        else:
            scan_masks_by_template[template_id] = zone_suppressed
            scan_mask_kinds_by_template[template_id] = "zone_suppressed"

    return GrayScanMasks(
        scan_masks_by_template=scan_masks_by_template,
        scan_mask_kinds_by_template=scan_mask_kinds_by_template,
        zone_mask=zone_dilated,
        evidence_mask=evidence_base,
        dark_threshold=thresholds.dark,
        zone_threshold=thresholds.zone,
        evidence_threshold=thresholds.evidence,
        raw_ink_pixels=raw_ink_pixels,
        suppressed_pixels=raw_ink_pixels - int(cv2.countNonZero(suppressed)),
        dark_ink_pixels=dark_ink_pixels,
        zone_ink_pixels=zone_ink_pixels,
        evidence_ink_pixels=evidence_ink_pixels,
        dark_suppressed_pixels=dark_ink_pixels - int(cv2.countNonZero(dark_suppressed)),
        zone_suppressed_pixels=zone_ink_pixels - int(cv2.countNonZero(zone_suppressed)),
    )
