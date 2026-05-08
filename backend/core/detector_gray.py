"""Gray-PDF detector strategy helpers.

The gray pipeline is intentionally isolated from the color pipeline.  Gray PDFs
need ink masks, dark-ink prefilters, wider scale search and a few rescue rules;
keeping those rules here makes the main detector easier to audit.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.detector_clustering import _bbox_metrics
from core.detector_config import (
    GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT,
    GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE,
    GRAY_COMPLEX_GEOMETRY_MIN_PURITY,
    GRAY_DARK_EVIDENCE_THRESHOLD,
    GRAY_DARK_INK_THRESHOLD,
    GRAY_DARK_ZONE_THRESHOLD,
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
    GRAY_MID_GEOMETRY_MIN_CONTEXT,
    GRAY_MID_GEOMETRY_MIN_COVERAGE,
    GRAY_MID_GEOMETRY_MIN_MATCH,
    GRAY_MID_GEOMETRY_MIN_PURITY,
    GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS,
    GRAY_MID_GEOMETRY_MIN_VERIFICATION,
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
from core.detector_masks import _context_purity, _ink_mask, _suppress_long_strokes
from core.detector_models import CandidateHit, TemplateInfo
from core.detector_selection import (
    candidate_quality_key,
    local_dominates,
    same_physical_place,
)


@dataclass(slots=True)
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


@dataclass(slots=True)
class GraySearchComponentIndex:
    foreground_pixels: int
    components: int
    stats: np.ndarray
    foreground_integral: np.ndarray


@dataclass(slots=True)
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


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    return max(0, int(rect[2])) * max(0, int(rect[3]))


def _rect_intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    return max(0, x2 - x1) * max(0, y2 - y1)


def _rect_union(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int] | None:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return _clamp_roi((x1, y1, x2 - x1, y2 - y1), image_shape)


def _integral_rect_sum(integral: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    x, y, w, h = rect
    return int(
        integral[y + h, x + w]
        - integral[y, x + w]
        - integral[y + h, x]
        + integral[y, x]
    )


def build_gray_search_component_index(plan_mask: np.ndarray) -> GraySearchComponentIndex:
    foreground_pixels = int(cv2.countNonZero(plan_mask))
    dilate_iterations = max(0, int(GRAY_SEARCH_COMPONENT_DILATE_ITERATIONS))
    seed_mask = (
        cv2.dilate(plan_mask, np.ones((3, 3), np.uint8), iterations=dilate_iterations)
        if dilate_iterations > 0
        else plan_mask
    )
    components, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        seed_mask,
        connectivity=8,
    )
    foreground_integral = cv2.integral((plan_mask > 0).astype(np.uint8, copy=False), sdepth=cv2.CV_32S)
    return GraySearchComponentIndex(
        foreground_pixels=foreground_pixels,
        components=int(components),
        stats=stats,
        foreground_integral=foreground_integral,
    )


def _coalesce_gray_rois(
    rois: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Collapse duplicate/nested gray search windows before expensive scans."""

    if len(rois) <= 1:
        return rois

    containment_threshold = max(0.0, min(1.0, float(GRAY_SEARCH_ROI_CONTAINMENT_THRESHOLD)))
    overlap_threshold = max(0.0, min(1.0, float(GRAY_SEARCH_ROI_OVERLAP_THRESHOLD)))
    kept: list[tuple[int, int, int, int]] = []

    for rect in sorted(dict.fromkeys(rois), key=lambda item: (_rect_area(item), item), reverse=True):
        rect_area = _rect_area(rect)
        if rect_area <= 0:
            continue

        should_skip = False
        for index, existing in enumerate(kept):
            existing_area = _rect_area(existing)
            if existing_area <= 0:
                continue

            overlap = _rect_intersection_area(rect, existing)
            if overlap / max(1, rect_area) >= containment_threshold:
                should_skip = True
                break

            smaller_area = max(1, min(rect_area, existing_area))
            if overlap / smaller_area >= overlap_threshold:
                merged = _rect_union(rect, existing, image_shape)
                if merged is not None:
                    kept[index] = merged
                should_skip = True
                break

        if not should_skip:
            kept.append(rect)

    # A merge can make later windows newly contained, so run one cheap cleanup pass.
    cleaned: list[tuple[int, int, int, int]] = []
    for rect in sorted(dict.fromkeys(kept), key=lambda item: (_rect_area(item), item), reverse=True):
        rect_area = _rect_area(rect)
        if rect_area <= 0:
            continue
        if any(
            _rect_intersection_area(rect, existing) / max(1, rect_area) >= containment_threshold
            for existing in cleaned
        ):
            continue
        cleaned.append(rect)

    return cleaned


def coalesce_gray_debug_rois(
    rois: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Expose gray ROI coalescing for debug overlays."""

    return _coalesce_gray_rois(rois, image_shape)


def _append_gray_tile_rois(
    rois: list[tuple[int, int, int, int]],
    plan_mask: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
    foreground_integral: np.ndarray | None = None,
    *,
    tile_size_override: int | None = None,
    max_tile_rois_override: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Add coarse gray tiles so symbols connected to walls still get scanned."""

    max_tile_rois = (
        int(max_tile_rois_override)
        if max_tile_rois_override is not None
        else int(GRAY_SEARCH_MAX_TILE_ROIS)
    )
    if max_tile_rois <= 0:
        return rois

    tile_size = max(
        64,
        int(tile_size_override) if tile_size_override is not None else int(GRAY_SEARCH_TILE_SIZE),
    )
    padding = max(0, int(GRAY_SEARCH_TILE_PADDING))
    min_foreground = max(1, int(GRAY_SEARCH_TILE_MIN_FOREGROUND))
    image_h, image_w = int(image_shape[0]), int(image_shape[1])

    tile_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for y in range(0, image_h, tile_size):
        tile_h = min(tile_size, image_h - y)
        if tile_h <= 0:
            continue
        for x in range(0, image_w, tile_size):
            tile_w = min(tile_size, image_w - x)
            if tile_w <= 0:
                continue

            foreground = (
                _integral_rect_sum(foreground_integral, (x, y, tile_w, tile_h))
                if foreground_integral is not None
                else int(cv2.countNonZero(plan_mask[y : y + tile_h, x : x + tile_w]))
            )
            if foreground < min_foreground:
                continue

            clamped = _clamp_roi(
                (
                    x - padding,
                    y - padding,
                    tile_w + 2 * padding,
                    tile_h + 2 * padding,
                ),
                image_shape,
            )
            if clamped is not None:
                tile_candidates.append((foreground, clamped))

    tile_candidates.sort(key=lambda item: item[0], reverse=True)
    existing = set(rois)
    for _foreground, rect in tile_candidates[:max_tile_rois]:
        if rect in existing:
            continue
        rois.append(rect)
        existing.add(rect)

    return rois


def build_gray_search_rois(
    plan_mask: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
    max_template_width: int,
    max_template_height: int,
    *,
    is_large_text_template: bool = False,
    component_index: GraySearchComponentIndex | None = None,
    tile_size_override: int | None = None,
    max_tile_rois_override: int | None = None,
) -> tuple[list[tuple[int, int, int, int]], bool, int, int]:
    """Build bounded ROIs for gray/ink plans."""

    foreground_pixels = (
        component_index.foreground_pixels
        if component_index is not None
        else int(cv2.countNonZero(plan_mask))
    )
    if foreground_pixels <= 0:
        return [], False, 0, foreground_pixels

    image_h, image_w = image_shape[:2]
    if component_index is None:
        component_index = build_gray_search_component_index(plan_mask)
    components = component_index.components
    stats = component_index.stats
    foreground_integral = component_index.foreground_integral
    if components <= 1:
        return [], False, 0, foreground_pixels

    def _limit_component_rects_spatially(
        rects: list[tuple[float, int, tuple[int, int, int, int]]],
        limit: int,
    ) -> list[tuple[float, int, tuple[int, int, int, int]]]:
        """Keep ROI coverage spread across the plan instead of only best-scoring islands."""

        if len(rects) <= limit:
            return rects

        tile_size = max(192, min(256, int(GRAY_SEARCH_TILE_SIZE)))
        per_cell_limit = 2
        best_by_cell: dict[
            tuple[int, int],
            list[tuple[float, int, tuple[int, int, int, int]]],
        ] = {}
        for item in rects:
            score, area, (x, y, w, h) = item
            cell = (int((x + w / 2) // tile_size), int((y + h / 2) // tile_size))
            bucket = best_by_cell.setdefault(cell, [])
            bucket.append(item)
            bucket.sort(key=lambda candidate: (candidate[0], -candidate[1]))
            del bucket[per_cell_limit:]

        selected: list[tuple[float, int, tuple[int, int, int, int]]] = sorted(
            (item for bucket in best_by_cell.values() for item in bucket),
            key=lambda item: (item[0], -item[1]),
        )[:limit]
        selected_rects = {item[2] for item in selected}

        if len(selected) < limit:
            for item in rects:
                if item[2] in selected_rects:
                    continue
                selected.append(item)
                selected_rects.add(item[2])
                if len(selected) >= limit:
                    break

        selected.sort(key=lambda item: (item[0], -item[1]))
        return selected

    template_area = max(1, int(max_template_width) * int(max_template_height))
    max_component_width = max(80, int(max_template_width * 4.0))
    max_component_height = max(80, int(max_template_height * 4.0))
    max_component_area = max(120, int(template_area * 8.0))
    component_padding_ratio = max(0.25, float(GRAY_SEARCH_COMPONENT_PADDING_RATIO))
    pad_x = max(6, int(round(max_template_width * component_padding_ratio)))
    pad_y = max(6, int(round(max_template_height * component_padding_ratio)))

    component_rects: list[tuple[float, int, tuple[int, int, int, int]]] = []
    target_area = max(1, int(max_template_width) * int(max_template_height))
    target_aspect = max_template_width / max(1, max_template_height)

    def _component_score(width: int, height: int) -> float:
        bbox_area = max(1, int(width) * int(height))
        aspect = width / max(1, height)
        area_score = abs(np.log((bbox_area + 1) / target_area))
        aspect_score = abs(np.log((aspect + 0.05) / max(0.05, target_aspect)))
        return float(area_score + 0.35 * aspect_score)

    def _append_large_component_tiles(x: int, y: int, w: int, h: int, area: int) -> None:
        """Split oversized dark components so attached symbols are still scanned."""

        tile_w = max(96, min(max_component_width, int(round(max_template_width * 3.2))))
        tile_h = max(96, min(max_component_height, int(round(max_template_height * 3.2))))
        step_x = max(24, int(round(tile_w * 0.55)))
        step_y = max(24, int(round(tile_h * 0.55)))

        start_x = max(0, x - pad_x)
        start_y = max(0, y - pad_y)
        stop_x = min(image_w, x + w + pad_x)
        stop_y = min(image_h, y + h + pad_y)

        tile_y = start_y
        while tile_y < stop_y:
            tile_x = start_x
            while tile_x < stop_x:
                clamped = _clamp_roi((tile_x, tile_y, tile_w, tile_h), image_shape)
                if clamped is not None:
                    rx, ry, rw, rh = clamped
                    foreground = _integral_rect_sum(foreground_integral, (rx, ry, rw, rh))
                    if foreground >= GRAY_SEARCH_TILE_MIN_FOREGROUND:
                        component_rects.append(
                            (
                                _component_score(rw, rh) + 0.12,
                                min(area, foreground),
                                clamped,
                            )
                        )
                tile_x += step_x
            tile_y += step_y

    for component_id in range(1, components):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 6:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        if area > max_component_area or w > max_component_width or h > max_component_height:
            _append_large_component_tiles(x, y, w, h, area)
            continue

        clamped = _clamp_roi((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), image_shape)
        if clamped is not None:
            component_rects.append((_component_score(w, h), area, clamped))

    if not component_rects:
        return [], False, 0, foreground_pixels

    component_rects.sort(key=lambda item: (item[0], -item[1]))
    component_rects = _limit_component_rects_spatially(
        component_rects,
        GRAY_SEARCH_MAX_ROIS,
    )
    rois = [rect for _score, _area, rect in component_rects]
    rois = _append_gray_tile_rois(
        rois,
        plan_mask,
        image_shape,
        foreground_integral,
        tile_size_override=(
            int(tile_size_override)
            if tile_size_override is not None
            else int(GRAY_SEARCH_LARGE_TEXT_TILE_SIZE)
            if is_large_text_template
            else None
        ),
        max_tile_rois_override=(
            int(max_tile_rois_override)
            if max_tile_rois_override is not None
            else int(GRAY_SEARCH_LARGE_TEXT_MAX_TILE_ROIS)
            if is_large_text_template
            else None
        ),
    )
    rois = _coalesce_gray_rois(rois, image_shape)
    roi_area = min(
        sum(_rect_area(roi) for roi in rois),
        max(1, int(image_w) * int(image_h)),
    )
    return rois, False, roi_area, foreground_pixels


def _gray_budget_geometry_score(
    hit: CandidateHit,
    plan_mask: np.ndarray,
) -> tuple[float, float] | None:
    """Cheap coverage/purity check used only to protect strong gray raw hits."""

    if hit.transformed_mask is None or hit.pixel_count <= 0:
        return None

    clamped = _clamp_roi(hit.bbox, plan_mask.shape)
    if clamped is None:
        return None

    x, y, w, h = clamped
    roi = plan_mask[y : y + h, x : x + w]
    if roi.shape != hit.transformed_mask.shape:
        return None

    roi_foreground = int(cv2.countNonZero(roi))
    if roi_foreground <= 0:
        return None

    intersection_mask = cv2.bitwise_and(roi, hit.transformed_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    coverage = intersection / max(1, hit.pixel_count)
    purity = intersection / max(1, roi_foreground)
    standard_geometry_failed = (
        hit.match_score < GRAY_STRONG_GEOMETRY_MIN_MATCH
        or coverage < GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        or purity < GRAY_STRONG_RESCUE_MIN_PURITY
    )
    strong_complex_geometry = (
        hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_RAW_SCAN_THRESHOLD
        and coverage >= GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE
        and purity >= GRAY_COMPLEX_GEOMETRY_MIN_PURITY
        and _context_purity(plan_mask, hit.bbox, intersection_mask) >= GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT
    )
    if standard_geometry_failed and not strong_complex_geometry:
        return None

    return coverage, purity


def gray_raw_budget(
    candidates: list[CandidateHit],
    templates: list[TemplateInfo],
    plan_masks_by_template: dict[int, np.ndarray] | None = None,
) -> tuple[list[CandidateHit], dict]:
    """Keep the strongest gray-mode raw hits before the slower overlap prefilter."""

    if len(candidates) <= GRAY_RAW_MAX_TOTAL_HITS:
        return candidates, {
            "before": len(candidates),
            "after": len(candidates),
            "removed": 0,
            "geometryProtected": 0,
            "geometryProtectedKept": 0,
            "geometryProtectedDropped": 0,
            "perTemplateReserved": 0,
            "perTemplateReservedKept": 0,
            "perTemplateReservedActive": 0,
            "perTemplateReserveLimit": int(max(0, GRAY_RAW_MIN_HITS_PER_TEMPLATE)),
            "topGenerators": [],
        }

    protected_scores: dict[int, tuple[float, float]] = {}
    if plan_masks_by_template is not None:
        for hit in candidates:
            if not (0 <= hit.template_id < len(templates)):
                continue
            plan_mask = plan_masks_by_template.get(hit.template_id)
            if plan_mask is None:
                continue
            score = _gray_budget_geometry_score(hit, plan_mask)
            if score is not None:
                protected_scores[id(hit)] = score

    def _budget_rank(hit: CandidateHit) -> tuple[float, float, float, float, float, float]:
        geometry = protected_scores.get(id(hit))
        if geometry is not None:
            coverage, purity = geometry
            return (
                1.0,
                coverage,
                purity,
                float(hit.match_score),
                -float(hit.bbox[1]),
                -float(hit.bbox[0]),
            )
        return (
            0.0,
            float(hit.match_score),
            0.0,
            0.0,
            -float(hit.bbox[1]),
            -float(hit.bbox[0]),
        )

    def _limit_with_geometry_reserve(
        hits: list[CandidateHit],
        limit: int,
        extra_reserved_ids: set[int] | None = None,
    ) -> list[CandidateHit]:
        """Apply a soft cap without dropping raw hits that already prove geometry."""

        reserved_ids = extra_reserved_ids or set()
        reserved = [hit for hit in hits if id(hit) in protected_scores or id(hit) in reserved_ids]
        unreserved = [
            hit for hit in hits if id(hit) not in protected_scores and id(hit) not in reserved_ids
        ]
        reserved.sort(key=_budget_rank, reverse=True)
        unreserved.sort(key=_budget_rank, reverse=True)
        if len(reserved) >= limit:
            return reserved
        return reserved + unreserved[: max(0, limit - len(reserved))]

    grouped_by_variant: dict[tuple[int, float, int, bool, str], list[CandidateHit]] = {}
    for hit in candidates:
        grouped_by_variant.setdefault(
            (hit.template_id, hit.scale, hit.rotation, hit.mirrored, hit.source),
            [],
        ).append(hit)

    variant_limited: list[CandidateHit] = []
    raw_counts_by_template: dict[int, int] = {}
    for hits in grouped_by_variant.values():
        template_id = hits[0].template_id
        raw_counts_by_template[template_id] = raw_counts_by_template.get(template_id, 0) + len(hits)
        variant_limited.extend(_limit_with_geometry_reserve(hits, GRAY_RAW_MAX_HITS_PER_VARIANT))

    grouped_by_template: dict[int, list[CandidateHit]] = {}
    for hit in variant_limited:
        grouped_by_template.setdefault(hit.template_id, []).append(hit)

    template_limited: list[CandidateHit] = []
    for hits in grouped_by_template.values():
        template_limited.extend(_limit_with_geometry_reserve(hits, GRAY_RAW_MAX_HITS_PER_TEMPLATE))

    per_template_reserved_ids: set[int] = set()
    per_template_reserved_total = 0
    per_template_reserve_limit = max(0, int(GRAY_RAW_MIN_HITS_PER_TEMPLATE))
    if per_template_reserve_limit > 0:
        regrouped_template_limited: dict[int, list[CandidateHit]] = {}
        for hit in template_limited:
            regrouped_template_limited.setdefault(hit.template_id, []).append(hit)
        for hits in regrouped_template_limited.values():
            ranked = sorted(hits, key=_budget_rank, reverse=True)
            for hit in ranked[:per_template_reserve_limit]:
                per_template_reserved_ids.add(id(hit))
            per_template_reserved_total += min(len(ranked), per_template_reserve_limit)

    active_reserved_ids = (
        per_template_reserved_ids
        if len(protected_scores) < int(GRAY_RAW_MAX_TOTAL_HITS)
        else set()
    )
    if len(template_limited) > GRAY_RAW_MAX_TOTAL_HITS:
        template_limited = _limit_with_geometry_reserve(
            template_limited,
            GRAY_RAW_MAX_TOTAL_HITS,
            active_reserved_ids,
        )

    top_generators = []
    for template_id, count in sorted(
        raw_counts_by_template.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:8]:
        top_generators.append(
            {
                "templateId": int(template_id),
                "templateName": templates[template_id].name if 0 <= template_id < len(templates) else "",
                "rawHits": int(count),
            }
        )

    return template_limited, {
        "before": len(candidates),
        "after": len(template_limited),
        "removed": max(0, len(candidates) - len(template_limited)),
        "geometryProtected": len(protected_scores),
        "geometryProtectedKept": sum(1 for hit in template_limited if id(hit) in protected_scores),
        "geometryProtectedDropped": sum(1 for hit in candidates if id(hit) in protected_scores)
        - sum(1 for hit in template_limited if id(hit) in protected_scores),
        "perTemplateReserved": per_template_reserved_total,
        "perTemplateReservedKept": sum(
            1 for hit in template_limited if id(hit) in per_template_reserved_ids
        ),
        "perTemplateReservedActive": len(active_reserved_ids),
        "perTemplateReserveLimit": int(per_template_reserve_limit),
        "topGenerators": top_generators,
    }


def is_gray_frame_raw_rescue_hit(hit: CandidateHit, templates: list[TemplateInfo]) -> bool:
    """Keep promising gray hits through raw de-duplication."""

    if hit.dominant_hsv is not None or not (0 <= hit.template_id < len(templates)):
        return False
    if (
        not hit.is_text_label
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_RAW_SCAN_THRESHOLD
    ):
        return True
    if (
        not hit.is_text_label
        and hit.pixel_count >= GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_MID_GEOMETRY_MIN_MATCH
    ):
        return True
    if (
        hit.is_text_label
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_LABEL_GEOMETRY_MIN_MATCH
    ):
        return True
    return (
        hit.scale <= 0.70
        and hit.match_score >= GRAY_STRONG_GEOMETRY_MIN_MATCH
    )


def _is_complex_gray_geometry_hit(hit: CandidateHit) -> bool:
    """Recognize larger gray symbols whose evidence is geometry, not scale."""

    return (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_RAW_SCAN_THRESHOLD
        and hit.verification_score >= (GRAY_STRONG_RESCUE_MIN_VERIFICATION - 0.03)
        and hit.coverage >= GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_COMPLEX_GEOMETRY_MIN_PURITY
        and hit.context_purity >= GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT
    )


def _is_mid_gray_geometry_hit(hit: CandidateHit) -> bool:
    """Recognize medium gray symbols that are too large for small-scale rescue."""

    return (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and hit.pixel_count >= GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_MID_GEOMETRY_MIN_MATCH
        and hit.verification_score >= GRAY_MID_GEOMETRY_MIN_VERIFICATION
        and hit.coverage >= GRAY_MID_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_MID_GEOMETRY_MIN_PURITY
        and hit.context_purity >= GRAY_MID_GEOMETRY_MIN_CONTEXT
    )


def _is_tiny_gray_geometry_hit(hit: CandidateHit) -> bool:
    """Rescue compact gray symbols that are real but score just below 0.60."""

    return (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and hit.pixel_count <= GRAY_TINY_GEOMETRY_MAX_TEMPLATE_PIXELS
        and hit.scale <= GRAY_TINY_GEOMETRY_MAX_SCALE
        and hit.match_score >= GRAY_TINY_GEOMETRY_MIN_MATCH
        and hit.verification_score >= GRAY_TINY_GEOMETRY_MIN_VERIFICATION
        and hit.coverage >= GRAY_TINY_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_TINY_GEOMETRY_MIN_PURITY
        and hit.context_purity >= GRAY_TINY_GEOMETRY_MIN_CONTEXT
    )


def _is_strong_gray_label_geometry_hit(hit: CandidateHit) -> bool:
    """Rescue large text/label templates only when they are very clean."""

    return (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_LABEL_GEOMETRY_MIN_MATCH
        and hit.verification_score >= GRAY_LABEL_GEOMETRY_MIN_VERIFICATION
        and hit.coverage >= GRAY_LABEL_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_LABEL_GEOMETRY_MIN_PURITY
        and hit.context_purity >= GRAY_LABEL_GEOMETRY_MIN_CONTEXT
    )


def _is_gray_frame_validated_rescue_hit(
    hit: CandidateHit,
    templates: list[TemplateInfo],
) -> bool:
    if hit.dominant_hsv is not None or not (0 <= hit.template_id < len(templates)):
        return False
    if (
        _is_complex_gray_geometry_hit(hit)
        or _is_mid_gray_geometry_hit(hit)
        or _is_tiny_gray_geometry_hit(hit)
        or _is_strong_gray_label_geometry_hit(hit)
    ):
        return True
    return (
        hit.scale <= 0.70
        and hit.match_score >= GRAY_STRONG_GEOMETRY_MIN_MATCH
        and hit.coverage >= GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_STRONG_RESCUE_MIN_PURITY
        and hit.verification_score >= GRAY_STRONG_RESCUE_MIN_VERIFICATION
    )


def _same_physical_hit(left: CandidateHit, right: CandidateHit) -> bool:
    return same_physical_place(left, right, mode="gray")


def _hit_area(hit: CandidateHit) -> int:
    return max(1, hit.bbox[2] * hit.bbox[3])


def _hit_center(hit: CandidateHit) -> tuple[float, float]:
    x, y, width, height = hit.bbox
    return x + width / 2.0, y + height / 2.0


def _center_inside_bbox(
    center: tuple[float, float],
    bbox: tuple[int, int, int, int],
    *,
    margin_ratio: float = 0.04,
) -> bool:
    x, y = center
    bx, by, bw, bh = bbox
    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio
    return (bx - pad_x) <= x <= (bx + bw + pad_x) and (by - pad_y) <= y <= (
        by + bh + pad_y
    )


def _is_gray_symbol_hit(hit: CandidateHit) -> bool:
    return hit.dominant_hsv is None and not hit.is_text_label


def _is_weak_gray_text_fragment(hit: CandidateHit) -> bool:
    # Narrow text-label symbols can be real even with modest purity: nearby
    # lettering and frames add foreground to the ROI.  Keep them when the
    # template itself is almost perfectly covered and verification is strong.
    strong_label_geometry = (
        hit.match_score >= 0.68
        and hit.verification_score >= 0.60
        and hit.coverage >= 0.94
        and hit.purity >= 0.48
    )
    return (
        hit.dominant_hsv is None
        and hit.is_text_label
        and not strong_label_geometry
        and hit.context_purity <= GRAY_WEAK_LABEL_MAX_CONTEXT
        and hit.purity <= GRAY_WEAK_LABEL_MAX_PURITY
    )


def _is_gray_rect_frame_hit(hit: CandidateHit, templates: list[TemplateInfo]) -> bool:
    return (
        _is_gray_symbol_hit(hit)
        and 0 <= hit.template_id < len(templates)
        and is_gray_rect_frame_template(templates[hit.template_id])
    )


def _is_same_template_duplicate_shadow(small: CandidateHit, large: CandidateHit) -> bool:
    if small.template_id != large.template_id:
        return False

    small_area = _hit_area(small)
    large_area = _hit_area(large)
    if large_area < small_area * 1.20:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(small.bbox, large.bbox)
    if inter_area <= 0:
        return False

    center_nested = _center_inside_bbox(_hit_center(small), large.bbox, margin_ratio=0.04)
    if not (iom >= 0.55 or (center_nested and center_distance <= 0.58)):
        return False

    min_large_coverage = 0.86 if large.is_text_label else 0.88
    if large.coverage < min_large_coverage:
        return False

    return large.verification_score + 0.12 >= small.verification_score


def _weak_gray_compact_fragment_loses_to_larger(
    compact: CandidateHit,
    large: CandidateHit,
) -> bool:
    if compact.dominant_hsv is not None or large.dominant_hsv is not None:
        return False
    if compact.is_text_label != large.is_text_label:
        return False

    compact_area = _hit_area(compact)
    large_area = _hit_area(large)
    if compact_area > 3600 or large_area < compact_area * 1.24:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(compact.bbox, large.bbox)
    if inter_area <= 0:
        return False
    compact_center_nested = _center_inside_bbox(
        _hit_center(compact),
        large.bbox,
        margin_ratio=0.05,
    )
    if not (iom >= 0.48 and (compact_center_nested or center_distance <= 0.44)):
        return False

    if large.verification_score + 0.08 < compact.verification_score:
        return False
    if large.match_score + 0.08 < compact.match_score:
        return False

    weak_compact_evidence = compact.context_purity <= 0.24 and compact.purity <= 0.56
    stronger_large_evidence = (
        large.purity >= compact.purity + 0.08
        and large.context_purity >= compact.context_purity + 0.05
    )
    return weak_compact_evidence and stronger_large_evidence


def _compact_gray_hit_beats_large_partial(
    compact: CandidateHit,
    large: CandidateHit,
) -> bool:
    if compact.dominant_hsv is not None or large.dominant_hsv is not None:
        return False

    compact_area = _hit_area(compact)
    large_area = _hit_area(large)
    if compact_area > 1450 or large_area < compact_area * 1.55:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(compact.bbox, large.bbox)
    if inter_area > 0:
        compact_nested = _center_inside_bbox(
            _hit_center(compact),
            large.bbox,
            margin_ratio=0.06,
        )
        if not (iom >= 0.62 or (compact_nested and center_distance <= 0.64)):
            return False
    else:
        cx, cy, cw, ch = compact.bbox
        lx, ly, lw, lh = large.bbox
        overlap_x = max(0, min(cx + cw, lx + lw) - max(cx, lx))
        overlap_y = max(0, min(cy + ch, ly + lh) - max(cy, ly))
        gap_x = max(0, max(cx, lx) - min(cx + cw, lx + lw))
        gap_y = max(0, max(cy, ly) - min(cy + ch, ly + lh))
        aligned_x = overlap_x / max(1, min(cw, lw)) >= 0.45 and gap_y <= max(
            3,
            min(ch, lh) * 0.12,
        )
        aligned_y = overlap_y / max(1, min(ch, lh)) >= 0.45 and gap_x <= max(
            3,
            min(cw, lw) * 0.12,
        )
        if not (aligned_x or aligned_y):
            return False

    if compact.coverage < 0.98 or compact.purity < 0.49:
        return False
    if compact.verification_score < 0.56:
        return False
    if compact.verification_score + 0.03 < large.verification_score:
        return False

    compact_is_cleaner = (
        compact.purity >= large.purity + 0.045
        and compact.context_purity >= large.context_purity + 0.015
    )
    large_is_weak_partial = large.purity <= 0.48 and large.context_purity <= 0.24
    return compact_is_cleaner or large_is_weak_partial


def _strong_compact_gray_hit_should_coexist(
    left: CandidateHit,
    right: CandidateHit,
) -> bool:
    """Keep compact symbols that sit on/near a larger gray label frame.

    Gray E9 contains real compact symbols mounted on A1/AW-style long labels.
    Treating every overlap as competing interpretations makes those compact
    symbols disappear.  Keep only high-evidence compact hits; weak interior
    fragments are still handled by the nested suppression pass.
    """

    left_area = _hit_area(left)
    right_area = _hit_area(right)
    if left_area == right_area:
        return False

    compact = left if left_area < right_area else right
    large = right if compact is left else left
    compact_area = min(left_area, right_area)
    large_area = max(left_area, right_area)
    if compact_area > 4200 or large_area < compact_area * 2.0:
        return False

    large_aspect = max(
        large.bbox[2] / max(1, large.bbox[3]),
        large.bbox[3] / max(1, large.bbox[2]),
    )
    if large_aspect < 1.55 and large_area < compact_area * 2.8:
        return False

    return (
        compact.coverage >= 0.92
        and compact.purity >= 0.58
        and compact.context_purity >= 0.34
        and compact.verification_score >= 0.60
    )


def _suppress_nested_gray_core_hits(hits: list[CandidateHit]) -> list[CandidateHit]:
    """Drop smaller gray sub-symbols when a fuller symbol covers the same ink."""

    if len(hits) < 2:
        return hits

    suppressed: set[int] = set()
    for small_idx, small in enumerate(hits):
        if small.dominant_hsv is not None:
            continue

        small_area = _hit_area(small)
        for large_idx, large in enumerate(hits):
            if small_idx == large_idx or large.dominant_hsv is not None:
                continue
            if _is_same_template_duplicate_shadow(small, large):
                suppressed.add(small_idx)
                break
            if _weak_gray_compact_fragment_loses_to_larger(small, large):
                suppressed.add(small_idx)
                break
            area_ratio = 1.25 if large.template_id == small.template_id else GRAY_FULLER_SYMBOL_MIN_AREA_RATIO
            if _hit_area(large) < small_area * area_ratio:
                continue
            if large.coverage < min(0.90, GRAY_FULLER_SYMBOL_MIN_COVERAGE):
                continue
            if large.purity < GRAY_FULLER_SYMBOL_MIN_PURITY:
                continue
            if large.verification_score + GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP < small.verification_score:
                continue
            if _strong_compact_gray_hit_should_coexist(small, large):
                continue
            small_center = _hit_center(small)
            if not (
                _same_physical_hit(small, large)
                or _center_inside_bbox(small_center, large.bbox, margin_ratio=0.03)
            ):
                continue

            suppressed.add(small_idx)
            break

    for large_idx, large in enumerate(hits):
        if large_idx in suppressed or large.dominant_hsv is not None:
            continue

        for compact_idx, compact in enumerate(hits):
            if compact_idx == large_idx or compact_idx in suppressed:
                continue
            if _compact_gray_hit_beats_large_partial(compact, large):
                suppressed.add(large_idx)
                break

    if not suppressed:
        return hits

    return [hit for index, hit in enumerate(hits) if index not in suppressed]


def _dedupe_gray_overlapping_alternatives(hits: list[CandidateHit]) -> list[CandidateHit]:
    """Keep one gray interpretation when several symbols explain the same ink."""

    if len(hits) < 2:
        return hits

    def _score_rank(hit: CandidateHit) -> float:
        area_bonus = min(0.45, float(np.log1p(_hit_area(hit))) * 0.05)
        return float(
            hit.verification_score
            + hit.match_score
            + hit.context_purity
            + 0.70 * hit.purity
            + area_bonus
        )

    def _fuller_rank(hit: CandidateHit) -> tuple[float, ...]:
        return (
            float(_hit_area(hit)),
            float(hit.pixel_count),
            float(hit.context_purity),
            float(hit.purity),
            float(hit.coverage),
            float(hit.verification_score),
            float(hit.match_score),
            0.0 if hit.mirrored else 1.0,
        )

    def _competing_winner(
        left: CandidateHit,
        right: CandidateHit,
    ) -> CandidateHit | None:
        if left.dominant_hsv is not None or right.dominant_hsv is not None:
            return None
        if left.is_text_label != right.is_text_label:
            return None
        if not _same_physical_hit(left, right):
            return None
        if _strong_compact_gray_hit_should_coexist(left, right):
            return None
        if local_dominates(left, right, mode="gray"):
            return left
        if local_dominates(right, left, mode="gray"):
            return right

        score_left = _score_rank(left)
        score_right = _score_rank(right)
        score_margin = abs(score_left - score_right)
        area_ratio = max(_hit_area(left), _hit_area(right)) / max(
            1,
            min(_hit_area(left), _hit_area(right)),
        )
        similar_size_gray_symbols = (
            not left.is_text_label
            and not right.is_text_label
            and area_ratio <= 1.25
            and score_margin < 0.22
        )
        if similar_size_gray_symbols:
            return max((left, right), key=_fuller_rank)
        return left if score_left >= score_right else right

    ordered_hits = sorted(
        hits,
        key=lambda hit: (
            _score_rank(hit),
            *candidate_quality_key(hit, mode="gray"),
            -float(hit.bbox[1]),
            -float(hit.bbox[0]),
        ),
        reverse=True,
    )

    selected: list[CandidateHit] = []
    for candidate in ordered_hits:
        candidate_survives = True
        next_selected: list[CandidateHit] = []
        for existing in selected:
            winner = _competing_winner(candidate, existing)
            if winner is existing:
                candidate_survives = False
                next_selected.append(existing)
            elif winner is candidate:
                continue
            else:
                next_selected.append(existing)
        if candidate_survives:
            next_selected.append(candidate)
        selected = next_selected

    selected.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.verification_score))
    return selected


def _filter_weak_gray_text_fragments(hits: list[CandidateHit]) -> list[CandidateHit]:
    """Drop label fragments that only explain a thin slice of a larger symbol."""

    return [hit for hit in hits if not _is_weak_gray_text_fragment(hit)]


def _merge_duplicate_gray_rect_frames(
    hits: list[CandidateHit],
    templates: list[TemplateInfo],
) -> tuple[list[CandidateHit], int]:
    """Merge shifted hollow-frame detections that represent one oversized frame."""

    if len(hits) < 2:
        return hits, 0

    parent = list(range(len(hits)))

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

    def same_frame(left: CandidateHit, right: CandidateHit) -> bool:
        if left.template_id != right.template_id:
            return False
        if not (_is_gray_rect_frame_hit(left, templates) and _is_gray_rect_frame_hit(right, templates)):
            return False
        if (left.rotation % 180) != (right.rotation % 180):
            return False
        if abs(float(left.scale) - float(right.scale)) > GRAY_RECT_FRAME_MERGE_MAX_SCALE_DELTA:
            return False

        _inter, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
        if iou <= 0.0:
            return False
        return (
            iom >= GRAY_RECT_FRAME_MERGE_IOM
            and center_distance <= GRAY_RECT_FRAME_MERGE_CENTER_DISTANCE
        )

    for left_index in range(len(hits)):
        for right_index in range(left_index + 1, len(hits)):
            if same_frame(hits[left_index], hits[right_index]):
                union(left_index, right_index)

    groups: dict[int, list[CandidateHit]] = {}
    for index, hit in enumerate(hits):
        groups.setdefault(find(index), []).append(hit)

    merged_count = 0
    output: list[CandidateHit] = []
    for group_hits in groups.values():
        if len(group_hits) == 1:
            output.append(group_hits[0])
            continue

        winner = max(
            group_hits,
            key=lambda hit: (
                float(hit.verification_score),
                float(hit.match_score),
                float(hit.coverage),
            ),
        )
        min_x = min(hit.bbox[0] for hit in group_hits)
        min_y = min(hit.bbox[1] for hit in group_hits)
        max_x = max(hit.bbox[0] + hit.bbox[2] for hit in group_hits)
        max_y = max(hit.bbox[1] + hit.bbox[3] for hit in group_hits)
        winner.bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        output.append(winner)
        merged_count += len(group_hits) - 1

    output.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.verification_score))
    return output, merged_count


def _is_gray_rescue_blocked_by_existing(
    hit: CandidateHit,
    existing: CandidateHit,
) -> bool:
    """Do not resurrect small interior ghosts already defeated by clustering."""

    if hit.dominant_hsv is not None or existing.dominant_hsv is not None:
        return False
    if local_dominates(existing, hit, mode="gray"):
        return True
    if existing.template_id == hit.template_id and _same_physical_hit(existing, hit):
        return candidate_quality_key(existing, mode="gray") >= candidate_quality_key(hit, mode="gray")

    if _compact_gray_hit_beats_large_partial(hit, existing):
        return False
    if _is_weak_gray_text_fragment(existing) and (
        _is_tiny_gray_geometry_hit(hit) or _is_mid_gray_geometry_hit(hit)
    ):
        return False

    hit_area = _hit_area(hit)
    existing_area = _hit_area(existing)
    if existing_area < hit_area * 2.4:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(hit.bbox, existing.bbox)
    if inter_area <= 0:
        return False

    center_nested = _center_inside_bbox(_hit_center(hit), existing.bbox)
    if not (iom >= 0.72 or (center_nested and center_distance <= 0.62)):
        return False

    if existing.verification_score < 0.50 and existing.match_score < 0.60:
        return False

    # True rescues have their own surrounding ink. Interior ghosts usually
    # borrow the parent/label geometry, so their local context is much weaker.
    return hit.context_purity <= 0.35 or hit.purity <= 0.55 or existing.is_text_label


def _is_gray_rescue_locally_dominated(
    hit: CandidateHit,
    competitor: CandidateHit,
) -> bool:
    """Skip a rescued interpretation beaten by a better local gray candidate."""

    return local_dominates(competitor, hit, mode="gray")


def rescue_validated_gray_frame_hits(
    final_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    templates: list[TemplateInfo],
) -> tuple[list[CandidateHit], int, dict[str, dict[str, object]]]:
    """Re-add strong gray detections lost by global NMS/clustering."""

    trace: dict[str, dict[str, object]] = {}

    def _trace_stage(
        stage: str,
        hits: list[CandidateHit],
        reasons: dict[int, str] | None = None,
    ) -> None:
        trace[stage] = {"hits": hits, "reasons": reasons or {}}

    def _rescue_rank(hit: CandidateHit) -> tuple[float, ...]:
        x, y, width, height = hit.bbox
        return (
            *candidate_quality_key(hit, mode="gray"),
            -float(y),
            -float(x),
            -float(width * height),
            -float(hit.template_id),
        )

    rescue_candidates = sorted(
        (hit for hit in validated_hits if _is_gray_frame_validated_rescue_hit(hit, templates)),
        key=_rescue_rank,
        reverse=True,
    )
    _trace_stage("rescue_candidates", rescue_candidates)

    rescued: list[CandidateHit] = []
    dominated: list[CandidateHit] = []
    dominated_reasons: dict[int, str] = {}
    blocked: list[CandidateHit] = []
    blocked_reasons: dict[int, str] = {}
    local_competitors = final_hits

    for hit in rescue_candidates:
        dominator = next(
            (
                competitor
                for competitor in local_competitors
                if competitor is not hit and _is_gray_rescue_locally_dominated(hit, competitor)
                and not (
                    competitor.template_id == hit.template_id
                    and competitor.bbox == hit.bbox
                    and candidate_quality_key(competitor, mode="gray")
                    == candidate_quality_key(hit, mode="gray")
                )
            ),
            None,
        )
        if dominator is not None:
            dominated.append(hit)
            dominated_reasons[id(hit)] = (
                f"dominated_by:{templates[dominator.template_id].name}"
                if 0 <= dominator.template_id < len(templates)
                else "dominated_by:unknown"
            )
            continue

        duplicate = next(
            (
                existing
                for existing in final_hits + rescued
                if existing.template_id == hit.template_id and _same_physical_hit(existing, hit)
            ),
            None,
        )
        if duplicate is not None:
            blocked.append(hit)
            blocked_reasons[id(hit)] = "duplicate_same_template"
            continue

        blocker = next(
            (
                existing
                for existing in final_hits + rescued
                if _is_gray_rescue_blocked_by_existing(hit, existing)
            ),
            None,
        )
        if blocker is not None:
            blocked.append(hit)
            blocked_reasons[id(hit)] = (
                f"blocked_by:{templates[blocker.template_id].name}"
                if 0 <= blocker.template_id < len(templates)
                else "blocked_by:unknown"
            )
            continue

        rescued.append(hit)

    _trace_stage("rescue_dominated", dominated, dominated_reasons)
    _trace_stage("rescue_blocked_existing", blocked, blocked_reasons)
    _trace_stage("rescue_added", rescued)

    combined = final_hits + rescued
    combined = _filter_weak_gray_text_fragments(combined)
    _trace_stage("post_gray_filter_weak_text", combined)

    combined = _suppress_nested_gray_core_hits(combined)
    _trace_stage("post_gray_suppress_nested", combined)

    combined = _dedupe_gray_overlapping_alternatives(combined)
    _trace_stage("post_gray_dedupe", combined)

    combined, _merged_count = _merge_duplicate_gray_rect_frames(combined, templates)
    _trace_stage("post_gray_merge_frames", combined)

    combined.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.verification_score))
    return combined, len(rescued), trace


def trace_unresolved_strong_gray_hits(
    final_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    templates: list[TemplateInfo],
) -> dict:
    """Explain strong validated gray hits that did not survive final output."""

    strong_hits = [
        hit for hit in validated_hits if _is_gray_frame_validated_rescue_hit(hit, templates)
    ]
    unresolved: list[dict] = []

    for hit in sorted(strong_hits, key=lambda item: item.verification_score, reverse=True):
        same_template_final = [
            existing
            for existing in final_hits
            if existing.template_id == hit.template_id and _same_physical_hit(existing, hit)
        ]
        if same_template_final:
            continue

        blockers: list[tuple[float, CandidateHit]] = []
        for existing in final_hits:
            inter_area, iou, iom, center_distance = _bbox_metrics(existing.bbox, hit.bbox)
            if inter_area <= 0:
                continue
            if iou >= 0.10 or iom >= 0.45 or center_distance <= 0.45:
                blockers.append((max(iou, iom), existing))
        blockers.sort(key=lambda item: (item[0], item[1].verification_score), reverse=True)

        blocker = blockers[0][1] if blockers else None
        unresolved.append(
            {
                "symbol": templates[hit.template_id].name,
                "bbox": [int(value) for value in hit.bbox],
                "match": round(float(hit.match_score), 3),
                "verification": round(float(hit.verification_score), 3),
                "coverage": round(float(hit.coverage), 3),
                "purity": round(float(hit.purity), 3),
                "contextPurity": round(float(hit.context_purity), 3),
                "rotation": int(hit.rotation),
                "scale": round(float(hit.scale), 3),
                "mirrored": bool(hit.mirrored),
                "blockedBy": (
                    {
                        "symbol": templates[blocker.template_id].name,
                        "bbox": [int(value) for value in blocker.bbox],
                        "match": round(float(blocker.match_score), 3),
                        "verification": round(float(blocker.verification_score), 3),
                    }
                    if blocker is not None
                    else None
                ),
            }
        )

    max_items = max(0, int(GRAY_STRONG_TRACE_MAX_ITEMS))
    return {
        "strongValidated": len(strong_hits),
        "unresolved": len(unresolved),
        "items": unresolved[:max_items],
    }
