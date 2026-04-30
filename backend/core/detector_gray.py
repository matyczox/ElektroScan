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
    GRAY_LEGEND_MIN_EVIDENCE_THRESHOLD,
    GRAY_LEGEND_MIN_ZONE_THRESHOLD,
    GRAY_LEGEND_ZONE_MARGIN,
    GRAY_RAW_MAX_HITS_PER_TEMPLATE,
    GRAY_RAW_MAX_HITS_PER_VARIANT,
    GRAY_RAW_MAX_TOTAL_HITS,
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
    GRAY_SEARCH_MAX_ROIS,
    GRAY_SEARCH_MAX_TILE_ROIS,
    GRAY_SEARCH_ROI_CONTAINMENT_THRESHOLD,
    GRAY_SEARCH_ROI_OVERLAP_THRESHOLD,
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
)
from core.detector_masks import _ink_mask, _suppress_long_strokes
from core.detector_models import CandidateHit, TemplateInfo


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

    return gray_template_area(template) >= GRAY_RAW_SCAN_MIN_TEMPLATE_AREA


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
) -> list[tuple[int, int, int, int]]:
    """Add coarse gray tiles so symbols connected to walls still get scanned."""

    if GRAY_SEARCH_MAX_TILE_ROIS <= 0:
        return rois

    tile_size = max(64, int(GRAY_SEARCH_TILE_SIZE))
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

            foreground = int(cv2.countNonZero(plan_mask[y : y + tile_h, x : x + tile_w]))
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
    for _foreground, rect in tile_candidates[:GRAY_SEARCH_MAX_TILE_ROIS]:
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
) -> tuple[list[tuple[int, int, int, int]], bool, int, int]:
    """Build bounded ROIs for gray/ink plans."""

    foreground_pixels = int(cv2.countNonZero(plan_mask))
    if foreground_pixels <= 0:
        return [], False, 0, foreground_pixels

    image_h, image_w = image_shape[:2]
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
    if components <= 1:
        return [], False, 0, foreground_pixels

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
    for component_id in range(1, components):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 6 or area > max_component_area:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        if w > max_component_width or h > max_component_height:
            continue

        clamped = _clamp_roi((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), image_shape)
        if clamped is not None:
            bbox_area = max(1, w * h)
            aspect = w / max(1, h)
            area_score = abs(np.log((bbox_area + 1) / target_area))
            aspect_score = abs(np.log((aspect + 0.05) / max(0.05, target_aspect)))
            score = float(area_score + 0.35 * aspect_score)
            component_rects.append((score, area, clamped))

    if not component_rects:
        return [], False, 0, foreground_pixels

    component_rects.sort(key=lambda item: (item[0], -item[1]))
    rois = [rect for _score, _area, rect in component_rects[:GRAY_SEARCH_MAX_ROIS]]
    rois = _append_gray_tile_rois(rois, plan_mask, image_shape)
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

    intersection = int(cv2.countNonZero(cv2.bitwise_and(roi, hit.transformed_mask)))
    coverage = intersection / max(1, hit.pixel_count)
    purity = intersection / max(1, roi_foreground)
    if (
        hit.match_score < GRAY_STRONG_GEOMETRY_MIN_MATCH
        or coverage < GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        or purity < GRAY_STRONG_RESCUE_MIN_PURITY
    ):
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

    def _budget_rank(hit: CandidateHit) -> tuple[float, float, float, float]:
        geometry = protected_scores.get(id(hit))
        if geometry is not None:
            coverage, purity = geometry
            return (1.0, coverage, purity, float(hit.match_score))
        return (0.0, float(hit.match_score), 0.0, 0.0)

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
        hits.sort(key=_budget_rank, reverse=True)
        variant_limited.extend(hits[:GRAY_RAW_MAX_HITS_PER_VARIANT])

    grouped_by_template: dict[int, list[CandidateHit]] = {}
    for hit in variant_limited:
        grouped_by_template.setdefault(hit.template_id, []).append(hit)

    template_limited: list[CandidateHit] = []
    for hits in grouped_by_template.values():
        hits.sort(key=_budget_rank, reverse=True)
        template_limited.extend(hits[:GRAY_RAW_MAX_HITS_PER_TEMPLATE])

    if len(template_limited) > GRAY_RAW_MAX_TOTAL_HITS:
        template_limited.sort(key=_budget_rank, reverse=True)
        template_limited = template_limited[:GRAY_RAW_MAX_TOTAL_HITS]

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
        "topGenerators": top_generators,
    }


def is_gray_frame_raw_rescue_hit(hit: CandidateHit, templates: list[TemplateInfo]) -> bool:
    """Keep promising gray hits through raw de-duplication."""

    if hit.dominant_hsv is not None or not (0 <= hit.template_id < len(templates)):
        return False
    return (
        hit.scale <= 0.70
        and hit.match_score >= GRAY_STRONG_GEOMETRY_MIN_MATCH
    )


def _is_gray_frame_validated_rescue_hit(
    hit: CandidateHit,
    templates: list[TemplateInfo],
) -> bool:
    if hit.dominant_hsv is not None or not (0 <= hit.template_id < len(templates)):
        return False
    return (
        hit.scale <= 0.70
        and hit.match_score >= GRAY_STRONG_GEOMETRY_MIN_MATCH
        and hit.coverage >= GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_STRONG_RESCUE_MIN_PURITY
        and hit.verification_score >= GRAY_STRONG_RESCUE_MIN_VERIFICATION
    )


def _same_physical_hit(left: CandidateHit, right: CandidateHit) -> bool:
    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False
    return iou >= 0.35 or iom >= 0.72 or center_distance <= 0.30


def _hit_area(hit: CandidateHit) -> int:
    return max(1, hit.bbox[2] * hit.bbox[3])


def _is_gray_symbol_hit(hit: CandidateHit) -> bool:
    return hit.dominant_hsv is None and not hit.is_text_label


def _is_gray_rect_frame_hit(hit: CandidateHit, templates: list[TemplateInfo]) -> bool:
    return (
        _is_gray_symbol_hit(hit)
        and 0 <= hit.template_id < len(templates)
        and is_gray_rect_frame_template(templates[hit.template_id])
    )


def _suppress_nested_gray_core_hits(hits: list[CandidateHit]) -> list[CandidateHit]:
    """Drop smaller gray sub-symbols when a fuller symbol covers the same ink."""

    if len(hits) < 2:
        return hits

    suppressed: set[int] = set()
    for small_idx, small in enumerate(hits):
        if not _is_gray_symbol_hit(small):
            continue

        small_area = _hit_area(small)
        for large_idx, large in enumerate(hits):
            if small_idx == large_idx or not _is_gray_symbol_hit(large):
                continue
            if _hit_area(large) < small_area * GRAY_FULLER_SYMBOL_MIN_AREA_RATIO:
                continue
            if large.coverage < GRAY_FULLER_SYMBOL_MIN_COVERAGE:
                continue
            if large.purity < GRAY_FULLER_SYMBOL_MIN_PURITY:
                continue
            if large.verification_score + GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP < small.verification_score:
                continue
            if not _same_physical_hit(small, large):
                continue

            suppressed.add(small_idx)
            break

    if not suppressed:
        return hits

    return [hit for index, hit in enumerate(hits) if index not in suppressed]


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


def rescue_validated_gray_frame_hits(
    final_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    templates: list[TemplateInfo],
) -> tuple[list[CandidateHit], int]:
    """Re-add strong gray detections lost by global NMS/clustering."""

    rescued: list[CandidateHit] = []
    for hit in validated_hits:
        if not _is_gray_frame_validated_rescue_hit(hit, templates):
            continue
        duplicate = any(
            existing.template_id == hit.template_id and _same_physical_hit(existing, hit)
            for existing in final_hits + rescued
        )
        if not duplicate:
            rescued.append(hit)

    if not rescued:
        suppressed = _suppress_nested_gray_core_hits(final_hits)
        merged, _merged_count = _merge_duplicate_gray_rect_frames(suppressed, templates)
        return merged, 0

    combined = final_hits + rescued
    combined = _suppress_nested_gray_core_hits(combined)
    combined, _merged_count = _merge_duplicate_gray_rect_frames(combined, templates)
    combined.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.verification_score))
    return combined, len(rescued)


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
