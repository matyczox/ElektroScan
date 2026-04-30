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
    GRAY_RAW_MAX_HITS_PER_TEMPLATE,
    GRAY_RAW_MAX_HITS_PER_VARIANT,
    GRAY_RAW_MAX_TOTAL_HITS,
    GRAY_RAW_SCAN_MIN_TEMPLATE_AREA,
    GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS,
    GRAY_RAW_SCAN_THRESHOLD,
    GRAY_SEARCH_COMPONENT_PADDING_RATIO,
    GRAY_SEARCH_MAX_ROIS,
    GRAY_SEARCH_MAX_TILE_ROIS,
    GRAY_SEARCH_TILE_MIN_FOREGROUND,
    GRAY_SEARCH_TILE_PADDING,
    GRAY_SEARCH_TILE_SIZE,
    GRAY_SPATIAL_FAIR_PEAKS_PER_ROI,
    GRAY_STRICT_SCAN_THRESHOLD,
    GRAY_STRONG_GEOMETRY_MIN_COVERAGE,
    GRAY_STRONG_GEOMETRY_MIN_PURITY,
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


def use_relaxed_gray_scan_threshold(template: TemplateInfo) -> bool:
    """Relax gray scan threshold only for genuinely large framed templates."""

    return gray_template_area(template) >= GRAY_RAW_SCAN_MIN_TEMPLATE_AREA


def use_raw_gray_scan_mask(template: TemplateInfo) -> bool:
    """Use dark raw ink for large/complex shapes whose frame is the signal."""

    return (
        use_relaxed_gray_scan_threshold(template)
        or gray_template_pixels(template) >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
    )


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
    return use_relaxed_gray_scan_threshold(template)


def gray_spatial_fair_peaks_per_roi() -> int:
    return GRAY_SPATIAL_FAIR_PEAKS_PER_ROI


def build_gray_scan_masks(
    *,
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    plan_masks_by_template: dict[int, np.ndarray],
    exclude_rects: list[tuple[int, int, int, int]],
    raw_dilated: np.ndarray,
) -> GrayScanMasks:
    """Build strict dark gray scan masks while keeping raw masks for validation."""

    raw_ink_pixels = int(cv2.countNonZero(raw_dilated))
    dark_base = _ink_mask(
        plan_image,
        dilate=False,
        threshold=GRAY_DARK_INK_THRESHOLD,
    )
    zone_base = _ink_mask(
        plan_image,
        dilate=False,
        threshold=GRAY_DARK_ZONE_THRESHOLD,
    )
    evidence_base = _ink_mask(
        plan_image,
        dilate=False,
        threshold=GRAY_DARK_EVIDENCE_THRESHOLD,
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
    kernel = np.ones((3, 3), np.uint8)
    seed_mask = cv2.dilate(plan_mask, kernel, iterations=1)
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
        coverage < GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        or purity < GRAY_STRONG_GEOMETRY_MIN_PURITY
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
            template = templates[hit.template_id]
            if not use_relaxed_gray_scan_threshold(template):
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
    """Keep promising framed gray hits through raw de-duplication."""

    if hit.dominant_hsv is not None or not (0 <= hit.template_id < len(templates)):
        return False
    template = templates[hit.template_id]
    return (
        use_relaxed_gray_scan_threshold(template)
        and hit.scale <= 0.70
        and hit.match_score >= GRAY_RAW_SCAN_THRESHOLD
    )


def _is_gray_frame_validated_rescue_hit(
    hit: CandidateHit,
    templates: list[TemplateInfo],
) -> bool:
    if hit.dominant_hsv is not None or not (0 <= hit.template_id < len(templates)):
        return False
    template = templates[hit.template_id]
    return (
        use_relaxed_gray_scan_threshold(template)
        and hit.scale <= 0.70
        and hit.coverage >= GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_STRONG_GEOMETRY_MIN_PURITY
        and hit.verification_score >= 0.60
    )


def _same_physical_hit(left: CandidateHit, right: CandidateHit) -> bool:
    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False
    return iou >= 0.35 or iom >= 0.72 or center_distance <= 0.30


def rescue_validated_gray_frame_hits(
    final_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    templates: list[TemplateInfo],
) -> tuple[list[CandidateHit], int]:
    """Re-add strong framed gray detections lost by global NMS/clustering."""

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
        return final_hits, 0

    combined = final_hits + rescued
    combined.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.verification_score))
    return combined, len(rescued)
