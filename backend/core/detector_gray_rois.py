"""Gray detector search ROI construction helpers."""

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
from core.detector_models import TemplateInfo
from core.detector_gray_masks import (
    _clamp_roi,
    adapt_gray_tile_roi_strategy_for_plan,
    gray_template_area,
    gray_tile_roi_strategy,
    use_large_text_tile_rois,
)

class GraySearchComponentIndex:
    foreground_pixels: int
    components: int
    stats: np.ndarray
    foreground_integral: np.ndarray


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
    component_supplement_rois: int = 0,
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
    all_component_rects = list(component_rects)
    component_rects = _limit_component_rects_spatially(
        all_component_rects,
        GRAY_SEARCH_MAX_ROIS,
    )
    supplement_limit = max(0, int(component_supplement_rois))
    if supplement_limit and len(all_component_rects) > len(component_rects):
        selected = {rect for _score, _area, rect in component_rects}
        covered_cells = {
            (
                int((rect[0] + rect[2] / 2) // max(192, int(GRAY_SEARCH_FAST_TILE_SIZE))),
                int((rect[1] + rect[3] / 2) // max(192, int(GRAY_SEARCH_FAST_TILE_SIZE))),
            )
            for rect in selected
        }
        added_cells: set[tuple[int, int]] = set()
        for score, area, rect in all_component_rects:
            if rect in selected or area < 24:
                continue
            cell = (
                int((rect[0] + rect[2] / 2) // max(192, int(GRAY_SEARCH_FAST_TILE_SIZE))),
                int((rect[1] + rect[3] / 2) // max(192, int(GRAY_SEARCH_FAST_TILE_SIZE))),
            )
            if cell in added_cells:
                continue
            # Text labels often split into several small disconnected glyph strokes.
            # Keep a few extra spatial sentinels so sparse labels do not vanish behind
            # denser neighbouring tiles before matchTemplate gets a chance to score them.
            if cell in covered_cells and score > 2.4 and area < 80:
                continue
            component_rects.append((score, area, rect))
            selected.add(rect)
            added_cells.add(cell)
            if len(added_cells) >= supplement_limit:
                break
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
