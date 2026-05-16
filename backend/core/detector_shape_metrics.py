"""Binary-mask shape and validation scoring helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.detector_config import (
    CONTEXT_MARGIN_RATIO,
    LABEL_CONTENT_MAX_RATIO,
    LABEL_CONTENT_MIN_PIXELS,
    LABEL_CONTENT_MIN_RATIO,
    LABEL_LINE_MIN_RATIO,
    LABEL_TEMPLATE_MIN_ASPECT_RATIO,
    LABEL_TEMPLATE_MIN_WIDTH,
)


def _thickness_normalized_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize stroke thickness while preserving the coarse symbol shape."""

    if mask.size == 0 or cv2.countNonZero(mask) == 0:
        return mask

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(opened) >= max(3, int(cv2.countNonZero(mask) * 0.35)):
        mask = opened
    return cv2.dilate(mask, kernel, iterations=1)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight bbox around foreground pixels in a binary mask."""

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _foreground_span_ratios(mask: np.ndarray) -> tuple[float, float]:
    bbox = _mask_bbox(mask)
    if bbox is None or mask.shape[0] <= 0 or mask.shape[1] <= 0:
        return 0.0, 0.0
    _x, _y, width, height = bbox
    return width / max(1, mask.shape[1]), height / max(1, mask.shape[0])


def _foreground_path_variance_ratio(mask: np.ndarray) -> float:
    """Return normalized centerline variance for elongated foreground shapes."""

    bbox = _mask_bbox(mask)
    if bbox is None:
        return 0.0

    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return 0.0

    crop = mask[y : y + height, x : x + width]
    centers: list[float] = []
    if width >= height:
        for col in range(width):
            ys = np.flatnonzero(crop[:, col] > 0)
            if ys.size:
                centers.append(float(np.mean(ys)))
        normalizer = max(1, height)
    else:
        for row in range(height):
            xs = np.flatnonzero(crop[row, :] > 0)
            if xs.size:
                centers.append(float(np.mean(xs)))
        normalizer = max(1, width)

    if len(centers) < 3:
        return 0.0
    return float(np.std(np.asarray(centers, dtype=np.float32))) / normalizer


def _tight_mask_crop(mask: np.ndarray | None) -> np.ndarray | None:
    """Crop a mask to its foreground bbox."""

    if mask is None:
        return None

    bbox = _mask_bbox(mask)
    if bbox is None:
        return None

    x, y, w, h = bbox
    return mask[y : y + h, x : x + w]


def _extract_label_content_mask(mask: np.ndarray) -> np.ndarray | None:
    """Extract glyph-like content from a label by removing long frame/line strokes."""

    height, width = mask.shape[:2]
    long_side = max(width, height)
    short_side = min(width, height)
    if long_side < LABEL_TEMPLATE_MIN_WIDTH:
        return None
    if short_side < max(18, int(round(LABEL_TEMPLATE_MIN_WIDTH * 0.45))):
        return None
    if long_side / max(1, short_side) < LABEL_TEMPLATE_MIN_ASPECT_RATIO:
        return None

    foreground_pixels = int(cv2.countNonZero(mask))
    if foreground_pixels <= 0:
        return None

    def collect_components(raw_content: np.ndarray, suffix_mode: bool) -> np.ndarray | None:
        components, labels, stats, _ = cv2.connectedComponentsWithStats(raw_content, connectivity=8)
        if suffix_mode and components <= 2:
            return None

        candidate_components: list[tuple[int, bool]] = []
        for component_id in range(1, components):
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < LABEL_CONTENT_MIN_PIXELS:
                continue

            center_x_ratio = (x + (w / 2.0)) / max(1, width)
            if suffix_mode and center_x_ratio < 0.35:
                continue

            aspect = w / max(1, h)
            inverse_aspect = h / max(1, w)
            if aspect > 7.0 and h <= 3:
                continue
            if inverse_aspect > 7.0 and w <= 3:
                continue
            if w >= int(width * 0.92) or h >= int(height * 0.92):
                continue

            bbox_area = max(1, w * h)
            fill_ratio = area / bbox_area
            horizontal_edge_marker = (
                fill_ratio >= 0.50
                and (y <= int(height * 0.12) or y + h >= int(height * 0.88))
                and h <= max(5, int(height * 0.16))
                and w <= max(8, int(width * 0.45))
            )
            vertical_edge_marker = (
                fill_ratio >= 0.50
                and (x <= int(width * 0.12) or x + w >= int(width * 0.88))
                and w <= max(5, int(width * 0.16))
                and h <= max(8, int(height * 0.45))
            )
            # Direction markers on framed labels can move around the label on the
            # plan. Keep the glyphs, but do not let a top/bottom/side arrow anchor
            # the content-only match to one legend orientation.
            is_edge_marker = horizontal_edge_marker or vertical_edge_marker
            candidate_components.append((component_id, is_edge_marker))

        non_marker_components = [
            component_id for component_id, is_marker in candidate_components if not is_marker
        ]
        if len(non_marker_components) >= 2:
            kept_component_ids = non_marker_components
        else:
            kept_component_ids = [component_id for component_id, _ in candidate_components]

        content = np.zeros_like(mask)
        for component_id in kept_component_ids:
            content[labels == component_id] = 255

        if len(kept_component_ids) < 2:
            return None

        content_pixels = int(cv2.countNonZero(content))
        if content_pixels < LABEL_CONTENT_MIN_PIXELS:
            return None

        content_ratio = content_pixels / max(1, foreground_pixels)
        if not (LABEL_CONTENT_MIN_RATIO <= content_ratio <= LABEL_CONTENT_MAX_RATIO):
            return None

        content_bbox = _mask_bbox(content)
        if content_bbox is None:
            return None

        content_x, _, content_w, content_h = content_bbox
        if content_w < max(6, int(round(width * 0.12))):
            return None
        if content_h < max(8, int(round(height * 0.20))):
            return None
        if not suffix_mode and content_x <= int(width * 0.10) and content_w < int(width * 0.55):
            return None

        return content

    horizontal_len = max(5, min(width, int(round(width * 0.42))))
    vertical_len = max(7, min(height, int(round(height * 0.72))))
    horizontal = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((1, horizontal_len), np.uint8),
    )
    vertical = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((vertical_len, 1), np.uint8),
    )
    line_mask = cv2.bitwise_or(horizontal, vertical)
    line_ratio = int(cv2.countNonZero(line_mask)) / max(1, foreground_pixels)
    if line_ratio >= LABEL_LINE_MIN_RATIO:
        content = collect_components(
            cv2.bitwise_and(mask, cv2.bitwise_not(line_mask)), suffix_mode=False
        )
        if content is not None:
            return content

    return collect_components(mask, suffix_mode=True)


def _label_content_score(
    roi: np.ndarray,
    template_content_mask: np.ndarray | None,
    template_content_pixels: int,
    validation_cache: Any | None = None,
) -> float:
    """Score how well the ROI matches the template glyph mask, ignoring frame strokes."""

    if template_content_mask is None or template_content_pixels <= 0:
        return 0.0
    if roi.shape != template_content_mask.shape:
        return 0.0

    content_bbox = (
        validation_cache.content_bbox(template_content_mask)
        if validation_cache is not None
        else _mask_bbox(template_content_mask)
    )
    if content_bbox is None:
        return 0.0

    intersection = int(cv2.countNonZero(cv2.bitwise_and(roi, template_content_mask)))
    coverage = intersection / max(1, template_content_pixels)

    x, y, w, h = content_bbox
    roi_content_window = roi[y : y + h, x : x + w]
    foreground = int(cv2.countNonZero(roi_content_window))
    if foreground <= 0:
        return 0.0

    purity = intersection / max(1, foreground)
    return max(0.0, min(1.0, (0.60 * coverage) + (0.40 * purity)))


def _roi_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Extract ROI from a binary mask."""

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    roi = mask[y : y + h, x : x + w]
    if roi.size == 0 or roi.shape[0] != h or roi.shape[1] != w:
        return None
    return roi


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Return the centroid of foreground pixels in a binary mask."""

    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return None

    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def _context_purity(
    plan_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    intersection_mask: np.ndarray,
    *,
    explained_pixels: int | None = None,
    validation_cache: Any | None = None,
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
