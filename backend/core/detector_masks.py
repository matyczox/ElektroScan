"""Masking, ROI and candidate validation helpers."""

from __future__ import annotations

import cv2
import numpy as np

from core.detector_config import (
    COLOR_HUE_REJECTION_THRESHOLD,
    COLOR_HUE_TOLERANCE,
    COLOR_SAT_TOLERANCE,
    COLOR_VAL_TOLERANCE,
    CONTEXT_MARGIN_RATIO,
    DILATE_KERNEL,
    HSV_LOWER,
    HSV_UPPER,
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


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Clamp a bbox to image bounds."""

    height = int(image_shape[0])
    width = int(image_shape[1])

    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2 - x1, y2 - y1)


def _hsv_mask(
    image_bgr: np.ndarray,
    dilate: bool = False,
    hsv_image: np.ndarray | None = None,
) -> np.ndarray:
    """Create a binary mask of colored pixels."""

    hsv = hsv_image if hsv_image is not None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    if dilate:
        mask = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    return mask


def _dominant_hsv_color(image_bgr: np.ndarray) -> tuple[int, int, int] | None:
    """Return the dominant HSV color among colored pixels."""

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    colored_pixels = hsv[mask > 0]
    if len(colored_pixels) == 0:
        return None

    h_med = int(np.median(colored_pixels[:, 0]))
    s_med = int(np.median(colored_pixels[:, 1]))
    v_med = int(np.median(colored_pixels[:, 2]))
    return (h_med, s_med, v_med)


def _color_mask_for_template(
    image_bgr: np.ndarray,
    dominant_hsv: tuple[int, int, int],
    dilate: bool = False,
    hsv_image: np.ndarray | None = None,
) -> np.ndarray:
    """Create a color-specific binary mask aligned to the template hue."""

    h, s, v = dominant_hsv
    lower1 = np.array(
        [
            max(0, h - COLOR_HUE_TOLERANCE),
            max(0, s - COLOR_SAT_TOLERANCE),
            max(0, v - COLOR_VAL_TOLERANCE),
        ]
    )
    upper1 = np.array(
        [
            min(180, h + COLOR_HUE_TOLERANCE),
            min(255, s + COLOR_SAT_TOLERANCE),
            min(255, v + COLOR_VAL_TOLERANCE),
        ]
    )

    hsv = hsv_image if hsv_image is not None else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower1, upper1)

    if h - COLOR_HUE_TOLERANCE < 0:
        lower2 = np.array([180 + h - COLOR_HUE_TOLERANCE, lower1[1], lower1[2]])
        upper2 = np.array([180, upper1[1], upper1[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
    elif h + COLOR_HUE_TOLERANCE > 180:
        lower2 = np.array([0, lower1[1], lower1[2]])
        upper2 = np.array([h + COLOR_HUE_TOLERANCE - 180, upper1[1], upper1[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

    if dilate:
        mask = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    return mask


def _odd_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _find_local_maxima(
    match_result: np.ndarray,
    threshold: float,
    template_width: int,
    template_height: int,
) -> list[tuple[int, int, float]]:
    """Return only local maxima instead of every pixel above threshold."""

    if match_result.size == 0:
        return []

    kernel_w = min(match_result.shape[1], _odd_size(max(3, int(template_width * LOCAL_MAX_KERNEL_RATIO))))
    kernel_h = min(match_result.shape[0], _odd_size(max(3, int(template_height * LOCAL_MAX_KERNEL_RATIO))))
    kernel = np.ones((kernel_h, kernel_w), np.uint8)

    local_max = cv2.dilate(match_result, kernel)
    mask = (match_result >= threshold) & (match_result >= (local_max - 1e-6))
    ys, xs = np.where(mask)

    peaks = [(int(x), int(y), float(match_result[y, x])) for y, x in zip(ys, xs)]
    peaks.sort(key=lambda item: item[2], reverse=True)
    return peaks


def _roi_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Extract ROI from a binary mask."""

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    roi = mask[y : y + h, x : x + w]
    if roi.size == 0 or roi.shape[0] != h or roi.shape[1] != w:
        return None
    return roi


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    """Return integer area for an x/y/w/h rectangle."""

    return max(0, int(rect[2])) * max(0, int(rect[3]))


def _rects_touch_or_overlap(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    gap: int = 0,
) -> bool:
    """Return True when two rectangles overlap or are close enough to merge."""

    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return not (
        lx + lw + gap < rx
        or rx + rw + gap < lx
        or ly + lh + gap < ry
        or ry + rh + gap < ly
    )


def _union_rect(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Return the smallest x/y/w/h rectangle containing both inputs."""

    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x0 = min(lx, rx)
    y0 = min(ly, ry)
    x1 = max(lx + lw, rx + rw)
    y1 = max(ly + lh, ry + rh)
    return (x0, y0, x1 - x0, y1 - y0)


def _merge_search_rois(
    rois: list[tuple[int, int, int, int]],
    gap: int = ROI_MERGE_GAP_PIXELS,
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping search windows so template matching does not duplicate work."""

    merged: list[tuple[int, int, int, int]] = []
    for rect in sorted(rois, key=lambda item: (item[0], item[1], item[2] * item[3])):
        absorbed = False
        for idx, existing in enumerate(merged):
            if _rects_touch_or_overlap(existing, rect, gap):
                merged[idx] = _union_rect(existing, rect)
                absorbed = True
                break
        if not absorbed:
            merged.append(rect)

    changed = True
    while changed:
        changed = False
        compacted: list[tuple[int, int, int, int]] = []
        for rect in merged:
            absorbed = False
            for idx, existing in enumerate(compacted):
                if _rects_touch_or_overlap(existing, rect, gap):
                    compacted[idx] = _union_rect(existing, rect)
                    absorbed = True
                    changed = True
                    break
            if not absorbed:
                compacted.append(rect)
        merged = compacted

    return merged


def _full_scan_roi(image_shape: tuple[int, int, int] | tuple[int, int]) -> tuple[int, int, int, int]:
    """Return a full-image scan rectangle."""

    return (0, 0, int(image_shape[1]), int(image_shape[0]))


def _build_search_rois(
    plan_mask: np.ndarray,
    image_shape: tuple[int, int, int] | tuple[int, int],
    max_template_width: int,
    max_template_height: int,
) -> tuple[list[tuple[int, int, int, int]], bool, int, int]:
    """Build per-request scan windows around colored foreground instead of white space."""

    full_roi = _full_scan_roi(image_shape)
    full_area = max(1, _rect_area(full_roi))
    foreground_pixels = int(cv2.countNonZero(plan_mask))
    if foreground_pixels <= 0:
        return [], False, 0, foreground_pixels

    kernel_size = _odd_size(ROI_COMPONENT_DILATE_PIXELS)
    seed_mask = cv2.dilate(
        plan_mask,
        np.ones((kernel_size, kernel_size), np.uint8),
        iterations=1,
    )
    components, _, stats, _ = cv2.connectedComponentsWithStats(seed_mask, connectivity=8)
    if components <= 1:
        return [], False, 0, foreground_pixels
    if components - 1 > ROI_MAX_COMPONENTS:
        return [full_roi], True, full_area, foreground_pixels

    pad_x = max(8, int(round(max_template_width * ROI_PADDING_RATIO)))
    pad_y = max(8, int(round(max_template_height * ROI_PADDING_RATIO)))
    rois: list[tuple[int, int, int, int]] = []

    for component_id in range(1, components):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < ROI_MIN_COMPONENT_PIXELS:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        clamped = _clamp_bbox((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), image_shape)
        if clamped is not None:
            rois.append(clamped)

    if not rois:
        return [], False, 0, foreground_pixels

    merged_rois = _merge_search_rois(rois)
    roi_area = sum(_rect_area(roi) for roi in merged_rois)
    if roi_area >= int(full_area * ROI_FULL_SCAN_AREA_RATIO):
        return [full_roi], True, full_area, foreground_pixels

    return merged_rois, False, roi_area, foreground_pixels


def _cached_dilated_mask(
    template_id: int,
    plan_mask: np.ndarray,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    """Return a cached one-pixel dilation of a full plan mask."""

    if template_id not in cache:
        cache[template_id] = cv2.dilate(plan_mask, DILATE_KERNEL, iterations=1)
    return cache[template_id]


def _hue_distance(hue_a: int, hue_b: int) -> int:
    """Circular hue distance in OpenCV's 0-180 HSV space."""

    diff = abs(hue_a - hue_b)
    return min(diff, 180 - diff)


def _roi_color_similarity(
    plan_image: np.ndarray,
    plan_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    dominant_hsv: tuple[int, int, int] | None,
) -> float:
    """Compare ROI hue with template hue and return a [0, 1] similarity score."""

    if dominant_hsv is None:
        return 1.0

    x, y, w, h = bbox
    roi_image = plan_image[y : y + h, x : x + w]
    roi_mask = plan_mask[y : y + h, x : x + w]
    if roi_image.size == 0 or roi_mask.size == 0:
        return 0.0

    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    colored_pixels = hsv_roi[roi_mask > 0]
    if len(colored_pixels) == 0:
        return 0.0

    roi_hue = int(np.median(colored_pixels[:, 0]))
    diff = _hue_distance(roi_hue, dominant_hsv[0])
    if diff > COLOR_HUE_REJECTION_THRESHOLD:
        return 0.0

    return max(0.0, 1.0 - (diff / COLOR_HUE_REJECTION_THRESHOLD))


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
) -> float:
    """Measure how much local foreground around the hit is explained by the template."""

    x, y, w, h = bbox
    margin = max(3, int(round(max(w, h) * CONTEXT_MARGIN_RATIO)))

    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(plan_mask.shape[1], x + w + margin)
    y1 = min(plan_mask.shape[0], y + h + margin)

    context_mask = plan_mask[y0:y1, x0:x1]
    if context_mask.size == 0:
        return 0.0

    context_foreground = int(cv2.countNonZero(context_mask))
    if context_foreground == 0:
        return 0.0

    explained_pixels = int(cv2.countNonZero(intersection_mask))
    return explained_pixels / context_foreground


def _validate_template_hit(
    hit: CandidateHit,
    plan_mask: np.ndarray,
    plan_image: np.ndarray,
) -> bool:
    """Validate a candidate by foreground overlap, purity and hue consistency."""

    if hit.transformed_mask is None:
        return True

    roi = _roi_mask(plan_mask, hit.bbox)
    if roi is None or roi.shape != hit.transformed_mask.shape:
        return False

    roi_foreground = int(cv2.countNonZero(roi))
    if roi_foreground == 0 or hit.pixel_count <= 0:
        return False

    intersection_mask = cv2.bitwise_and(roi, hit.transformed_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    coverage = intersection / hit.pixel_count
    purity = intersection / roi_foreground

    if coverage < MIN_COVERAGE_RATIO or purity < MIN_PURITY_RATIO:
        return False

    if (
        context_purity := _context_purity(plan_mask, hit.bbox, intersection_mask)
    ) <= 0.0:
        return False

    if (
        context_purity < NOISY_PARTIAL_CONTEXT_THRESHOLD
        and coverage < NOISY_PARTIAL_COVERAGE_THRESHOLD
        and purity < NOISY_PARTIAL_PURITY_THRESHOLD
    ):
        return False

    template_centroid = _mask_centroid(hit.transformed_mask)
    intersection_centroid = _mask_centroid(intersection_mask)
    if template_centroid is None or intersection_centroid is None:
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
        return False

    if hit.match_score < LOW_MATCH_STRICT_THRESHOLD and context_purity < MIN_CONTEXT_PURITY:
        return False

    color_similarity = _roi_color_similarity(plan_image, plan_mask, hit.bbox, hit.dominant_hsv)
    if color_similarity <= 0.0:
        return False

    verification_score = (
        0.45 * hit.match_score
        + 0.20 * coverage
        + 0.15 * purity
        + 0.20 * context_purity
    )

    if verification_score < MIN_VERIFICATION_SCORE:
        return False

    hit.coverage = round(coverage, 4)
    hit.purity = round(purity, 4)
    hit.context_purity = round(context_purity, 4)
    hit.color_similarity = round(color_similarity, 4)
    hit.verification_score = round(verification_score, 4)
    return True
