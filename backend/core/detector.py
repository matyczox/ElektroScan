"""
detector.py - CPU-friendly symbol detection for electrical plans.
"""

from __future__ import annotations

import glob
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz
import numpy as np


# Configuration constants


def _safe_cpu_count() -> int:
    """Return a sane positive CPU count fallback."""

    return max(1, int(os.cpu_count() or 1))


def _default_detector_workers() -> int:
    """Use available logical cores while keeping OpenCV internal threading low."""

    return _safe_cpu_count()


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read an integer env var safely."""

    raw = os.getenv(name)
    if raw is None or raw == "":
        return max(minimum, default)
    try:
        return max(minimum, int(raw))
    except ValueError:
        return max(minimum, default)


DETECTOR_SCAN_MAX_WORKERS = _env_int(
    "ELEKTROSCAN_DETECTOR_SCAN_WORKERS",
    _default_detector_workers(),
)
DETECTOR_POSTPROCESS_MAX_WORKERS = _env_int(
    "ELEKTROSCAN_DETECTOR_POSTPROCESS_WORKERS",
    _default_detector_workers(),
)
OPENCV_NUM_THREADS = _env_int("ELEKTROSCAN_OPENCV_THREADS", 1)

try:
    cv2.setNumThreads(OPENCV_NUM_THREADS)
except Exception:
    pass

HSV_LOWER = np.array([0, 30, 50])
HSV_UPPER = np.array([180, 255, 255])

DILATE_KERNEL = np.ones((3, 3), np.uint8)

ROTATIONS = [
    (0, None),
    (90, cv2.ROTATE_90_CLOCKWISE),
    (180, cv2.ROTATE_180),
    (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
]
SCALES = [0.90, 1.00, 1.10]

THRESHOLD_PRECISE = 0.55
THRESHOLD_DILATED = 0.45

MAX_PEAKS_PER_VARIANT = 1500
MIN_TEMPLATE_PIXELS = 20

MIN_COVERAGE_RATIO = 0.24
MIN_PURITY_RATIO = 0.08
MIN_CONTEXT_PURITY = 0.72
MIN_VERIFICATION_SCORE = 0.40
MAX_CENTROID_OFFSET_RATIO = 0.18
LOW_MATCH_STRICT_THRESHOLD = 0.58
CONTEXT_MARGIN_RATIO = 0.40

LOCAL_MAX_KERNEL_RATIO = 0.25

ROI_COMPONENT_DILATE_PIXELS = 9
ROI_MIN_COMPONENT_PIXELS = 6
ROI_MAX_COMPONENTS = 1200
ROI_MERGE_GAP_PIXELS = 4
ROI_PADDING_RATIO = 1.15
ROI_FULL_SCAN_AREA_RATIO = 0.70

PRECISE_KEYWORDS = ["gniazdo", "wypust"]
MIRRORED_VARIANT_PREFIXES = {"06", "07", "09", "10", "11", "12"}

COLOR_HUE_TOLERANCE = 18
COLOR_SAT_TOLERANCE = 80
COLOR_VAL_TOLERANCE = 80
COLOR_HUE_REJECTION_THRESHOLD = 36
SOCKET_07_EXTRA_MIN_COVERAGE = 0.30
SOCKET_07_PROMOTION_SEARCH_RADIUS = 4
SOCKET_PROMOTED_MAX_VERIFICATION_DROP = 0.05
SWITCH_10_EXTRA_MIN_COVERAGE = 0.18
SWITCH_12_EXTRA_MIN_COVERAGE = 0.18
SWITCH_PROMOTION_SEARCH_RADIUS = 8
SWITCH_FAMILY_MIN_CHILD_COVERAGE = 0.62
SWITCH_FAMILY_MIN_CROP_PURITY = 0.90
SWITCH_PROMOTED_MIN_PURITY = 0.58
SWITCH_PROMOTED_MIN_CONTEXT_PURITY = 0.22
SWITCH_PROMOTED_MIN_VERIFICATION = 0.62
SWITCH_10_PROMOTED_MAX_VERIFICATION_DROP = 0.06
SWITCH_12_PROMOTED_MAX_VERIFICATION_DROP = 0.18
SWITCH_PARENT_FALLBACK_SEARCH_RADIUS = 18
PROMOTED_PARENT_MIN_VERIFICATION = 0.68
PROMOTED_PARENT_OVERRIDE_MARGIN = 0.16
PROMOTED_PARENT_MIN_AREA_RATIO = 1.10
NOISY_PARTIAL_CONTEXT_THRESHOLD = 0.36
NOISY_PARTIAL_COVERAGE_THRESHOLD = 0.67
NOISY_PARTIAL_PURITY_THRESHOLD = 0.88

PREFILTER_NMS_MIN_CANDIDATES = 250
PREFILTER_NMS_IOU_THRESHOLD = 0.85
RAW_PREFILTER_MIN_CANDIDATES = 1200
RAW_PREFILTER_IOU_THRESHOLD = 0.98
RAW_PREFILTER_IOM_THRESHOLD = 0.995
RAW_PREFILTER_CENTER_DISTANCE_RATIO = 0.04

CLUSTER_IOU_THRESHOLD = 0.30
CLUSTER_IOM_THRESHOLD = 0.72
CLUSTER_CENTER_DISTANCE_RATIO = 0.28
CROSS_COLOR_CLUSTER_IOU_THRESHOLD = 0.55
CROSS_COLOR_CLUSTER_IOM_THRESHOLD = 0.88
CROSS_COLOR_CENTER_DISTANCE_RATIO = 0.16

DEFAULT_PDF_DPI = 300
PDF_TEXT_MIN_TOKEN_LENGTH = 2
PDF_TEXT_MAX_TOKEN_LENGTH = 6
LEGEND_KEYWORD = "LEGENDA"
LEGEND_WIDTH_PT = 300
LEGEND_HEIGHT_PT = 550

# Data structures

@dataclass
class Detection:
    """Single detected symbol on the plan."""

    symbol_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    source: str = "template"
    rotation: int = 0
    scale: float = 1.0
    mirrored: bool = False
    coverage: float = 0.0
    purity: float = 0.0
    context_purity: float = 0.0
    color_similarity: float = 1.0
    verification_score: float = 0.0


@dataclass
class DetectionResult:
    """Grouped detections for one symbol type."""

    symbol_name: str
    count: int
    color: str = "#10b981"
    detections: list[Detection] = field(default_factory=list)


@dataclass
class TemplateInfo:
    """Loaded template with metadata used during matching."""

    path: str
    name: str
    pixel_count: int
    mask: np.ndarray
    requires_precision: bool
    image_bgr: np.ndarray
    dominant_hsv: tuple[int, int, int] | None
    text_tokens: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TemplateVariant:
    """One concrete template variant after scale and rotation."""

    template_id: int
    scale: float
    rotation: int
    mirrored: bool
    transformed_mask: np.ndarray
    pixel_count: int
    width: int
    height: int


@dataclass
class CandidateHit:
    """Candidate detection produced by template matching or PDF text lookup."""

    template_id: int
    scale: float
    rotation: int
    mirrored: bool
    transformed_mask: np.ndarray | None
    pixel_count: int
    bbox: tuple[int, int, int, int]
    match_score: float
    dominant_hsv: tuple[int, int, int] | None
    source: str = "template"
    coverage: float = 0.0
    purity: float = 0.0
    context_purity: float = 1.0
    color_similarity: float = 1.0
    verification_score: float = 0.0
    promoted_from_template_id: int | None = None


@dataclass
class TargetedPromotionRule:
    """Pair-specific promotion from a smaller template to a larger one."""

    child_template_id: int
    parent_template_id: int
    scale: float
    rotation: int
    mirrored: bool
    offset_x: int
    offset_y: int
    extension_mask: np.ndarray
    extension_pixels: int
    min_extra_coverage: float
    allow_rotation_mismatch: bool = False


# HSV helpers

def _hsv_mask(image_bgr: np.ndarray, dilate: bool = False) -> np.ndarray:
    """Create a binary mask of colored pixels."""

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
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

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
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


# Template preparation

def _normalize_template_name(name: str) -> str:
    """Strip numeric prefixes generated during legend extraction."""

    return re.sub(r"^\d+_+", "", name)


def _template_numeric_prefix(name: str) -> str | None:
    """Return the numeric prefix from a template filename stem, if present."""

    match = re.match(r"^(\d+)_", name)
    return match.group(1) if match else None


def _derive_text_tokens(name: str) -> list[str]:
    """Extract short text tokens that can be searched directly in the PDF."""

    normalized = _normalize_template_name(name)
    candidate = normalized.strip().upper()

    # Only enable PDF-text lookup for templates that are themselves short text
    # labels, e.g. "MSW". Extracted legend names are long descriptive phrases,
    # and splitting them into fragments like "TM" or "INT" causes false routing.
    if not re.fullmatch(r"[A-Z0-9]+", candidate):
        return []
    if not (PDF_TEXT_MIN_TOKEN_LENGTH <= len(candidate) <= PDF_TEXT_MAX_TOKEN_LENGTH):
        return []
    if not re.search(r"[A-Z]", candidate):
        return []

    return [candidate]


def _prepare_variants(template_id: int, template: TemplateInfo) -> list[TemplateVariant]:
    """Precompute all scale/rotation variants for one template."""

    variants: list[TemplateVariant] = []
    base_mask = template.mask
    template_prefix = _template_numeric_prefix(Path(template.path).name)
    allow_mirror = template_prefix in MIRRORED_VARIANT_PREFIXES

    for scale in SCALES:
        if scale != 1.0:
            new_w = max(1, int(round(base_mask.shape[1] * scale)))
            new_h = max(1, int(round(base_mask.shape[0] * scale)))
            scaled_mask = cv2.resize(base_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_mask = base_mask

        mask_sources = [(False, scaled_mask)]
        if allow_mirror:
            mask_sources.append((True, cv2.flip(scaled_mask, 1)))

        for mirrored, source_mask in mask_sources:
            for rotation, rotate_code in ROTATIONS:
                rot_mask = cv2.rotate(source_mask, rotate_code) if rotate_code is not None else source_mask
                pixel_count = int(cv2.countNonZero(rot_mask))
                if pixel_count == 0:
                    continue

                variants.append(
                    TemplateVariant(
                        template_id=template_id,
                        scale=scale,
                        rotation=rotation,
                        mirrored=mirrored,
                        transformed_mask=rot_mask,
                        pixel_count=pixel_count,
                        width=int(rot_mask.shape[1]),
                        height=int(rot_mask.shape[0]),
                    )
                )

    return variants


def _build_socket_07_promotions(
    templates: list[TemplateInfo],
    variants_by_template: dict[int, list[TemplateVariant]],
) -> dict[tuple[int, float, int, bool], list[TargetedPromotionRule]]:
    """Build targeted family-promotion rules from contained symbols to fuller parents."""

    template_ids_by_prefix = {
        prefix: template_id
        for template_id, template in enumerate(templates)
        for prefix in [_template_numeric_prefix(Path(template.path).name)]
        if prefix is not None
    }

    promotions: dict[tuple[int, float, int, bool], list[TargetedPromotionRule]] = {}
    family_specs = [
        ("06", "07", 0.95, 0.82, SOCKET_07_EXTRA_MIN_COVERAGE, False),
        ("09", "07", 0.82, 0.90, SOCKET_07_EXTRA_MIN_COVERAGE, False),
        ("11", "10", SWITCH_FAMILY_MIN_CHILD_COVERAGE, SWITCH_FAMILY_MIN_CROP_PURITY, SWITCH_10_EXTRA_MIN_COVERAGE, True),
        ("11", "12", SWITCH_FAMILY_MIN_CHILD_COVERAGE, SWITCH_FAMILY_MIN_CROP_PURITY, SWITCH_12_EXTRA_MIN_COVERAGE, True),
    ]

    for (
        child_prefix,
        parent_prefix,
        min_child_coverage,
        min_crop_purity,
        min_extra_coverage,
        allow_rotation_mismatch,
    ) in family_specs:
        child_id = template_ids_by_prefix.get(child_prefix)
        parent_id = template_ids_by_prefix.get(parent_prefix)
        if child_id is None or parent_id is None:
            continue
        parent_variants = list(variants_by_template.get(parent_id, []))

        for child_variant in variants_by_template.get(child_id, []):
            child_key = (child_id, child_variant.scale, child_variant.rotation, child_variant.mirrored)
            for parent_mirrored in (False, True):
                for parent_variant in parent_variants:
                    if not allow_rotation_mismatch and parent_variant.rotation != child_variant.rotation:
                        continue
                    if parent_variant.mirrored != parent_mirrored:
                        continue
                    if child_variant.width > parent_variant.width or child_variant.height > parent_variant.height:
                        continue

                    result = cv2.matchTemplate(
                        parent_variant.transformed_mask,
                        child_variant.transformed_mask,
                        cv2.TM_CCORR_NORMED,
                    )
                    _, _, _, max_loc = cv2.minMaxLoc(result)
                    offset_x, offset_y = int(max_loc[0]), int(max_loc[1])
                    crop = parent_variant.transformed_mask[
                        offset_y : offset_y + child_variant.height,
                        offset_x : offset_x + child_variant.width,
                    ]
                    if crop.shape != child_variant.transformed_mask.shape:
                        continue

                    intersection = int(cv2.countNonZero(cv2.bitwise_and(crop, child_variant.transformed_mask)))
                    child_coverage = intersection / max(1, child_variant.pixel_count)
                    crop_pixels = int(cv2.countNonZero(crop))
                    crop_purity = intersection / max(1, crop_pixels)
                    if child_coverage < min_child_coverage or crop_purity < min_crop_purity:
                        continue

                    child_canvas = np.zeros_like(parent_variant.transformed_mask)
                    child_canvas[
                        offset_y : offset_y + child_variant.height,
                        offset_x : offset_x + child_variant.width,
                    ] = child_variant.transformed_mask
                    extension_mask = cv2.bitwise_and(
                        parent_variant.transformed_mask,
                        cv2.bitwise_not(child_canvas),
                    )
                    extension_pixels = int(cv2.countNonZero(extension_mask))
                    if extension_pixels <= 0:
                        continue

                    promotions.setdefault(child_key, []).append(
                        TargetedPromotionRule(
                            child_template_id=child_id,
                            parent_template_id=parent_id,
                            scale=parent_variant.scale,
                            rotation=parent_variant.rotation,
                            mirrored=parent_mirrored,
                            offset_x=offset_x,
                            offset_y=offset_y,
                            extension_mask=extension_mask,
                            extension_pixels=extension_pixels,
                            min_extra_coverage=min_extra_coverage,
                            allow_rotation_mismatch=allow_rotation_mismatch,
                        )
                    )

    return promotions


def load_templates(folder: str) -> list[TemplateInfo]:
    """Load template PNG files and their metadata."""

    paths = glob.glob(os.path.join(folder, "*.png"))
    templates: list[TemplateInfo] = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue

        name = Path(path).stem
        name_lower = name.lower()
        requires_precision = any(keyword in name_lower for keyword in PRECISE_KEYWORDS)

        mask = _hsv_mask(img, dilate=not requires_precision)
        pixel_count = int(cv2.countNonZero(mask))

        if pixel_count <= MIN_TEMPLATE_PIXELS:
            continue

        templates.append(
            TemplateInfo(
                path=path,
                name=name,
                pixel_count=pixel_count,
                mask=mask,
                requires_precision=requires_precision,
                image_bgr=img,
                dominant_hsv=_dominant_hsv_color(img),
                text_tokens=_derive_text_tokens(name),
            )
        )

    templates.sort(key=lambda item: item.pixel_count, reverse=True)
    return templates


# Matching helpers

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


def _maybe_promote_socket_06_to_07(
    hit: CandidateHit,
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    plan_masks: dict[int, np.ndarray],
    dilated_plan_masks: dict[int, np.ndarray],
    variants_lookup: dict[tuple[int, float, int, bool], TemplateVariant],
    promotions: dict[tuple[int, float, int, bool], list[TargetedPromotionRule]],
) -> CandidateHit:
    """Apply cheap family promotions when parent-only extension pixels are present."""

    rules = promotions.get((hit.template_id, hit.scale, hit.rotation, hit.mirrored))
    if not rules:
        return hit

    best_promoted: CandidateHit | None = None
    best_key: tuple[float, float, float] | None = None

    for rule in rules:
        parent_variant = variants_lookup.get(
            (rule.parent_template_id, rule.scale, rule.rotation, rule.mirrored)
        )
        parent_plan_mask = plan_masks.get(rule.parent_template_id)
        if parent_variant is None or parent_plan_mask is None:
            continue
        parent_prefix = _template_numeric_prefix(Path(templates[rule.parent_template_id].path).name)
        if parent_prefix == "07":
            promotion_plan_mask = _cached_dilated_mask(
                rule.parent_template_id,
                parent_plan_mask,
                dilated_plan_masks,
            )
        else:
            promotion_plan_mask = parent_plan_mask
        extension_plan_mask = (
            _cached_dilated_mask(
                rule.parent_template_id,
                parent_plan_mask,
                dilated_plan_masks,
            )
            if parent_prefix in {"10", "12"}
            else promotion_plan_mask
        )
        search_radius = (
            SWITCH_PROMOTION_SEARCH_RADIUS
            if parent_prefix in {"10", "12"}
            else SOCKET_07_PROMOTION_SEARCH_RADIUS
        )

        base_parent_x = hit.bbox[0] - rule.offset_x
        base_parent_y = hit.bbox[1] - rule.offset_y

        for delta_y in range(-search_radius, search_radius + 1):
            for delta_x in range(-search_radius, search_radius + 1):
                parent_bbox = (
                    base_parent_x + delta_x,
                    base_parent_y + delta_y,
                    parent_variant.width,
                    parent_variant.height,
                )
                parent_roi = _roi_mask(promotion_plan_mask, parent_bbox)
                extension_roi = _roi_mask(extension_plan_mask, parent_bbox)
                if parent_roi is None or parent_roi.shape != parent_variant.transformed_mask.shape:
                    continue
                if extension_roi is None or extension_roi.shape != parent_variant.transformed_mask.shape:
                    continue

                extra_overlap = int(cv2.countNonZero(cv2.bitwise_and(extension_roi, rule.extension_mask)))
                extra_coverage = extra_overlap / max(1, rule.extension_pixels)
                if extra_coverage < rule.min_extra_coverage:
                    continue

                try:
                    local_match = float(
                        cv2.matchTemplate(
                            parent_roi,
                            parent_variant.transformed_mask,
                            cv2.TM_CCORR_NORMED,
                        )[0][0]
                    )
                except cv2.error:
                    continue

                promoted_hit = CandidateHit(
                    template_id=rule.parent_template_id,
                    scale=rule.scale,
                    rotation=rule.rotation,
                    mirrored=rule.mirrored,
                    transformed_mask=parent_variant.transformed_mask,
                    pixel_count=parent_variant.pixel_count,
                    bbox=parent_bbox,
                    match_score=local_match,
                    dominant_hsv=templates[rule.parent_template_id].dominant_hsv,
                    source=f"template_promoted_{rule.child_template_id}_to_{rule.parent_template_id}",
                    promoted_from_template_id=hit.template_id,
                )
                if not _validate_template_hit(promoted_hit, promotion_plan_mask, plan_image):
                    continue
                if parent_prefix == "07":
                    if promoted_hit.verification_score < (hit.verification_score - SOCKET_PROMOTED_MAX_VERIFICATION_DROP):
                        continue
                if parent_prefix in {"10", "12"}:
                    max_drop = (
                        SWITCH_12_PROMOTED_MAX_VERIFICATION_DROP
                        if parent_prefix == "12"
                        else SWITCH_10_PROMOTED_MAX_VERIFICATION_DROP
                    )
                    if (
                        promoted_hit.purity < SWITCH_PROMOTED_MIN_PURITY
                        or promoted_hit.context_purity < SWITCH_PROMOTED_MIN_CONTEXT_PURITY
                        or promoted_hit.verification_score < SWITCH_PROMOTED_MIN_VERIFICATION
                        or promoted_hit.verification_score
                        < (hit.verification_score - max_drop)
                    ):
                        continue
                    if (
                        parent_prefix == "10"
                        and rule.allow_rotation_mismatch
                        and promoted_hit.verification_score + 0.02 < hit.verification_score
                    ):
                        continue

                candidate_key = (
                    float(extra_coverage),
                    float(promoted_hit.verification_score),
                    float(promoted_hit.match_score),
                )
                if best_key is None or candidate_key > best_key:
                    best_promoted = promoted_hit
                    best_key = candidate_key

    return best_promoted or hit


def _maybe_promote_switch_parent_search(
    hit: CandidateHit,
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    plan_masks: dict[int, np.ndarray],
    dilated_plan_masks: dict[int, np.ndarray],
    variants_lookup: dict[tuple[int, float, int, bool], TemplateVariant],
    promotions: dict[tuple[int, float, int, bool], list[TargetedPromotionRule]],
    stats: dict[str, int] | None = None,
) -> CandidateHit:
    """Run the expensive 11 -> 10/12 parent search only on prefiltered hits."""

    if hit.transformed_mask is None:
        return hit

    child_prefix = _template_numeric_prefix(Path(templates[hit.template_id].path).name)
    if child_prefix != "11":
        return hit

    rules = promotions.get((hit.template_id, hit.scale, hit.rotation, hit.mirrored), [])
    if not rules:
        return hit

    child_center = _box_center(hit.bbox)
    child_area = max(1, hit.bbox[2] * hit.bbox[3])
    fallback_best: CandidateHit | None = None
    fallback_key: tuple[float, float, float] | None = None
    if stats is not None:
        stats["parent_search_input_hits"] = stats.get("parent_search_input_hits", 0) + 1

    for rule in rules:
        parent_prefix = _template_numeric_prefix(Path(templates[rule.parent_template_id].path).name)
        if parent_prefix not in {"10", "12"}:
            continue

        parent_variant = variants_lookup.get(
            (rule.parent_template_id, rule.scale, rule.rotation, rule.mirrored)
        )
        parent_plan_mask = plan_masks.get(rule.parent_template_id)
        if parent_variant is None or parent_plan_mask is None:
            continue

        parent_area = max(1, parent_variant.width * parent_variant.height)
        if parent_area < child_area * PROMOTED_PARENT_MIN_AREA_RATIO:
            continue

        extension_plan_mask = _cached_dilated_mask(
            rule.parent_template_id,
            parent_plan_mask,
            dilated_plan_masks,
        )
        base_x = int(round(child_center[0] - parent_variant.width / 2.0))
        base_y = int(round(child_center[1] - parent_variant.height / 2.0))

        for delta_y in range(-SWITCH_PARENT_FALLBACK_SEARCH_RADIUS, SWITCH_PARENT_FALLBACK_SEARCH_RADIUS + 1):
            for delta_x in range(-SWITCH_PARENT_FALLBACK_SEARCH_RADIUS, SWITCH_PARENT_FALLBACK_SEARCH_RADIUS + 1):
                parent_bbox = (
                    base_x + delta_x,
                    base_y + delta_y,
                    parent_variant.width,
                    parent_variant.height,
                )
                extension_roi = _roi_mask(extension_plan_mask, parent_bbox)
                if extension_roi is None or extension_roi.shape != parent_variant.transformed_mask.shape:
                    continue

                extra_overlap = int(cv2.countNonZero(cv2.bitwise_and(extension_roi, rule.extension_mask)))
                extra_coverage = extra_overlap / max(1, rule.extension_pixels)
                if extra_coverage < rule.min_extra_coverage:
                    continue

                parent_roi = _roi_mask(parent_plan_mask, parent_bbox)
                if parent_roi is None or parent_roi.shape != parent_variant.transformed_mask.shape:
                    continue

                if not _center_inside_box(child_center, parent_bbox, margin_ratio=0.08):
                    continue

                inter_area, _, iom, _ = _bbox_metrics(hit.bbox, parent_bbox)
                if inter_area <= 0 or iom < 0.40:
                    continue

                if stats is not None:
                    stats["parent_search_candidates"] = stats.get("parent_search_candidates", 0) + 1
                try:
                    local_match = float(
                        cv2.matchTemplate(
                            parent_roi,
                            parent_variant.transformed_mask,
                            cv2.TM_CCORR_NORMED,
                        )[0][0]
                    )
                except cv2.error:
                    continue

                promoted_hit = CandidateHit(
                    template_id=rule.parent_template_id,
                    scale=parent_variant.scale,
                    rotation=parent_variant.rotation,
                    mirrored=parent_variant.mirrored,
                    transformed_mask=parent_variant.transformed_mask,
                    pixel_count=parent_variant.pixel_count,
                    bbox=parent_bbox,
                    match_score=local_match,
                    dominant_hsv=templates[rule.parent_template_id].dominant_hsv,
                    source=f"template_parent_search_{hit.template_id}_to_{rule.parent_template_id}",
                    promoted_from_template_id=hit.template_id,
                )
                if not _validate_template_hit(promoted_hit, parent_plan_mask, plan_image):
                    continue

                max_drop = (
                    SWITCH_12_PROMOTED_MAX_VERIFICATION_DROP
                    if parent_prefix == "12"
                    else SWITCH_10_PROMOTED_MAX_VERIFICATION_DROP
                )
                if (
                    promoted_hit.purity < SWITCH_PROMOTED_MIN_PURITY
                    or promoted_hit.context_purity < SWITCH_PROMOTED_MIN_CONTEXT_PURITY
                    or promoted_hit.verification_score < SWITCH_PROMOTED_MIN_VERIFICATION
                    or promoted_hit.verification_score < (hit.verification_score - max_drop)
                ):
                    continue

                if (
                    parent_prefix == "10"
                    and promoted_hit.verification_score + 0.02 < hit.verification_score
                ):
                    continue

                candidate_key = (
                    float(extra_coverage),
                    float(promoted_hit.verification_score),
                    float(promoted_hit.match_score),
                )
                if fallback_key is None or candidate_key > fallback_key:
                    fallback_best = promoted_hit
                    fallback_key = candidate_key

    return fallback_best or hit


def _bbox_metrics(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> tuple[int, float, float, float]:
    """Return intersection area, IoU, IoM and normalized center distance."""

    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    union_area = area_a + area_b - inter_area

    iou = inter_area / union_area if union_area > 0 else 0.0
    iom = inter_area / min(area_a, area_b)

    center_a = (ax + aw / 2.0, ay + ah / 2.0)
    center_b = (bx + bw / 2.0, by + bh / 2.0)
    center_distance = float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    ref_distance = max(1.0, min(np.hypot(aw, ah), np.hypot(bw, bh)))
    normalized_center_distance = center_distance / ref_distance

    return inter_area, iou, iom, normalized_center_distance


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return the geometric center of a bbox."""

    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _center_inside_box(
    center: tuple[float, float],
    box: tuple[int, int, int, int],
    margin_ratio: float = 0.05,
) -> bool:
    """Check whether a center point lies inside a bbox with a small safety margin."""

    x, y = center
    bx, by, bw, bh = box
    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio
    return (bx - pad_x) <= x <= (bx + bw + pad_x) and (by - pad_y) <= y <= (by + bh + pad_y)


def _should_cluster(hit_a: CandidateHit, hit_b: CandidateHit) -> bool:
    """Decide whether two candidates describe the same physical object."""

    inter_area, iou, iom, center_distance = _bbox_metrics(hit_a.bbox, hit_b.bbox)
    if inter_area <= 0:
        return False

    center_a = _box_center(hit_a.bbox)
    center_b = _box_center(hit_b.bbox)
    centers_nested = _center_inside_box(center_a, hit_b.bbox) or _center_inside_box(center_b, hit_a.bbox)

    if (
        hit_a.dominant_hsv is not None
        and hit_b.dominant_hsv is not None
        and _hue_distance(hit_a.dominant_hsv[0], hit_b.dominant_hsv[0]) > (COLOR_HUE_TOLERANCE + 6)
    ):
        return (
            centers_nested
            and iou >= CROSS_COLOR_CLUSTER_IOU_THRESHOLD
            and iom >= CROSS_COLOR_CLUSTER_IOM_THRESHOLD
            and center_distance <= CROSS_COLOR_CENTER_DISTANCE_RATIO
        )

    if iou >= CLUSTER_IOU_THRESHOLD and center_distance <= CLUSTER_CENTER_DISTANCE_RATIO:
        return True

    if iom >= CLUSTER_IOM_THRESHOLD and centers_nested and center_distance <= (CLUSTER_CENTER_DISTANCE_RATIO * 1.15):
        return True

    return False


def _prefilter_candidates(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Use a conservative NMS only when the candidate set becomes very large."""

    if len(candidates) < PREFILTER_NMS_MIN_CANDIDATES:
        return candidates

    boxes = [list(hit.bbox) for hit in candidates]
    scores = [float(hit.verification_score or hit.match_score) for hit in candidates]

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.0,
        nms_threshold=PREFILTER_NMS_IOU_THRESHOLD,
    )
    if len(indices) == 0:
        return candidates

    keep = set(indices.flatten().tolist())
    return [candidate for idx, candidate in enumerate(candidates) if idx in keep]


def _raw_candidates_overlap_strongly(left: CandidateHit, right: CandidateHit) -> bool:
    """Detect duplicate raw peaks for the same template before expensive validation."""

    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False

    if iou >= RAW_PREFILTER_IOU_THRESHOLD:
        return True

    return (
        iom >= RAW_PREFILTER_IOM_THRESHOLD
        and center_distance <= RAW_PREFILTER_CENTER_DISTANCE_RATIO
    )


def _prefilter_raw_template_hits(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Drop near-identical raw candidates only inside the same template family member."""

    if len(candidates) < RAW_PREFILTER_MIN_CANDIDATES:
        return candidates

    grouped: dict[int, list[CandidateHit]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.template_id, []).append(candidate)

    filtered: list[CandidateHit] = []
    for template_hits in grouped.values():
        kept: list[CandidateHit] = []
        for candidate in sorted(template_hits, key=lambda hit: hit.match_score, reverse=True):
            if any(_raw_candidates_overlap_strongly(candidate, existing) for existing in kept):
                continue
            kept.append(candidate)
        filtered.extend(kept)

    filtered.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.match_score))
    return filtered


def _candidate_rank_key(hit: CandidateHit) -> tuple[float, float, float, int]:
    """Return the default winner ranking inside a cluster."""

    return (
        float(hit.verification_score),
        float(hit.color_similarity),
        float(hit.match_score),
        1 if hit.source == "pdf_text" else 0,
    )


def _select_cluster_winner(
    group_hits: list[CandidateHit],
    parent_ids_by_child: dict[int, set[int]],
) -> CandidateHit:
    """Pick one winner per cluster, preferring promoted fuller symbols over simpler cores."""

    base_winner = max(group_hits, key=_candidate_rank_key)
    override_candidates: list[CandidateHit] = []

    for hit in group_hits:
        child_id = hit.promoted_from_template_id
        if child_id is None:
            continue
        if hit.template_id not in parent_ids_by_child.get(child_id, set()):
            continue
        if hit.verification_score < PROMOTED_PARENT_MIN_VERIFICATION:
            continue

        child_hits = [candidate for candidate in group_hits if candidate.template_id == child_id]
        if not child_hits:
            continue

        best_child = max(child_hits, key=_candidate_rank_key)
        child_area = max(1, best_child.bbox[2] * best_child.bbox[3])
        parent_area = max(1, hit.bbox[2] * hit.bbox[3])
        if parent_area < child_area * PROMOTED_PARENT_MIN_AREA_RATIO:
            continue

        if hit.verification_score + PROMOTED_PARENT_OVERRIDE_MARGIN < best_child.verification_score:
            continue

        override_candidates.append(hit)

    if override_candidates:
        return max(override_candidates, key=_candidate_rank_key)

    return base_winner


def _cluster_candidates(
    candidates: list[CandidateHit],
    parent_ids_by_child: dict[int, set[int]] | None = None,
) -> list[CandidateHit]:
    """Cluster class-agnostic overlaps and keep one winner per physical place."""

    if not candidates:
        return []

    parent_ids_by_child = parent_ids_by_child or {}

    parent = list(range(len(candidates)))

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

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if _should_cluster(candidates[left], candidates[right]):
                union(left, right)

    groups: dict[int, list[CandidateHit]] = {}
    for idx, candidate in enumerate(candidates):
        groups.setdefault(find(idx), []).append(candidate)

    winners: list[CandidateHit] = []
    for group_hits in groups.values():
        winners.append(_select_cluster_winner(group_hits, parent_ids_by_child))

    winners.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.verification_score))
    return winners


# PDF text helpers

def _apply_hidden_layers(doc: fitz.Document, hidden_layers: list[str] | None) -> None:
    """Disable selected PDF layers before text lookup."""

    if not hidden_layers:
        return

    try:
        ui_configs = doc.layer_ui_configs()
        if not ui_configs:
            return
        for config in ui_configs:
            if config.get("text") in hidden_layers:
                doc.set_layer_ui_config(config["number"], action=2)
    except Exception as exc:  # pragma: no cover - depends on PDF features
        print(f"Warning: could not apply hidden layers for text search: {exc}")


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


def _overlaps_excluded_zones(
    bbox: tuple[int, int, int, int],
    exclude_rects: list[tuple[int, int, int, int]],
) -> bool:
    """Return True when bbox intersects an excluded rectangle."""

    for exclude_box in exclude_rects:
        inter_area, _, iom, _ = _bbox_metrics(bbox, exclude_box)
        if inter_area > 0 and iom > 0.10:
            return True
    return False


def _quad_rotation(quad: fitz.Quad) -> int:
    """Estimate text rotation from a quad when available."""

    try:
        dx = quad.ur.x - quad.ul.x
        dy = quad.ur.y - quad.ul.y
        angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        return int((round(angle / 90.0) * 90) % 360)
    except Exception:
        return 0


def _collect_pdf_text_hits(
    pdf_path: str,
    templates: list[TemplateInfo],
    plan_image_shape: tuple[int, int, int] | tuple[int, int],
    dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
) -> dict[int, list[CandidateHit]]:
    """Find short alphanumeric symbols directly in the PDF text layer."""

    if not pdf_path or not os.path.exists(pdf_path):
        return {}

    exclude_rects = exclude_rects or []
    hits_by_template: dict[int, list[CandidateHit]] = {}
    token_rect_cache: dict[str, list[fitz.Quad | fitz.Rect]] = {}

    doc = fitz.open(pdf_path)
    try:
        _apply_hidden_layers(doc, hidden_layers)
        page = doc.load_page(0)
        scale = dpi / 72.0
        page_words = page.get_text("words")

        def get_token_hits(token: str) -> list[fitz.Quad | fitz.Rect]:
            cache_key = token.upper()
            if cache_key in token_rect_cache:
                return token_rect_cache[cache_key]

            quads = page.search_for(token, quads=True)
            if quads:
                token_rect_cache[cache_key] = list(quads)
                return token_rect_cache[cache_key]

            matches: list[fitz.Rect] = []
            for word in page_words:
                if len(word) < 5:
                    continue
                if str(word[4]).upper() == cache_key:
                    matches.append(fitz.Rect(word[0], word[1], word[2], word[3]))

            token_rect_cache[cache_key] = matches
            return matches

        for template_id, template in enumerate(templates):
            if not template.text_tokens:
                continue

            seen_boxes: set[tuple[int, int, int, int]] = set()
            template_hits: list[CandidateHit] = []

            for token in template.text_tokens:
                for hit in get_token_hits(token):
                    rect = hit.rect if hasattr(hit, "rect") else hit
                    bbox = _clamp_bbox(
                        (
                            int(round(rect.x0 * scale)),
                            int(round(rect.y0 * scale)),
                            int(round(rect.width * scale)),
                            int(round(rect.height * scale)),
                        ),
                        plan_image_shape,
                    )
                    if bbox is None or bbox in seen_boxes:
                        continue
                    if _overlaps_excluded_zones(bbox, exclude_rects):
                        continue

                    seen_boxes.add(bbox)
                    template_hits.append(
                        CandidateHit(
                            template_id=template_id,
                            scale=1.0,
                            rotation=_quad_rotation(hit) if hasattr(hit, "ul") else 0,
                            mirrored=False,
                            transformed_mask=None,
                            pixel_count=max(1, bbox[2] * bbox[3]),
                            bbox=bbox,
                            match_score=1.0,
                            dominant_hsv=template.dominant_hsv,
                            source="pdf_text",
                            coverage=1.0,
                            purity=1.0,
                            color_similarity=1.0,
                            verification_score=1.0,
                        )
                    )

            if template_hits:
                hits_by_template[template_id] = template_hits
    finally:
        doc.close()

    return hits_by_template


def _estimate_legend_exclude_rect(
    pdf_path: str,
    image_shape: tuple[int, int, int] | tuple[int, int],
    dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
) -> tuple[int, int, int, int] | None:
    """Estimate the legend bbox on the rendered plan so it can be excluded."""

    if not pdf_path or not os.path.exists(pdf_path):
        return None

    doc = fitz.open(pdf_path)
    try:
        _apply_hidden_layers(doc, hidden_layers)
        page = doc.load_page(0)
        found = page.search_for(LEGEND_KEYWORD)
        if not found:
            return None

        anchor = found[0]
        scale = dpi / 72.0
        bbox = _clamp_bbox(
            (
                int(round((anchor.x0 - 20) * scale)),
                int(round(anchor.y1 * scale)),
                int(round(LEGEND_WIDTH_PT * scale)),
                int(round(LEGEND_HEIGHT_PT * scale)),
            ),
            image_shape,
        )
        return bbox
    except Exception as exc:  # pragma: no cover - depends on input PDF
        print(f"Warning: could not estimate legend area: {exc}")
        return None
    finally:
        doc.close()


# Detection

def detect_symbols(
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    subtract_legend: bool = True,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
    pdf_path: str | None = None,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
    debug_profile: dict | None = None,
) -> list[DetectionResult]:
    """
    Detect symbols on a rendered plan using template matching plus PDF-text fallback.
    """

    exclude_rects = list(exclude_rects or [])

    if not templates:
        return []

    timings: dict[str, float] = {}

    legend_rect = _estimate_legend_exclude_rect(
        pdf_path=pdf_path or "",
        image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
    )
    if legend_rect is not None:
        exclude_rects.append(legend_rect)

    color_masks_cache: dict[str, np.ndarray] = {}

    def _get_plan_mask(template: TemplateInfo) -> np.ndarray:
        if template.dominant_hsv is not None:
            cache_key = f"{template.dominant_hsv}_{template.requires_precision}"
            if cache_key not in color_masks_cache:
                mask = _color_mask_for_template(
                    plan_image,
                    template.dominant_hsv,
                    dilate=not template.requires_precision,
                )
                for ex, ey, ew, eh in exclude_rects:
                    cv2.rectangle(mask, (ex, ey), (ex + ew, ey + eh), 0, -1)
                color_masks_cache[cache_key] = mask
            return color_masks_cache[cache_key]

        fallback = _hsv_mask(plan_image, dilate=False)
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(fallback, (ex, ey), (ex + ew, ey + eh), 0, -1)
        return fallback

    phase_start = time.perf_counter()
    pdf_hits_by_template = _collect_pdf_text_hits(
        pdf_path=pdf_path or "",
        templates=templates,
        plan_image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
        exclude_rects=exclude_rects,
    )
    pdf_candidates = [hit for hits in pdf_hits_by_template.values() for hit in hits]
    timings["pdf_text"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    variants_by_template = {
        template_id: _prepare_variants(template_id, template)
        for template_id, template in enumerate(templates)
    }
    variants_lookup = {
        (variant.template_id, variant.scale, variant.rotation, variant.mirrored): variant
        for variants in variants_by_template.values()
        for variant in variants
    }
    socket_07_promotions = _build_socket_07_promotions(templates, variants_by_template)
    parent_ids_by_child: dict[int, set[int]] = {}
    for rules in socket_07_promotions.values():
        for rule in rules:
            parent_ids_by_child.setdefault(rule.child_template_id, set()).add(rule.parent_template_id)
    plan_masks_by_template = {
        template_id: _get_plan_mask(template)
        for template_id, template in enumerate(templates)
    }
    plan_mask_foregrounds = {
        template_id: int(cv2.countNonZero(plan_mask))
        for template_id, plan_mask in plan_masks_by_template.items()
    }
    max_variant_size_by_template = {
        template_id: (
            max((variant.width for variant in variants), default=templates[template_id].mask.shape[1]),
            max((variant.height for variant in variants), default=templates[template_id].mask.shape[0]),
        )
        for template_id, variants in variants_by_template.items()
    }
    search_rois_by_template: dict[int, list[tuple[int, int, int, int]]] = {}
    search_roi_stats_by_template: dict[int, tuple[bool, int, int]] = {}
    for template_id, plan_mask in plan_masks_by_template.items():
        max_width, max_height = max_variant_size_by_template[template_id]
        rois, uses_full_scan, roi_area, foreground_pixels = _build_search_rois(
            plan_mask,
            plan_image.shape,
            max_width,
            max_height,
        )
        search_rois_by_template[template_id] = rois
        search_roi_stats_by_template[template_id] = (uses_full_scan, roi_area, foreground_pixels)
    dilated_plan_masks_by_template: dict[int, np.ndarray] = {}
    timings["prepare"] = time.perf_counter() - phase_start

    diagnostics = {
        "raw_peaks": 0,
        "raw_prefilter_hits": 0,
        "raw_prefilter_removed": 0,
        "prepared_variants": sum(len(variants) for variants in variants_by_template.values()),
        "skipped_empty_color_masks": 0,
        "validated_template_hits": 0,
        "promoted_targeted_hits": 0,
        "parent_search_input_hits": 0,
        "parent_search_candidates": 0,
        "promoted_parent_search_hits": 0,
        "pdf_text_hits": len(pdf_candidates),
        "prefilter_hits": 0,
        "pre_parent_clusters": 0,
        "final_hits": 0,
        "search_rois": sum(len(rois) for rois in search_rois_by_template.values()),
        "full_scan_templates": sum(1 for uses_full, _, _ in search_roi_stats_by_template.values() if uses_full),
        "roi_area_pixels": sum(area for _, area, _ in search_roi_stats_by_template.values()),
        "roi_foreground_pixels": sum(pixels for _, _, pixels in search_roi_stats_by_template.values()),
    }

    def _scan_template(template_id: int) -> list[CandidateHit]:
        template = templates[template_id]
        threshold = THRESHOLD_PRECISE if template.requires_precision else THRESHOLD_DILATED
        plan_mask = plan_masks_by_template[template_id]
        search_rois = search_rois_by_template.get(template_id, [])
        if plan_mask_foregrounds.get(template_id, 0) < MIN_TEMPLATE_PIXELS or not search_rois:
            return []

        template_hits: list[CandidateHit] = []
        for variant in variants_by_template.get(template_id, []):
            if variant.height > plan_mask.shape[0] or variant.width > plan_mask.shape[1]:
                continue

            variant_peaks: list[tuple[int, int, float]] = []
            too_many_peaks = False
            for roi_x, roi_y, roi_w, roi_h in search_rois:
                if variant.height > roi_h or variant.width > roi_w:
                    continue

                roi_plan_mask = plan_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                match_result = cv2.matchTemplate(
                    roi_plan_mask,
                    variant.transformed_mask,
                    cv2.TM_CCOEFF_NORMED,
                )
                peaks = _find_local_maxima(
                    match_result,
                    threshold=threshold,
                    template_width=variant.width,
                    template_height=variant.height,
                )
                if peaks:
                    variant_peaks.extend((roi_x + px, roi_y + py, score) for px, py, score in peaks)
                if len(variant_peaks) > MAX_PEAKS_PER_VARIANT:
                    too_many_peaks = True
                    break

            if too_many_peaks:
                continue

            for px, py, score in variant_peaks:
                template_hits.append(
                    CandidateHit(
                        template_id=template_id,
                        scale=variant.scale,
                        rotation=variant.rotation,
                        mirrored=variant.mirrored,
                        transformed_mask=variant.transformed_mask,
                        pixel_count=variant.pixel_count,
                        bbox=(px, py, variant.width, variant.height),
                        match_score=score,
                        dominant_hsv=template.dominant_hsv,
                        source="template",
                    )
                )

        return template_hits

    template_ids_to_scan = [
        template_id
        for template_id in variants_by_template
        if plan_mask_foregrounds.get(template_id, 0) >= MIN_TEMPLATE_PIXELS
    ]
    diagnostics["skipped_empty_color_masks"] = len(variants_by_template) - len(template_ids_to_scan)
    raw_template_hits: list[CandidateHit] = []
    phase_start = time.perf_counter()
    scan_workers = max(1, min(len(template_ids_to_scan), DETECTOR_SCAN_MAX_WORKERS))
    if template_ids_to_scan:
        with ThreadPoolExecutor(max_workers=scan_workers) as pool:
            for hits in pool.map(_scan_template, template_ids_to_scan):
                raw_template_hits.extend(hits)
    else:
        scan_workers = 0
    timings["scan"] = time.perf_counter() - phase_start

    diagnostics["raw_peaks"] = len(raw_template_hits)
    phase_start = time.perf_counter()
    raw_before_prefilter = len(raw_template_hits)
    raw_template_hits = _prefilter_raw_template_hits(raw_template_hits)
    diagnostics["raw_prefilter_hits"] = len(raw_template_hits)
    diagnostics["raw_prefilter_removed"] = raw_before_prefilter - len(raw_template_hits)
    timings["raw_prefilter"] = time.perf_counter() - phase_start

    validated_candidates: list[CandidateHit] = list(pdf_candidates)
    phase_start = time.perf_counter()
    postprocess_workers = max(1, DETECTOR_POSTPROCESS_MAX_WORKERS)

    def _validate_and_promote_hit(hit: CandidateHit) -> tuple[CandidateHit, CandidateHit] | None:
        plan_mask = plan_masks_by_template[hit.template_id]
        if _validate_template_hit(hit, plan_mask, plan_image):
            promoted_hit = _maybe_promote_socket_06_to_07(
                hit,
                plan_image,
                templates,
                plan_masks_by_template,
                dilated_plan_masks_by_template,
                variants_lookup,
                socket_07_promotions,
            )
            return hit, promoted_hit
        return None

    validated_hits: list[CandidateHit] = []
    validation_workers = max(1, min(len(raw_template_hits), postprocess_workers))
    if raw_template_hits:
        with ThreadPoolExecutor(max_workers=validation_workers) as pool:
            for validation_result in pool.map(_validate_and_promote_hit, raw_template_hits):
                if validation_result is None:
                    continue
                original_hit, promoted_hit = validation_result
                if promoted_hit.template_id != original_hit.template_id or promoted_hit.bbox != original_hit.bbox:
                    diagnostics["promoted_targeted_hits"] += 1
                validated_hits.append(promoted_hit)
    validated_candidates.extend(validated_hits)

    diagnostics["validated_template_hits"] = len(validated_candidates) - len(pdf_candidates)
    timings["validation_targeted"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    prefiltered_candidates = _prefilter_candidates(validated_candidates)
    diagnostics["prefilter_hits"] = len(prefiltered_candidates)
    timings["prefilter"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    pre_parent_candidates = _cluster_candidates(prefiltered_candidates, parent_ids_by_child)
    diagnostics["pre_parent_clusters"] = len(pre_parent_candidates)
    timings["pre_parent_clustering"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    def _search_parent_hit(hit: CandidateHit) -> tuple[CandidateHit, dict[str, int]]:
        local_stats: dict[str, int] = {}
        promoted_hit = _maybe_promote_switch_parent_search(
            hit,
            plan_image,
            templates,
            plan_masks_by_template,
            dilated_plan_masks_by_template,
            variants_lookup,
            socket_07_promotions,
            local_stats,
        )
        return promoted_hit, local_stats

    parent_search_candidates: list[CandidateHit] = []
    parent_search_workers = max(1, min(len(pre_parent_candidates), postprocess_workers))
    if pre_parent_candidates:
        with ThreadPoolExecutor(max_workers=parent_search_workers) as pool:
            for hit, (promoted_hit, local_stats) in zip(pre_parent_candidates, pool.map(_search_parent_hit, pre_parent_candidates)):
                diagnostics["parent_search_input_hits"] += local_stats.get("parent_search_input_hits", 0)
                diagnostics["parent_search_candidates"] += local_stats.get("parent_search_candidates", 0)
                if promoted_hit.template_id != hit.template_id or promoted_hit.bbox != hit.bbox:
                    diagnostics["promoted_parent_search_hits"] += 1
                parent_search_candidates.append(promoted_hit)
    timings["parent_search"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    final_hits = _cluster_candidates(parent_search_candidates, parent_ids_by_child)
    diagnostics["final_hits"] = len(final_hits)
    timings["clustering"] = time.perf_counter() - phase_start

    timings_ms = {
        name: round(seconds * 1000.0, 3)
        for name, seconds in timings.items()
    }
    if debug_profile is not None:
        debug_profile.clear()
        debug_profile.update(
            {
                "timingsMs": timings_ms,
                "counters": {key: int(value) for key, value in diagnostics.items()},
                "threading": {
                    "scanWorkers": int(scan_workers),
                    "configuredScanWorkers": int(DETECTOR_SCAN_MAX_WORKERS),
                    "validationWorkers": int(validation_workers if raw_template_hits else 0),
                    "parentSearchWorkers": int(parent_search_workers if pre_parent_candidates else 0),
                    "configuredPostprocessWorkers": int(DETECTOR_POSTPROCESS_MAX_WORKERS),
                    "opencvThreads": int(OPENCV_NUM_THREADS),
                    "cpuCount": int(_safe_cpu_count()),
                },
                "searchRoi": {
                    "totalRois": int(diagnostics["search_rois"]),
                    "fullScanTemplates": int(diagnostics["full_scan_templates"]),
                    "roiAreaPixels": int(diagnostics["roi_area_pixels"]),
                    "foregroundPixels": int(diagnostics["roi_foreground_pixels"]),
                    "fullImageAreaPixels": int(plan_image.shape[0] * plan_image.shape[1]),
                },
                "slowestPhase": max(timings_ms.items(), key=lambda item: item[1])[0]
                if timings_ms
                else None,
            }
        )

    print(
        "Detection diagnostics:"
        f" prepared_variants={diagnostics['prepared_variants']},"
        f" skipped_empty_color_masks={diagnostics['skipped_empty_color_masks']},"
        f" raw_peaks={diagnostics['raw_peaks']},"
        f" raw_after_prefilter={diagnostics['raw_prefilter_hits']}(-{diagnostics['raw_prefilter_removed']}),"
        f" validated_template_hits={diagnostics['validated_template_hits']},"
        f" promoted_targeted_hits={diagnostics['promoted_targeted_hits']},"
        f" parent_search_input_hits={diagnostics['parent_search_input_hits']},"
        f" parent_search_candidates={diagnostics['parent_search_candidates']},"
        f" promoted_parent_search_hits={diagnostics['promoted_parent_search_hits']},"
        f" pdf_text_hits={diagnostics['pdf_text_hits']},"
        f" after_prefilter={diagnostics['prefilter_hits']},"
        f" pre_parent_clusters={diagnostics['pre_parent_clusters']},"
        f" final_clusters={diagnostics['final_hits']},"
        f" rois={diagnostics['search_rois']} full_scan_templates={diagnostics['full_scan_templates']},"
        f" threads=scan:{scan_workers}/{DETECTOR_SCAN_MAX_WORKERS}|post:{postprocess_workers}/{DETECTOR_POSTPROCESS_MAX_WORKERS}|opencv:{OPENCV_NUM_THREADS},"
        f" timings_ms="
        f"pdf_text:{timings_ms['pdf_text']:.0f}|"
        f"prepare:{timings_ms['prepare']:.0f}|"
        f"scan:{timings_ms['scan']:.0f}|"
        f"raw_prefilter:{timings_ms['raw_prefilter']:.0f}|"
        f"validation_targeted:{timings_ms['validation_targeted']:.0f}|"
        f"prefilter:{timings_ms['prefilter']:.0f}|"
        f"pre_parent_clustering:{timings_ms['pre_parent_clustering']:.0f}|"
        f"parent_search:{timings_ms['parent_search']:.0f}|"
        f"clustering:{timings_ms['clustering']:.0f}"
    )

    per_template: dict[int, list[Detection]] = {}
    for hit in final_hits:
        x, y, w, h = [int(value) for value in hit.bbox]
        detection = Detection(
            symbol_name=templates[hit.template_id].name,
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=round(hit.match_score, 3),
            source=hit.source,
            rotation=hit.rotation,
            scale=hit.scale,
            mirrored=hit.mirrored,
            coverage=round(hit.coverage, 3),
            purity=round(hit.purity, 3),
            context_purity=round(hit.context_purity, 3),
            color_similarity=round(hit.color_similarity, 3),
            verification_score=round(hit.verification_score, 3),
        )
        per_template.setdefault(hit.template_id, []).append(detection)

    results: list[DetectionResult] = []
    for template_id, detections in per_template.items():
        detections.sort(key=lambda det: (det.verification_score, det.confidence), reverse=True)

        count = len(detections)
        if subtract_legend and legend_rect is None:
            count = max(0, count - 1)

        if count <= 0:
            continue

        results.append(
            DetectionResult(
                symbol_name=templates[template_id].name,
                count=count,
                color="#22c55e",
                detections=detections[:count] if subtract_legend and legend_rect is None else detections,
            )
        )

    results.sort(key=lambda result: result.symbol_name.lower())
    return results


def draw_results(
    plan_image: np.ndarray,
    results: list[DetectionResult],
) -> np.ndarray:
    """Draw detection boxes on a copy of the plan image."""

    output = plan_image.copy()

    for result in results:
        color = np.random.randint(0, 255, size=3).tolist()
        for det in result.detections:
            cv2.rectangle(
                output,
                (det.x, det.y),
                (det.x + det.width, det.y + det.height),
                color,
                2,
            )

    return output


if __name__ == "__main__":
    import sys

    plan_path = sys.argv[1] if len(sys.argv) > 1 else "wygenerowany_plan_300dpi.png"
    templates_dir = sys.argv[2] if len(sys.argv) > 2 else "templates"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "wynik.png"

    print(f"Loading plan: {plan_path}")
    plan = cv2.imread(plan_path)
    if plan is None:
        print(f"Error: cannot read {plan_path}")
        sys.exit(1)

    print(f"Loading templates from: {templates_dir}")
    templates = load_templates(templates_dir)
    print(f"Loaded {len(templates)} templates.\n")

    print(f"{'NAME':<45} | {'TYPE':<10} | {'COUNT':>5}")
    print("-" * 68)

    results = detect_symbols(plan, templates)

    total = 0
    for result in results:
        mode = "[PRECISE]" if any(word in result.symbol_name.lower() for word in PRECISE_KEYWORDS) else "[DILATE]"
        print(f"{result.symbol_name[:43]:<45} | {mode:<10} | {result.count:>5}")
        total += result.count

    print("-" * 68)
    print(f"{'TOTAL':<45} | {'':10} | {total:>5}")

    output_image = draw_results(plan, results)
    cv2.imwrite(output_path, output_image)
    print(f"\nSaved result: {output_path}")
