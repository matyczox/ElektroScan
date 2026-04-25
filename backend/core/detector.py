"""
detector.py - CPU-friendly symbol detection for electrical plans.
"""

from __future__ import annotations

import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import fitz
import numpy as np


# Configuration constants

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
MIN_VERIFICATION_SCORE = 0.40

LOCAL_MAX_KERNEL_RATIO = 0.35

PRECISE_KEYWORDS = ["gniazdo", "wypust"]

COLOR_HUE_TOLERANCE = 18
COLOR_SAT_TOLERANCE = 80
COLOR_VAL_TOLERANCE = 80
COLOR_HUE_REJECTION_THRESHOLD = 36

PREFILTER_NMS_MIN_CANDIDATES = 250
PREFILTER_NMS_IOU_THRESHOLD = 0.85

CLUSTER_IOU_THRESHOLD = 0.18
CLUSTER_IOM_THRESHOLD = 0.50
CLUSTER_CENTER_DISTANCE_RATIO = 0.40

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
    transformed_mask: np.ndarray | None
    pixel_count: int
    bbox: tuple[int, int, int, int]
    match_score: float
    dominant_hsv: tuple[int, int, int] | None
    source: str = "template"
    coverage: float = 0.0
    purity: float = 0.0
    color_similarity: float = 1.0
    verification_score: float = 0.0


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

    for scale in SCALES:
        if scale != 1.0:
            new_w = max(1, int(round(base_mask.shape[1] * scale)))
            new_h = max(1, int(round(base_mask.shape[0] * scale)))
            scaled_mask = cv2.resize(base_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_mask = base_mask

        for rotation, rotate_code in ROTATIONS:
            rot_mask = cv2.rotate(scaled_mask, rotate_code) if rotate_code is not None else scaled_mask
            pixel_count = int(cv2.countNonZero(rot_mask))
            if pixel_count == 0:
                continue

            variants.append(
                TemplateVariant(
                    template_id=template_id,
                    scale=scale,
                    rotation=rotation,
                    transformed_mask=rot_mask,
                    pixel_count=pixel_count,
                    width=int(rot_mask.shape[1]),
                    height=int(rot_mask.shape[0]),
                )
            )

    return variants


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

    intersection = int(cv2.countNonZero(cv2.bitwise_and(roi, hit.transformed_mask)))
    coverage = intersection / hit.pixel_count
    purity = intersection / roi_foreground

    if coverage < MIN_COVERAGE_RATIO or purity < MIN_PURITY_RATIO:
        return False

    color_similarity = _roi_color_similarity(plan_image, plan_mask, hit.bbox, hit.dominant_hsv)
    if color_similarity <= 0.0:
        return False

    verification_score = (
        0.60 * hit.match_score
        + 0.25 * coverage
        + 0.15 * purity
    )

    if verification_score < MIN_VERIFICATION_SCORE:
        return False

    hit.coverage = round(coverage, 4)
    hit.purity = round(purity, 4)
    hit.color_similarity = round(color_similarity, 4)
    hit.verification_score = round(verification_score, 4)
    return True


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


def _should_cluster(hit_a: CandidateHit, hit_b: CandidateHit) -> bool:
    """Decide whether two candidates describe the same physical object."""

    inter_area, iou, iom, center_distance = _bbox_metrics(hit_a.bbox, hit_b.bbox)
    if inter_area <= 0:
        return False

    if iom >= CLUSTER_IOM_THRESHOLD:
        return True

    return (
        iou >= CLUSTER_IOU_THRESHOLD
        and center_distance <= CLUSTER_CENTER_DISTANCE_RATIO
    )


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


def _cluster_candidates(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Cluster class-agnostic overlaps and keep one winner per physical place."""

    if not candidates:
        return []

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
        winners.append(
            max(
                group_hits,
                key=lambda hit: (
                    hit.verification_score,
                    hit.color_similarity,
                    hit.match_score,
                    1 if hit.source == "pdf_text" else 0,
                ),
            )
        )

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
) -> list[DetectionResult]:
    """
    Detect symbols on a rendered plan using template matching plus PDF-text fallback.
    """

    exclude_rects = list(exclude_rects or [])

    if not templates:
        return []

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

    pdf_hits_by_template = _collect_pdf_text_hits(
        pdf_path=pdf_path or "",
        templates=templates,
        plan_image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
        exclude_rects=exclude_rects,
    )
    pdf_candidates = [hit for hits in pdf_hits_by_template.values() for hit in hits]

    variants_by_template = {
        template_id: _prepare_variants(template_id, template)
        for template_id, template in enumerate(templates)
    }

    diagnostics = {
        "raw_peaks": 0,
        "validated_template_hits": 0,
        "pdf_text_hits": len(pdf_candidates),
        "prefilter_hits": 0,
        "final_hits": 0,
    }

    def _scan_template(template_id: int) -> list[CandidateHit]:
        template = templates[template_id]
        threshold = THRESHOLD_PRECISE if template.requires_precision else THRESHOLD_DILATED
        plan_mask = _get_plan_mask(template)

        template_hits: list[CandidateHit] = []
        for variant in variants_by_template.get(template_id, []):
            if variant.height > plan_mask.shape[0] or variant.width > plan_mask.shape[1]:
                continue

            match_result = cv2.matchTemplate(plan_mask, variant.transformed_mask, cv2.TM_CCOEFF_NORMED)
            peaks = _find_local_maxima(
                match_result,
                threshold=threshold,
                template_width=variant.width,
                template_height=variant.height,
            )
            if len(peaks) > MAX_PEAKS_PER_VARIANT:
                continue

            for px, py, score in peaks:
                template_hits.append(
                    CandidateHit(
                        template_id=template_id,
                        scale=variant.scale,
                        rotation=variant.rotation,
                        transformed_mask=variant.transformed_mask,
                        pixel_count=variant.pixel_count,
                        bbox=(px, py, variant.width, variant.height),
                        match_score=score,
                        dominant_hsv=template.dominant_hsv,
                        source="template",
                    )
                )

        return template_hits

    template_ids_to_scan = list(variants_by_template.keys())
    raw_template_hits: list[CandidateHit] = []
    if template_ids_to_scan:
        num_workers = max(1, min(len(template_ids_to_scan), os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for hits in pool.map(_scan_template, template_ids_to_scan):
                raw_template_hits.extend(hits)

    diagnostics["raw_peaks"] = len(raw_template_hits)

    validated_candidates: list[CandidateHit] = list(pdf_candidates)
    for hit in raw_template_hits:
        plan_mask = _get_plan_mask(templates[hit.template_id])
        if _validate_template_hit(hit, plan_mask, plan_image):
            validated_candidates.append(hit)

    diagnostics["validated_template_hits"] = len(validated_candidates) - len(pdf_candidates)

    prefiltered_candidates = _prefilter_candidates(validated_candidates)
    diagnostics["prefilter_hits"] = len(prefiltered_candidates)

    final_hits = _cluster_candidates(prefiltered_candidates)
    diagnostics["final_hits"] = len(final_hits)

    print(
        "Detection diagnostics:"
        f" raw_peaks={diagnostics['raw_peaks']},"
        f" validated_template_hits={diagnostics['validated_template_hits']},"
        f" pdf_text_hits={diagnostics['pdf_text_hits']},"
        f" after_prefilter={diagnostics['prefilter_hits']},"
        f" final_clusters={diagnostics['final_hits']}"
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
