"""ROI-level detector diagnostics.

This module is intentionally separate from the production detector path.  It
answers a different question: "if the user manually points at this object, what
does the engine see there and why would it accept/reject nearby templates?"
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from core.detector_config import (
    GRAY_DARK_EVIDENCE_THRESHOLD,
    GRAY_DARK_ZONE_THRESHOLD,
    GRAY_ELONGATED_SCAN_MAX_TEMPLATE_PIXELS,
    GRAY_ELONGATED_SCAN_THRESHOLD,
    GRAY_SCALES,
    GRAY_RAW_SCAN_MIN_TEMPLATE_AREA,
    GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS,
    GRAY_RAW_SCAN_THRESHOLD,
    GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
    GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    GRAY_STRICT_SCAN_THRESHOLD,
    SCALES,
    THRESHOLD_DILATED,
    THRESHOLD_PRECISE,
)
from core.detector_masks import (
    _color_mask_for_template,
    _context_purity,
    _ink_mask,
    _roi_mask,
    _suppress_long_strokes,
    _validate_template_hit,
)
from core.detector_models import CandidateHit, TemplateInfo
from core.detector_templates import _prepare_variants


def _clamp_roi(
    roi: tuple[int, int, int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int] | None:
    x, y, w, h = roi
    image_h = int(image_shape[0])
    image_w = int(image_shape[1])
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(image_w, int(round(x + w)))
    y2 = min(image_h, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _crop(mask: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return mask[y : y + h, x : x + w]


def _image_data_url(image: np.ndarray, ext: str = ".png") -> str:
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        return ""
    return f"data:image/{ext.lstrip('.')};base64,{base64.b64encode(buffer).decode('utf-8')}"


def _mask_data_url(mask: np.ndarray) -> str:
    if mask.ndim == 2:
        image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        image = mask
    return _image_data_url(image)


def _hit_overlap_metrics(
    hit: CandidateHit,
    plan_mask: np.ndarray,
) -> tuple[float, float, float]:
    if hit.transformed_mask is None:
        return 1.0, 1.0, 1.0
    roi_mask = _roi_mask(plan_mask, hit.bbox)
    if roi_mask is None or roi_mask.shape != hit.transformed_mask.shape:
        return 0.0, 0.0, 0.0
    roi_foreground = int(cv2.countNonZero(roi_mask))
    if roi_foreground <= 0 or hit.pixel_count <= 0:
        return 0.0, 0.0, 0.0
    intersection_mask = cv2.bitwise_and(roi_mask, hit.transformed_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    coverage = intersection / max(1, hit.pixel_count)
    purity = intersection / max(1, roi_foreground)
    context = _context_purity(plan_mask, hit.bbox, intersection_mask)
    return coverage, purity, context


def _use_raw_gray_scan_mask(template: TemplateInfo) -> bool:
    height, width = template.mask.shape[:2]
    area = int(width * height)
    pixels = int(getattr(template, "pixel_count", 0) or cv2.countNonZero(template.mask))
    return area >= GRAY_RAW_SCAN_MIN_TEMPLATE_AREA or pixels >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS


def _use_relaxed_gray_scan_threshold(template: TemplateInfo) -> bool:
    height, width = template.mask.shape[:2]
    return int(width * height) >= GRAY_RAW_SCAN_MIN_TEMPLATE_AREA


def _use_lenient_gray_elongated_scan_threshold(template: TemplateInfo) -> bool:
    height, width = template.mask.shape[:2]
    aspect = max(width / max(1, height), height / max(1, width))
    pixels = int(getattr(template, "pixel_count", 0) or cv2.countNonZero(template.mask))
    return aspect >= 2.0 and pixels <= GRAY_ELONGATED_SCAN_MAX_TEMPLATE_PIXELS


def _scan_mask_for_template(
    *,
    detector_profile: str,
    plan_image: np.ndarray,
    plan_hsv: np.ndarray,
    template: TemplateInfo,
    gray_raw_mask: np.ndarray,
    gray_scan_mask: np.ndarray,
    gray_dark_raw_mask: np.ndarray,
    gray_dark_scan_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (validation_mask, scan_mask, scan_mask_kind) for one template."""

    if detector_profile == "gray" or template.dominant_hsv is None:
        if _use_raw_gray_scan_mask(template):
            return gray_raw_mask, gray_dark_raw_mask, "zone_raw"
        return gray_raw_mask, gray_dark_scan_mask, "zone_suppressed"

    color_mask = _color_mask_for_template(
        plan_image,
        template.dominant_hsv,
        dilate=not template.requires_precision,
        hsv_image=plan_hsv,
    )
    return color_mask, color_mask, "color"


def inspect_roi(
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    roi: tuple[int, int, int, int],
    *,
    detector_profile: str = "color",
    top_n: int = 15,
) -> dict[str, Any]:
    """Inspect the selected ROI against all templates and return ranked diagnostics."""

    clamped = _clamp_roi(roi, plan_image.shape)
    if clamped is None:
        return {
            "roi": {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]},
            "error": "empty_roi",
            "candidates": [],
        }

    detector_profile = detector_profile if detector_profile in {"color", "gray"} else "color"
    used_scales = list(GRAY_SCALES) if detector_profile == "gray" else list(SCALES)
    top_n = max(1, min(50, int(top_n)))
    x, y, w, h = clamped

    plan_hsv = cv2.cvtColor(plan_image, cv2.COLOR_BGR2HSV)
    gray_raw_mask = _ink_mask(plan_image, dilate=True)
    gray_scan_mask = _suppress_long_strokes(
        gray_raw_mask,
        GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
        GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    )
    gray_dark_raw_mask = _ink_mask(
        plan_image,
        dilate=True,
        threshold=GRAY_DARK_ZONE_THRESHOLD,
    )
    gray_evidence_mask = _ink_mask(
        plan_image,
        dilate=False,
        threshold=GRAY_DARK_EVIDENCE_THRESHOLD,
    )
    gray_dark_scan_mask = _suppress_long_strokes(
        gray_dark_raw_mask,
        GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
        GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    )

    candidates: list[dict[str, Any]] = []
    raw_hits_by_scale: dict[float, int] = {}
    rejected_by_reason: dict[str, int] = {}
    variant_count = 0

    for template_id, template in enumerate(templates):
        validation_mask, scan_mask, scan_mask_kind = _scan_mask_for_template(
            detector_profile=detector_profile,
            plan_image=plan_image,
            plan_hsv=plan_hsv,
            template=template,
            gray_raw_mask=gray_raw_mask,
            gray_scan_mask=gray_scan_mask,
            gray_dark_raw_mask=gray_dark_raw_mask,
            gray_dark_scan_mask=gray_dark_scan_mask,
        )
        roi_scan = _crop(scan_mask, clamped)
        if cv2.countNonZero(roi_scan) <= 0:
            continue

        threshold = THRESHOLD_PRECISE if template.requires_precision else THRESHOLD_DILATED
        if detector_profile == "gray":
            if _use_relaxed_gray_scan_threshold(template):
                threshold = max(threshold, GRAY_RAW_SCAN_THRESHOLD)
            elif _use_lenient_gray_elongated_scan_threshold(template):
                threshold = GRAY_ELONGATED_SCAN_THRESHOLD
            else:
                threshold = max(threshold, GRAY_STRICT_SCAN_THRESHOLD)

        for variant in _prepare_variants(template_id, template, scales=used_scales):
            variant_count += 1
            if variant.width > w or variant.height > h:
                continue
            result = cv2.matchTemplate(
                roi_scan,
                variant.transformed_mask,
                cv2.TM_CCOEFF_NORMED,
            )
            if result.size == 0:
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            match_score = float(max_val)
            bbox = (
                int(x + max_loc[0]),
                int(y + max_loc[1]),
                int(variant.width),
                int(variant.height),
            )
            hit = CandidateHit(
                template_id=template_id,
                scale=variant.scale,
                rotation=variant.rotation,
                mirrored=variant.mirrored,
                transformed_mask=variant.transformed_mask,
                content_mask=variant.content_mask,
                pixel_count=variant.pixel_count,
                content_pixel_count=variant.content_pixel_count,
                content_bbox=variant.content_bbox,
                bbox=bbox,
                match_score=round(match_score, 4),
                dominant_hsv=None if detector_profile == "gray" else template.dominant_hsv,
                source="roi_inspector",
                is_text_label=template.is_text_label,
            )
            if match_score < threshold:
                # Keep a histogram so we know if a scale is alive even when it
                # does not reach the normal detector threshold.
                if match_score >= 0.18:
                    raw_hits_by_scale[variant.scale] = raw_hits_by_scale.get(variant.scale, 0) + 1
                    coverage, purity, context = _hit_overlap_metrics(hit, validation_mask)
                    candidates.append(
                        {
                            "symbolName": template.name,
                            "accepted": False,
                            "reason": "below_threshold",
                            "match": round(match_score, 4),
                            "verification": 0.0,
                            "coverage": round(float(coverage), 4),
                            "purity": round(float(purity), 4),
                            "contextPurity": round(float(context), 4),
                            "scale": float(variant.scale),
                            "rotation": int(variant.rotation),
                            "mirrored": bool(variant.mirrored),
                            "bbox": {
                                "x": bbox[0],
                                "y": bbox[1],
                                "width": bbox[2],
                                "height": bbox[3],
                            },
                            "templatePixels": int(variant.pixel_count),
                            "templateId": int(template_id),
                            "threshold": round(float(threshold), 4),
                            "scanMask": scan_mask_kind,
                        }
                    )
                continue

            raw_hits_by_scale[variant.scale] = raw_hits_by_scale.get(variant.scale, 0) + 1
            reasons: dict[str, int] = {}
            accepted = _validate_template_hit(
                hit,
                validation_mask,
                plan_image,
                reasons=reasons,
                evidence_mask=gray_evidence_mask if detector_profile == "gray" else None,
            )
            reason = next(iter(reasons), "accepted" if accepted else "unknown")
            if not accepted:
                rejected_by_reason[reason] = rejected_by_reason.get(reason, 0) + 1
                coverage, purity, context = _hit_overlap_metrics(hit, validation_mask)
            else:
                coverage, purity, context = hit.coverage, hit.purity, hit.context_purity

            candidates.append(
                {
                    "symbolName": template.name,
                    "accepted": accepted,
                    "reason": reason,
                    "match": round(match_score, 4),
                    "verification": round(float(hit.verification_score), 4),
                    "coverage": round(float(coverage), 4),
                    "purity": round(float(purity), 4),
                    "contextPurity": round(float(context), 4),
                    "scale": float(variant.scale),
                    "rotation": int(variant.rotation),
                    "mirrored": bool(variant.mirrored),
                    "bbox": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2],
                        "height": bbox[3],
                    },
                    "templatePixels": int(variant.pixel_count),
                    "templateId": int(template_id),
                    "scanMask": scan_mask_kind,
                }
            )

    candidates.sort(
        key=lambda item: (
            1 if item["accepted"] else 0,
            float(item["verification"]),
            float(item["match"]),
            float(item["coverage"]),
        ),
        reverse=True,
    )

    roi_image = plan_image[y : y + h, x : x + w]
    roi_raw_mask = _crop(gray_raw_mask, clamped)
    roi_scan_mask = _crop(gray_scan_mask, clamped)
    roi_dark_raw_mask = _crop(gray_dark_raw_mask, clamped)
    roi_dark_scan_mask = _crop(gray_dark_scan_mask, clamped)

    return {
        "roi": {"x": x, "y": y, "width": w, "height": h},
        "profile": detector_profile,
        "usedScales": used_scales,
        "templates": len(templates),
        "variantsChecked": variant_count,
        "rawHitsByScale": {f"{scale:.2f}": count for scale, count in sorted(raw_hits_by_scale.items())},
        "rejectedByReason": rejected_by_reason,
        "roiInkPixels": int(cv2.countNonZero(roi_raw_mask)),
        "roiScanPixels": int(cv2.countNonZero(roi_scan_mask)),
        "roiDarkInkPixels": int(cv2.countNonZero(roi_dark_raw_mask)),
        "roiDarkScanPixels": int(cv2.countNonZero(roi_dark_scan_mask)),
        "grayDarkInkThreshold": int(GRAY_DARK_ZONE_THRESHOLD),
        "grayDarkEvidenceThreshold": int(GRAY_DARK_EVIDENCE_THRESHOLD),
        "roiDarkEvidencePixels": int(cv2.countNonZero(_crop(gray_evidence_mask, clamped))),
        "roiImage": _image_data_url(roi_image),
        "roiRawMask": _mask_data_url(roi_raw_mask),
        "roiScanMask": _mask_data_url(roi_scan_mask),
        "roiDarkRawMask": _mask_data_url(roi_dark_raw_mask),
        "roiDarkScanMask": _mask_data_url(roi_dark_scan_mask),
        "candidates": candidates[:top_n],
    }
