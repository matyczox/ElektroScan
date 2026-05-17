"""Scan eligibility and peak-budget policies for template scanning."""

from __future__ import annotations

from core.detector_config import (
    COLOR_MAX_PEAKS_PER_VARIANT,
    COLOR_NEAR_THRESHOLD_RECOVERY_ENABLED,
    COLOR_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT,
    COLOR_NEAR_THRESHOLD_RECOVERY_MAX_ROI_AREA,
    COLOR_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA,
    COLOR_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA,
    GRAY_COMPACT_TEXT_DIAGONAL_MAX_ROI_AREA,
    GRAY_COMPACT_TEXT_DIAGONAL_MAX_ROI_ASPECT,
    GRAY_COMPACT_TEXT_DIAGONAL_MIN_ROI_DENSITY,
    GRAY_INTERRUPTED_LABEL_RECOVERY_ENABLED,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ASPECT,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ROI_AREA,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_TEMPLATE_AREA,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA,
    GRAY_NEAR_THRESHOLD_RECOVERY_ENABLED,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ROI_AREA,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA,
    LABEL_CONTENT_SCAN_MIN_PIXELS,
)
from core.detector_masks import _tight_mask_crop
from core.detector_models import TemplateInfo, TemplateVariant


def _is_content_scan_eligible(variant: TemplateVariant) -> bool:
    """Keep OCR-like scans limited to useful label glyph masks."""

    if variant.content_mask is None or variant.content_bbox is None:
        return False
    if variant.content_pixel_count < LABEL_CONTENT_SCAN_MIN_PIXELS:
        return False

    content_crop = _tight_mask_crop(variant.content_mask)
    if content_crop is None:
        return False
    return content_crop.shape != variant.transformed_mask.shape


def _needs_directional_text_content_scan(template: TemplateInfo) -> bool:
    """Return true for label templates whose marker can move around text."""

    if not template.is_text_label or template.content_bbox is None:
        return False
    height, width = template.mask.shape[:2]
    _x, y, content_width, content_height = template.content_bbox
    return (
        y >= height * 0.25
        and content_width >= width * 0.55
        and content_height >= height * 0.45
    )


def _gray_near_threshold_recovery_eligible(
    *,
    detector_profile: str,
    variant: TemplateVariant,
    roi_w: int,
    roi_h: int,
    roi_foreground: int,
) -> bool:
    if detector_profile != "gray" or not GRAY_NEAR_THRESHOLD_RECOVERY_ENABLED:
        return False
    variant_area = int(variant.width) * int(variant.height)
    if (
        variant_area < int(GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA)
        or variant_area > int(GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA)
    ):
        return False
    if int(roi_w) * int(roi_h) > int(GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ROI_AREA):
        return False
    aspect = max(
        float(variant.width) / max(1.0, float(variant.height)),
        float(variant.height) / max(1.0, float(variant.width)),
    )
    if aspect > float(GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT):
        return False
    return roi_foreground >= max(1, int(variant.pixel_count * 0.45))


def _gray_interrupted_label_recovery_eligible(
    *,
    detector_profile: str,
    template: TemplateInfo,
    variant: TemplateVariant,
    scan_mask_kind: str,
    roi_w: int,
    roi_h: int,
    roi_foreground: int,
) -> bool:
    if detector_profile != "gray" or not GRAY_INTERRUPTED_LABEL_RECOVERY_ENABLED:
        return False
    if not template.is_text_label:
        return False
    if variant.mirrored:
        return False
    if scan_mask_kind != "zone_raw":
        return False
    variant_area = int(variant.width) * int(variant.height)
    if (
        variant_area < int(GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA)
        or variant_area > int(GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_TEMPLATE_AREA)
    ):
        return False
    if int(roi_w) * int(roi_h) > int(GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ROI_AREA):
        return False
    aspect = max(
        float(variant.width) / max(1.0, float(variant.height)),
        float(variant.height) / max(1.0, float(variant.width)),
    )
    if aspect > float(GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ASPECT):
        return False
    return roi_foreground >= max(1, int(variant.pixel_count * 0.40))


def _color_near_threshold_recovery_eligible(
    *,
    detector_profile: str,
    template: TemplateInfo,
    variant: TemplateVariant,
    roi_w: int,
    roi_h: int,
    roi_foreground: int,
) -> bool:
    if detector_profile != "color" or not COLOR_NEAR_THRESHOLD_RECOVERY_ENABLED:
        return False
    if template.dominant_hsv is None:
        return False
    if variant.scale < 0.90:
        return False
    variant_area = int(variant.width) * int(variant.height)
    if (
        variant_area < int(COLOR_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA)
        or variant_area > int(COLOR_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA)
    ):
        return False
    if int(roi_w) * int(roi_h) > int(COLOR_NEAR_THRESHOLD_RECOVERY_MAX_ROI_AREA):
        return False
    aspect = max(
        float(variant.width) / max(1.0, float(variant.height)),
        float(variant.height) / max(1.0, float(variant.width)),
    )
    if aspect > float(COLOR_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT):
        return False
    return roi_foreground >= 1


def _is_compact_text_diagonal_variant(
    detector_profile: str,
    template: TemplateInfo,
    variant: TemplateVariant,
) -> bool:
    return (
        detector_profile == "gray"
        and template.dominant_hsv is None
        and template.is_text_label
        and int(variant.rotation) % 90 != 0
    )


def _compact_text_diagonal_roi_eligible(
    *,
    detector_profile: str,
    template: TemplateInfo,
    variant: TemplateVariant,
    roi_strategy: str,
    roi_w: int,
    roi_h: int,
    roi_foreground: int,
) -> bool:
    if not _is_compact_text_diagonal_variant(detector_profile, template, variant):
        return True

    if roi_strategy not in {"fast_compact", "fast_compact_connected"}:
        return False

    roi_area = int(roi_w) * int(roi_h)
    if roi_area <= 0 or roi_area > int(GRAY_COMPACT_TEXT_DIAGONAL_MAX_ROI_AREA):
        return False

    roi_aspect = max(
        float(roi_w) / max(1.0, float(roi_h)),
        float(roi_h) / max(1.0, float(roi_w)),
    )
    if roi_aspect > float(GRAY_COMPACT_TEXT_DIAGONAL_MAX_ROI_ASPECT):
        return False

    roi_density = float(roi_foreground) / max(1.0, float(roi_area))
    return roi_density >= float(GRAY_COMPACT_TEXT_DIAGONAL_MIN_ROI_DENSITY)


def _select_spatially_fair_peaks(
    peaks: list[tuple[int, int, float]],
    *,
    limit: int,
    template_width: int,
    template_height: int,
) -> list[tuple[int, int, float]]:
    """Keep high-scoring peaks spread across a large gray ROI."""

    if len(peaks) <= limit:
        return peaks

    min_dx = max(1.0, float(template_width) * 0.70)
    min_dy = max(1.0, float(template_height) * 0.70)
    selected: list[tuple[int, int, float]] = []
    selected_keys: set[tuple[int, int]] = set()
    for peak in peaks:
        px, py = peak[0], peak[1]
        if all(abs(px - sx) >= min_dx or abs(py - sy) >= min_dy for sx, sy, *_ in selected):
            selected.append(peak)
            selected_keys.add((px, py))
            if len(selected) >= limit:
                break

    if len(selected) < limit:
        for peak in peaks:
            key = (peak[0], peak[1])
            if key in selected_keys:
                continue
            selected.append(peak)
            if len(selected) >= limit:
                break

    selected.sort(key=lambda item: item[2], reverse=True)
    return selected


def _cap_color_variant_peaks(
    peaks: list[tuple[int, int, float]],
    *,
    detector_profile: str,
    limit: int,
    template_width: int,
    template_height: int,
) -> list[tuple[int, int, float]]:
    """Limit color raw peaks before creating heavy CandidateHit objects."""

    if detector_profile != "color" or limit <= 0 or len(peaks) <= limit:
        return peaks
    return _select_spatially_fair_peaks(
        peaks,
        limit=limit,
        template_width=template_width,
        template_height=template_height,
    )
