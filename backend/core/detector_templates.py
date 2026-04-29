"""Template loading and variant preparation."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np

from core.detector_config import (
    MIN_TEMPLATE_PIXELS,
    MIRRORED_VARIANT_PREFIXES,
    PDF_TEXT_MAX_TOKEN_LENGTH,
    PDF_TEXT_MIN_TOKEN_LENGTH,
    PRECISE_KEYWORDS,
    ROTATIONS,
    SCALES,
    SOCKET_07_EXTRA_MIN_COVERAGE,
    SWITCH_10_EXTRA_MIN_COVERAGE,
    SWITCH_12_EXTRA_MIN_COVERAGE,
    SWITCH_FAMILY_MIN_CHILD_COVERAGE,
    SWITCH_FAMILY_MIN_CROP_PURITY,
)
from core.detector_masks import (
    _dominant_hsv_color,
    _extract_label_content_mask,
    _hsv_mask,
    _mask_bbox,
)
from core.detector_models import TargetedPromotionRule, TemplateInfo, TemplateVariant


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

    # Only enable PDF-text lookup for templates whose whole name is a short
    # alphanumeric label. Descriptive legend names are intentionally ignored.
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
    allow_mirror = template.is_text_label or template_prefix in MIRRORED_VARIANT_PREFIXES

    for scale in SCALES:
        if scale != 1.0:
            new_w = max(1, int(round(base_mask.shape[1] * scale)))
            new_h = max(1, int(round(base_mask.shape[0] * scale)))
            scaled_mask = cv2.resize(base_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            scaled_content_mask = (
                cv2.resize(template.content_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                if template.content_mask is not None
                else None
            )
        else:
            scaled_mask = base_mask
            scaled_content_mask = template.content_mask

        mask_sources = [(False, scaled_mask)]
        if allow_mirror:
            mask_sources.append((True, cv2.flip(scaled_mask, 1)))

        for mirrored, source_mask in mask_sources:
            source_content_mask = scaled_content_mask
            if mirrored and source_content_mask is not None:
                source_content_mask = cv2.flip(source_content_mask, 1)

            for rotation, rotate_code in ROTATIONS:
                rot_mask = (
                    cv2.rotate(source_mask, rotate_code) if rotate_code is not None else source_mask
                )
                rot_content_mask = (
                    cv2.rotate(source_content_mask, rotate_code)
                    if rotate_code is not None and source_content_mask is not None
                    else source_content_mask
                )
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
                        content_mask=rot_content_mask,
                        pixel_count=pixel_count,
                        content_pixel_count=(
                            int(cv2.countNonZero(rot_content_mask))
                            if rot_content_mask is not None
                            else 0
                        ),
                        content_bbox=(
                            _mask_bbox(rot_content_mask) if rot_content_mask is not None else None
                        ),
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
        (
            "11",
            "10",
            SWITCH_FAMILY_MIN_CHILD_COVERAGE,
            SWITCH_FAMILY_MIN_CROP_PURITY,
            SWITCH_10_EXTRA_MIN_COVERAGE,
            True,
        ),
        (
            "11",
            "12",
            SWITCH_FAMILY_MIN_CHILD_COVERAGE,
            SWITCH_FAMILY_MIN_CROP_PURITY,
            SWITCH_12_EXTRA_MIN_COVERAGE,
            True,
        ),
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
            child_key = (
                child_id,
                child_variant.scale,
                child_variant.rotation,
                child_variant.mirrored,
            )
            for parent_mirrored in (False, True):
                for parent_variant in parent_variants:
                    if (
                        not allow_rotation_mismatch
                        and parent_variant.rotation != child_variant.rotation
                    ):
                        continue
                    if parent_variant.mirrored != parent_mirrored:
                        continue
                    if (
                        child_variant.width > parent_variant.width
                        or child_variant.height > parent_variant.height
                    ):
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

                    intersection = int(
                        cv2.countNonZero(cv2.bitwise_and(crop, child_variant.transformed_mask))
                    )
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

        precise_mask = _hsv_mask(img, dilate=False)
        content_mask = _extract_label_content_mask(precise_mask)
        content_pixel_count = int(cv2.countNonZero(content_mask)) if content_mask is not None else 0
        if content_mask is not None:
            requires_precision = False

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
                content_mask=content_mask,
                content_pixel_count=content_pixel_count,
                content_bbox=_mask_bbox(content_mask) if content_mask is not None else None,
                is_text_label=content_mask is not None,
            )
        )

    templates.sort(key=lambda item: item.pixel_count, reverse=True)
    return templates
