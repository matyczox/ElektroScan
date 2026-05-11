"""Template loading and variant preparation."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np

from core.detector_config import (
    GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MAX_ASPECT,
    GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MAX_SCALE,
    GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MAX_TEMPLATE_PIXELS,
    GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MIN_ASPECT,
    GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MIN_SCALE,
    GRAY_COMPACT_TEXT_DIAGONAL_ROTATIONS,
    GRAY_DIAGONAL_ROTATION_MAX_TEMPLATE_PIXELS,
    GRAY_DIAGONAL_ROTATION_MIN_ASPECT,
    GRAY_DIAGONAL_ROTATION_MIN_SCALE,
    GRAY_DIAGONAL_ROTATIONS,
    GRAY_NON_TEXT_DIAGONAL_ROTATIONS_ENABLED,
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
    _ink_mask,
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


def _is_color_symbol_core_misread_as_label(
    mask: np.ndarray,
    content_mask: np.ndarray | None,
    *,
    foreground_pixels: int,
) -> bool:
    """Detect pictogram symbols whose central shape looks like label content.

    Some colored legend symbols are composed of a compact core plus an attached
    marker line. A one-pixel crop difference can make the content extractor keep
    the core and drop the marker, turning e.g. an X-with-tail symbol into a
    text-label-like plain X. Treat those as normal pictograms so they compete
    against their neighboring plain symbols with the full geometry.
    """

    if content_mask is None or foreground_pixels <= 0:
        return False

    mask_bbox = _mask_bbox(mask)
    content_bbox = _mask_bbox(content_mask)
    if mask_bbox is None or content_bbox is None:
        return False

    _mx, _my, mask_width, mask_height = mask_bbox
    _cx, _cy, content_width, content_height = content_bbox
    if mask_width <= 0 or mask_height <= 0 or content_width <= 0 or content_height <= 0:
        return False

    content_aspect = max(
        content_width / max(1, content_height),
        content_height / max(1, content_width),
    )
    full_aspect = max(
        mask_width / max(1, mask_height),
        mask_height / max(1, mask_width),
    )
    content_ratio = int(cv2.countNonZero(content_mask)) / max(1, foreground_pixels)
    content_width_ratio = content_width / max(1, mask_width)
    content_height_ratio = content_height / max(1, mask_height)

    return (
        full_aspect >= 1.35
        and content_aspect <= 1.30
        and content_width_ratio >= 0.78
        and content_height_ratio <= 0.66
        and content_ratio >= 0.30
    )


def _prepare_variants(
    template_id: int,
    template: TemplateInfo,
    scales: list[float] | tuple[float, ...] | None = None,
    *,
    include_gray_diagonal_rotations: bool = False,
    disable_text_mirror: bool = False,
) -> list[TemplateVariant]:
    """Precompute all scale/rotation variants for one template."""

    variants: list[TemplateVariant] = []
    base_mask = template.mask
    template_prefix = _template_numeric_prefix(Path(template.path).name)

    def _mask_aspect(mask: np.ndarray) -> float:
        bbox = _mask_bbox(mask)
        if bbox is None:
            return 1.0
        _x, _y, width, height = bbox
        return max(width / max(1, height), height / max(1, width))

    allow_mirror = (
        (template.is_text_label and not disable_text_mirror)
        or (not template.is_text_label and template_prefix in MIRRORED_VARIANT_PREFIXES)
    )

    base_aspect = _mask_aspect(base_mask)
    use_diagonal_rotations = (
        include_gray_diagonal_rotations
        and GRAY_NON_TEXT_DIAGONAL_ROTATIONS_ENABLED
        and template.dominant_hsv is None
        and not template.is_text_label
        and template.pixel_count <= GRAY_DIAGONAL_ROTATION_MAX_TEMPLATE_PIXELS
        and base_aspect >= GRAY_DIAGONAL_ROTATION_MIN_ASPECT
    )
    use_compact_text_diagonal_rotations = (
        include_gray_diagonal_rotations
        and template.dominant_hsv is None
        and template.is_text_label
        and template.pixel_count <= GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MAX_TEMPLATE_PIXELS
        and GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MIN_ASPECT
        <= base_aspect
        <= GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MAX_ASPECT
    )
    base_rotation_specs = list(ROTATIONS)
    diagonal_rotation_specs = base_rotation_specs + [
        (angle, None) for angle in GRAY_DIAGONAL_ROTATIONS
    ]
    compact_text_diagonal_rotation_specs = base_rotation_specs + [
        (angle, None) for angle in GRAY_COMPACT_TEXT_DIAGONAL_ROTATIONS
    ]

    def _rotate_mask(mask: np.ndarray | None, rotation: int, rotate_code) -> np.ndarray | None:
        if mask is None:
            return None
        if rotate_code is not None:
            return cv2.rotate(mask, rotate_code)
        if rotation == 0:
            return mask

        height, width = mask.shape[:2]
        center = (width / 2.0, height / 2.0)
        matrix = cv2.getRotationMatrix2D(center, -float(rotation), 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_width = int(round((height * sin) + (width * cos)))
        new_height = int(round((height * cos) + (width * sin)))
        matrix[0, 2] += (new_width / 2.0) - center[0]
        matrix[1, 2] += (new_height / 2.0) - center[1]
        rotated = cv2.warpAffine(
            mask,
            matrix,
            (max(1, new_width), max(1, new_height)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return ((rotated > 0) * 255).astype(np.uint8)

    scale_list = list(scales) if scales is not None else list(SCALES)
    for scale in scale_list:
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

        if use_diagonal_rotations and scale >= GRAY_DIAGONAL_ROTATION_MIN_SCALE:
            rotation_specs = diagonal_rotation_specs
        elif (
            use_compact_text_diagonal_rotations
            and scale >= GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MIN_SCALE
            and scale <= GRAY_COMPACT_TEXT_DIAGONAL_ROTATION_MAX_SCALE
        ):
            rotation_specs = compact_text_diagonal_rotation_specs
        else:
            rotation_specs = base_rotation_specs
        mask_sources = [(False, scaled_mask)]
        if allow_mirror:
            mask_sources.append((True, cv2.flip(scaled_mask, 1)))

        for mirrored, source_mask in mask_sources:
            source_content_mask = scaled_content_mask
            if mirrored and source_content_mask is not None:
                source_content_mask = cv2.flip(source_content_mask, 1)

            for rotation, rotate_code in rotation_specs:
                rot_mask = _rotate_mask(source_mask, rotation, rotate_code)
                rot_content_mask = _rotate_mask(source_content_mask, rotation, rotate_code)
                if rot_mask is None:
                    continue
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
        is_gray_template = int(cv2.countNonZero(precise_mask)) <= MIN_TEMPLATE_PIXELS
        if is_gray_template:
            precise_mask = _ink_mask(img, dilate=False)

        content_mask = _extract_label_content_mask(precise_mask)
        content_pixel_count = int(cv2.countNonZero(content_mask)) if content_mask is not None else 0
        if (
            not is_gray_template
            and _is_color_symbol_core_misread_as_label(
                precise_mask,
                content_mask,
                foreground_pixels=int(cv2.countNonZero(precise_mask)),
            )
        ):
            content_mask = None
            content_pixel_count = 0
        if content_mask is not None:
            requires_precision = False

        mask = (
            _ink_mask(img, dilate=not requires_precision)
            if is_gray_template
            else _hsv_mask(img, dilate=not requires_precision)
        )
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
                dominant_hsv=None if is_gray_template else _dominant_hsv_color(img),
                text_tokens=_derive_text_tokens(name),
                content_mask=content_mask,
                content_pixel_count=content_pixel_count,
                content_bbox=_mask_bbox(content_mask) if content_mask is not None else None,
                is_text_label=content_mask is not None,
            )
        )

    templates.sort(key=lambda item: item.pixel_count, reverse=True)
    return templates
