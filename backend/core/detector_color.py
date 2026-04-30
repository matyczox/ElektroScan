"""Color-PDF detector strategy helpers.

Color plans should stay as simple as possible: use HSV masks, keep the narrow
scale range, and avoid gray-specific ink heuristics.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.detector_masks import _color_mask_for_template
from core.detector_models import TemplateInfo


def color_mask_cache_key(template: TemplateInfo) -> str | None:
    if template.dominant_hsv is None:
        return None
    return f"{template.dominant_hsv}_{template.requires_precision}"


def build_color_plan_mask(
    *,
    plan_image: np.ndarray,
    plan_hsv: np.ndarray,
    template: TemplateInfo,
    exclude_rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Build a color-specific HSV mask for one template."""

    mask = _color_mask_for_template(
        plan_image,
        template.dominant_hsv,
        dilate=not template.requires_precision,
        hsv_image=plan_hsv,
    )
    for ex, ey, ew, eh in exclude_rects:
        cv2.rectangle(mask, (ex, ey), (ex + ew, ey + eh), 0, -1)
    return mask
