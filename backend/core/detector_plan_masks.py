"""Plan-level mask cache used by the detector pipeline."""

from __future__ import annotations

import cv2
import numpy as np

from core import detector_color as color_strategy
from core.detector_mask_builders import _hsv_mask, _ink_mask
from core.detector_models import TemplateInfo


class PlanMaskCache:
    """Lazy cache for scan masks that depend on the rendered plan."""

    def __init__(
        self,
        *,
        plan_image: np.ndarray,
        plan_hsv: np.ndarray | None,
        exclude_rects: list[tuple[int, int, int, int]],
        detector_profile: str,
    ) -> None:
        self.plan_image = plan_image
        self.plan_hsv = plan_hsv
        self.exclude_rects = exclude_rects
        self.detector_profile = detector_profile
        self.color_masks_cache: dict[str, np.ndarray] = {}
        self.ink_mask_cache: np.ndarray | None = None
        self.dilated_ink_mask_cache: np.ndarray | None = None
        self.empty_plan_mask_cache: np.ndarray | None = None

    def get_ink_plan_mask(self, *, dilate: bool) -> np.ndarray:
        if self.ink_mask_cache is None:
            self.ink_mask_cache = _ink_mask(self.plan_image, dilate=False)
            for ex, ey, ew, eh in self.exclude_rects:
                cv2.rectangle(self.ink_mask_cache, (ex, ey), (ex + ew, ey + eh), 0, -1)
        if not dilate:
            return self.ink_mask_cache
        if self.dilated_ink_mask_cache is None:
            self.dilated_ink_mask_cache = cv2.dilate(
                self.ink_mask_cache,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
        return self.dilated_ink_mask_cache

    def get_empty_plan_mask(self) -> np.ndarray:
        if self.empty_plan_mask_cache is None:
            self.empty_plan_mask_cache = np.zeros(self.plan_image.shape[:2], dtype=np.uint8)
        return self.empty_plan_mask_cache

    def build_color_mask(self, template: TemplateInfo) -> np.ndarray:
        if self.plan_hsv is None:
            raise ValueError("plan_hsv is required for color mask building")
        return color_strategy.build_color_plan_mask(
            plan_image=self.plan_image,
            plan_hsv=self.plan_hsv,
            template=template,
            exclude_rects=self.exclude_rects,
        )

    def get_plan_mask(self, template: TemplateInfo) -> np.ndarray:
        if self.detector_profile == "gray":
            return self.get_ink_plan_mask(dilate=True)

        if template.dominant_hsv is None:
            return self.get_empty_plan_mask()

        cache_key = color_strategy.color_mask_cache_key(template)
        if cache_key not in self.color_masks_cache:
            mask = self.build_color_mask(template)
            if cache_key is not None:
                self.color_masks_cache[cache_key] = mask
            return mask
        return self.color_masks_cache[cache_key]

    def fallback_color_mask(self) -> np.ndarray:
        fallback = _hsv_mask(self.plan_image, dilate=False, hsv_image=self.plan_hsv)
        for ex, ey, ew, eh in self.exclude_rects:
            cv2.rectangle(fallback, (ex, ey), (ex + ew, ey + eh), 0, -1)
        return fallback

