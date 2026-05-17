"""Validation-time mask cache and shared validation evidence helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from core.detector_config import (
    CONTEXT_MARGIN_RATIO,
    GRAY_RECT_FRAME_LONG_EDGE_MIN_RUN,
    GRAY_RECT_FRAME_MAX_CENTER_DENSITY,
    GRAY_RECT_FRAME_MAX_DENSITY,
    GRAY_RECT_FRAME_MIN_ASPECT,
    GRAY_RECT_FRAME_MIN_DENSITY,
    GRAY_RECT_FRAME_SHORT_EDGE_STRONG_RUN,
    GRAY_RECT_FRAME_SHORT_EDGE_WEAK_RUN,
    GRAY_RECT_FRAME_STRONG_EDGE_COVERAGE,
    GRAY_RECT_FRAME_WEAK_EDGE_COVERAGE,
)
from core.detector_models import CandidateHit
from core.detector_shape_metrics import (
    _mask_bbox,
    _mask_centroid,
    _roi_mask,
    _thickness_normalized_mask,
)


@dataclass(slots=True)
class ValidationMaskCache:
    """Per-validation-run cache for template-mask derived metrics."""

    normalized_masks: dict[int, np.ndarray]
    normalized_pixels: dict[int, int]
    centroids: dict[int, tuple[float, float] | None]
    content_bboxes: dict[int, tuple[int, int, int, int] | None]
    foreground_integrals: dict[int, np.ndarray]

    @classmethod
    def build(
        cls,
        hits: list[CandidateHit],
        plan_masks: Iterable[np.ndarray] = (),
    ) -> "ValidationMaskCache":
        transformed_masks: dict[int, np.ndarray] = {}
        content_masks: dict[int, np.ndarray] = {}
        for hit in hits:
            if hit.transformed_mask is not None:
                transformed_masks[id(hit.transformed_mask)] = hit.transformed_mask
            if hit.content_mask is not None:
                content_masks[id(hit.content_mask)] = hit.content_mask

        foreground_integrals: dict[int, np.ndarray] = {}
        for mask in plan_masks:
            if not isinstance(mask, np.ndarray):
                continue
            mask_id = id(mask)
            if mask_id in foreground_integrals:
                continue
            foreground_integrals[mask_id] = cv2.integral(
                (mask > 0).astype(np.uint8, copy=False),
                sdepth=cv2.CV_32S,
            )

        normalized_masks: dict[int, np.ndarray] = {}
        normalized_pixels: dict[int, int] = {}
        centroids: dict[int, tuple[float, float] | None] = {}
        for mask_id, mask in transformed_masks.items():
            normalized = _thickness_normalized_mask(mask)
            normalized_masks[mask_id] = normalized
            normalized_pixels[mask_id] = max(1, int(cv2.countNonZero(normalized)))
            centroids[mask_id] = _mask_centroid(mask)

        return cls(
            normalized_masks=normalized_masks,
            normalized_pixels=normalized_pixels,
            centroids=centroids,
            content_bboxes={
                mask_id: _mask_bbox(mask)
                for mask_id, mask in content_masks.items()
            },
            foreground_integrals=foreground_integrals,
        )

    def normalized_mask(self, mask: np.ndarray) -> np.ndarray:
        cached = self.normalized_masks.get(id(mask))
        if cached is not None:
            return cached
        return _thickness_normalized_mask(mask)

    def normalized_pixel_count(self, mask: np.ndarray) -> int:
        cached = self.normalized_pixels.get(id(mask))
        if cached is not None:
            return cached
        return max(1, int(cv2.countNonZero(self.normalized_mask(mask))))

    def centroid(self, mask: np.ndarray) -> tuple[float, float] | None:
        cached = self.centroids.get(id(mask))
        if id(mask) in self.centroids:
            return cached
        return _mask_centroid(mask)

    def content_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        cached = self.content_bboxes.get(id(mask))
        if id(mask) in self.content_bboxes:
            return cached
        return _mask_bbox(mask)

    def foreground_count(
        self,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> int:
        integral = self.foreground_integrals.get(id(mask))
        if integral is None:
            roi = _roi_mask(mask, bbox)
            return 0 if roi is None else int(cv2.countNonZero(roi))

        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
        return int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def _is_gray_rect_frame_candidate(hit: CandidateHit) -> bool:
    if hit.transformed_mask is None:
        return False

    width, height = hit.bbox[2], hit.bbox[3]
    aspect = max(width / max(1, height), height / max(1, width))
    density = hit.pixel_count / max(1, width * height)
    return (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and aspect >= GRAY_RECT_FRAME_MIN_ASPECT
        and GRAY_RECT_FRAME_MIN_DENSITY <= density <= GRAY_RECT_FRAME_MAX_DENSITY
    )


def _gray_rect_frame_evidence_ok(roi: np.ndarray, template_mask: np.ndarray) -> bool:
    """Check that a hollow frame has real ink on its perimeter and an empty middle."""

    height, width = template_mask.shape[:2]
    band = max(2, min(height, width) // 5)
    intersection = cv2.bitwise_and(roi, template_mask)
    edge_slices = (
        (slice(0, band), slice(None)),
        (slice(height - band, height), slice(None)),
        (slice(None), slice(0, band)),
        (slice(None), slice(width - band, width)),
    )

    edge_coverages: list[float] = []
    for edge_slice in edge_slices:
        template_pixels = cv2.countNonZero(template_mask[edge_slice])
        intersection_pixels = cv2.countNonZero(intersection[edge_slice])
        edge_coverages.append(intersection_pixels / max(1, template_pixels))

    inner = roi[band : height - band, band : width - band]
    center_density = cv2.countNonZero(inner) / max(1, inner.size) if inner.size else 0.0

    def max_run(values: np.ndarray) -> int:
        best = 0
        current = 0
        for value in values.astype(bool, copy=False).tolist():
            if value:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    top_run = max_run((intersection[:band, :] > 0).any(axis=0)) / max(1, width)
    bottom_run = max_run((intersection[height - band : height, :] > 0).any(axis=0)) / max(1, width)
    left_run = max_run((intersection[:, :band] > 0).any(axis=1)) / max(1, height)
    right_run = max_run((intersection[:, width - band : width] > 0).any(axis=1)) / max(1, height)

    if width >= height:
        long_edges = (top_run, bottom_run)
        short_edges = (left_run, right_run)
    else:
        long_edges = (left_run, right_run)
        short_edges = (top_run, bottom_run)

    continuous_frame = (
        min(long_edges) >= GRAY_RECT_FRAME_LONG_EDGE_MIN_RUN
        and max(short_edges) >= GRAY_RECT_FRAME_SHORT_EDGE_STRONG_RUN
        and min(short_edges) >= GRAY_RECT_FRAME_SHORT_EDGE_WEAK_RUN
    )

    return (
        sum(score >= GRAY_RECT_FRAME_STRONG_EDGE_COVERAGE for score in edge_coverages) >= 3
        and all(score >= GRAY_RECT_FRAME_WEAK_EDGE_COVERAGE for score in edge_coverages)
        and center_density <= GRAY_RECT_FRAME_MAX_CENTER_DENSITY
        and continuous_frame
    )


def _context_purity(
    plan_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    intersection_mask: np.ndarray,
    *,
    explained_pixels: int | None = None,
    validation_cache: ValidationMaskCache | None = None,
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
