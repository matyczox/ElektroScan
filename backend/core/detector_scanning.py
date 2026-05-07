"""Template scanning phase for the detector pipeline."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import cv2
import numpy as np

from core import detector_gray as gray_strategy
from core.detector_config import (
    DETECTOR_SCAN_MAX_WORKERS,
    LABEL_CONTENT_SCAN_MIN_PIXELS,
    MAX_PEAKS_PER_VARIANT,
    MAX_TEXT_CONTENT_PEAKS_PER_VARIANT,
    MIN_TEMPLATE_PIXELS,
    TEXT_CONTENT_THRESHOLD,
    THRESHOLD_DILATED,
    THRESHOLD_PRECISE,
)
from core.detector_masks import _find_local_maxima, _tight_mask_crop
from core.detector_models import CandidateHit, TemplateInfo, TemplateVariant


@dataclass(slots=True)
class ScanResult:
    raw_template_hits: list[CandidateHit]
    scan_workers: int
    skipped_empty_color_masks: int
    timing_seconds: float
    raw_hits_by_mask_kind: dict[str, int]


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
        px, py, _score = peak
        if all(abs(px - sx) >= min_dx or abs(py - sy) >= min_dy for sx, sy, _ in selected):
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


def scan_template_candidates(
    *,
    templates: list[TemplateInfo],
    variants_by_template: dict[int, list[TemplateVariant]],
    scan_masks_by_template: dict[int, np.ndarray],
    scan_mask_kinds_by_template: dict[int, str] | None = None,
    search_rois_by_template: dict[int, list[tuple[int, int, int, int]]],
    plan_mask_foregrounds: dict[int, int],
    detector_profile: str,
    progress_callback: Callable[[str, float, str], None],
) -> ScanResult:
    """Run matchTemplate over prepared scan masks and ROIs."""

    def _scan_template(template_id: int) -> list[CandidateHit]:
        template = templates[template_id]
        threshold = THRESHOLD_PRECISE if template.requires_precision else THRESHOLD_DILATED
        if detector_profile == "gray":
            threshold = gray_strategy.gray_scan_threshold(template, threshold)

        scan_mask = scan_masks_by_template[template_id]
        scan_mask_kind = (
            scan_mask_kinds_by_template.get(template_id, "unknown")
            if scan_mask_kinds_by_template is not None
            else "unknown"
        )
        search_rois = search_rois_by_template.get(template_id, [])
        if plan_mask_foregrounds.get(template_id, 0) < MIN_TEMPLATE_PIXELS or not search_rois:
            return []

        spatial_fair_peaks = (
            detector_profile == "gray" and gray_strategy.use_gray_spatial_fair_peaks(template)
        )

        template_hits: list[CandidateHit] = []
        for variant in variants_by_template.get(template_id, []):
            if detector_profile == "gray" and not gray_strategy.should_scan_gray_variant(
                template,
                variant.scale,
                scan_mask_kind,
            ):
                continue
            if variant.height > scan_mask.shape[0] or variant.width > scan_mask.shape[1]:
                continue

            variant_peaks: list[tuple[int, int, float]] = []
            too_many_peaks = False
            for roi_x, roi_y, roi_w, roi_h in search_rois:
                if variant.height > roi_h or variant.width > roi_w:
                    continue

                roi_plan_mask = scan_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
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
                if (
                    spatial_fair_peaks
                    and len(peaks) > gray_strategy.gray_spatial_fair_peaks_per_roi()
                ):
                    peaks = _select_spatially_fair_peaks(
                        peaks,
                        limit=gray_strategy.gray_spatial_fair_peaks_per_roi(),
                        template_width=variant.width,
                        template_height=variant.height,
                    )
                if peaks:
                    variant_peaks.extend((roi_x + px, roi_y + py, score) for px, py, score in peaks)
                if not spatial_fair_peaks and len(variant_peaks) > MAX_PEAKS_PER_VARIANT:
                    too_many_peaks = True
                    break

            if too_many_peaks:
                variant_peaks.sort(key=lambda item: item[2], reverse=True)
                variant_peaks = variant_peaks[:MAX_PEAKS_PER_VARIANT]

            for px, py, score in variant_peaks:
                template_hits.append(
                    CandidateHit(
                        template_id=template_id,
                        scale=variant.scale,
                        rotation=variant.rotation,
                        mirrored=variant.mirrored,
                        transformed_mask=variant.transformed_mask,
                        content_mask=variant.content_mask,
                        pixel_count=variant.pixel_count,
                        content_pixel_count=variant.content_pixel_count,
                        content_bbox=variant.content_bbox,
                        bbox=(px, py, variant.width, variant.height),
                        match_score=score,
                        dominant_hsv=None if detector_profile == "gray" else template.dominant_hsv,
                        source="template",
                        is_text_label=template.is_text_label,
                    )
                )

            if (
                detector_profile == "gray"
                or not template.is_text_label
                or not _is_content_scan_eligible(variant)
            ):
                continue

            content_crop = _tight_mask_crop(variant.content_mask)
            if content_crop is None or variant.content_bbox is None:
                continue

            content_x, content_y, content_w, content_h = variant.content_bbox
            content_peaks: list[tuple[int, int, float]] = []
            too_many_content_peaks = False
            for roi_x, roi_y, roi_w, roi_h in search_rois:
                if content_h > roi_h or content_w > roi_w:
                    continue

                roi_plan_mask = scan_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                match_result = cv2.matchTemplate(
                    roi_plan_mask,
                    content_crop,
                    cv2.TM_CCOEFF_NORMED,
                )
                peaks = _find_local_maxima(
                    match_result,
                    threshold=TEXT_CONTENT_THRESHOLD,
                    template_width=content_w,
                    template_height=content_h,
                )
                if peaks:
                    content_peaks.extend((roi_x + px, roi_y + py, score) for px, py, score in peaks)
                if len(content_peaks) > MAX_TEXT_CONTENT_PEAKS_PER_VARIANT:
                    too_many_content_peaks = True
                    break

            if too_many_content_peaks:
                content_peaks.sort(key=lambda item: item[2], reverse=True)
                content_peaks = content_peaks[:MAX_TEXT_CONTENT_PEAKS_PER_VARIANT]

            for px, py, score in content_peaks:
                template_hits.append(
                    CandidateHit(
                        template_id=template_id,
                        scale=variant.scale,
                        rotation=variant.rotation,
                        mirrored=variant.mirrored,
                        transformed_mask=variant.content_mask,
                        content_mask=variant.content_mask,
                        pixel_count=variant.content_pixel_count,
                        content_pixel_count=variant.content_pixel_count,
                        content_bbox=variant.content_bbox,
                        bbox=(px - content_x, py - content_y, variant.width, variant.height),
                        match_score=score,
                        dominant_hsv=template.dominant_hsv,
                        source="template_content",
                        is_text_label=True,
                    )
                )

        return template_hits

    template_ids_to_scan = [
        template_id
        for template_id in variants_by_template
        if plan_mask_foregrounds.get(template_id, 0) >= MIN_TEMPLATE_PIXELS
    ]
    skipped_empty_color_masks = len(variants_by_template) - len(template_ids_to_scan)

    raw_template_hits: list[CandidateHit] = []
    raw_hits_by_mask_kind: dict[str, int] = {}
    phase_start = time.perf_counter()
    scan_workers = max(1, min(len(template_ids_to_scan), DETECTOR_SCAN_MAX_WORKERS))
    if template_ids_to_scan:
        completed_scans = 0
        total_scans = max(1, len(template_ids_to_scan))
        with ThreadPoolExecutor(max_workers=scan_workers) as pool:
            for hits in pool.map(_scan_template, template_ids_to_scan):
                raw_template_hits.extend(hits)
                for hit in hits:
                    mask_kind = (
                        scan_mask_kinds_by_template.get(hit.template_id, "raw")
                        if scan_mask_kinds_by_template is not None
                        else "raw"
                    )
                    raw_hits_by_mask_kind[mask_kind] = raw_hits_by_mask_kind.get(mask_kind, 0) + 1
                completed_scans += 1
                if (
                    completed_scans == 1
                    or completed_scans % 2 == 0
                    or completed_scans == total_scans
                ):
                    progress_callback(
                        "scan",
                        25 + 35 * completed_scans / total_scans,
                        f"Skan template {completed_scans}/{total_scans}",
                    )
    else:
        scan_workers = 0

    return ScanResult(
        raw_template_hits=raw_template_hits,
        scan_workers=scan_workers,
        skipped_empty_color_masks=skipped_empty_color_masks,
        timing_seconds=time.perf_counter() - phase_start,
        raw_hits_by_mask_kind=raw_hits_by_mask_kind,
    )
