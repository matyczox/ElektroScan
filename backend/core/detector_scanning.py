"""Template scanning phase for the detector pipeline."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np

from core import detector_gray as gray_strategy
from core.detector_config import (
    DETECTOR_SCAN_MAX_WORKERS,
    GRAY_SCAN_MAX_WORKERS,
    LABEL_CONTENT_SCAN_MIN_PIXELS,
    MAX_PEAKS_PER_VARIANT,
    MAX_TEXT_CONTENT_PEAKS_PER_VARIANT,
    MIN_TEMPLATE_PIXELS,
    OPENCV_NUM_THREADS,
    GRAY_SCAN_MIN_ROI_FOREGROUND_RATIO,
    TEXT_CONTENT_THRESHOLD,
    THRESHOLD_DILATED,
    THRESHOLD_PRECISE,
    _safe_cpu_count,
)
from core.detector_diagnostics import aggregate_scan_profile, scan_roi_bucket
from core.detector_masks import _find_local_maxima, _tight_mask_crop
from core.detector_models import CandidateHit, TemplateInfo, TemplateVariant


@dataclass(slots=True)
class ScanResult:
    raw_template_hits: list[CandidateHit]
    scan_workers: int
    skipped_empty_color_masks: int
    timing_seconds: float
    raw_hits_by_mask_kind: dict[str, int]
    scan_strategy: str
    opencv_threads: int
    configured_scan_workers: int
    scan_tasks: int
    scan_task_rois: int
    scan_profile: dict


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
    """Return true for label templates whose marker can move around text.

    Large framed/text-like symbols in the gray Viking fixtures validate through
    the full template. Running an extra content-only scan for every rotation and
    scale creates tens of thousands of matchTemplate calls and many duplicates.
    F-like labels, however, need content-only rescue because their direction
    marker can appear above/below the text on the plan.
    """

    if not template.is_text_label or template.content_bbox is None:
        return False
    height, width = template.mask.shape[:2]
    _x, y, content_width, content_height = template.content_bbox
    return (
        y >= height * 0.25
        and content_width >= width * 0.55
        and content_height >= height * 0.45
    )


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
    roi_strategies_by_template: dict[int, str] | None = None,
    search_rois_by_template: dict[int, list[tuple[int, int, int, int]]],
    plan_mask_foregrounds: dict[int, int],
    detector_profile: str,
    progress_callback: Callable[[str, float, str], None],
    collect_profile: bool = False,
) -> ScanResult:
    """Run matchTemplate over prepared scan masks and ROIs."""

    scan_profile_records: list[dict] = []
    scan_profile_lock = Lock()
    foreground_integrals_by_mask_id: dict[int, np.ndarray] = {}
    foreground_integral_lock = Lock()

    def _record_roi_scan(stats: dict, roi_w: int, roi_h: int, output_pixels: int) -> str:
        pixels = int(roi_w) * int(roi_h)
        stats["calls"] += 1
        stats["pixels"] += pixels
        stats["outputPixels"] += int(output_pixels)
        bucket = scan_roi_bucket(roi_w, roi_h)
        bucket_stats = stats["roiBuckets"].setdefault(
            bucket,
            {"calls": 0, "pixels": 0, "outputPixels": 0, "rawPeaks": 0},
        )
        bucket_stats["calls"] += 1
        bucket_stats["pixels"] += pixels
        bucket_stats["outputPixels"] += int(output_pixels)
        return bucket

    def _foreground_integral_for_scan_mask(scan_mask: np.ndarray) -> np.ndarray:
        key = id(scan_mask)
        cached = foreground_integrals_by_mask_id.get(key)
        if cached is not None:
            return cached

        foreground_mask = (scan_mask > 0).astype(np.uint8, copy=False)
        foreground_integral = cv2.integral(foreground_mask, sdepth=cv2.CV_32S)
        with foreground_integral_lock:
            return foreground_integrals_by_mask_id.setdefault(key, foreground_integral)

    def _scan_variant(
        template_id: int,
        template: TemplateInfo,
        variant: TemplateVariant,
        threshold: float,
        scan_mask: np.ndarray,
        scan_mask_kind: str,
        roi_strategy: str,
        search_rois: list[tuple[int, int, int, int, int]],
        spatial_fair_peaks: bool,
    ) -> list[CandidateHit]:
        """Scan one prepared variant across all ROIs.

        Variant-level jobs keep all CPU cores busy even when a PDF has only a
        handful of templates, while preserving the same candidate logic.
        """

        template_hits: list[CandidateHit] = []
        stats = (
            {
                "templateId": int(template_id),
                "templateName": Path(str(template.path)).name,
                "scale": round(float(variant.scale), 3),
                "rotation": int(variant.rotation),
                "mirrored": bool(variant.mirrored),
                "maskKind": scan_mask_kind,
                "calls": 0,
                "pixels": 0,
                "outputPixels": 0,
                "rawPeaks": 0,
                "contentCalls": 0,
                "contentRawPeaks": 0,
                "emittedHits": 0,
                "roiBuckets": {},
            }
            if collect_profile
            else None
        )

        def _finish() -> list[CandidateHit]:
            if stats is not None:
                stats["emittedHits"] = len(template_hits)
                with scan_profile_lock:
                    scan_profile_records.append(stats)
            return template_hits

        variant_peaks: list[tuple[int, int, float]] = []
        too_many_peaks = False
        min_roi_foreground = (
            variant.pixel_count * float(GRAY_SCAN_MIN_ROI_FOREGROUND_RATIO)
            if detector_profile == "gray"
            else 0.0
        )
        for roi_x, roi_y, roi_w, roi_h, roi_foreground in search_rois:
            if variant.height > roi_h or variant.width > roi_w:
                continue
            if roi_foreground < min_roi_foreground:
                continue

            roi_plan_mask = scan_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            match_result = cv2.matchTemplate(
                roi_plan_mask,
                variant.transformed_mask,
                cv2.TM_CCOEFF_NORMED,
            )
            roi_bucket = ""
            if stats is not None:
                roi_bucket = _record_roi_scan(stats, roi_w, roi_h, int(match_result.size))
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
                if stats is not None:
                    stats["rawPeaks"] += len(peaks)
                    stats["roiBuckets"][roi_bucket]["rawPeaks"] += len(peaks)
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
                    roi_strategy=roi_strategy,
                )
            )

        if (
            not _needs_directional_text_content_scan(template)
            or not _is_content_scan_eligible(variant)
        ):
            return _finish()

        content_crop = _tight_mask_crop(variant.content_mask)
        if content_crop is None or variant.content_bbox is None:
            return _finish()

        content_x, content_y, content_w, content_h = variant.content_bbox
        content_peaks: list[tuple[int, int, float]] = []
        too_many_content_peaks = False
        min_content_roi_foreground = (
            variant.content_pixel_count * float(GRAY_SCAN_MIN_ROI_FOREGROUND_RATIO)
            if detector_profile == "gray"
            else 0.0
        )
        for roi_x, roi_y, roi_w, roi_h, roi_foreground in search_rois:
            if content_h > roi_h or content_w > roi_w:
                continue
            if roi_foreground < min_content_roi_foreground:
                continue

            roi_plan_mask = scan_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            match_result = cv2.matchTemplate(
                roi_plan_mask,
                content_crop,
                cv2.TM_CCOEFF_NORMED,
            )
            roi_bucket = ""
            if stats is not None:
                roi_bucket = _record_roi_scan(stats, roi_w, roi_h, int(match_result.size))
                stats["contentCalls"] += 1
            peaks = _find_local_maxima(
                match_result,
                threshold=TEXT_CONTENT_THRESHOLD,
                template_width=content_w,
                template_height=content_h,
            )
            if peaks:
                if stats is not None:
                    stats["rawPeaks"] += len(peaks)
                    stats["contentRawPeaks"] += len(peaks)
                    stats["roiBuckets"][roi_bucket]["rawPeaks"] += len(peaks)
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
                    dominant_hsv=None if detector_profile == "gray" else template.dominant_hsv,
                    source="template_content",
                    is_text_label=True,
                    roi_strategy=roi_strategy,
                )
            )

        return _finish()

    def _scan_template_variants(template_id: int) -> list[
        tuple[
            int,
            TemplateInfo,
            TemplateVariant,
            float,
            np.ndarray,
            str,
            str,
            list[tuple[int, int, int, int, int]],
            bool,
        ]
    ]:
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
        roi_strategy = (
            roi_strategies_by_template.get(template_id, "")
            if roi_strategies_by_template is not None
            else ""
        )
        search_rois = search_rois_by_template.get(template_id, [])
        if plan_mask_foregrounds.get(template_id, 0) < MIN_TEMPLATE_PIXELS or not search_rois:
            return []

        foreground_integral = _foreground_integral_for_scan_mask(scan_mask)
        search_rois_with_foreground = []
        for roi_x, roi_y, roi_w, roi_h in search_rois:
            roi_foreground = int(
                foreground_integral[roi_y + roi_h, roi_x + roi_w]
                - foreground_integral[roi_y, roi_x + roi_w]
                - foreground_integral[roi_y + roi_h, roi_x]
                + foreground_integral[roi_y, roi_x]
            )
            search_rois_with_foreground.append((roi_x, roi_y, roi_w, roi_h, roi_foreground))

        spatial_fair_peaks = (
            detector_profile == "gray" and gray_strategy.use_gray_spatial_fair_peaks(template)
        )

        tasks: list[
            tuple[
                int,
                TemplateInfo,
                TemplateVariant,
                float,
                np.ndarray,
                str,
                str,
                list[tuple[int, int, int, int, int]],
                bool,
            ]
        ] = []
        for variant in variants_by_template.get(template_id, []):
            if detector_profile == "gray" and not gray_strategy.should_scan_gray_variant(
                template,
                variant.scale,
                scan_mask_kind,
            ):
                continue
            if variant.height > scan_mask.shape[0] or variant.width > scan_mask.shape[1]:
                continue
            tasks.append(
                (
                    template_id,
                    template,
                    variant,
                    threshold,
                    scan_mask,
                    scan_mask_kind,
                    roi_strategy,
                    search_rois_with_foreground,
                    spatial_fair_peaks,
                )
            )

        return tasks

    def _scan_template(template_id: int) -> list[CandidateHit]:
        """Scan all variants for one template in the current worker."""

        template_hits: list[CandidateHit] = []
        for task in _scan_template_variants(template_id):
            template_hits.extend(_scan_variant(*task))
        return template_hits

    template_ids_to_scan = [
        template_id
        for template_id in variants_by_template
        if plan_mask_foregrounds.get(template_id, 0) >= MIN_TEMPLATE_PIXELS
    ]
    skipped_empty_color_masks = len(variants_by_template) - len(template_ids_to_scan)

    variant_scan_tasks = [
        task
        for template_id in template_ids_to_scan
        for task in _scan_template_variants(template_id)
    ]

    raw_template_hits: list[CandidateHit] = []
    raw_hits_by_mask_kind: dict[str, int] = {}
    phase_start = time.perf_counter()
    use_template_tasks = False
    max_scan_workers = (
        GRAY_SCAN_MAX_WORKERS if detector_profile == "gray" else DETECTOR_SCAN_MAX_WORKERS
    )
    if use_template_tasks:
        scan_items = template_ids_to_scan
        scan_workers = max(1, min(len(scan_items), _safe_cpu_count(), max_scan_workers))
        scan_label = "template"
        scan_fn = _scan_template
        scan_strategy = "template"
        active_opencv_threads = 1
        scan_task_rois = sum(len(search_rois_by_template.get(template_id, [])) for template_id in scan_items)
    else:
        scan_items = variant_scan_tasks
        scan_workers = max(1, min(len(scan_items), max_scan_workers))
        scan_label = "wariantow"
        scan_fn = lambda args: _scan_variant(*args)
        scan_strategy = "variant"
        active_opencv_threads = OPENCV_NUM_THREADS
        scan_task_rois = sum(len(task[7]) for task in variant_scan_tasks)

    if scan_items:
        completed_scans = 0
        total_scans = max(1, len(scan_items))
        progress_step = max(1, total_scans // 20)
        previous_opencv_threads = cv2.getNumThreads()
        try:
            cv2.setNumThreads(active_opencv_threads)
            with ThreadPoolExecutor(max_workers=scan_workers) as pool:
                for hits in pool.map(scan_fn, scan_items):
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
                        or completed_scans % progress_step == 0
                        or completed_scans == total_scans
                    ):
                        progress_callback(
                            "scan",
                            25 + 35 * completed_scans / total_scans,
                            f"Skan {scan_label} {completed_scans}/{total_scans}",
                        )
        finally:
            cv2.setNumThreads(previous_opencv_threads)
    else:
        scan_workers = 0
        scan_strategy = "none"
        active_opencv_threads = OPENCV_NUM_THREADS

    return ScanResult(
        raw_template_hits=raw_template_hits,
        scan_workers=scan_workers,
        skipped_empty_color_masks=skipped_empty_color_masks,
        timing_seconds=time.perf_counter() - phase_start,
        raw_hits_by_mask_kind=raw_hits_by_mask_kind,
        scan_strategy=scan_strategy,
        opencv_threads=active_opencv_threads,
        configured_scan_workers=max_scan_workers,
        scan_tasks=len(scan_items),
        scan_task_rois=scan_task_rois,
        scan_profile=aggregate_scan_profile(scan_profile_records) if collect_profile else {},
    )
