"""
detector.py - CPU-friendly symbol detection for electrical plans.

This module now orchestrates the pipeline. The heavy helpers live in sibling
modules so detector behavior stays easier to audit and tune.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import os

import cv2
import numpy as np

from core import detector_color as color_strategy
from core import detector_gray as gray_strategy
from core.detector_clustering import (
    _cluster_candidates,
    _prefilter_candidates,
    _prefilter_raw_template_hits,
)
from core.detector_config import (
    DEFAULT_PDF_DPI,
    DETECTOR_POSTPROCESS_MAX_WORKERS,
    DETECTOR_SCAN_MAX_WORKERS,
    GRAY_DARK_EVIDENCE_THRESHOLD,
    GRAY_DARK_INK_THRESHOLD,
    GRAY_DARK_ZONE_THRESHOLD,
    GRAY_SCALES,
    GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
    GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    GRAY_TINY_FRAGMENT_MAX_DIMENSION,
    GRAY_TINY_FRAGMENT_MAX_SCALE,
    OPENCV_NUM_THREADS,
    SCALES,
    _safe_cpu_count,
)
from core.detector_diagnostics import (
    build_candidate_stage_counts,
    build_hit_flow_profile,
    build_roi_strategy_profile,
)
from core.detector_masks import (
    _build_search_rois,
    _hsv_mask,
    _ink_mask,
)
from core.detector_models import (
    CandidateHit,
    Detection,
    DetectionResult,
    TemplateInfo,
)
from core.detector_pdf import (
    _collect_pdf_text_exclude_rects,
    _collect_pdf_text_hits,
    _estimate_legend_exclude_rect,
    _estimate_title_block_exclude_rects,
)
from core.detector_parent_search import search_parent_candidates
from core.detector_scanning import scan_template_candidates
from core.detector_templates import _build_socket_07_promotions, _prepare_variants
from core.detector_validation import validate_template_candidates


def _detect_symbols_pipeline(
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    subtract_legend: bool = True,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
    pdf_path: str | None = None,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
    debug_profile: dict | None = None,
    detector_profile: str = "color",
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> list[DetectionResult]:
    """
    Detect symbols on a rendered plan using template matching plus PDF-text fallback.
    """

    exclude_rects = list(exclude_rects or [])

    if not templates:
        return []

    timings: dict[str, float] = {}

    def _progress(stage: str, percent: float, detail: str = "") -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, max(0.0, min(100.0, float(percent))), detail)
        except Exception:
            pass

    detector_profile = detector_profile if detector_profile in {"color", "gray"} else "color"
    initial_debug_profile = dict(debug_profile or {})
    collect_performance_profile = bool(initial_debug_profile.get("performanceProfile"))
    ablation_value = str(
        initial_debug_profile.get("ablation") or os.getenv("ELEKTROSCAN_ABLATION", "")
    ).strip().lower()
    ablation_no_text_mirror = ablation_value in {
        "no-text-mirror",
        "no_text_mirror",
        "notextmirror",
    } or os.getenv("ELEKTROSCAN_ABLATION_NO_TEXT_MIRROR", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    gray_text_mirror_override = os.getenv("ELEKTROSCAN_GRAY_TEXT_MIRROR", "").strip().lower()
    gray_force_text_mirror = ablation_value in {
        "text-mirror",
        "text_mirror",
        "with-text-mirror",
        "with_text_mirror",
    } or gray_text_mirror_override in {"1", "true", "yes", "on"}
    disable_text_mirror = ablation_no_text_mirror or (
        detector_profile == "gray" and not gray_force_text_mirror
    )

    trace_input = initial_debug_profile.get("candidateTrace") or initial_debug_profile.get("trace") or {}
    if not isinstance(trace_input, dict):
        trace_input = {}

    def _trace_values(value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(part).strip() for part in value if str(part).strip()]
        return [str(value).strip()] if str(value).strip() else []

    def _trace_points(value: object) -> list[tuple[float, float]]:
        raw_values: list[object]
        if value is None:
            raw_values = []
        elif isinstance(value, str):
            raw_values = [part.strip() for part in value.split(";") if part.strip()]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]

        points: list[tuple[float, float]] = []
        for item in raw_values:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                points.append((float(item[0]), float(item[1])))
                continue
            parts = str(item).replace(":", ",").split(",")
            if len(parts) >= 2:
                points.append((float(parts[0].strip()), float(parts[1].strip())))
        return points

    trace_symbols = set(_trace_values(trace_input.get("symbols") or trace_input.get("symbol")))
    trace_symbols.update(_trace_values(os.getenv("ELEKTROSCAN_TRACE_SYMBOLS")))
    trace_points = _trace_points(trace_input.get("points") or trace_input.get("point"))
    trace_points.extend(_trace_points(os.getenv("ELEKTROSCAN_TRACE_POINTS")))
    trace_radius = float(trace_input.get("radius") or os.getenv("ELEKTROSCAN_TRACE_RADIUS", 80))
    trace_max_items = int(trace_input.get("maxItems") or os.getenv("ELEKTROSCAN_TRACE_MAX_ITEMS", 40))
    candidate_trace_enabled = bool(trace_symbols or trace_points)
    candidate_trace: dict[str, dict] = {}

    def _trace_symbol_matches(symbol_name: str) -> bool:
        if not trace_symbols:
            return True
        return any(
            symbol_name == requested
            or symbol_name.startswith(f"{requested}_")
            or symbol_name.startswith(requested)
            for requested in trace_symbols
        )

    def _trace_box_distance(bbox: tuple[int, int, int, int]) -> float:
        if not trace_points:
            return 0.0
        x, y, w, h = bbox
        best = float("inf")
        for px, py in trace_points:
            dx = max(float(x) - px, 0.0, px - float(x + w))
            dy = max(float(y) - py, 0.0, py - float(y + h))
            best = min(best, float(np.hypot(dx, dy)))
        return best

    def _record_candidate_trace(
        stage: str,
        hits: list[CandidateHit],
        reason_by_id: dict[int, str] | None = None,
    ) -> None:
        if not candidate_trace_enabled:
            return

        matched: list[tuple[float, CandidateHit]] = []
        for hit in hits:
            if not (0 <= hit.template_id < len(templates)):
                continue
            symbol_name = templates[hit.template_id].name
            if not _trace_symbol_matches(symbol_name):
                continue
            distance = _trace_box_distance(hit.bbox)
            if trace_points and distance > trace_radius:
                continue
            matched.append((distance, hit))

        matched.sort(
            key=lambda item: (
                item[0],
                -float(item[1].verification_score),
                -float(item[1].match_score),
            )
        )
        items = []
        reason_by_id = reason_by_id or {}
        for distance, hit in matched[:trace_max_items]:
            x, y, w, h = hit.bbox
            item = {
                "symbolName": templates[hit.template_id].name,
                "templateId": int(hit.template_id),
                "bbox": [int(x), int(y), int(w), int(h)],
                "match": round(float(hit.match_score), 3),
                "verification": round(float(hit.verification_score), 3),
                "coverage": round(float(hit.coverage), 3),
                "purity": round(float(hit.purity), 3),
                "context": round(float(hit.context_purity), 3),
                "contentScore": round(float(hit.content_score), 3),
                "pixelCount": int(hit.pixel_count),
                "scale": round(float(hit.scale), 3),
                "rotation": int(hit.rotation),
                "mirrored": bool(hit.mirrored),
                "source": str(hit.source),
                "isTextLabel": bool(hit.is_text_label),
                "roiStrategy": str(hit.roi_strategy),
                "distance": round(float(distance), 3),
            }
            reason = reason_by_id.get(id(hit))
            if reason:
                item["reason"] = reason
            items.append(item)
        candidate_trace[stage] = {
            "totalCandidates": int(len(hits)),
            "matchedCandidates": int(len(matched)),
            "items": items,
        }

    legend_rect = _estimate_legend_exclude_rect(
        pdf_path=pdf_path or "",
        image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
    )
    if legend_rect is not None:
        exclude_rects.append(legend_rect)

    gray_text_exclude_rects: list[tuple[int, int, int, int]] = []
    gray_title_exclude_rects: list[tuple[int, int, int, int]] = []
    if detector_profile == "gray":
        gray_text_exclude_rects = _collect_pdf_text_exclude_rects(
            pdf_path=pdf_path or "",
            image_shape=plan_image.shape,
            dpi=pdf_dpi,
            hidden_layers=hidden_layers,
        )
        gray_title_exclude_rects = _estimate_title_block_exclude_rects(
            pdf_path=pdf_path or "",
            image_shape=plan_image.shape,
            dpi=pdf_dpi,
            hidden_layers=hidden_layers,
        )
        exclude_rects.extend(gray_title_exclude_rects)

    color_masks_cache: dict[str, np.ndarray] = {}
    ink_mask_cache: np.ndarray | None = None
    empty_plan_mask_cache: np.ndarray | None = None

    def _get_ink_plan_mask(*, dilate: bool) -> np.ndarray:
        nonlocal ink_mask_cache
        if ink_mask_cache is None:
            ink_mask_cache = _ink_mask(plan_image, dilate=False)
            for ex, ey, ew, eh in exclude_rects:
                cv2.rectangle(ink_mask_cache, (ex, ey), (ex + ew, ey + eh), 0, -1)
        return cv2.dilate(ink_mask_cache, np.ones((3, 3), np.uint8), iterations=1) if dilate else ink_mask_cache

    def _get_empty_plan_mask() -> np.ndarray:
        nonlocal empty_plan_mask_cache
        if empty_plan_mask_cache is None:
            empty_plan_mask_cache = np.zeros(plan_image.shape[:2], dtype=np.uint8)
        return empty_plan_mask_cache

    def _get_plan_mask(template: TemplateInfo) -> np.ndarray:
        if detector_profile == "gray":
            return _get_ink_plan_mask(dilate=True)

        if template.dominant_hsv is None:
            return _get_empty_plan_mask()

        if template.dominant_hsv is not None:
            cache_key = color_strategy.color_mask_cache_key(template)
            if cache_key not in color_masks_cache:
                mask = color_strategy.build_color_plan_mask(
                    plan_image=plan_image,
                    plan_hsv=plan_hsv,
                    template=template,
                    exclude_rects=exclude_rects,
                )
                if cache_key is not None:
                    color_masks_cache[cache_key] = mask
                return mask
            return color_masks_cache[cache_key]

        fallback = _hsv_mask(plan_image, dilate=False, hsv_image=plan_hsv)
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(fallback, (ex, ey), (ex + ew, ey + eh), 0, -1)
        return fallback

    phase_start = time.perf_counter()
    _progress("pdf_text", 12, "Odczyt pomocniczych tekstow PDF")
    pdf_hits_by_template = _collect_pdf_text_hits(
        pdf_path=pdf_path or "",
        templates=templates,
        plan_image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
        exclude_rects=exclude_rects,
    )
    pdf_candidates = [hit for hits in pdf_hits_by_template.values() for hit in hits]
    timings["pdf_text"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    _progress("prepare", 18, "Przygotowanie masek i wariantow")
    plan_hsv = cv2.cvtColor(plan_image, cv2.COLOR_BGR2HSV) if detector_profile == "color" else None

    variant_workers = max(1, min(len(templates), DETECTOR_POSTPROCESS_MAX_WORKERS))
    used_scales = list(GRAY_SCALES) if detector_profile == "gray" else list(SCALES)
    with ThreadPoolExecutor(max_workers=variant_workers) as pool:
        prepared_variant_items = list(
            pool.map(
                lambda item: (
                    item[0],
                    _prepare_variants(
                        item[0],
                        item[1],
                        scales=used_scales,
                        include_gray_diagonal_rotations=detector_profile == "gray",
                        disable_text_mirror=disable_text_mirror,
                    ),
                ),
                enumerate(templates),
            )
        )
    variants_by_template = dict(prepared_variant_items)
    if detector_profile == "gray":
        variants_by_template = {
            template_id: [
                variant
                for variant in variants
                if not (
                    not templates[template_id].is_text_label
                    and variant.scale <= GRAY_TINY_FRAGMENT_MAX_SCALE
                    and max(variant.width, variant.height) <= GRAY_TINY_FRAGMENT_MAX_DIMENSION
                )
            ]
            for template_id, variants in variants_by_template.items()
        }
    variants_lookup = {
        (variant.template_id, variant.scale, variant.rotation, variant.mirrored): variant
        for variants in variants_by_template.values()
        for variant in variants
    }
    socket_07_promotions = _build_socket_07_promotions(templates, variants_by_template)
    parent_ids_by_child: dict[int, set[int]] = {}
    for rules in socket_07_promotions.values():
        for rule in rules:
            parent_ids_by_child.setdefault(rule.child_template_id, set()).add(
                rule.parent_template_id
            )
    plan_masks_by_template: dict[int, np.ndarray] = {}
    template_by_mask_key: dict[str, TemplateInfo] = {}
    if detector_profile == "color":
        for template in templates:
            cache_key = color_strategy.color_mask_cache_key(template)
            if cache_key is not None:
                template_by_mask_key.setdefault(cache_key, template)
    unique_mask_keys = set(template_by_mask_key)

    def _build_cached_color_mask(cache_key: str) -> tuple[str, np.ndarray]:
        mask = color_strategy.build_color_plan_mask(
            plan_image=plan_image,
            plan_hsv=plan_hsv,
            template=template_by_mask_key[cache_key],
            exclude_rects=exclude_rects,
        )
        return cache_key, mask

    if unique_mask_keys:
        mask_workers = max(1, min(len(unique_mask_keys), DETECTOR_POSTPROCESS_MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=mask_workers) as pool:
            color_masks_cache.update(dict(pool.map(_build_cached_color_mask, unique_mask_keys)))

    for template_id, template in enumerate(templates):
        plan_masks_by_template[template_id] = _get_plan_mask(template)
    plan_mask_foregrounds = {
        template_id: int(cv2.countNonZero(plan_mask))
        for template_id, plan_mask in plan_masks_by_template.items()
    }
    max_variant_size_by_template = {
        template_id: (
            max(
                (variant.width for variant in variants),
                default=templates[template_id].mask.shape[1],
            ),
            max(
                (variant.height for variant in variants),
                default=templates[template_id].mask.shape[0],
            ),
        )
        for template_id, variants in variants_by_template.items()
    }

    # Gray stroke suppression - build dedicated dark scan masks before ROI.
    # Gray ROIs should be seeded from the same dark ink mask used by scanning,
    # otherwise pale architectural lines still create too many search windows.
    gray_suppressed_pixels = 0
    gray_raw_ink_pixels = 0
    gray_dark_ink_pixels = 0
    gray_dark_zone_pixels = 0
    gray_dark_evidence_pixels = 0
    gray_dark_threshold = GRAY_DARK_INK_THRESHOLD
    gray_dark_zone_threshold = GRAY_DARK_ZONE_THRESHOLD
    gray_dark_evidence_threshold = GRAY_DARK_EVIDENCE_THRESHOLD
    gray_dark_suppressed_pixels = 0
    gray_dark_zone_suppressed_pixels = 0
    scan_masks_by_template: dict[int, np.ndarray]
    scan_mask_kinds_by_template: dict[int, str]
    gray_zone_mask: np.ndarray | None = None
    gray_evidence_mask: np.ndarray | None = None
    timings["gray_suppress"] = 0.0
    if detector_profile == "gray":
        _t_suppress = time.perf_counter()
        _raw_dilated = _get_ink_plan_mask(dilate=True)
        gray_scan_masks = gray_strategy.build_gray_scan_masks(
            plan_image=plan_image,
            templates=templates,
            plan_masks_by_template=plan_masks_by_template,
            exclude_rects=exclude_rects,
            raw_dilated=_raw_dilated,
        )
        gray_raw_ink_pixels = gray_scan_masks.raw_ink_pixels
        gray_suppressed_pixels = gray_scan_masks.suppressed_pixels
        gray_dark_ink_pixels = gray_scan_masks.dark_ink_pixels
        gray_dark_zone_pixels = gray_scan_masks.zone_ink_pixels
        gray_dark_evidence_pixels = gray_scan_masks.evidence_ink_pixels
        gray_dark_threshold = gray_scan_masks.dark_threshold
        gray_dark_zone_threshold = gray_scan_masks.zone_threshold
        gray_dark_evidence_threshold = gray_scan_masks.evidence_threshold
        gray_dark_suppressed_pixels = gray_scan_masks.dark_suppressed_pixels
        gray_dark_zone_suppressed_pixels = gray_scan_masks.zone_suppressed_pixels
        scan_masks_by_template = gray_scan_masks.scan_masks_by_template
        scan_mask_kinds_by_template = gray_scan_masks.scan_mask_kinds_by_template
        gray_zone_mask = gray_scan_masks.zone_mask
        gray_evidence_mask = gray_scan_masks.evidence_mask
        timings["gray_suppress"] = time.perf_counter() - _t_suppress
    else:
        scan_masks_by_template = plan_masks_by_template
        scan_mask_kinds_by_template = {template_id: "color" for template_id in plan_masks_by_template}

    search_rois_by_template: dict[int, list[tuple[int, int, int, int]]] = {}
    search_roi_stats_by_template: dict[int, tuple[bool, int, int]] = {}
    search_roi_strategy_by_template: dict[int, str] = {}
    gray_search_component_index = (
        gray_strategy.build_gray_search_component_index(gray_zone_mask)
        if detector_profile == "gray" and gray_zone_mask is not None
        else None
    )

    def _prepare_search_roi(
        item: tuple[int, np.ndarray]
    ) -> tuple[int, list[tuple[int, int, int, int]], tuple[bool, int, int]]:
        template_id, plan_mask = item
        max_width, max_height = max_variant_size_by_template[template_id]
        if detector_profile == "gray":
            roi_seed_mask = gray_zone_mask if gray_zone_mask is not None else plan_mask
            roi_strategy, tile_size, max_tile_rois = gray_strategy.gray_tile_roi_strategy(
                templates[template_id]
            )
            rois, uses_full_scan, roi_area, foreground_pixels = gray_strategy.build_gray_search_rois(
                roi_seed_mask,
                plan_image.shape,
                max_width,
                max_height,
                is_large_text_template=gray_strategy.use_large_text_tile_rois(
                    templates[template_id]
                ),
                component_index=gray_search_component_index,
                tile_size_override=tile_size,
                max_tile_rois_override=max_tile_rois,
            )
        else:
            roi_strategy = "color"
            rois, uses_full_scan, roi_area, foreground_pixels = _build_search_rois(
                plan_mask,
                plan_image.shape,
                max_width,
                max_height,
            )
        return template_id, rois, (uses_full_scan, roi_area, foreground_pixels), roi_strategy

    roi_workers = max(1, min(len(plan_masks_by_template), DETECTOR_POSTPROCESS_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=roi_workers) as pool:
        roi_items = list(pool.map(_prepare_search_roi, plan_masks_by_template.items()))
    for template_id, rois, stats, roi_strategy in roi_items:
        search_rois_by_template[template_id] = rois
        search_roi_stats_by_template[template_id] = stats
        search_roi_strategy_by_template[template_id] = roi_strategy
    dilated_plan_masks_by_template: dict[int, np.ndarray] = {}
    timings["prepare"] = time.perf_counter() - phase_start

    diagnostics = {
        "detector_profile": 1 if detector_profile == "gray" else 0,
        "gray_text_exclude_rects": len(gray_text_exclude_rects),
        "gray_title_exclude_rects": len(gray_title_exclude_rects),
        "gray_suppressed_pixels": gray_suppressed_pixels,
        "gray_suppressed_ratio": round(gray_suppressed_pixels / max(1, gray_raw_ink_pixels), 3),
        "gray_dark_ink_pixels": gray_dark_ink_pixels,
        "gray_dark_zone_pixels": gray_dark_zone_pixels,
        "gray_dark_evidence_pixels": gray_dark_evidence_pixels,
        "gray_dark_threshold": gray_dark_threshold,
        "gray_dark_zone_threshold": gray_dark_zone_threshold,
        "gray_dark_evidence_threshold": gray_dark_evidence_threshold,
        "gray_dark_suppressed_pixels": gray_dark_suppressed_pixels,
        "gray_dark_suppressed_ratio": round(
            gray_dark_suppressed_pixels / max(1, gray_dark_ink_pixels), 3
        ),
        "gray_dark_zone_suppressed_pixels": gray_dark_zone_suppressed_pixels,
        "gray_dark_zone_suppressed_ratio": round(
            gray_dark_zone_suppressed_pixels / max(1, gray_dark_zone_pixels), 3
        ),
        "raw_peaks": 0,
        "raw_budget_hits": 0,
        "raw_budget_removed": 0,
        "raw_prefilter_hits": 0,
        "raw_prefilter_removed": 0,
        "prepared_variants": sum(len(variants) for variants in variants_by_template.values()),
        "gray_text_mirror_enabled": int(detector_profile == "gray" and not disable_text_mirror),
        "color_no_hsv_templates": sum(
            1
            for template in templates
            if detector_profile == "color" and template.dominant_hsv is None
        ),
        "skipped_empty_color_masks": 0,
        "validated_template_hits": 0,
        "promoted_targeted_hits": 0,
        "parent_search_input_hits": 0,
        "parent_search_candidates": 0,
        "promoted_parent_search_hits": 0,
        "pdf_text_hits": len(pdf_candidates),
        "prefilter_hits": 0,
        "pre_parent_clusters": 0,
        "final_hits": 0,
        "search_rois": sum(len(rois) for rois in search_rois_by_template.values()),
        "gray_roi_fast_templates": sum(
            1
            for strategy in search_roi_strategy_by_template.values()
            if strategy in {"large_text_fast", "fast_compact"}
        ),
        "gray_roi_fast_rois": sum(
            len(search_rois_by_template[template_id])
            for template_id, strategy in search_roi_strategy_by_template.items()
            if strategy in {"large_text_fast", "fast_compact"}
        ),
        "gray_roi_safe_templates": sum(
            1 for strategy in search_roi_strategy_by_template.values() if strategy == "safe_elongated"
        ),
        "gray_roi_safe_rois": sum(
            len(search_rois_by_template[template_id])
            for template_id, strategy in search_roi_strategy_by_template.items()
            if strategy == "safe_elongated"
        ),
        "full_scan_templates": sum(
            1 for uses_full, _, _ in search_roi_stats_by_template.values() if uses_full
        ),
        "roi_area_pixels": sum(area for _, area, _ in search_roi_stats_by_template.values()),
        "roi_foreground_pixels": sum(
            pixels for _, _, pixels in search_roi_stats_by_template.values()
        ),
    }

    scan_result = scan_template_candidates(
        templates=templates,
        variants_by_template=variants_by_template,
        scan_masks_by_template=scan_masks_by_template,
        scan_mask_kinds_by_template=scan_mask_kinds_by_template,
        search_rois_by_template=search_rois_by_template,
        roi_strategies_by_template=search_roi_strategy_by_template,
        plan_mask_foregrounds=plan_mask_foregrounds,
        detector_profile=detector_profile,
        progress_callback=_progress,
        collect_profile=collect_performance_profile,
    )
    raw_template_hits = scan_result.raw_template_hits
    scan_workers = scan_result.scan_workers
    diagnostics["skipped_empty_color_masks"] = scan_result.skipped_empty_color_masks
    diagnostics["scan_tasks"] = scan_result.scan_tasks
    diagnostics["scan_task_rois"] = scan_result.scan_task_rois
    timings["scan"] = scan_result.timing_seconds
    diagnostics["raw_peaks"] = len(raw_template_hits)
    raw_scan_hits = raw_template_hits
    _record_candidate_trace("raw_scan", raw_template_hits)

    # Per-scale histograms for diagnostics: count raw_peaks at each scale
    # before budget/prefilter so we can see whether matchTemplate fires at all
    # at a given scale (especially important on gray plans).
    raw_peaks_by_scale: dict[float, int] = {}
    if debug_profile is not None:
        for _hit in raw_template_hits:
            raw_peaks_by_scale[_hit.scale] = raw_peaks_by_scale.get(_hit.scale, 0) + 1

    gray_budget_profile: dict = {}
    phase_start = time.perf_counter()
    if detector_profile == "gray":
        raw_template_hits, gray_budget_profile = gray_strategy.gray_raw_budget(
            raw_template_hits,
            templates,
            plan_masks_by_template,
        )
        diagnostics["raw_budget_hits"] = len(raw_template_hits)
        diagnostics["raw_budget_removed"] = int(gray_budget_profile.get("removed", 0))
    else:
        diagnostics["raw_budget_hits"] = len(raw_template_hits)
    timings["raw_budget"] = time.perf_counter() - phase_start
    raw_budget_hits = raw_template_hits
    _record_candidate_trace("raw_budget", raw_template_hits)

    phase_start = time.perf_counter()
    raw_before_prefilter = len(raw_template_hits)
    gray_frame_raw_rescue_hits = (
        [hit for hit in raw_template_hits if gray_strategy.is_gray_frame_raw_rescue_hit(hit, templates)]
        if detector_profile == "gray"
        else []
    )
    detail = f"Odsiew slabych kandydatow ({raw_before_prefilter})"
    if gray_budget_profile.get("removed"):
        detail += f", ucieto {gray_budget_profile['removed']}"
    _progress("raw_prefilter", 62, detail)
    raw_template_hits = _prefilter_raw_template_hits(raw_template_hits)
    if gray_frame_raw_rescue_hits:
        existing_ids = {id(hit) for hit in raw_template_hits}
        restored = [hit for hit in gray_frame_raw_rescue_hits if id(hit) not in existing_ids]
        if restored:
            raw_template_hits.extend(restored)
    else:
        restored = []
    diagnostics["raw_prefilter_hits"] = len(raw_template_hits)
    diagnostics["raw_prefilter_removed"] = raw_before_prefilter - len(raw_template_hits)
    diagnostics["gray_frame_raw_rescue_hits"] = len(gray_frame_raw_rescue_hits)
    diagnostics["gray_frame_raw_rescue_restored"] = len(restored)
    raw_prefilter_hits = raw_template_hits
    timings["raw_prefilter"] = time.perf_counter() - phase_start
    _record_candidate_trace("raw_prefilter", raw_template_hits)

    candidates_by_scale: dict[float, int] = {}
    if debug_profile is not None:
        for _hit in raw_template_hits:
            candidates_by_scale[_hit.scale] = candidates_by_scale.get(_hit.scale, 0) + 1

    validated_candidates: list[CandidateHit] = list(pdf_candidates)
    postprocess_workers = max(1, DETECTOR_POSTPROCESS_MAX_WORKERS)
    validation_result = validate_template_candidates(
        raw_template_hits=raw_template_hits,
        plan_image=plan_image,
        templates=templates,
        plan_masks_by_template=plan_masks_by_template,
        dilated_plan_masks_by_template=dilated_plan_masks_by_template,
        variants_lookup=variants_lookup,
        socket_07_promotions=socket_07_promotions,
        plan_hsv=plan_hsv,
        postprocess_workers=postprocess_workers,
        progress_callback=_progress,
        gray_evidence_mask=gray_evidence_mask if detector_profile == "gray" else None,
        gray_relaxed_evidence_mask=gray_zone_mask if detector_profile == "gray" else None,
    )
    validated_hits = validation_result.validated_hits
    rejection_reasons = validation_result.rejection_reasons
    validation_workers = validation_result.validation_workers
    diagnostics["promoted_targeted_hits"] += validation_result.promoted_targeted_hits
    validated_candidates.extend(validated_hits)
    _record_candidate_trace("validation_accepted", validated_hits)
    _record_candidate_trace("validation_rejected", validation_result.rejected_hits)

    accepted_by_scale: dict[float, int] = {}
    if debug_profile is not None:
        for _hit in validated_hits:
            accepted_by_scale[_hit.scale] = accepted_by_scale.get(_hit.scale, 0) + 1

    diagnostics["validated_template_hits"] = len(validated_candidates) - len(pdf_candidates)
    timings["validation_targeted"] = validation_result.timing_seconds

    phase_start = time.perf_counter()
    _progress("prefilter", 84, "Klastrowanie kandydatow")
    prefiltered_candidates = _prefilter_candidates(validated_candidates)
    diagnostics["prefilter_hits"] = len(prefiltered_candidates)
    timings["prefilter"] = time.perf_counter() - phase_start
    _record_candidate_trace("prefilter", prefiltered_candidates)

    phase_start = time.perf_counter()
    pre_parent_candidates = _cluster_candidates(
        prefiltered_candidates,
        parent_ids_by_child,
        mode=detector_profile,
    )
    diagnostics["pre_parent_clusters"] = len(pre_parent_candidates)
    timings["pre_parent_clustering"] = time.perf_counter() - phase_start
    _record_candidate_trace("pre_parent_clusters", pre_parent_candidates)

    parent_search_result = search_parent_candidates(
        pre_parent_candidates=pre_parent_candidates,
        detector_profile=detector_profile,
        plan_image=plan_image,
        templates=templates,
        plan_masks_by_template=plan_masks_by_template,
        dilated_plan_masks_by_template=dilated_plan_masks_by_template,
        variants_lookup=variants_lookup,
        socket_07_promotions=socket_07_promotions,
        plan_hsv=plan_hsv,
        postprocess_workers=postprocess_workers,
        progress_callback=_progress,
    )
    parent_search_candidates = parent_search_result.candidates
    parent_search_workers = parent_search_result.workers
    diagnostics["parent_search_input_hits"] += parent_search_result.input_hits
    diagnostics["parent_search_candidates"] += parent_search_result.attempted_candidates
    diagnostics["promoted_parent_search_hits"] += parent_search_result.promoted_hits
    timings["parent_search"] = parent_search_result.timing_seconds
    _record_candidate_trace("parent_search", parent_search_candidates)

    phase_start = time.perf_counter()
    _progress("final_clustering", 94, "Finalne laczenie wynikow")
    final_hits = _cluster_candidates(
        parent_search_candidates,
        parent_ids_by_child,
        mode=detector_profile,
    )
    if detector_profile == "gray":
        final_hits, rescued_gray_frames, gray_rescue_trace = gray_strategy.rescue_validated_gray_frame_hits(
            final_hits,
            validated_hits,
            templates,
        )
        for stage_name, stage_trace in gray_rescue_trace.items():
            _record_candidate_trace(
                stage_name,
                stage_trace["hits"],
                stage_trace.get("reasons"),
            )
        gray_unresolved_strong_hits = gray_strategy.trace_unresolved_strong_gray_hits(
            final_hits,
            validated_hits,
            templates,
        )
    else:
        rescued_gray_frames = 0
        gray_unresolved_strong_hits = {"strongValidated": 0, "unresolved": 0, "items": []}
    diagnostics["gray_frame_final_rescued"] = rescued_gray_frames
    diagnostics["final_hits"] = len(final_hits)
    timings["clustering"] = time.perf_counter() - phase_start
    _record_candidate_trace("final", final_hits)

    timings_ms = {name: round(seconds * 1000.0, 3) for name, seconds in timings.items()}
    if debug_profile is not None:
        candidate_stage_counts = build_candidate_stage_counts(
            pdf_candidates=pdf_candidates,
            raw_scan_hits=raw_scan_hits,
            raw_budget_hits=raw_budget_hits,
            raw_prefilter_hits=raw_prefilter_hits,
            validation_rejected_hits=validation_result.rejected_hits,
            validated_hits=validated_hits,
            prefiltered_hits=prefiltered_candidates,
            pre_parent_hits=pre_parent_candidates,
            parent_search_hits=parent_search_candidates,
            final_hits=final_hits,
            rescued_gray_frames=rescued_gray_frames,
        )
        hit_flow_profile = {}
        roi_strategy_profile = {}
        if collect_performance_profile:
            hit_flow_profile = build_hit_flow_profile(
                templates=templates,
                raw_scan_hits=raw_scan_hits,
                raw_budget_hits=raw_budget_hits,
                raw_prefilter_hits=raw_prefilter_hits,
                validated_hits=validated_hits,
                pre_cluster_hits=pre_parent_candidates,
                final_hits=final_hits,
            )
            roi_strategy_profile = build_roi_strategy_profile(
                templates=templates,
                variants_by_template=variants_by_template,
                search_rois_by_template=search_rois_by_template,
                search_roi_stats_by_template=search_roi_stats_by_template,
                search_roi_strategy_by_template=search_roi_strategy_by_template,
                raw_scan_hits=raw_scan_hits,
                raw_budget_hits=raw_budget_hits,
                raw_prefilter_hits=raw_prefilter_hits,
                validated_hits=validated_hits,
                final_hits=final_hits,
            )
        debug_profile.clear()
        debug_profile.update(
            {
                "profileFlags": {
                    "debugProfile": True,
                    "performanceProfile": bool(collect_performance_profile),
                    "candidateTrace": bool(candidate_trace_enabled),
                },
                "timingsMs": timings_ms,
                "counters": {key: int(value) for key, value in diagnostics.items()},
                "profile": detector_profile,
                "threading": {
                    "scanStrategy": scan_result.scan_strategy,
                    "scanWorkers": int(scan_workers),
                    "scanTasks": int(scan_result.scan_tasks),
                    "scanTaskRois": int(scan_result.scan_task_rois),
                    "configuredScanWorkers": int(scan_result.configured_scan_workers),
                    "validationWorkers": int(validation_workers if raw_template_hits else 0),
                    "parentSearchWorkers": int(
                        parent_search_workers if pre_parent_candidates else 0
                    ),
                    "configuredPostprocessWorkers": int(DETECTOR_POSTPROCESS_MAX_WORKERS),
                    "opencvThreads": int(scan_result.opencv_threads),
                    "configuredOpencvThreads": int(OPENCV_NUM_THREADS),
                    "cpuCount": int(_safe_cpu_count()),
                },
                "searchRoi": {
                    "totalRois": int(diagnostics["search_rois"]),
                    "fullScanTemplates": int(diagnostics["full_scan_templates"]),
                    "roiAreaPixels": int(diagnostics["roi_area_pixels"]),
                    "foregroundPixels": int(diagnostics["roi_foreground_pixels"]),
                    "fullImageAreaPixels": int(plan_image.shape[0] * plan_image.shape[1]),
                },
                "grayRawBudget": gray_budget_profile,
                "candidateTrace": candidate_trace if candidate_trace_enabled else {},
                "candidateStageCounts": candidate_stage_counts,
                "scanProfile": scan_result.scan_profile,
                "hitFlowProfile": hit_flow_profile,
                "roiStrategyProfile": roi_strategy_profile,
                "ablation": {"noTextMirror": bool(ablation_no_text_mirror)},
                "grayVariantStrategy": {
                    "textMirrorEnabled": bool(
                        detector_profile == "gray" and not disable_text_mirror
                    ),
                    "textMirrorOverride": bool(gray_force_text_mirror),
                },
                "grayStrokeSuppression": {
                    "removedPixels": int(diagnostics["gray_suppressed_pixels"]),
                    "rawPixels": int(gray_raw_ink_pixels),
                    "ratio": float(diagnostics["gray_suppressed_ratio"]),
                    "darkThreshold": int(diagnostics["gray_dark_threshold"]),
                    "darkPixels": int(diagnostics["gray_dark_ink_pixels"]),
                    "zoneThreshold": int(diagnostics["gray_dark_zone_threshold"]),
                    "zonePixels": int(diagnostics["gray_dark_zone_pixels"]),
                    "evidenceThreshold": int(diagnostics["gray_dark_evidence_threshold"]),
                    "evidencePixels": int(diagnostics["gray_dark_evidence_pixels"]),
                    "zoneRemovedPixels": int(diagnostics["gray_dark_zone_suppressed_pixels"]),
                    "zoneRemovedRatio": float(diagnostics["gray_dark_zone_suppressed_ratio"]),
                    "darkRemovedPixels": int(diagnostics["gray_dark_suppressed_pixels"]),
                    "darkRemovedRatio": float(diagnostics["gray_dark_suppressed_ratio"]),
                    "timingMs": round(timings_ms.get("gray_suppress", 0.0), 3),
                    "horizontalKernelPx": GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
                    "verticalKernelPx": GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
                },
                "scaleHistogram": {
                    "usedScales": [round(float(s), 3) for s in used_scales],
                    "rawPeaks": {
                        round(float(s), 3): int(c) for s, c in sorted(raw_peaks_by_scale.items())
                    },
                    "candidates": {
                        round(float(s), 3): int(c) for s, c in sorted(candidates_by_scale.items())
                    },
                    "accepted": {
                        round(float(s), 3): int(c) for s, c in sorted(accepted_by_scale.items())
                    },
                },
                "scanMaskRawHits": {
                    str(k): int(v) for k, v in sorted(scan_result.raw_hits_by_mask_kind.items())
                },
                "grayUnresolvedStrongHits": gray_unresolved_strong_hits,
                "rejectionReasons": {k: int(v) for k, v in rejection_reasons.items()},
                "slowestPhase": (
                    max(timings_ms.items(), key=lambda item: item[1])[0] if timings_ms else None
                ),
            }
        )

    if debug_profile is not None:
        def _fmt_scale_hist(hist: dict[float, int]) -> str:
            if not hist:
                return "{}"
            return "{" + ",".join(f"{s:.2f}:{hist.get(s, 0)}" for s in used_scales) + "}"

        rejection_summary = (
            ",".join(f"{k}:{v}" for k, v in sorted(rejection_reasons.items(), key=lambda kv: -kv[1]))
            if rejection_reasons
            else "-"
        )
        scan_mask_summary = (
            ",".join(f"{k}:{v}" for k, v in sorted(scan_result.raw_hits_by_mask_kind.items()))
            if scan_result.raw_hits_by_mask_kind
            else "-"
        )

        print(
            "Detection diagnostics:"
            f" profile={detector_profile},"
            f" templates={len(templates)},"
            f" used_scales=[{','.join(f'{s:.2f}' for s in used_scales)}],"
            f" prepared_variants={diagnostics['prepared_variants']},"
            f" color_no_hsv_templates={diagnostics['color_no_hsv_templates']},"
            f" skipped_empty_color_masks={diagnostics['skipped_empty_color_masks']},"
            f" raw_peaks={diagnostics['raw_peaks']} per_scale={_fmt_scale_hist(raw_peaks_by_scale)},"
            f" raw_peaks_by_mask={scan_mask_summary},"
            f" raw_budget={diagnostics['raw_budget_hits']}(-{diagnostics['raw_budget_removed']}),"
            f" raw_budget_protected={int(gray_budget_profile.get('geometryProtectedKept', 0))}/{int(gray_budget_profile.get('geometryProtected', 0))},"
            f" raw_after_prefilter={diagnostics['raw_prefilter_hits']}(-{diagnostics['raw_prefilter_removed']})"  # noqa: E501
            f" candidates_per_scale={_fmt_scale_hist(candidates_by_scale)},"
            f" validated_template_hits={diagnostics['validated_template_hits']}"
            f" accepted_per_scale={_fmt_scale_hist(accepted_by_scale)},"
            f" rejects=nms:{diagnostics['raw_prefilter_removed']}|suppression(budget):{diagnostics['raw_budget_removed']}|validation:[{rejection_summary}],"  # noqa: E501
            f" promoted_targeted_hits={diagnostics['promoted_targeted_hits']},"
            f" parent_search_input_hits={diagnostics['parent_search_input_hits']},"
            f" parent_search_candidates={diagnostics['parent_search_candidates']},"
            f" promoted_parent_search_hits={diagnostics['promoted_parent_search_hits']},"
            f" pdf_text_hits={diagnostics['pdf_text_hits']},"
            f" after_prefilter={diagnostics['prefilter_hits']},"
            f" pre_parent_clusters={diagnostics['pre_parent_clusters']},"
            f" final_clusters={diagnostics['final_hits']},"
            f" rois={diagnostics['search_rois']} full_scan_templates={diagnostics['full_scan_templates']},"  # noqa: E501
            f" scan_tasks={diagnostics['scan_tasks']} task_rois={diagnostics['scan_task_rois']},"
            f" gray_suppress={diagnostics['gray_suppressed_pixels']}px({diagnostics['gray_suppressed_ratio']:.0%}),"
            f" gray_dark<{diagnostics['gray_dark_threshold']}={diagnostics['gray_dark_ink_pixels']}px,"
            f" gray_zone<{diagnostics['gray_dark_zone_threshold']}={diagnostics['gray_dark_zone_pixels']}px,"
            f" gray_evidence<{diagnostics['gray_dark_evidence_threshold']}={diagnostics['gray_dark_evidence_pixels']}px,"
            f" gray_zone_suppress={diagnostics['gray_dark_zone_suppressed_pixels']}px({diagnostics['gray_dark_zone_suppressed_ratio']:.0%}),"
            f" gray_dark_suppress={diagnostics['gray_dark_suppressed_pixels']}px({diagnostics['gray_dark_suppressed_ratio']:.0%}),"
            f" threads={scan_result.scan_strategy}:scan:{scan_workers}/{scan_result.configured_scan_workers}|post:{postprocess_workers}/{DETECTOR_POSTPROCESS_MAX_WORKERS}|opencv:{scan_result.opencv_threads},"  # noqa: E501
            f" timings_ms="
            f"pdf_text:{timings_ms['pdf_text']:.0f}|"
            f"prepare:{timings_ms['prepare']:.0f}|"
            f"gray_suppress:{timings_ms['gray_suppress']:.0f}|"
            f"scan:{timings_ms['scan']:.0f}|"
            f"raw_prefilter:{timings_ms['raw_prefilter']:.0f}|"
            f"validation_targeted:{timings_ms['validation_targeted']:.0f}|"
            f"prefilter:{timings_ms['prefilter']:.0f}|"
            f"pre_parent_clustering:{timings_ms['pre_parent_clustering']:.0f}|"
            f"parent_search:{timings_ms['parent_search']:.0f}|"
            f"clustering:{timings_ms['clustering']:.0f}"
        )

    per_template: dict[int, list[Detection]] = {}
    _progress("format_results", 99, "Formatowanie wynikow")
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
            mirrored=hit.mirrored,
            coverage=round(hit.coverage, 3),
            purity=round(hit.purity, 3),
            context_purity=round(hit.context_purity, 3),
            color_similarity=round(hit.color_similarity, 3),
            verification_score=round(hit.verification_score, 3),
            is_text_label=hit.is_text_label,
            content_score=round(hit.content_score, 3),
            content_bbox=(
                (
                    x + hit.content_bbox[0],
                    y + hit.content_bbox[1],
                    hit.content_bbox[2],
                    hit.content_bbox[3],
                )
                if hit.content_bbox is not None
                else None
            ),
            content_source=hit.source if hit.is_text_label else "",
            roi_strategy=hit.roi_strategy,
        )
        per_template.setdefault(hit.template_id, []).append(detection)

    results: list[DetectionResult] = []
    for template_id, detections in per_template.items():
        detections.sort(key=lambda det: (det.verification_score, det.confidence), reverse=True)

        count = len(detections)
        # Gray PDFs use explicit legend/text/plan-zone exclusions.  Blindly
        # subtracting one hit per symbol here can hide valid plan detections
        # that already passed validation, especially when the legend is outside
        # the selected plan zone.
        blind_legend_subtract = (
            detector_profile != "gray" and subtract_legend and legend_rect is None
        )
        if blind_legend_subtract:
            count = max(0, count - 1)

        if count <= 0:
            continue

        results.append(
            DetectionResult(
                symbol_name=templates[template_id].name,
                count=count,
                color="#22c55e",
                detections=(
                    detections[:count] if blind_legend_subtract else detections
                ),
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


