"""
detector.py - CPU-friendly symbol detection for electrical plans.

This module now orchestrates the pipeline. The heavy helpers live in sibling
modules so detector behavior stays easier to audit and tune.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from core import detector_color as color_strategy
from core import detector_gray as gray_strategy
from core.detector_color_resolvers import apply_color_postprocess
from core.detector_clustering import (
    _cluster_candidates,
    _dedupe_raw_template_hits_before_validation,
    _prefilter_candidates,
    _prefilter_raw_template_hits,
    _suppress_color_local_fragments,
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
from core.detector_context import DetectionTemplateContext
from core.detector_diagnostics import (
    build_candidate_stage_counts,
    build_hit_flow_profile,
    build_roi_strategy_profile,
)
from core.detector_masks import _build_search_rois
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
from core.detector_pdf_policy import (
    apply_color_pdf_text_class_hints,
    filter_pdf_text_fallbacks,
)
from core.detector_parent_search import search_parent_candidates
from core.detector_plan_masks import PlanMaskCache
from core.detector_pipeline_debug import emit_pipeline_debug_profile
from core.detector_result_formatting import build_detection_results
from core.detector_runtime_options import build_runtime_options
from core.detector_scanning import scan_template_candidates
from core.detector_templates import (
    _build_socket_07_promotions,
    _prepare_variants,
    _template_numeric_prefix,
)
from core.detector_trace import CandidateTraceRecorder
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
    runtime_options = build_runtime_options(
        debug_profile=debug_profile,
        detector_profile=detector_profile,
    )
    initial_debug_profile = runtime_options.initial_debug_profile
    collect_performance_profile = runtime_options.collect_performance_profile
    ablation_no_text_mirror = runtime_options.ablation_no_text_mirror
    gray_force_text_mirror = runtime_options.gray_force_text_mirror
    disable_text_mirror = runtime_options.disable_text_mirror
    candidate_trace_recorder = CandidateTraceRecorder(
        templates=templates,
        trace_symbols=runtime_options.trace_symbols,
        trace_points=runtime_options.trace_points,
        trace_radius=runtime_options.trace_radius,
        max_items=runtime_options.trace_max_items,
    )
    _record_candidate_trace = candidate_trace_recorder.record

    template_context = DetectionTemplateContext(templates)
    _template_tokens = template_context.template_tokens
    _is_visual_pdf_text_blocked = template_context.is_visual_pdf_text_blocked

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
    has_pdf_text_assist = len(pdf_candidates) > 0
    timings["pdf_text"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    _progress("prepare", 18, "Przygotowanie masek i wariantow")
    plan_hsv = cv2.cvtColor(plan_image, cv2.COLOR_BGR2HSV) if detector_profile == "color" else None
    plan_mask_cache = PlanMaskCache(
        plan_image=plan_image,
        plan_hsv=plan_hsv,
        exclude_rects=exclude_rects,
        detector_profile=detector_profile,
    )

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
    socket_07_promotions = (
        _build_socket_07_promotions(templates, variants_by_template)
        if detector_profile in {"gray", "color"}
        else {}
    )
    validation_promotions = socket_07_promotions if detector_profile in {"gray", "color"} else {}
    parent_ids_by_child: dict[int, set[int]] = {}
    for rules in socket_07_promotions.values():
        for rule in rules:
            parent_ids_by_child.setdefault(rule.child_template_id, set()).add(
                rule.parent_template_id
            )
    if detector_profile == "color":
        template_ids_by_prefix: dict[str, list[int]] = {}
        for template_id, template in enumerate(templates):
            prefix = _template_numeric_prefix(template.name)
            if prefix is None:
                continue
            template_ids_by_prefix.setdefault(prefix, []).append(template_id)
        for child_prefix, parent_prefixes in {
            "06": ("07",),
            "09": ("07",),
            "11": ("10", "12"),
        }.items():
            for child_id in template_ids_by_prefix.get(child_prefix, []):
                for parent_prefix in parent_prefixes:
                    parent_ids_by_child.setdefault(child_id, set()).update(
                        template_ids_by_prefix.get(parent_prefix, [])
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
        return cache_key, plan_mask_cache.build_color_mask(template_by_mask_key[cache_key])

    if unique_mask_keys:
        mask_workers = max(1, min(len(unique_mask_keys), DETECTOR_POSTPROCESS_MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=mask_workers) as pool:
            plan_mask_cache.color_masks_cache.update(
                dict(pool.map(_build_cached_color_mask, unique_mask_keys))
            )

    for template_id, template in enumerate(templates):
        plan_masks_by_template[template_id] = plan_mask_cache.get_plan_mask(template)
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
        _raw_dilated = plan_mask_cache.get_ink_plan_mask(dilate=True)
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
            roi_strategy, tile_size, max_tile_rois = (
                gray_strategy.adapt_gray_tile_roi_strategy_for_plan(
                    roi_strategy,
                    tile_size,
                    max_tile_rois,
                    gray_search_component_index,
                )
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
                component_supplement_rois=80 if templates[template_id].is_text_label else 0,
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
        "gray_near_threshold_recovery_candidates": 0,
        "gray_near_threshold_recovery_accepted": 0,
        "gray_near_threshold_recovery_rejected": 0,
        "gray_interrupted_recovery_candidates": 0,
        "gray_interrupted_recovery_accepted": 0,
        "gray_interrupted_recovery_rejected": 0,
        "color_recovery_candidates": 0,
        "color_recovery_accepted": 0,
        "color_recovery_rejected": 0,
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
        "pdf_text_label_disambiguation": 0,
        "color_family_rescued": 0,
        "prefilter_hits": 0,
        "pre_parent_clusters": 0,
        "final_hits": 0,
        "search_rois": sum(len(rois) for rois in search_rois_by_template.values()),
        "gray_roi_fast_templates": sum(
            1
            for strategy in search_roi_strategy_by_template.values()
            if strategy in {"large_text_fast", "fast_compact", "fast_compact_connected"}
        ),
        "gray_roi_fast_rois": sum(
            len(search_rois_by_template[template_id])
            for template_id, strategy in search_roi_strategy_by_template.items()
            if strategy in {"large_text_fast", "fast_compact", "fast_compact_connected"}
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
    diagnostics["gray_near_threshold_recovery_candidates"] = sum(
        1 for hit in raw_template_hits if hit.source == "template_near_threshold"
    )
    diagnostics["gray_interrupted_recovery_candidates"] = sum(
        1 for hit in raw_template_hits if hit.source == "template_interrupted_recovery"
    )
    diagnostics["color_recovery_candidates"] = sum(
        1 for hit in raw_template_hits if hit.source == "template_color_recovery"
    )
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
    raw_template_hits = _dedupe_raw_template_hits_before_validation(raw_template_hits)
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

    validated_candidates: list[CandidateHit] = []
    postprocess_workers = max(1, DETECTOR_POSTPROCESS_MAX_WORKERS)
    validation_result = validate_template_candidates(
        raw_template_hits=raw_template_hits,
        plan_image=plan_image,
        templates=templates,
        plan_masks_by_template=plan_masks_by_template,
        dilated_plan_masks_by_template=dilated_plan_masks_by_template,
        variants_lookup=variants_lookup,
        socket_07_promotions=validation_promotions,
        plan_hsv=plan_hsv,
        postprocess_workers=postprocess_workers,
        progress_callback=_progress,
        gray_evidence_mask=gray_evidence_mask if detector_profile == "gray" else None,
        gray_relaxed_evidence_mask=gray_zone_mask if detector_profile == "gray" else None,
    )
    validated_hits = validation_result.validated_hits
    rejection_reasons = validation_result.rejection_reasons
    validation_workers = validation_result.validation_workers
    color_pdf_text_hinted = apply_color_pdf_text_class_hints(
        validated_hits,
        pdf_candidates,
        detector_profile=detector_profile,
        template_tokens=_template_tokens,
    )
    label_disambiguation_count = 0
    pdf_candidates, removed_pdf_text_fallbacks = filter_pdf_text_fallbacks(
        pdf_candidates,
        validated_hits,
        detector_profile=detector_profile,
        is_visual_pdf_text_blocked=_is_visual_pdf_text_blocked,
    )
    validated_candidates.extend(pdf_candidates)
    diagnostics["promoted_targeted_hits"] += validation_result.promoted_targeted_hits
    diagnostics["gray_near_threshold_recovery_accepted"] = sum(
        1 for hit in validated_hits if hit.source == "template_near_threshold"
    )
    diagnostics["gray_near_threshold_recovery_rejected"] = sum(
        1 for hit in validation_result.rejected_hits if hit.source == "template_near_threshold"
    )
    diagnostics["gray_interrupted_recovery_accepted"] = sum(
        1 for hit in validated_hits if hit.source == "template_interrupted_recovery"
    )
    diagnostics["gray_interrupted_recovery_rejected"] = sum(
        1
        for hit in validation_result.rejected_hits
        if hit.source == "template_interrupted_recovery"
    )
    diagnostics["color_recovery_accepted"] = sum(
        1 for hit in validated_hits if hit.source == "template_color_recovery"
    )
    diagnostics["color_recovery_rejected"] = sum(
        1 for hit in validation_result.rejected_hits if hit.source == "template_color_recovery"
    )
    validated_candidates.extend(validated_hits)
    _record_candidate_trace("validation_accepted", validated_hits)
    _record_candidate_trace(
        "validation_rejected",
        validation_result.rejected_hits,
        validation_result.rejection_reason_by_hit_id,
    )

    accepted_by_scale: dict[float, int] = {}
    if debug_profile is not None:
        for _hit in validated_hits:
            accepted_by_scale[_hit.scale] = accepted_by_scale.get(_hit.scale, 0) + 1

    diagnostics["validated_template_hits"] = len(validated_candidates) - len(pdf_candidates)
    diagnostics["pdf_text_fallback_removed"] = len(removed_pdf_text_fallbacks)
    diagnostics["pdf_text_class_hints"] = color_pdf_text_hinted
    diagnostics["pdf_text_label_disambiguation"] = label_disambiguation_count
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
        prefer_direct_color_family_parent=True,
    )
    pre_parent_ids = {id(hit) for hit in pre_parent_candidates}
    pre_parent_suppressed = [hit for hit in prefiltered_candidates if id(hit) not in pre_parent_ids]
    diagnostics["pre_parent_clusters"] = len(pre_parent_candidates)
    timings["pre_parent_clustering"] = time.perf_counter() - phase_start
    _record_candidate_trace("pre_parent_clusters", pre_parent_candidates)
    _record_candidate_trace(
        "pre_parent_cluster_suppressed",
        pre_parent_suppressed,
        {id(hit): "cluster_suppressed" for hit in pre_parent_suppressed},
    )

    if detector_profile == "color":
        parent_search_input = [
            hit
            for hit in prefiltered_candidates
            if (hit.template_id, hit.scale, hit.rotation, hit.mirrored) in socket_07_promotions
            and hit.dominant_hsv is not None
            and hit.source == "template"
            and hit.match_score >= 0.48
            and hit.coverage >= 0.65
            and hit.purity >= 0.55
        ]
    else:
        parent_search_input = pre_parent_candidates
    parent_search_result = search_parent_candidates(
        pre_parent_candidates=parent_search_input,
        detector_profile=detector_profile,
        plan_image=plan_image,
        templates=templates,
        plan_masks_by_template=plan_masks_by_template,
        dilated_plan_masks_by_template=dilated_plan_masks_by_template,
        variants_lookup=variants_lookup,
        socket_07_promotions=socket_07_promotions,
        plan_hsv=plan_hsv,
        allow_color_switch_10=True,
        postprocess_workers=postprocess_workers,
        progress_callback=_progress,
    )
    if detector_profile == "color":
        promoted_parent_hits = [
            hit
            for hit in parent_search_result.candidates
            if hit.source.startswith("template_parent_search_")
            or hit.source.startswith("template_promoted_")
        ]
        parent_search_candidates = pre_parent_candidates + promoted_parent_hits
    else:
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
        prefer_direct_color_family_parent=True,
    )
    final_cluster_ids = {id(hit) for hit in final_hits}
    final_cluster_suppressed = [
        hit for hit in parent_search_candidates if id(hit) not in final_cluster_ids
    ]
    _record_candidate_trace(
        "final_cluster_suppressed",
        final_cluster_suppressed,
        {id(hit): "same_place_loser" for hit in final_cluster_suppressed},
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
        (
            final_hits,
            color_fragment_suppressed,
            color_fragment_suppression_reasons,
        ) = _suppress_color_local_fragments(final_hits)
        _record_candidate_trace(
            "color_fragment_suppressed",
            color_fragment_suppressed,
            color_fragment_suppression_reasons,
        )
        final_hits, color_postprocess_counts = apply_color_postprocess(
            detector_profile=detector_profile,
            final_hits=final_hits,
            prefiltered_candidates=prefiltered_candidates,
            rejected_hits=validation_result.rejected_hits,
            rejection_reason_by_hit_id=validation_result.rejection_reason_by_hit_id,
            pdf_candidates=pdf_candidates,
            removed_pdf_text_fallbacks=removed_pdf_text_fallbacks,
            templates=templates,
            template_context=template_context,
            plan_masks_by_template=plan_masks_by_template,
        )
        diagnostics.update(color_postprocess_counts)
        rescued_gray_frames = 0
        gray_unresolved_strong_hits = {"strongValidated": 0, "unresolved": 0, "items": []}
    diagnostics["gray_frame_final_rescued"] = rescued_gray_frames
    diagnostics["final_hits"] = len(final_hits)
    timings["clustering"] = time.perf_counter() - phase_start
    _record_candidate_trace("final", final_hits)

    emit_pipeline_debug_profile(locals())

    _progress("format_results", 99, "Formatowanie wynikow")
    return build_detection_results(
        final_hits=final_hits,
        templates=templates,
        detector_profile=detector_profile,
        plan_masks_by_template=plan_masks_by_template,
        subtract_legend=subtract_legend,
        legend_rect=legend_rect,
    )


