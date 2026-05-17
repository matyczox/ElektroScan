"""Debug profile emission for detector pipeline runs."""

from __future__ import annotations

from typing import Any

from core.detector_config import (
    DETECTOR_POSTPROCESS_MAX_WORKERS,
    GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
    GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    OPENCV_NUM_THREADS,
    _safe_cpu_count,
)
from core.detector_diagnostics import (
    build_candidate_stage_counts,
    build_hit_flow_profile,
    build_roi_strategy_profile,
)


def emit_pipeline_debug_profile(ctx: dict[str, Any]) -> None:
    debug_profile = ctx.get("debug_profile")
    if debug_profile is None:
        return

    diagnostics = ctx["diagnostics"]
    timings = ctx["timings"]
    timings_ms = {name: round(seconds * 1000.0, 3) for name, seconds in timings.items()}
    templates = ctx["templates"]
    detector_profile = ctx["detector_profile"]
    validation_result = ctx["validation_result"]
    scan_result = ctx["scan_result"]
    final_hits = ctx["final_hits"]
    validated_hits = ctx["validated_hits"]
    raw_scan_hits = ctx["raw_scan_hits"]
    raw_budget_hits = ctx["raw_budget_hits"]
    raw_prefilter_hits = ctx["raw_prefilter_hits"]
    prefiltered_candidates = ctx["prefiltered_candidates"]
    pre_parent_candidates = ctx["pre_parent_candidates"]
    parent_search_candidates = ctx["parent_search_candidates"]
    rescued_gray_frames = ctx["rescued_gray_frames"]
    collect_performance_profile = ctx["collect_performance_profile"]
    variants_by_template = ctx["variants_by_template"]
    search_rois_by_template = ctx["search_rois_by_template"]
    search_roi_stats_by_template = ctx["search_roi_stats_by_template"]
    search_roi_strategy_by_template = ctx["search_roi_strategy_by_template"]
    used_scales = ctx["used_scales"]
    raw_peaks_by_scale = ctx["raw_peaks_by_scale"]
    candidates_by_scale = ctx["candidates_by_scale"]
    accepted_by_scale = ctx["accepted_by_scale"]
    rejection_reasons = ctx["rejection_reasons"]
    gray_budget_profile = ctx["gray_budget_profile"]
    candidate_trace_recorder = ctx["candidate_trace_recorder"]

    candidate_stage_counts = build_candidate_stage_counts(
        pdf_candidates=ctx["pdf_candidates"],
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
                "candidateTrace": bool(candidate_trace_recorder.enabled),
            },
            "timingsMs": timings_ms,
            "counters": {key: int(value) for key, value in diagnostics.items()},
            "profile": detector_profile,
            "threading": {
                "scanStrategy": scan_result.scan_strategy,
                "scanWorkers": int(ctx["scan_workers"]),
                "scanTasks": int(scan_result.scan_tasks),
                "scanTaskRois": int(scan_result.scan_task_rois),
                "configuredScanWorkers": int(scan_result.configured_scan_workers),
                "validationWorkers": int(
                    ctx["validation_workers"] if ctx["raw_template_hits"] else 0
                ),
                "parentSearchWorkers": int(
                    ctx["parent_search_workers"] if pre_parent_candidates else 0
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
                "fullImageAreaPixels": int(
                    ctx["plan_image"].shape[0] * ctx["plan_image"].shape[1]
                ),
            },
            "grayRawBudget": gray_budget_profile,
            "candidateTrace": (
                candidate_trace_recorder.data if candidate_trace_recorder.enabled else {}
            ),
            "candidateStageCounts": candidate_stage_counts,
            "scanProfile": scan_result.scan_profile,
            "grayNearThresholdRecovery": {
                "candidates": int(diagnostics["gray_near_threshold_recovery_candidates"]),
                "accepted": int(diagnostics["gray_near_threshold_recovery_accepted"]),
                "rejected": int(diagnostics["gray_near_threshold_recovery_rejected"]),
                "rejectionReasons": {
                    str(key): int(value)
                    for key, value in validation_result.rejection_reasons_by_source.get(
                        "template_near_threshold",
                        {},
                    ).items()
                },
            },
            "grayInterruptedRecovery": {
                "candidates": int(diagnostics["gray_interrupted_recovery_candidates"]),
                "accepted": int(diagnostics["gray_interrupted_recovery_accepted"]),
                "rejected": int(diagnostics["gray_interrupted_recovery_rejected"]),
                "rejectionReasons": {
                    str(key): int(value)
                    for key, value in validation_result.rejection_reasons_by_source.get(
                        "template_interrupted_recovery",
                        {},
                    ).items()
                },
            },
            "hitFlowProfile": hit_flow_profile,
            "roiStrategyProfile": roi_strategy_profile,
            "ablation": {"noTextMirror": bool(ctx["ablation_no_text_mirror"])},
            "grayVariantStrategy": {
                "textMirrorEnabled": bool(
                    detector_profile == "gray" and not ctx["disable_text_mirror"]
                ),
                "textMirrorOverride": bool(ctx["gray_force_text_mirror"]),
            },
            "grayStrokeSuppression": {
                "removedPixels": int(diagnostics["gray_suppressed_pixels"]),
                "rawPixels": int(ctx["gray_raw_ink_pixels"]),
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
                    round(float(s), 3): int(c)
                    for s, c in sorted(raw_peaks_by_scale.items())
                },
                "candidates": {
                    round(float(s), 3): int(c)
                    for s, c in sorted(candidates_by_scale.items())
                },
                "accepted": {
                    round(float(s), 3): int(c)
                    for s, c in sorted(accepted_by_scale.items())
                },
            },
            "scanMaskRawHits": {
                str(k): int(v) for k, v in sorted(scan_result.raw_hits_by_mask_kind.items())
            },
            "grayUnresolvedStrongHits": ctx["gray_unresolved_strong_hits"],
            "rejectionReasons": {k: int(v) for k, v in rejection_reasons.items()},
            "slowestPhase": (
                max(timings_ms.items(), key=lambda item: item[1])[0] if timings_ms else None
            ),
        }
    )

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
        f" gray_near_threshold={diagnostics['gray_near_threshold_recovery_candidates']}/"
        f"{diagnostics['gray_near_threshold_recovery_accepted']}/"
        f"{diagnostics['gray_near_threshold_recovery_rejected']},"
        f" gray_interrupted={diagnostics['gray_interrupted_recovery_candidates']}/"
        f"{diagnostics['gray_interrupted_recovery_accepted']}/"
        f"{diagnostics['gray_interrupted_recovery_rejected']},"
        f" raw_peaks_by_mask={scan_mask_summary},"
        f" raw_budget={diagnostics['raw_budget_hits']}(-{diagnostics['raw_budget_removed']}),"
        f" raw_budget_protected={int(gray_budget_profile.get('geometryProtectedKept', 0))}/{int(gray_budget_profile.get('geometryProtected', 0))},"
        f" raw_after_prefilter={diagnostics['raw_prefilter_hits']}(-{diagnostics['raw_prefilter_removed']})"
        f" candidates_per_scale={_fmt_scale_hist(candidates_by_scale)},"
        f" validated_template_hits={diagnostics['validated_template_hits']}"
        f" accepted_per_scale={_fmt_scale_hist(accepted_by_scale)},"
        f" rejects=nms:{diagnostics['raw_prefilter_removed']}|suppression(budget):{diagnostics['raw_budget_removed']}|validation:[{rejection_summary}],"
        f" promoted_targeted_hits={diagnostics['promoted_targeted_hits']},"
        f" parent_search_input_hits={diagnostics['parent_search_input_hits']},"
        f" parent_search_candidates={diagnostics['parent_search_candidates']},"
        f" promoted_parent_search_hits={diagnostics['promoted_parent_search_hits']},"
        f" pdf_text_hits={diagnostics['pdf_text_hits']},"
        f" after_prefilter={diagnostics['prefilter_hits']},"
        f" pre_parent_clusters={diagnostics['pre_parent_clusters']},"
        f" final_clusters={diagnostics['final_hits']},"
        f" rois={diagnostics['search_rois']} full_scan_templates={diagnostics['full_scan_templates']},"
        f" scan_tasks={diagnostics['scan_tasks']} task_rois={diagnostics['scan_task_rois']},"
        f" gray_suppress={diagnostics['gray_suppressed_pixels']}px({diagnostics['gray_suppressed_ratio']:.0%}),"
        f" gray_dark<{diagnostics['gray_dark_threshold']}={diagnostics['gray_dark_ink_pixels']}px,"
        f" gray_zone<{diagnostics['gray_dark_zone_threshold']}={diagnostics['gray_dark_zone_pixels']}px,"
        f" gray_evidence<{diagnostics['gray_dark_evidence_threshold']}={diagnostics['gray_dark_evidence_pixels']}px,"
        f" gray_zone_suppress={diagnostics['gray_dark_zone_suppressed_pixels']}px({diagnostics['gray_dark_zone_suppressed_ratio']:.0%}),"
        f" gray_dark_suppress={diagnostics['gray_dark_suppressed_pixels']}px({diagnostics['gray_dark_suppressed_ratio']:.0%}),"
        f" threads={scan_result.scan_strategy}:scan:{ctx['scan_workers']}/{scan_result.configured_scan_workers}|post:{ctx['postprocess_workers']}/{DETECTOR_POSTPROCESS_MAX_WORKERS}|opencv:{scan_result.opencv_threads},"
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
