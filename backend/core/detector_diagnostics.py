"""Debug/profile builders for the detector pipeline.

Keep reporting-only aggregation out of the hot orchestration path.  These
helpers must not decide which candidates survive; they only summarize already
computed stages for perf JSON, traces and local regression output.
"""

from __future__ import annotations

from pathlib import Path

from core.detector_models import CandidateHit, TemplateInfo, TemplateVariant


def template_profile_name(templates: list[TemplateInfo], template_id: int) -> str:
    if 0 <= template_id < len(templates):
        return Path(str(templates[template_id].path)).name
    return str(template_id)


def count_hits_by_template(hits: list[CandidateHit]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for hit in hits:
        counts[int(hit.template_id)] = counts.get(int(hit.template_id), 0) + 1
    return counts


def top_hit_variants(
    hits: list[CandidateHit],
    templates: list[TemplateInfo],
    *,
    limit: int = 10,
) -> list[dict]:
    counts: dict[tuple[int, float, int, bool, str], int] = {}
    for hit in hits:
        key = (
            int(hit.template_id),
            round(float(hit.scale), 3),
            int(hit.rotation),
            bool(hit.mirrored),
            str(hit.source),
        )
        counts[key] = counts.get(key, 0) + 1

    rows: list[dict] = []
    for (template_id, scale, rotation, mirrored, source), count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:limit]:
        rows.append(
            {
                "templateId": template_id,
                "templateName": template_profile_name(templates, template_id),
                "scale": scale,
                "rotation": rotation,
                "mirrored": mirrored,
                "source": source,
                "hits": int(count),
            }
        )
    return rows


def build_hit_flow_profile(
    *,
    templates: list[TemplateInfo],
    raw_scan_hits: list[CandidateHit],
    raw_budget_hits: list[CandidateHit],
    raw_prefilter_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    pre_cluster_hits: list[CandidateHit],
    final_hits: list[CandidateHit],
) -> dict:
    raw_scan_counts = count_hits_by_template(raw_scan_hits)
    raw_budget_counts = count_hits_by_template(raw_budget_hits)
    raw_prefilter_counts = count_hits_by_template(raw_prefilter_hits)
    validated_counts = count_hits_by_template(validated_hits)
    pre_cluster_counts = count_hits_by_template(pre_cluster_hits)
    final_counts = count_hits_by_template(final_hits)
    template_ids = sorted(
        set(raw_scan_counts)
        | set(raw_budget_counts)
        | set(raw_prefilter_counts)
        | set(validated_counts)
        | set(pre_cluster_counts)
        | set(final_counts)
    )

    return {
        "byTemplate": [
            {
                "templateId": template_id,
                "templateName": template_profile_name(templates, template_id),
                "rawScan": int(raw_scan_counts.get(template_id, 0)),
                "rawBudget": int(raw_budget_counts.get(template_id, 0)),
                "rawPrefilter": int(raw_prefilter_counts.get(template_id, 0)),
                "validated": int(validated_counts.get(template_id, 0)),
                "preCluster": int(pre_cluster_counts.get(template_id, 0)),
                "final": int(final_counts.get(template_id, 0)),
                "finalYield": round(
                    final_counts.get(template_id, 0)
                    / max(1, validated_counts.get(template_id, 0)),
                    4,
                ),
                "validatedPerFinal": round(
                    validated_counts.get(template_id, 0)
                    / max(1, final_counts.get(template_id, 0)),
                    2,
                ),
            }
            for template_id in template_ids
        ],
        "topValidatedVariants": top_hit_variants(validated_hits, templates),
        "topFinalVariants": top_hit_variants(final_hits, templates),
    }


def build_candidate_stage_counts(
    *,
    pdf_candidates: list[CandidateHit],
    raw_scan_hits: list[CandidateHit],
    raw_budget_hits: list[CandidateHit],
    raw_prefilter_hits: list[CandidateHit],
    validation_rejected_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    prefiltered_hits: list[CandidateHit],
    pre_parent_hits: list[CandidateHit],
    parent_search_hits: list[CandidateHit],
    final_hits: list[CandidateHit],
    rescued_gray_frames: int,
) -> dict[str, int]:
    return {
        "pdfText": int(len(pdf_candidates)),
        "rawScan": int(len(raw_scan_hits)),
        "rawBudget": int(len(raw_budget_hits)),
        "rawBudgetRemoved": int(len(raw_scan_hits) - len(raw_budget_hits)),
        "rawPrefilter": int(len(raw_prefilter_hits)),
        "rawPrefilterRemoved": int(len(raw_budget_hits) - len(raw_prefilter_hits)),
        "validationAccepted": int(len(validated_hits)),
        "validationRejected": int(len(validation_rejected_hits)),
        "validatedTotal": int(len(pdf_candidates) + len(validated_hits)),
        "prefilter": int(len(prefiltered_hits)),
        "preParentClusters": int(len(pre_parent_hits)),
        "parentSearch": int(len(parent_search_hits)),
        "grayFinalRescued": int(rescued_gray_frames),
        "final": int(len(final_hits)),
    }


def build_roi_strategy_profile(
    *,
    templates: list[TemplateInfo],
    variants_by_template: dict[int, list[TemplateVariant]],
    search_rois_by_template: dict[int, list[tuple[int, int, int, int]]],
    search_roi_stats_by_template: dict[int, tuple[bool, int, int]],
    search_roi_strategy_by_template: dict[int, str],
    raw_scan_hits: list[CandidateHit],
    raw_budget_hits: list[CandidateHit],
    raw_prefilter_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    final_hits: list[CandidateHit],
) -> dict:
    raw_scan_counts = count_hits_by_template(raw_scan_hits)
    raw_budget_counts = count_hits_by_template(raw_budget_hits)
    raw_prefilter_counts = count_hits_by_template(raw_prefilter_hits)
    validated_counts = count_hits_by_template(validated_hits)
    final_counts = count_hits_by_template(final_hits)
    strategy_rows: dict[str, dict[str, int | str]] = {}
    template_rows: list[dict] = []

    template_ids = sorted(
        set(search_rois_by_template)
        | set(search_roi_strategy_by_template)
        | set(raw_scan_counts)
        | set(validated_counts)
        | set(final_counts)
    )
    for template_id in template_ids:
        strategy = str(search_roi_strategy_by_template.get(template_id, "unknown"))
        _uses_full, roi_area, foreground_pixels = search_roi_stats_by_template.get(
            template_id,
            (False, 0, 0),
        )
        row = {
            "templateId": int(template_id),
            "templateName": template_profile_name(templates, template_id),
            "strategy": strategy,
            "variants": int(len(variants_by_template.get(template_id, []))),
            "rois": int(len(search_rois_by_template.get(template_id, []))),
            "roiAreaPixels": int(roi_area),
            "foregroundPixels": int(foreground_pixels),
            "rawScan": int(raw_scan_counts.get(template_id, 0)),
            "rawBudget": int(raw_budget_counts.get(template_id, 0)),
            "rawPrefilter": int(raw_prefilter_counts.get(template_id, 0)),
            "validated": int(validated_counts.get(template_id, 0)),
            "final": int(final_counts.get(template_id, 0)),
        }
        row["validatedPerFinal"] = round(row["validated"] / max(1, row["final"]), 2)
        template_rows.append(row)

        aggregate = strategy_rows.setdefault(
            strategy,
            {
                "strategy": strategy,
                "templates": 0,
                "variants": 0,
                "rois": 0,
                "roiAreaPixels": 0,
                "foregroundPixels": 0,
                "rawScan": 0,
                "rawBudget": 0,
                "rawPrefilter": 0,
                "validated": 0,
                "final": 0,
            },
        )
        aggregate["templates"] = int(aggregate["templates"]) + 1
        for key in (
            "variants",
            "rois",
            "roiAreaPixels",
            "foregroundPixels",
            "rawScan",
            "rawBudget",
            "rawPrefilter",
            "validated",
            "final",
        ):
            aggregate[key] = int(aggregate[key]) + int(row[key])

    by_strategy = []
    for row in strategy_rows.values():
        validated = int(row["validated"])
        final = int(row["final"])
        row["validatedPerFinal"] = round(validated / max(1, final), 2)
        by_strategy.append(row)

    by_strategy.sort(key=lambda item: (-int(item["roiAreaPixels"]), str(item["strategy"])))
    template_rows.sort(key=lambda item: (-int(item["roiAreaPixels"]), item["templateName"]))
    return {
        "byStrategy": by_strategy,
        "byTemplate": template_rows,
    }
