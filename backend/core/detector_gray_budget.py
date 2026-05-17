"""Gray detector raw-hit budget and ranking helpers."""

from __future__ import annotations

import numpy as np

from core.detector_config import (
    GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT,
    GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE,
    GRAY_COMPLEX_GEOMETRY_MIN_PURITY,
    GRAY_DARK_EVIDENCE_THRESHOLD,
    GRAY_DARK_INK_THRESHOLD,
    GRAY_DARK_ZONE_THRESHOLD,
    GRAY_DIAGONAL_TEXT_LABEL_MAX_ASPECT,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_AREA,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_CONTEXT,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_COVERAGE,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_MATCH,
    GRAY_DIAGONAL_TEXT_LABEL_MIN_PURITY,
    GRAY_ELONGATED_SCAN_MAX_TEMPLATE_PIXELS,
    GRAY_ELONGATED_SCAN_THRESHOLD,
    GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP,
    GRAY_FULLER_SYMBOL_MIN_AREA_RATIO,
    GRAY_FULLER_SYMBOL_MIN_COVERAGE,
    GRAY_FULLER_SYMBOL_MIN_PURITY,
    GRAY_LEGEND_DARK_MARGIN,
    GRAY_LEGEND_EVIDENCE_MARGIN,
    GRAY_LEGEND_INK_PERCENTILE,
    GRAY_LABEL_GEOMETRY_MIN_CONTEXT,
    GRAY_LABEL_GEOMETRY_MIN_COVERAGE,
    GRAY_LABEL_GEOMETRY_MIN_MATCH,
    GRAY_LABEL_GEOMETRY_MIN_PURITY,
    GRAY_LABEL_GEOMETRY_MIN_VERIFICATION,
    GRAY_LEGEND_MIN_EVIDENCE_THRESHOLD,
    GRAY_LEGEND_MIN_ZONE_THRESHOLD,
    GRAY_LEGEND_ZONE_MARGIN,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ASPECT,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_TEMPLATE_AREA,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_CONTEXT,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_COVERAGE,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_MATCH,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_PURITY,
    GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA,
    GRAY_MID_GEOMETRY_MIN_CONTEXT,
    GRAY_MID_GEOMETRY_MIN_COVERAGE,
    GRAY_MID_GEOMETRY_MIN_MATCH,
    GRAY_MID_GEOMETRY_MIN_PURITY,
    GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS,
    GRAY_MID_GEOMETRY_MIN_VERIFICATION,
    GRAY_LINE_CROSSED_LABEL_MIN_CONTEXT,
    GRAY_LINE_CROSSED_LABEL_MIN_COVERAGE,
    GRAY_LINE_CROSSED_LABEL_MIN_MATCH,
    GRAY_LINE_CROSSED_LABEL_MIN_PURITY,
    GRAY_LINE_CROSSED_LABEL_MIN_SCALE,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_CONTEXT,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_COVERAGE,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_MATCH,
    GRAY_LINE_CROSSED_LABEL_ALT_MIN_PURITY,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT,
    GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_CONTEXT,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_COVERAGE,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_MATCH,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_PURITY,
    GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA,
    GRAY_RAW_MAX_HITS_PER_TEMPLATE,
    GRAY_RAW_MAX_HITS_PER_VARIANT,
    GRAY_RAW_MAX_TOTAL_HITS,
    GRAY_RAW_MIN_HITS_PER_TEMPLATE,
    GRAY_RAW_SCAN_MIN_TEMPLATE_AREA,
    GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS,
    GRAY_RAW_SCAN_THRESHOLD,
    GRAY_RECT_FRAME_MAX_CENTER_DENSITY,
    GRAY_RECT_FRAME_MAX_DENSITY,
    GRAY_RECT_FRAME_MAX_RAW_SCAN_SCALE,
    GRAY_RECT_FRAME_MERGE_CENTER_DISTANCE,
    GRAY_RECT_FRAME_MERGE_IOM,
    GRAY_RECT_FRAME_MERGE_MAX_SCALE_DELTA,
    GRAY_RECT_FRAME_MIN_ASPECT,
    GRAY_RECT_FRAME_MIN_DENSITY,
    GRAY_RECT_FRAME_MIN_RAW_SCAN_SCALE,
    GRAY_RECT_FRAME_STRONG_EDGE_COVERAGE,
    GRAY_RECT_FRAME_WEAK_EDGE_COVERAGE,
    GRAY_SEARCH_COMPONENT_PADDING_RATIO,
    GRAY_SEARCH_COMPONENT_DILATE_ITERATIONS,
    GRAY_SEARCH_CONNECTED_FAST_COMPONENTS_MAX,
    GRAY_SEARCH_CONNECTED_FAST_MAX_TILE_ROIS,
    GRAY_SEARCH_FAST_MAX_TILE_ROIS,
    GRAY_SEARCH_FAST_TILE_SIZE,
    GRAY_SEARCH_LARGE_TEXT_MAX_TILE_ROIS,
    GRAY_SEARCH_LARGE_TEXT_MIN_TEMPLATE_AREA,
    GRAY_SEARCH_LARGE_TEXT_TILE_SIZE,
    GRAY_SEARCH_MAX_ROIS,
    GRAY_SEARCH_MAX_TILE_ROIS,
    GRAY_SEARCH_ROI_CONTAINMENT_THRESHOLD,
    GRAY_SEARCH_ROI_OVERLAP_THRESHOLD,
    GRAY_SEARCH_SAFE_ELONGATED_ASPECT,
    GRAY_SEARCH_TILE_MIN_FOREGROUND,
    GRAY_SEARCH_TILE_PADDING,
    GRAY_SEARCH_TILE_SIZE,
    GRAY_SPATIAL_FAIR_PEAKS_PER_ROI,
    GRAY_STRICT_SCAN_THRESHOLD,
    GRAY_STRONG_GEOMETRY_MIN_COVERAGE,
    GRAY_STRONG_GEOMETRY_MIN_MATCH,
    GRAY_STRONG_RESCUE_MIN_PURITY,
    GRAY_STRONG_RESCUE_MIN_VERIFICATION,
    GRAY_STRONG_TRACE_MAX_ITEMS,
    GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
    GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    GRAY_TINY_GEOMETRY_MAX_SCALE,
    GRAY_TINY_GEOMETRY_MAX_TEMPLATE_PIXELS,
    GRAY_TINY_GEOMETRY_MIN_CONTEXT,
    GRAY_TINY_GEOMETRY_MIN_COVERAGE,
    GRAY_TINY_GEOMETRY_MIN_MATCH,
    GRAY_TINY_GEOMETRY_MIN_PURITY,
    GRAY_TINY_GEOMETRY_MIN_VERIFICATION,
    GRAY_WEAK_LABEL_MAX_CONTEXT,
    GRAY_WEAK_LABEL_MAX_PURITY,
)
from core.detector_masks import _context_purity
from core.detector_models import CandidateHit, TemplateInfo
from core.detector_gray_masks import (
    _clamp_roi,
    gray_template_area,
    gray_template_pixels,
    is_gray_rect_frame_template,
    use_raw_gray_scan_mask,
)

def _gray_budget_geometry_score(
    hit: CandidateHit,
    plan_mask: np.ndarray,
) -> tuple[float, float] | None:
    """Cheap coverage/purity check used only to protect strong gray raw hits."""

    if hit.transformed_mask is None or hit.pixel_count <= 0:
        return None

    clamped = _clamp_roi(hit.bbox, plan_mask.shape)
    if clamped is None:
        return None

    x, y, w, h = clamped
    roi = plan_mask[y : y + h, x : x + w]
    if roi.shape != hit.transformed_mask.shape:
        return None

    roi_foreground = int(cv2.countNonZero(roi))
    if roi_foreground <= 0:
        return None

    intersection_mask = cv2.bitwise_and(roi, hit.transformed_mask)
    intersection = int(cv2.countNonZero(intersection_mask))
    coverage = intersection / max(1, hit.pixel_count)
    purity = intersection / max(1, roi_foreground)
    standard_geometry_failed = (
        hit.match_score < GRAY_STRONG_GEOMETRY_MIN_MATCH
        or coverage < GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        or purity < GRAY_STRONG_RESCUE_MIN_PURITY
    )
    hit_aspect = max(hit.bbox[2] / max(1, hit.bbox[3]), hit.bbox[3] / max(1, hit.bbox[2]))
    near_threshold_geometry = (
        hit.source == "template_near_threshold"
        and hit.match_score >= GRAY_NEAR_THRESHOLD_RECOVERY_MIN_MATCH
        and GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA
        <= hit.bbox[2] * hit.bbox[3]
        <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA
        and hit_aspect <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT
        and coverage >= GRAY_NEAR_THRESHOLD_RECOVERY_MIN_COVERAGE
        and purity >= GRAY_NEAR_THRESHOLD_RECOVERY_MIN_PURITY
        and _context_purity(plan_mask, hit.bbox, intersection_mask)
        >= GRAY_NEAR_THRESHOLD_RECOVERY_MIN_CONTEXT
    )
    line_crossed_near_threshold_common = (
        hit.is_text_label
        and hit.source == "template_near_threshold"
        and hit.scale >= GRAY_LINE_CROSSED_LABEL_MIN_SCALE
        and GRAY_NEAR_THRESHOLD_RECOVERY_MIN_TEMPLATE_AREA
        <= hit.bbox[2] * hit.bbox[3]
        <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_TEMPLATE_AREA
        and hit_aspect <= GRAY_NEAR_THRESHOLD_RECOVERY_MAX_ASPECT
    )
    line_crossed_context = (
        _context_purity(plan_mask, hit.bbox, intersection_mask)
        if line_crossed_near_threshold_common
        else 0.0
    )
    line_crossed_near_threshold_geometry = (
        line_crossed_near_threshold_common
        and (
            (
                hit.match_score >= GRAY_LINE_CROSSED_LABEL_MIN_MATCH
                and coverage >= GRAY_LINE_CROSSED_LABEL_MIN_COVERAGE
                and purity >= GRAY_LINE_CROSSED_LABEL_MIN_PURITY
                and line_crossed_context >= GRAY_LINE_CROSSED_LABEL_MIN_CONTEXT
            )
            or (
                hit.match_score >= GRAY_LINE_CROSSED_LABEL_ALT_MIN_MATCH
                and coverage >= GRAY_LINE_CROSSED_LABEL_ALT_MIN_COVERAGE
                and purity >= GRAY_LINE_CROSSED_LABEL_ALT_MIN_PURITY
                and line_crossed_context >= GRAY_LINE_CROSSED_LABEL_ALT_MIN_CONTEXT
            )
        )
    )
    diagonal_text_label_geometry = (
        hit.is_text_label
        and hit.rotation % 90 != 0
        and hit.match_score >= GRAY_DIAGONAL_TEXT_LABEL_MIN_MATCH
        and GRAY_DIAGONAL_TEXT_LABEL_MIN_AREA
        <= hit.bbox[2] * hit.bbox[3]
        and hit_aspect <= GRAY_DIAGONAL_TEXT_LABEL_MAX_ASPECT
        and coverage >= GRAY_DIAGONAL_TEXT_LABEL_MIN_COVERAGE
        and purity >= GRAY_DIAGONAL_TEXT_LABEL_MIN_PURITY
        and _context_purity(plan_mask, hit.bbox, intersection_mask)
        >= GRAY_DIAGONAL_TEXT_LABEL_MIN_CONTEXT
    )
    interrupted_label_geometry = (
        hit.source == "template_interrupted_recovery"
        and hit.match_score >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_MATCH
        and GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_TEMPLATE_AREA
        <= hit.bbox[2] * hit.bbox[3]
        <= GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_TEMPLATE_AREA
        and hit_aspect <= GRAY_INTERRUPTED_LABEL_RECOVERY_MAX_ASPECT
        and coverage >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_COVERAGE
        and purity >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_PURITY
        and _context_purity(plan_mask, hit.bbox, intersection_mask)
        >= GRAY_INTERRUPTED_LABEL_RECOVERY_MIN_CONTEXT
    )
    strong_complex_geometry = (
        hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_RAW_SCAN_THRESHOLD
        and coverage >= GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE
        and purity >= GRAY_COMPLEX_GEOMETRY_MIN_PURITY
        and _context_purity(plan_mask, hit.bbox, intersection_mask) >= GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT
    )
    if (
        standard_geometry_failed
        and not strong_complex_geometry
        and not near_threshold_geometry
        and not line_crossed_near_threshold_geometry
        and not diagonal_text_label_geometry
        and not interrupted_label_geometry
    ):
        return None

    return coverage, purity


def gray_raw_budget(
    candidates: list[CandidateHit],
    templates: list[TemplateInfo],
    plan_masks_by_template: dict[int, np.ndarray] | None = None,
) -> tuple[list[CandidateHit], dict]:
    """Keep the strongest gray-mode raw hits before the slower overlap prefilter."""

    if len(candidates) <= GRAY_RAW_MAX_TOTAL_HITS:
        return candidates, {
            "before": len(candidates),
            "after": len(candidates),
            "removed": 0,
            "geometryProtected": 0,
            "geometryProtectedKept": 0,
            "geometryProtectedDropped": 0,
            "perTemplateReserved": 0,
            "perTemplateReservedKept": 0,
            "perTemplateReservedActive": 0,
            "perTemplateReserveLimit": int(max(0, GRAY_RAW_MIN_HITS_PER_TEMPLATE)),
            "topGenerators": [],
        }

    protected_scores: dict[int, tuple[float, float]] = {}
    if plan_masks_by_template is not None:
        for hit in candidates:
            if not (0 <= hit.template_id < len(templates)):
                continue
            plan_mask = plan_masks_by_template.get(hit.template_id)
            if plan_mask is None:
                continue
            score = _gray_budget_geometry_score(hit, plan_mask)
            if score is not None:
                protected_scores[id(hit)] = score

    def _budget_rank(hit: CandidateHit) -> tuple[float, float, float, float, float, float]:
        geometry = protected_scores.get(id(hit))
        if geometry is not None:
            coverage, purity = geometry
            return (
                1.0,
                coverage,
                purity,
                float(hit.match_score),
                -float(hit.bbox[1]),
                -float(hit.bbox[0]),
            )
        return (
            0.0,
            float(hit.match_score),
            0.0,
            0.0,
            -float(hit.bbox[1]),
            -float(hit.bbox[0]),
        )

    def _limit_with_geometry_reserve(
        hits: list[CandidateHit],
        limit: int,
        extra_reserved_ids: set[int] | None = None,
    ) -> list[CandidateHit]:
        """Apply a soft cap without dropping raw hits that already prove geometry."""

        reserved_ids = extra_reserved_ids or set()
        reserved = [hit for hit in hits if id(hit) in protected_scores or id(hit) in reserved_ids]
        unreserved = [
            hit for hit in hits if id(hit) not in protected_scores and id(hit) not in reserved_ids
        ]
        reserved.sort(key=_budget_rank, reverse=True)
        unreserved.sort(key=_budget_rank, reverse=True)
        if len(reserved) >= limit:
            return reserved
        return reserved + unreserved[: max(0, limit - len(reserved))]

    grouped_by_variant: dict[tuple[int, float, int, bool, str], list[CandidateHit]] = {}
    for hit in candidates:
        grouped_by_variant.setdefault(
            (hit.template_id, hit.scale, hit.rotation, hit.mirrored, hit.source),
            [],
        ).append(hit)

    variant_limited: list[CandidateHit] = []
    raw_counts_by_template: dict[int, int] = {}
    for hits in grouped_by_variant.values():
        template_id = hits[0].template_id
        raw_counts_by_template[template_id] = raw_counts_by_template.get(template_id, 0) + len(hits)
        variant_limited.extend(_limit_with_geometry_reserve(hits, GRAY_RAW_MAX_HITS_PER_VARIANT))

    grouped_by_template: dict[int, list[CandidateHit]] = {}
    for hit in variant_limited:
        grouped_by_template.setdefault(hit.template_id, []).append(hit)

    template_limited: list[CandidateHit] = []
    for hits in grouped_by_template.values():
        template_limited.extend(_limit_with_geometry_reserve(hits, GRAY_RAW_MAX_HITS_PER_TEMPLATE))

    per_template_reserved_ids: set[int] = set()
    per_template_reserved_total = 0
    per_template_reserve_limit = max(0, int(GRAY_RAW_MIN_HITS_PER_TEMPLATE))
    if per_template_reserve_limit > 0:
        regrouped_template_limited: dict[int, list[CandidateHit]] = {}
        for hit in template_limited:
            regrouped_template_limited.setdefault(hit.template_id, []).append(hit)
        for hits in regrouped_template_limited.values():
            ranked = sorted(hits, key=_budget_rank, reverse=True)
            for hit in ranked[:per_template_reserve_limit]:
                per_template_reserved_ids.add(id(hit))
            per_template_reserved_total += min(len(ranked), per_template_reserve_limit)

    active_reserved_ids = (
        per_template_reserved_ids
        if len(protected_scores) < int(GRAY_RAW_MAX_TOTAL_HITS)
        else set()
    )
    if len(template_limited) > GRAY_RAW_MAX_TOTAL_HITS:
        template_limited = _limit_with_geometry_reserve(
            template_limited,
            GRAY_RAW_MAX_TOTAL_HITS,
            active_reserved_ids,
        )

    top_generators = []
    for template_id, count in sorted(
        raw_counts_by_template.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:8]:
        top_generators.append(
            {
                "templateId": int(template_id),
                "templateName": templates[template_id].name if 0 <= template_id < len(templates) else "",
                "rawHits": int(count),
            }
        )

    return template_limited, {
        "before": len(candidates),
        "after": len(template_limited),
        "removed": max(0, len(candidates) - len(template_limited)),
        "geometryProtected": len(protected_scores),
        "geometryProtectedKept": sum(1 for hit in template_limited if id(hit) in protected_scores),
        "geometryProtectedDropped": sum(1 for hit in candidates if id(hit) in protected_scores)
        - sum(1 for hit in template_limited if id(hit) in protected_scores),
        "perTemplateReserved": per_template_reserved_total,
        "perTemplateReservedKept": sum(
            1 for hit in template_limited if id(hit) in per_template_reserved_ids
        ),
        "perTemplateReservedActive": len(active_reserved_ids),
        "perTemplateReserveLimit": int(per_template_reserve_limit),
        "topGenerators": top_generators,
    }
