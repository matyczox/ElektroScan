"""Gray detector rescue, dedupe and postprocess helpers."""

from __future__ import annotations

import numpy as np

from core.detector_clustering import _bbox_metrics
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
from core.detector_selection import (
    candidate_quality_key,
    local_dominates,
    same_physical_place,
)
from core.detector_gray_masks import gray_template_area, gray_template_pixels, is_gray_rect_frame_template
from core.detector_gray_rules import (
    _center_inside_bbox,
    _compact_gray_hit_beats_large_partial,
    _deep_same_ink_overlap,
    _hit_area,
    _hit_center,
    _is_complex_gray_geometry_hit,
    _is_full_gray_text_label_hit,
    _is_gray_frame_validated_rescue_hit,
    _is_gray_rect_frame_hit,
    _is_gray_symbol_hit,
    _is_mid_gray_geometry_hit,
    _is_same_template_duplicate_shadow,
    _is_strong_gray_label_geometry_hit,
    _is_tiny_gray_geometry_hit,
    _is_weak_gray_text_fragment,
    _nested_gray_fragment_loses_to_stronger_hit,
    _same_physical_hit,
    _same_template_close_shadow_loses_to_larger,
    _strong_compact_gray_hit_should_coexist,
    _weak_gray_compact_fragment_loses_to_larger,
    is_gray_frame_raw_rescue_hit,
)











































def _suppress_nested_gray_core_hits(hits: list[CandidateHit]) -> list[CandidateHit]:
    """Drop smaller gray sub-symbols when a fuller symbol covers the same ink."""

    if len(hits) < 2:
        return hits

    suppressed: set[int] = set()
    for small_idx, small in enumerate(hits):
        if small.dominant_hsv is not None:
            continue

        small_area = _hit_area(small)
        for large_idx, large in enumerate(hits):
            if small_idx == large_idx or large.dominant_hsv is not None:
                continue
            if _same_template_close_shadow_loses_to_larger(small, large):
                suppressed.add(small_idx)
                break
            if _is_same_template_duplicate_shadow(small, large):
                suppressed.add(small_idx)
                break
            if _nested_gray_fragment_loses_to_stronger_hit(small, large):
                suppressed.add(small_idx)
                break
            if _weak_gray_compact_fragment_loses_to_larger(small, large):
                suppressed.add(small_idx)
                break
            area_ratio = 1.25 if large.template_id == small.template_id else GRAY_FULLER_SYMBOL_MIN_AREA_RATIO
            if _hit_area(large) < small_area * area_ratio:
                continue
            if large.coverage < min(0.90, GRAY_FULLER_SYMBOL_MIN_COVERAGE):
                continue
            if large.purity < GRAY_FULLER_SYMBOL_MIN_PURITY:
                continue
            if large.verification_score + GRAY_FULLER_SYMBOL_MAX_VERIFICATION_DROP < small.verification_score:
                continue
            if _strong_compact_gray_hit_should_coexist(small, large):
                continue
            small_center = _hit_center(small)
            if not (
                _same_physical_hit(small, large)
                or _center_inside_bbox(small_center, large.bbox, margin_ratio=0.03)
            ):
                continue

            suppressed.add(small_idx)
            break

    for large_idx, large in enumerate(hits):
        if large_idx in suppressed or large.dominant_hsv is not None:
            continue

        for compact_idx, compact in enumerate(hits):
            if compact_idx == large_idx or compact_idx in suppressed:
                continue
            if _compact_gray_hit_beats_large_partial(compact, large):
                suppressed.add(large_idx)
                break

    if not suppressed:
        return hits

    return [hit for index, hit in enumerate(hits) if index not in suppressed]


def _dedupe_gray_overlapping_alternatives(hits: list[CandidateHit]) -> list[CandidateHit]:
    """Keep one gray interpretation when several symbols explain the same ink."""

    if len(hits) < 2:
        return hits

    def _score_rank(hit: CandidateHit) -> float:
        area_bonus = min(0.45, float(np.log1p(_hit_area(hit))) * 0.05)
        return float(
            hit.verification_score
            + hit.match_score
            + hit.context_purity
            + 0.70 * hit.purity
            + area_bonus
        )

    def _fuller_rank(hit: CandidateHit) -> tuple[float, ...]:
        return (
            float(_hit_area(hit)),
            float(hit.pixel_count),
            float(hit.context_purity),
            float(hit.purity),
            float(hit.coverage),
            float(hit.verification_score),
            float(hit.match_score),
            0.0 if hit.mirrored else 1.0,
        )

    def _near_identical_gray_bbox(left: CandidateHit, right: CandidateHit) -> bool:
        inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
        if inter_area <= 0:
            return False

        left_area = _hit_area(left)
        right_area = _hit_area(right)
        area_ratio = max(left_area, right_area) / max(1, min(left_area, right_area))
        lw, lh = left.bbox[2], left.bbox[3]
        rw, rh = right.bbox[2], right.bbox[3]
        width_ratio = max(lw, rw) / max(1, min(lw, rw))
        height_ratio = max(lh, rh) / max(1, min(lh, rh))
        return (
            iou >= 0.88
            and iom >= 0.94
            and center_distance <= 0.12
            and area_ratio <= 1.18
            and width_ratio <= 1.15
            and height_ratio <= 1.15
        )

    def _competing_winner(
        left: CandidateHit,
        right: CandidateHit,
    ) -> CandidateHit | None:
        if left.dominant_hsv is not None or right.dominant_hsv is not None:
            return None
        left_full_label = _is_full_gray_text_label_hit(left)
        right_full_label = _is_full_gray_text_label_hit(right)
        if left_full_label != right_full_label:
            full_label = left if left_full_label else right
            other = right if full_label is left else left
            full_area = _hit_area(full_label)
            other_area = _hit_area(other)
            inter_area, _iou, iom, center_distance = _bbox_metrics(full_label.bbox, other.bbox)
            if inter_area > 0 and full_area >= other_area * 1.15 and (
                iom >= 0.45 or center_distance <= 0.55
            ):
                other_is_low_purity_content_fragment = (
                    other.source == "template_content"
                    and other.purity <= 0.35
                    and other.context_purity <= 0.18
                    and full_area >= other_area * 1.45
                    and full_label.purity >= other.purity + 0.30
                )
                if other_is_low_purity_content_fragment and (
                    full_label.verification_score + 0.14 >= other.verification_score
                    and full_label.match_score + 0.22 >= other.match_score
                    and full_label.coverage + 0.08 >= other.coverage
                ):
                    return full_label
        if _near_identical_gray_bbox(left, right):
            left_rank = (_score_rank(left), *candidate_quality_key(left, mode="gray"))
            right_rank = (_score_rank(right), *candidate_quality_key(right, mode="gray"))
            return left if left_rank >= right_rank else right
        if left.is_text_label != right.is_text_label:
            if _nested_gray_fragment_loses_to_stronger_hit(left, right):
                return right
            if _nested_gray_fragment_loses_to_stronger_hit(right, left):
                return left
            return None
        if not _same_physical_hit(left, right):
            return None
        if _strong_compact_gray_hit_should_coexist(left, right):
            return None
        if local_dominates(left, right, mode="gray"):
            return left
        if local_dominates(right, left, mode="gray"):
            return right

        score_left = _score_rank(left)
        score_right = _score_rank(right)
        score_margin = abs(score_left - score_right)
        area_ratio = max(_hit_area(left), _hit_area(right)) / max(
            1,
            min(_hit_area(left), _hit_area(right)),
        )
        similar_size_gray_symbols = (
            not left.is_text_label
            and not right.is_text_label
            and area_ratio <= 1.25
            and score_margin < 0.22
        )
        if similar_size_gray_symbols:
            fuller = max((left, right), key=_fuller_rank)
            other = right if fuller is left else left
            fuller_is_quality_tie = score_margin < 0.10
            fuller_not_substantially_weaker = (
                fuller.verification_score + 0.035 >= other.verification_score
                and fuller.match_score + 0.055 >= other.match_score
                and fuller.coverage + 0.050 >= other.coverage
                and fuller.purity + 0.040 >= other.purity
            )
            if fuller_is_quality_tie or fuller_not_substantially_weaker:
                return fuller
        return left if score_left >= score_right else right

    ordered_hits = sorted(
        hits,
        key=lambda hit: (
            _score_rank(hit),
            *candidate_quality_key(hit, mode="gray"),
            -float(hit.bbox[1]),
            -float(hit.bbox[0]),
        ),
        reverse=True,
    )

    selected: list[CandidateHit] = []
    for candidate in ordered_hits:
        candidate_survives = True
        next_selected: list[CandidateHit] = []
        for existing in selected:
            winner = _competing_winner(candidate, existing)
            if winner is existing:
                candidate_survives = False
                next_selected.append(existing)
            elif winner is candidate:
                continue
            else:
                next_selected.append(existing)
        if candidate_survives:
            next_selected.append(candidate)
        selected = next_selected

    selected.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.verification_score))
    return selected


def _filter_weak_gray_text_fragments(hits: list[CandidateHit]) -> list[CandidateHit]:
    """Drop label fragments that only explain a thin slice of a larger symbol."""

    return [hit for hit in hits if not _is_weak_gray_text_fragment(hit)]


def _merge_duplicate_gray_rect_frames(
    hits: list[CandidateHit],
    templates: list[TemplateInfo],
) -> tuple[list[CandidateHit], int]:
    """Merge shifted hollow-frame detections that represent one oversized frame."""

    if len(hits) < 2:
        return hits, 0

    parent = list(range(len(hits)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    def same_frame(left: CandidateHit, right: CandidateHit) -> bool:
        if left.template_id != right.template_id:
            return False
        if not (_is_gray_rect_frame_hit(left, templates) and _is_gray_rect_frame_hit(right, templates)):
            return False
        if (left.rotation % 180) != (right.rotation % 180):
            return False
        if abs(float(left.scale) - float(right.scale)) > GRAY_RECT_FRAME_MERGE_MAX_SCALE_DELTA:
            return False

        _inter, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
        if iou <= 0.0:
            return False
        return (
            iom >= GRAY_RECT_FRAME_MERGE_IOM
            and center_distance <= GRAY_RECT_FRAME_MERGE_CENTER_DISTANCE
        )

    for left_index in range(len(hits)):
        for right_index in range(left_index + 1, len(hits)):
            if same_frame(hits[left_index], hits[right_index]):
                union(left_index, right_index)

    groups: dict[int, list[CandidateHit]] = {}
    for index, hit in enumerate(hits):
        groups.setdefault(find(index), []).append(hit)

    merged_count = 0
    output: list[CandidateHit] = []
    for group_hits in groups.values():
        if len(group_hits) == 1:
            output.append(group_hits[0])
            continue

        winner = max(
            group_hits,
            key=lambda hit: (
                float(hit.verification_score),
                float(hit.match_score),
                float(hit.coverage),
            ),
        )
        min_x = min(hit.bbox[0] for hit in group_hits)
        min_y = min(hit.bbox[1] for hit in group_hits)
        max_x = max(hit.bbox[0] + hit.bbox[2] for hit in group_hits)
        max_y = max(hit.bbox[1] + hit.bbox[3] for hit in group_hits)
        winner.bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        output.append(winner)
        merged_count += len(group_hits) - 1

    output.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.verification_score))
    return output, merged_count


def _is_gray_rescue_blocked_by_existing(
    hit: CandidateHit,
    existing: CandidateHit,
) -> bool:
    """Do not resurrect small interior ghosts already defeated by clustering."""

    if hit.dominant_hsv is not None or existing.dominant_hsv is not None:
        return False
    if local_dominates(existing, hit, mode="gray"):
        return True
    if existing.template_id == hit.template_id and _same_physical_hit(existing, hit):
        return candidate_quality_key(existing, mode="gray") >= candidate_quality_key(hit, mode="gray")

    if _compact_gray_hit_beats_large_partial(hit, existing):
        return False
    if _is_weak_gray_text_fragment(existing) and (
        _is_tiny_gray_geometry_hit(hit) or _is_mid_gray_geometry_hit(hit)
    ):
        return False

    hit_area = _hit_area(hit)
    existing_area = _hit_area(existing)
    if existing_area < hit_area * 2.4:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(hit.bbox, existing.bbox)
    if inter_area <= 0:
        return False

    center_nested = _center_inside_bbox(_hit_center(hit), existing.bbox)
    if not (iom >= 0.72 or (center_nested and center_distance <= 0.62)):
        return False

    if existing.verification_score < 0.50 and existing.match_score < 0.60:
        return False

    # True rescues have their own surrounding ink. Interior ghosts usually
    # borrow the parent/label geometry, so their local context is much weaker.
    return hit.context_purity <= 0.35 or hit.purity <= 0.55 or existing.is_text_label


def _is_gray_rescue_locally_dominated(
    hit: CandidateHit,
    competitor: CandidateHit,
) -> bool:
    """Skip a rescued interpretation beaten by a better local gray candidate."""

    return local_dominates(competitor, hit, mode="gray")


def rescue_validated_gray_frame_hits(
    final_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    templates: list[TemplateInfo],
) -> tuple[list[CandidateHit], int, dict[str, dict[str, object]]]:
    """Re-add strong gray detections lost by global NMS/clustering."""

    trace: dict[str, dict[str, object]] = {}

    def _trace_stage(
        stage: str,
        hits: list[CandidateHit],
        reasons: dict[int, str] | None = None,
    ) -> None:
        trace[stage] = {"hits": hits, "reasons": reasons or {}}

    def _rescue_rank(hit: CandidateHit) -> tuple[float, ...]:
        x, y, width, height = hit.bbox
        return (
            *candidate_quality_key(hit, mode="gray"),
            -float(y),
            -float(x),
            -float(width * height),
            -float(hit.template_id),
        )

    rescue_candidates = sorted(
        (hit for hit in validated_hits if _is_gray_frame_validated_rescue_hit(hit, templates)),
        key=_rescue_rank,
        reverse=True,
    )
    _trace_stage("rescue_candidates", rescue_candidates)

    rescued: list[CandidateHit] = []
    dominated: list[CandidateHit] = []
    dominated_reasons: dict[int, str] = {}
    blocked: list[CandidateHit] = []
    blocked_reasons: dict[int, str] = {}
    local_competitors = final_hits

    for hit in rescue_candidates:
        dominator = next(
            (
                competitor
                for competitor in local_competitors
                if competitor is not hit and _is_gray_rescue_locally_dominated(hit, competitor)
                and not (
                    competitor.template_id == hit.template_id
                    and competitor.bbox == hit.bbox
                    and candidate_quality_key(competitor, mode="gray")
                    == candidate_quality_key(hit, mode="gray")
                )
            ),
            None,
        )
        if dominator is not None:
            dominated.append(hit)
            dominated_reasons[id(hit)] = (
                f"dominated_by:{templates[dominator.template_id].name}"
                if 0 <= dominator.template_id < len(templates)
                else "dominated_by:unknown"
            )
            continue

        duplicate = next(
            (
                existing
                for existing in final_hits + rescued
                if existing.template_id == hit.template_id and _same_physical_hit(existing, hit)
            ),
            None,
        )
        if duplicate is not None:
            blocked.append(hit)
            blocked_reasons[id(hit)] = "duplicate_same_template"
            continue

        blocker = next(
            (
                existing
                for existing in final_hits + rescued
                if _is_gray_rescue_blocked_by_existing(hit, existing)
            ),
            None,
        )
        if blocker is not None:
            blocked.append(hit)
            blocked_reasons[id(hit)] = (
                f"blocked_by:{templates[blocker.template_id].name}"
                if 0 <= blocker.template_id < len(templates)
                else "blocked_by:unknown"
            )
            continue

        rescued.append(hit)

    _trace_stage("rescue_dominated", dominated, dominated_reasons)
    _trace_stage("rescue_blocked_existing", blocked, blocked_reasons)
    _trace_stage("rescue_added", rescued)

    combined = final_hits + rescued
    weak_text_fragments = [hit for hit in combined if _is_weak_gray_text_fragment(hit)]
    if weak_text_fragments:
        _trace_stage(
            "post_gray_filter_weak_text_removed",
            weak_text_fragments,
            {id(hit): "small_text_fragment" for hit in weak_text_fragments},
        )
    combined = _filter_weak_gray_text_fragments(combined)
    _trace_stage("post_gray_filter_weak_text", combined)

    combined = _suppress_nested_gray_core_hits(combined)
    _trace_stage("post_gray_suppress_nested", combined)

    combined = _dedupe_gray_overlapping_alternatives(combined)
    _trace_stage("post_gray_dedupe", combined)

    combined, _merged_count = _merge_duplicate_gray_rect_frames(combined, templates)
    _trace_stage("post_gray_merge_frames", combined)

    combined.sort(key=lambda item: (item.bbox[1], item.bbox[0], -item.verification_score))
    return combined, len(rescued), trace


def trace_unresolved_strong_gray_hits(
    final_hits: list[CandidateHit],
    validated_hits: list[CandidateHit],
    templates: list[TemplateInfo],
) -> dict:
    """Explain strong validated gray hits that did not survive final output."""

    strong_hits = [
        hit for hit in validated_hits if _is_gray_frame_validated_rescue_hit(hit, templates)
    ]
    unresolved: list[dict] = []

    for hit in sorted(strong_hits, key=lambda item: item.verification_score, reverse=True):
        same_template_final = [
            existing
            for existing in final_hits
            if existing.template_id == hit.template_id and _same_physical_hit(existing, hit)
        ]
        if same_template_final:
            continue

        blockers: list[tuple[float, CandidateHit]] = []
        for existing in final_hits:
            inter_area, iou, iom, center_distance = _bbox_metrics(existing.bbox, hit.bbox)
            if inter_area <= 0:
                continue
            if iou >= 0.10 or iom >= 0.45 or center_distance <= 0.45:
                blockers.append((max(iou, iom), existing))
        blockers.sort(key=lambda item: (item[0], item[1].verification_score), reverse=True)

        blocker = blockers[0][1] if blockers else None
        unresolved.append(
            {
                "symbol": templates[hit.template_id].name,
                "bbox": [int(value) for value in hit.bbox],
                "match": round(float(hit.match_score), 3),
                "verification": round(float(hit.verification_score), 3),
                "coverage": round(float(hit.coverage), 3),
                "purity": round(float(hit.purity), 3),
                "contextPurity": round(float(hit.context_purity), 3),
                "rotation": int(hit.rotation),
                "scale": round(float(hit.scale), 3),
                "mirrored": bool(hit.mirrored),
                "blockedBy": (
                    {
                        "symbol": templates[blocker.template_id].name,
                        "bbox": [int(value) for value in blocker.bbox],
                        "match": round(float(blocker.match_score), 3),
                        "verification": round(float(blocker.verification_score), 3),
                    }
                    if blocker is not None
                    else None
                ),
            }
        )

    max_items = max(0, int(GRAY_STRONG_TRACE_MAX_ITEMS))
    return {
        "strongValidated": len(strong_hits),
        "unresolved": len(unresolved),
        "items": unresolved[:max_items],
    }
