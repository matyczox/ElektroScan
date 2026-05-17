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

def is_gray_frame_raw_rescue_hit(hit: CandidateHit, templates: list[TemplateInfo]) -> bool:
    """Keep promising gray hits through raw de-duplication."""

    if hit.dominant_hsv is not None or not (0 <= hit.template_id < len(templates)):
        return False
    if (
        not hit.is_text_label
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_RAW_SCAN_THRESHOLD
    ):
        return True
    if (
        not hit.is_text_label
        and hit.pixel_count >= GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_MID_GEOMETRY_MIN_MATCH
    ):
        return True
    if (
        hit.is_text_label
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_LABEL_GEOMETRY_MIN_MATCH
    ):
        return True
    return (
        hit.scale <= 0.70
        and hit.match_score >= GRAY_STRONG_GEOMETRY_MIN_MATCH
    )


def _is_complex_gray_geometry_hit(hit: CandidateHit) -> bool:
    """Recognize larger gray symbols whose evidence is geometry, not scale."""

    return (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_RAW_SCAN_THRESHOLD
        and hit.verification_score >= (GRAY_STRONG_RESCUE_MIN_VERIFICATION - 0.03)
        and hit.coverage >= GRAY_COMPLEX_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_COMPLEX_GEOMETRY_MIN_PURITY
        and hit.context_purity >= GRAY_COMPLEX_GEOMETRY_MIN_CONTEXT
    )


def _is_mid_gray_geometry_hit(hit: CandidateHit) -> bool:
    """Recognize medium gray symbols that are too large for small-scale rescue."""

    return (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and hit.pixel_count >= GRAY_MID_GEOMETRY_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_MID_GEOMETRY_MIN_MATCH
        and hit.verification_score >= GRAY_MID_GEOMETRY_MIN_VERIFICATION
        and hit.coverage >= GRAY_MID_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_MID_GEOMETRY_MIN_PURITY
        and hit.context_purity >= GRAY_MID_GEOMETRY_MIN_CONTEXT
    )


def _is_tiny_gray_geometry_hit(hit: CandidateHit) -> bool:
    """Rescue compact gray symbols that are real but score just below 0.60."""

    return (
        hit.dominant_hsv is None
        and not hit.is_text_label
        and hit.pixel_count <= GRAY_TINY_GEOMETRY_MAX_TEMPLATE_PIXELS
        and hit.scale <= GRAY_TINY_GEOMETRY_MAX_SCALE
        and hit.match_score >= GRAY_TINY_GEOMETRY_MIN_MATCH
        and hit.verification_score >= GRAY_TINY_GEOMETRY_MIN_VERIFICATION
        and hit.coverage >= GRAY_TINY_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_TINY_GEOMETRY_MIN_PURITY
        and hit.context_purity >= GRAY_TINY_GEOMETRY_MIN_CONTEXT
    )


def _is_strong_gray_label_geometry_hit(hit: CandidateHit) -> bool:
    """Rescue large text/label templates only when they are very clean."""

    return (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.pixel_count >= GRAY_RAW_SCAN_MIN_TEMPLATE_PIXELS
        and hit.match_score >= GRAY_LABEL_GEOMETRY_MIN_MATCH
        and hit.verification_score >= GRAY_LABEL_GEOMETRY_MIN_VERIFICATION
        and hit.coverage >= GRAY_LABEL_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_LABEL_GEOMETRY_MIN_PURITY
        and hit.context_purity >= GRAY_LABEL_GEOMETRY_MIN_CONTEXT
    )


def _is_gray_frame_validated_rescue_hit(
    hit: CandidateHit,
    templates: list[TemplateInfo],
) -> bool:
    if hit.dominant_hsv is not None or not (0 <= hit.template_id < len(templates)):
        return False
    if (
        _is_complex_gray_geometry_hit(hit)
        or _is_mid_gray_geometry_hit(hit)
        or _is_tiny_gray_geometry_hit(hit)
        or _is_strong_gray_label_geometry_hit(hit)
    ):
        return True
    return (
        hit.scale <= 0.70
        and hit.match_score >= GRAY_STRONG_GEOMETRY_MIN_MATCH
        and hit.coverage >= GRAY_STRONG_GEOMETRY_MIN_COVERAGE
        and hit.purity >= GRAY_STRONG_RESCUE_MIN_PURITY
        and hit.verification_score >= GRAY_STRONG_RESCUE_MIN_VERIFICATION
    )


def _same_physical_hit(left: CandidateHit, right: CandidateHit) -> bool:
    return same_physical_place(left, right, mode="gray")


def _hit_area(hit: CandidateHit) -> int:
    return max(1, hit.bbox[2] * hit.bbox[3])


def _hit_center(hit: CandidateHit) -> tuple[float, float]:
    x, y, width, height = hit.bbox
    return x + width / 2.0, y + height / 2.0


def _center_inside_bbox(
    center: tuple[float, float],
    bbox: tuple[int, int, int, int],
    *,
    margin_ratio: float = 0.04,
) -> bool:
    x, y = center
    bx, by, bw, bh = bbox
    pad_x = bw * margin_ratio
    pad_y = bh * margin_ratio
    return (bx - pad_x) <= x <= (bx + bw + pad_x) and (by - pad_y) <= y <= (
        by + bh + pad_y
    )


def _is_gray_symbol_hit(hit: CandidateHit) -> bool:
    return hit.dominant_hsv is None and not hit.is_text_label


def _is_weak_gray_text_fragment(hit: CandidateHit) -> bool:
    # Narrow text-label symbols can be real even with modest purity: nearby
    # lettering and frames add foreground to the ROI.  Keep them when the
    # template itself is almost perfectly covered and verification is strong.
    strong_label_geometry = (
        hit.match_score >= 0.68
        and hit.verification_score >= 0.60
        and hit.coverage >= 0.94
        and hit.purity >= 0.48
    )
    strong_content_label_geometry = (
        hit.source == "template_content"
        and hit.scale >= 0.85
        and hit.match_score >= 0.72
        and hit.verification_score >= 0.74
        and hit.coverage >= 0.95
        and hit.content_score >= 0.82
    )
    full_content_label_geometry = (
        hit.source == "template_content"
        and hit.scale >= 0.85
        and hit.bbox[2] * hit.bbox[3] >= 1000
        and max(hit.bbox[2] / max(1, hit.bbox[3]), hit.bbox[3] / max(1, hit.bbox[2]))
        >= 1.30
        and hit.verification_score >= 0.70
        and hit.coverage >= 0.94
        and hit.purity >= 0.32
        and hit.context_purity >= 0.18
    )
    full_template_label_geometry = (
        hit.source == "template"
        and hit.scale >= 0.85
        and hit.bbox[2] * hit.bbox[3] >= 1000
        and hit.match_score >= 0.60
        and hit.verification_score >= 0.60
        and hit.coverage >= 0.90
        and hit.purity >= 0.50
        and hit.context_purity >= 0.18
    )
    interrupted_content_label_geometry = (
        hit.source == "template_content"
        and hit.scale >= 0.85
        and hit.bbox[2] * hit.bbox[3] >= 1000
        and hit.match_score >= 0.69
        and hit.verification_score >= 0.68
        and hit.coverage >= 0.84
        and hit.purity >= 0.28
        and hit.context_purity >= 0.16
        and hit.content_score >= 0.76
    )
    diagonal_content_label_geometry = (
        hit.source == "template_content"
        and hit.rotation % 90 != 0
        and hit.scale >= 0.85
        and hit.bbox[2] * hit.bbox[3] >= 1000
        and hit.match_score >= 0.58
        and hit.verification_score >= 0.66
        and hit.coverage >= 0.95
        and hit.content_score >= 0.74
        and hit.context_purity >= 0.14
    )
    tiny_text_fragment = (
        hit.dominant_hsv is None
        and hit.is_text_label
        and hit.source == "template"
        and hit.scale <= 0.55
        and max(hit.bbox[2], hit.bbox[3]) <= 26
        and hit.context_purity <= 0.30
        and hit.match_score <= 0.62
        and hit.verification_score <= 0.76
    )
    return (
        tiny_text_fragment
        or (
            hit.dominant_hsv is None
            and hit.is_text_label
            and not strong_content_label_geometry
            and not full_content_label_geometry
            and not full_template_label_geometry
            and not interrupted_content_label_geometry
            and not diagonal_content_label_geometry
            and not strong_label_geometry
            and hit.context_purity <= GRAY_WEAK_LABEL_MAX_CONTEXT
            and hit.purity <= GRAY_WEAK_LABEL_MAX_PURITY
        )
    )


def _is_full_gray_text_label_hit(hit: CandidateHit) -> bool:
    if hit.dominant_hsv is not None or not hit.is_text_label:
        return False
    if hit.scale < 0.85:
        return False
    area = _hit_area(hit)
    aspect = max(hit.bbox[2] / max(1, hit.bbox[3]), hit.bbox[3] / max(1, hit.bbox[2]))
    if area < 1000 or aspect < 1.35:
        return False
    if hit.coverage < 0.88 or hit.purity < 0.34:
        return False
    if hit.content_score < 0.70 or hit.verification_score < 0.66:
        return False
    if hit.source == "template":
        return hit.match_score >= 0.62
    if hit.source == "template_content":
        return (
            hit.match_score >= 0.70
            and hit.content_score >= 0.82
        ) or (
            hit.verification_score >= 0.70
            and hit.coverage >= 0.94
            and hit.purity >= 0.36
            and hit.context_purity >= 0.18
        )
    return False


def _is_gray_rect_frame_hit(hit: CandidateHit, templates: list[TemplateInfo]) -> bool:
    return (
        _is_gray_symbol_hit(hit)
        and 0 <= hit.template_id < len(templates)
        and is_gray_rect_frame_template(templates[hit.template_id])
    )


def _is_same_template_duplicate_shadow(small: CandidateHit, large: CandidateHit) -> bool:
    if small.template_id != large.template_id:
        return False

    small_area = _hit_area(small)
    large_area = _hit_area(large)
    if large_area < small_area * 1.20:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(small.bbox, large.bbox)
    if inter_area <= 0:
        return False

    center_nested = _center_inside_bbox(_hit_center(small), large.bbox, margin_ratio=0.04)
    if not (iom >= 0.55 or (center_nested and center_distance <= 0.58)):
        return False

    min_large_coverage = 0.86 if large.is_text_label else 0.88
    if large.coverage < min_large_coverage:
        return False

    return large.verification_score + 0.12 >= small.verification_score


def _deep_same_ink_overlap(fragment: CandidateHit, larger: CandidateHit) -> bool:
    inter_area, _iou, iom, _center_distance = _bbox_metrics(fragment.bbox, larger.bbox)
    if inter_area <= 0:
        return False
    return iom >= 0.72


def _nested_gray_fragment_loses_to_stronger_hit(
    fragment: CandidateHit,
    stronger: CandidateHit,
) -> bool:
    if fragment.dominant_hsv is not None or stronger.dominant_hsv is not None:
        return False
    fragment_area = _hit_area(fragment)
    stronger_area = _hit_area(stronger)
    if stronger_area < fragment_area * 1.12:
        return False
    inter_area, _iou, iom, _center_distance = _bbox_metrics(fragment.bbox, stronger.bbox)
    if inter_area <= 0 or not _deep_same_ink_overlap(fragment, stronger):
        return False

    fragment_center_nested = _center_inside_bbox(
        _hit_center(fragment),
        stronger.bbox,
        margin_ratio=0.03,
    )
    deeply_contained_fragment = (
        iom >= 0.90
        and fragment_center_nested
        and stronger_area >= fragment_area * 1.75
    )
    if deeply_contained_fragment:
        fragment_has_independent_evidence = (
            fragment.context_purity >= 0.43
            and fragment.purity >= 0.70
            and fragment.verification_score >= 0.66
        )
        stronger_is_not_weaker = (
            stronger.verification_score + 0.07 >= fragment.verification_score
            and stronger.match_score + 0.08 >= fragment.match_score
            and stronger.coverage >= 0.72
            and stronger.coverage + 0.20 >= fragment.coverage
        )
        if stronger_is_not_weaker and not fragment_has_independent_evidence:
            return True

    return (
        stronger.verification_score >= fragment.verification_score + 0.07
        and stronger.match_score >= fragment.match_score + 0.08
        and stronger.purity >= fragment.purity + 0.06
        and stronger.coverage >= fragment.coverage - 0.04
    )


def _same_template_close_shadow_loses_to_larger(
    small: CandidateHit,
    large: CandidateHit,
) -> bool:
    if small.template_id != large.template_id:
        return False
    if small.dominant_hsv is not None or large.dominant_hsv is not None:
        return False

    small_area = _hit_area(small)
    large_area = _hit_area(large)
    if small_area > 900 or large_area < small_area * 3.0:
        return False

    sx, sy, sw, sh = small.bbox
    lx, ly, lw, lh = large.bbox
    overlap_x = max(0, min(sx + sw, lx + lw) - max(sx, lx))
    overlap_y = max(0, min(sy + sh, ly + lh) - max(sy, ly))
    gap_x = max(0, max(sx, lx) - min(sx + sw, lx + lw))
    gap_y = max(0, max(sy, ly) - min(sy + sh, ly + lh))
    aligned_x = overlap_x / max(1, min(sw, lw)) >= 0.60 and gap_y <= max(6, min(sh, lh) * 0.35)
    aligned_y = overlap_y / max(1, min(sh, lh)) >= 0.60 and gap_x <= max(18, min(sw, lw) * 0.80)
    if not (aligned_x or aligned_y):
        return False

    large_diag = max(1.0, float(np.hypot(lw, lh)))
    center_gap = float(np.hypot(_hit_center(small)[0] - _hit_center(large)[0], _hit_center(small)[1] - _hit_center(large)[1]))
    if center_gap / large_diag > 0.90:
        return False

    if small.context_purity > 0.24 or small.verification_score > 0.62:
        return False
    return (
        large.verification_score >= small.verification_score + 0.055
        and large.match_score >= small.match_score + 0.09
        and large.coverage >= small.coverage - 0.02
        and large.purity >= small.purity + 0.02
    )


def _weak_gray_compact_fragment_loses_to_larger(
    compact: CandidateHit,
    large: CandidateHit,
) -> bool:
    if compact.dominant_hsv is not None or large.dominant_hsv is not None:
        return False
    if compact.is_text_label != large.is_text_label:
        return False

    compact_area = _hit_area(compact)
    large_area = _hit_area(large)
    if compact_area > 3600 or large_area < compact_area * 1.24:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(compact.bbox, large.bbox)
    if inter_area <= 0:
        return False
    compact_center_nested = _center_inside_bbox(
        _hit_center(compact),
        large.bbox,
        margin_ratio=0.05,
    )
    if not (iom >= 0.48 and (compact_center_nested or center_distance <= 0.44)):
        return False

    if large.verification_score + 0.08 < compact.verification_score:
        return False
    if large.match_score + 0.08 < compact.match_score:
        return False

    weak_compact_evidence = compact.context_purity <= 0.24 and compact.purity <= 0.56
    stronger_large_evidence = (
        large.purity >= compact.purity + 0.08
        and large.context_purity >= compact.context_purity + 0.05
    )
    return weak_compact_evidence and stronger_large_evidence


def _compact_gray_hit_beats_large_partial(
    compact: CandidateHit,
    large: CandidateHit,
) -> bool:
    if compact.dominant_hsv is not None or large.dominant_hsv is not None:
        return False

    compact_area = _hit_area(compact)
    large_area = _hit_area(large)
    if compact_area > 1450 or large_area < compact_area * 1.55:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(compact.bbox, large.bbox)
    if inter_area > 0:
        compact_nested = _center_inside_bbox(
            _hit_center(compact),
            large.bbox,
            margin_ratio=0.06,
        )
        if not (iom >= 0.62 or (compact_nested and center_distance <= 0.64)):
            return False
    else:
        cx, cy, cw, ch = compact.bbox
        lx, ly, lw, lh = large.bbox
        overlap_x = max(0, min(cx + cw, lx + lw) - max(cx, lx))
        overlap_y = max(0, min(cy + ch, ly + lh) - max(cy, ly))
        gap_x = max(0, max(cx, lx) - min(cx + cw, lx + lw))
        gap_y = max(0, max(cy, ly) - min(cy + ch, ly + lh))
        aligned_x = overlap_x / max(1, min(cw, lw)) >= 0.45 and gap_y <= max(
            3,
            min(ch, lh) * 0.12,
        )
        aligned_y = overlap_y / max(1, min(ch, lh)) >= 0.45 and gap_x <= max(
            3,
            min(cw, lw) * 0.12,
        )
        if not (aligned_x or aligned_y):
            return False

    if compact.coverage < 0.98 or compact.purity < 0.49:
        return False
    if compact.verification_score < 0.56:
        return False
    if compact.verification_score + 0.03 < large.verification_score:
        return False

    compact_is_cleaner = (
        compact.purity >= large.purity + 0.045
        and compact.context_purity >= large.context_purity + 0.015
    )
    large_is_weak_partial = large.purity <= 0.48 and large.context_purity <= 0.24
    return compact_is_cleaner or large_is_weak_partial


def _strong_compact_gray_hit_should_coexist(
    left: CandidateHit,
    right: CandidateHit,
) -> bool:
    """Keep compact symbols that sit on/near a larger gray label frame.

    Gray E9 contains real compact symbols mounted on A1/AW-style long labels.
    Treating every overlap as competing interpretations makes those compact
    symbols disappear.  Keep only high-evidence compact hits; weak interior
    fragments are still handled by the nested suppression pass.
    """

    left_area = _hit_area(left)
    right_area = _hit_area(right)
    if left_area == right_area:
        return False

    compact = left if left_area < right_area else right
    large = right if compact is left else left
    compact_area = min(left_area, right_area)
    large_area = max(left_area, right_area)
    if compact_area > 4200 or large_area < compact_area * 2.0:
        return False

    if _nested_gray_fragment_loses_to_stronger_hit(compact, large):
        return False

    large_aspect = max(
        large.bbox[2] / max(1, large.bbox[3]),
        large.bbox[3] / max(1, large.bbox[2]),
    )
    if large_aspect < 1.55 and large_area < compact_area * 2.8:
        return False

    return (
        compact.coverage >= 0.92
        and compact.purity >= 0.58
        and compact.context_purity >= 0.34
        and compact.verification_score >= 0.60
    )
