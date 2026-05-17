"""Color-path postprocess orchestration for detector candidates."""

from __future__ import annotations

import numpy as np

from core.detector_context import (
    DetectionTemplateContext,
    bbox_iom,
    center_distance,
    center_inside,
    expanded_box,
    hue_close,
    token_family,
)
from core.detector_label_resolver import apply_final_label_resolver, suppress_weak_short_l_fragments
from core.detector_long_l_resolver import (
    promote_long_l_over_tb11_conflicts,
    rescue_color_family_hits,
    suppress_long_l_rescue_over_tb11_waves,
    suppress_tb11_long_l_conflicts,
)
from core.detector_magenta_resolver import reconcile_magenta_family_hits, rescue_magenta_family_hits
from core.detector_models import CandidateHit, TemplateInfo
from core.detector_postprocess import dedupe_final_hits


def apply_color_postprocess(
    *,
    detector_profile: str,
    final_hits: list[CandidateHit],
    prefiltered_candidates: list[CandidateHit],
    rejected_hits: list[CandidateHit],
    rejection_reason_by_hit_id: dict[int, str],
    pdf_candidates: list[CandidateHit],
    removed_pdf_text_fallbacks: list[CandidateHit],
    templates: list[TemplateInfo],
    template_context: DetectionTemplateContext,
    plan_masks_by_template: dict[int, np.ndarray],
) -> tuple[list[CandidateHit], dict[str, int]]:
    """Apply behavior-preserving color-family postprocess stages."""

    all_candidates = prefiltered_candidates + rejected_hits

    final_hits, color_magenta_reclassed = reconcile_magenta_family_hits(
        final_hits,
        prefiltered_candidates,
        detector_profile=detector_profile,
        _is_magenta_family_template=template_context.is_magenta_family_template,
        _bbox_iom=bbox_iom,
        _center_distance=center_distance,
        _center_inside=center_inside,
    )
    final_hits, color_magenta_local_rescued = rescue_magenta_family_hits(
        final_hits,
        all_candidates,
        detector_profile=detector_profile,
        templates=templates,
        plan_masks_by_template=plan_masks_by_template,
        _is_magenta_family_template=template_context.is_magenta_family_template,
        _magenta_template_code=template_context.magenta_template_code,
        _bbox_iom=bbox_iom,
        _center_distance=center_distance,
        _expanded_box=expanded_box,
    )
    final_hits, color_family_rescued = rescue_color_family_hits(
        final_hits,
        all_candidates,
        rejection_reason_by_hit_id,
        detector_profile=detector_profile,
        _l_label_group=template_context.l_label_group,
        _bbox_iom=bbox_iom,
        _center_distance=center_distance,
        _is_tb11_wave_template=template_context.is_tb11_wave_template,
    )
    final_hits, long_l_over_tb11_promoted = promote_long_l_over_tb11_conflicts(
        final_hits,
        all_candidates,
        detector_profile=detector_profile,
        _l_label_group=template_context.l_label_group,
        _bbox_iom=bbox_iom,
        _center_distance=center_distance,
        _is_tb11_wave_template=template_context.is_tb11_wave_template,
    )
    final_hits, tb11_long_l_suppressed = suppress_tb11_long_l_conflicts(
        final_hits,
        all_candidates,
        detector_profile=detector_profile,
        _l_label_group=template_context.l_label_group,
        _bbox_iom=bbox_iom,
        _center_distance=center_distance,
        _is_tb11_wave_template=template_context.is_tb11_wave_template,
    )
    final_hits, final_label_disambiguation = apply_final_label_resolver(
        final_hits,
        final_hits + all_candidates,
        pdf_candidates + removed_pdf_text_fallbacks,
        detector_profile=detector_profile,
        templates=templates,
        plan_masks_by_template=plan_masks_by_template,
        _template_primary_token=template_context.template_primary_token,
        _token_family=token_family,
        _template_token_family=template_context.template_token_family,
        _l_label_group=template_context.l_label_group,
        _bbox_iom=bbox_iom,
        _center_distance=center_distance,
        _center_inside=center_inside,
        _expanded_box=expanded_box,
        _hue_close=hue_close,
    )
    final_hits, long_l_over_true_tb11_suppressed = suppress_long_l_rescue_over_tb11_waves(
        final_hits,
        pdf_candidates + removed_pdf_text_fallbacks,
        detector_profile=detector_profile,
        plan_masks_by_template=plan_masks_by_template,
        _l_label_group=template_context.l_label_group,
        _template_token_family=template_context.template_token_family,
        _is_tb11_wave_template=template_context.is_tb11_wave_template,
        _bbox_iom=bbox_iom,
        _center_distance=center_distance,
        _center_inside=center_inside,
        _expanded_box=expanded_box,
        _hue_close=hue_close,
    )
    final_hits, weak_short_l_suppressed = suppress_weak_short_l_fragments(
        final_hits,
        detector_profile=detector_profile,
        _template_name=template_context.template_name,
        _center_distance=center_distance,
    )
    final_hits, duplicate_final_suppressed = dedupe_final_hits(final_hits)
    return final_hits, {
        "color_magenta_reclassed": color_magenta_reclassed,
        "color_magenta_local_rescued": color_magenta_local_rescued,
        "color_family_rescued": color_family_rescued,
        "long_l_over_tb11_promoted": long_l_over_tb11_promoted,
        "tb11_long_l_suppressed": tb11_long_l_suppressed,
        "final_label_disambiguation": final_label_disambiguation,
        "long_l_over_true_tb11_suppressed": long_l_over_true_tb11_suppressed,
        "weak_short_l_suppressed": weak_short_l_suppressed,
        "duplicate_final_suppressed": duplicate_final_suppressed,
    }
