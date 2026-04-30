"""Family promotion rules for fuller symbol variants."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.detector_clustering import _bbox_metrics, _box_center, _center_inside_box
from core.detector_config import (
    PROMOTED_PARENT_MIN_AREA_RATIO,
    SOCKET_07_PROMOTION_SEARCH_RADIUS,
    SOCKET_07_STRONG_EXTRA_COVERAGE,
    SOCKET_07_STRONG_MAX_VERIFICATION_DROP,
    SOCKET_07_STRONG_MIN_CONTEXT_PURITY,
    SOCKET_07_STRONG_MIN_COVERAGE,
    SOCKET_PROMOTED_MAX_VERIFICATION_DROP,
    SWITCH_10_PROMOTED_MAX_VERIFICATION_DROP,
    SWITCH_12_PROMOTED_MAX_VERIFICATION_DROP,
    SWITCH_PARENT_FALLBACK_SEARCH_RADIUS,
    SWITCH_PROMOTED_MIN_CONTEXT_PURITY,
    SWITCH_PROMOTED_MIN_PURITY,
    SWITCH_PROMOTED_MIN_VERIFICATION,
    SWITCH_PROMOTION_SEARCH_RADIUS,
)
from core.detector_masks import _cached_dilated_mask, _roi_mask, _validate_template_hit
from core.detector_models import CandidateHit, TargetedPromotionRule, TemplateInfo, TemplateVariant
from core.detector_templates import _template_numeric_prefix


def _maybe_promote_socket_06_to_07(
    hit: CandidateHit,
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    plan_masks: dict[int, np.ndarray],
    dilated_plan_masks: dict[int, np.ndarray],
    variants_lookup: dict[tuple[int, float, int, bool], TemplateVariant],
    promotions: dict[tuple[int, float, int, bool], list[TargetedPromotionRule]],
) -> CandidateHit:
    """Apply cheap family promotions when parent-only extension pixels are present."""

    rules = promotions.get((hit.template_id, hit.scale, hit.rotation, hit.mirrored))
    if not rules:
        return hit

    best_promoted: CandidateHit | None = None
    best_key: tuple[float, float, float] | None = None

    for rule in rules:
        parent_variant = variants_lookup.get(
            (rule.parent_template_id, rule.scale, rule.rotation, rule.mirrored)
        )
        parent_plan_mask = plan_masks.get(rule.parent_template_id)
        if parent_variant is None or parent_plan_mask is None:
            continue
        child_prefix = _template_numeric_prefix(Path(templates[rule.child_template_id].path).name)
        parent_prefix = _template_numeric_prefix(Path(templates[rule.parent_template_id].path).name)
        if parent_prefix == "07":
            promotion_plan_mask = _cached_dilated_mask(
                rule.parent_template_id,
                parent_plan_mask,
                dilated_plan_masks,
            )
        else:
            promotion_plan_mask = parent_plan_mask
        extension_plan_mask = (
            _cached_dilated_mask(
                rule.parent_template_id,
                parent_plan_mask,
                dilated_plan_masks,
            )
            if parent_prefix in {"10", "12"}
            else promotion_plan_mask
        )
        search_radius = (
            SWITCH_PROMOTION_SEARCH_RADIUS
            if parent_prefix in {"10", "12"}
            else SOCKET_07_PROMOTION_SEARCH_RADIUS
        )

        base_parent_x = hit.bbox[0] - rule.offset_x
        base_parent_y = hit.bbox[1] - rule.offset_y

        for delta_y in range(-search_radius, search_radius + 1):
            for delta_x in range(-search_radius, search_radius + 1):
                parent_bbox = (
                    base_parent_x + delta_x,
                    base_parent_y + delta_y,
                    parent_variant.width,
                    parent_variant.height,
                )
                parent_roi = _roi_mask(promotion_plan_mask, parent_bbox)
                extension_roi = _roi_mask(extension_plan_mask, parent_bbox)
                if parent_roi is None or parent_roi.shape != parent_variant.transformed_mask.shape:
                    continue
                if (
                    extension_roi is None
                    or extension_roi.shape != parent_variant.transformed_mask.shape
                ):
                    continue

                extra_overlap = int(
                    cv2.countNonZero(cv2.bitwise_and(extension_roi, rule.extension_mask))
                )
                extra_coverage = extra_overlap / max(1, rule.extension_pixels)
                if extra_coverage < rule.min_extra_coverage:
                    continue

                try:
                    local_match = float(
                        cv2.matchTemplate(
                            parent_roi,
                            parent_variant.transformed_mask,
                            cv2.TM_CCORR_NORMED,
                        )[0][0]
                    )
                except cv2.error:
                    continue

                promoted_hit = CandidateHit(
                    template_id=rule.parent_template_id,
                    scale=rule.scale,
                    rotation=rule.rotation,
                    mirrored=rule.mirrored,
                    transformed_mask=parent_variant.transformed_mask,
                    content_mask=parent_variant.content_mask,
                    pixel_count=parent_variant.pixel_count,
                    content_pixel_count=parent_variant.content_pixel_count,
                    content_bbox=parent_variant.content_bbox,
                    bbox=parent_bbox,
                    match_score=local_match,
                    dominant_hsv=templates[rule.parent_template_id].dominant_hsv,
                    source=f"template_promoted_{rule.child_template_id}_to_{rule.parent_template_id}",  # noqa: E501
                    is_text_label=templates[rule.parent_template_id].is_text_label,
                    promoted_from_template_id=hit.template_id,
                )
                if not _validate_template_hit(promoted_hit, promotion_plan_mask, plan_image):
                    continue
                if parent_prefix == "07":
                    max_drop = SOCKET_PROMOTED_MAX_VERIFICATION_DROP
                    if (
                        child_prefix == "06"
                        and extra_coverage >= SOCKET_07_STRONG_EXTRA_COVERAGE
                        and promoted_hit.coverage >= SOCKET_07_STRONG_MIN_COVERAGE
                        and promoted_hit.context_purity >= SOCKET_07_STRONG_MIN_CONTEXT_PURITY
                    ):
                        max_drop = SOCKET_07_STRONG_MAX_VERIFICATION_DROP
                    if promoted_hit.verification_score < (hit.verification_score - max_drop):
                        continue
                if parent_prefix in {"10", "12"}:
                    max_drop = (
                        SWITCH_12_PROMOTED_MAX_VERIFICATION_DROP
                        if parent_prefix == "12"
                        else SWITCH_10_PROMOTED_MAX_VERIFICATION_DROP
                    )
                    if (
                        promoted_hit.purity < SWITCH_PROMOTED_MIN_PURITY
                        or promoted_hit.context_purity < SWITCH_PROMOTED_MIN_CONTEXT_PURITY
                        or promoted_hit.verification_score < SWITCH_PROMOTED_MIN_VERIFICATION
                        or promoted_hit.verification_score < (hit.verification_score - max_drop)
                    ):
                        continue
                    if (
                        parent_prefix == "10"
                        and rule.allow_rotation_mismatch
                        and promoted_hit.verification_score + 0.02 < hit.verification_score
                    ):
                        continue

                candidate_key = (
                    float(extra_coverage),
                    float(promoted_hit.verification_score),
                    float(promoted_hit.match_score),
                )
                if best_key is None or candidate_key > best_key:
                    best_promoted = promoted_hit
                    best_key = candidate_key

    return best_promoted or hit


def _maybe_promote_switch_parent_search(
    hit: CandidateHit,
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    plan_masks: dict[int, np.ndarray],
    dilated_plan_masks: dict[int, np.ndarray],
    variants_lookup: dict[tuple[int, float, int, bool], TemplateVariant],
    promotions: dict[tuple[int, float, int, bool], list[TargetedPromotionRule]],
    stats: dict[str, int] | None = None,
) -> CandidateHit:
    """Run the expensive 11 -> 10/12 parent search only on prefiltered hits."""

    if hit.transformed_mask is None:
        return hit

    child_prefix = _template_numeric_prefix(Path(templates[hit.template_id].path).name)
    if child_prefix != "11":
        return hit

    rules = promotions.get((hit.template_id, hit.scale, hit.rotation, hit.mirrored), [])
    if not rules:
        return hit

    child_center = _box_center(hit.bbox)
    child_area = max(1, hit.bbox[2] * hit.bbox[3])
    fallback_best: CandidateHit | None = None
    fallback_key: tuple[float, float, float] | None = None
    if stats is not None:
        stats["parent_search_input_hits"] = stats.get("parent_search_input_hits", 0) + 1

    for rule in rules:
        parent_prefix = _template_numeric_prefix(Path(templates[rule.parent_template_id].path).name)
        if parent_prefix not in {"10", "12"}:
            continue

        parent_variant = variants_lookup.get(
            (rule.parent_template_id, rule.scale, rule.rotation, rule.mirrored)
        )
        parent_plan_mask = plan_masks.get(rule.parent_template_id)
        if parent_variant is None or parent_plan_mask is None:
            continue

        parent_area = max(1, parent_variant.width * parent_variant.height)
        if parent_area < child_area * PROMOTED_PARENT_MIN_AREA_RATIO:
            continue

        extension_plan_mask = _cached_dilated_mask(
            rule.parent_template_id,
            parent_plan_mask,
            dilated_plan_masks,
        )
        base_x = int(round(child_center[0] - parent_variant.width / 2.0))
        base_y = int(round(child_center[1] - parent_variant.height / 2.0))

        for delta_y in range(
            -SWITCH_PARENT_FALLBACK_SEARCH_RADIUS, SWITCH_PARENT_FALLBACK_SEARCH_RADIUS + 1
        ):
            for delta_x in range(
                -SWITCH_PARENT_FALLBACK_SEARCH_RADIUS, SWITCH_PARENT_FALLBACK_SEARCH_RADIUS + 1
            ):
                parent_bbox = (
                    base_x + delta_x,
                    base_y + delta_y,
                    parent_variant.width,
                    parent_variant.height,
                )
                extension_roi = _roi_mask(extension_plan_mask, parent_bbox)
                if (
                    extension_roi is None
                    or extension_roi.shape != parent_variant.transformed_mask.shape
                ):
                    continue

                extra_overlap = int(
                    cv2.countNonZero(cv2.bitwise_and(extension_roi, rule.extension_mask))
                )
                extra_coverage = extra_overlap / max(1, rule.extension_pixels)
                if extra_coverage < rule.min_extra_coverage:
                    continue

                parent_roi = _roi_mask(parent_plan_mask, parent_bbox)
                if parent_roi is None or parent_roi.shape != parent_variant.transformed_mask.shape:
                    continue

                if not _center_inside_box(child_center, parent_bbox, margin_ratio=0.08):
                    continue

                inter_area, _, iom, _ = _bbox_metrics(hit.bbox, parent_bbox)
                if inter_area <= 0 or iom < 0.40:
                    continue

                if stats is not None:
                    stats["parent_search_candidates"] = stats.get("parent_search_candidates", 0) + 1
                try:
                    local_match = float(
                        cv2.matchTemplate(
                            parent_roi,
                            parent_variant.transformed_mask,
                            cv2.TM_CCORR_NORMED,
                        )[0][0]
                    )
                except cv2.error:
                    continue

                promoted_hit = CandidateHit(
                    template_id=rule.parent_template_id,
                    scale=parent_variant.scale,
                    rotation=parent_variant.rotation,
                    mirrored=parent_variant.mirrored,
                    transformed_mask=parent_variant.transformed_mask,
                    content_mask=parent_variant.content_mask,
                    pixel_count=parent_variant.pixel_count,
                    content_pixel_count=parent_variant.content_pixel_count,
                    content_bbox=parent_variant.content_bbox,
                    bbox=parent_bbox,
                    match_score=local_match,
                    dominant_hsv=templates[rule.parent_template_id].dominant_hsv,
                    source=f"template_parent_search_{hit.template_id}_to_{rule.parent_template_id}",
                    is_text_label=templates[rule.parent_template_id].is_text_label,
                    promoted_from_template_id=hit.template_id,
                )
                if not _validate_template_hit(promoted_hit, parent_plan_mask, plan_image):
                    continue

                max_drop = (
                    SWITCH_12_PROMOTED_MAX_VERIFICATION_DROP
                    if parent_prefix == "12"
                    else SWITCH_10_PROMOTED_MAX_VERIFICATION_DROP
                )
                if (
                    promoted_hit.purity < SWITCH_PROMOTED_MIN_PURITY
                    or promoted_hit.context_purity < SWITCH_PROMOTED_MIN_CONTEXT_PURITY
                    or promoted_hit.verification_score < SWITCH_PROMOTED_MIN_VERIFICATION
                    or promoted_hit.verification_score < (hit.verification_score - max_drop)
                ):
                    continue

                if (
                    parent_prefix == "10"
                    and promoted_hit.verification_score + 0.02 < hit.verification_score
                ):
                    continue

                candidate_key = (
                    float(extra_coverage),
                    float(promoted_hit.verification_score),
                    float(promoted_hit.match_score),
                )
                if fallback_key is None or candidate_key > fallback_key:
                    fallback_best = promoted_hit
                    fallback_key = candidate_key

    return fallback_best or hit
