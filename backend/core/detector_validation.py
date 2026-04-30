"""Template hit validation phase for the detector pipeline."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np

from core.detector_masks import _validate_template_hit
from core.detector_models import CandidateHit, TargetedPromotionRule, TemplateInfo, TemplateVariant
from core.detector_promotions import _maybe_promote_socket_06_to_07


@dataclass(slots=True)
class ValidationResult:
    validated_hits: list[CandidateHit]
    rejected_hits: list[CandidateHit]
    rejection_reasons: dict[str, int]
    promoted_targeted_hits: int
    validation_workers: int
    timing_seconds: float


def validate_template_candidates(
    *,
    raw_template_hits: list[CandidateHit],
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    plan_masks_by_template: dict[int, np.ndarray],
    dilated_plan_masks_by_template: dict[int, np.ndarray],
    variants_lookup: dict[tuple[int, float, int, bool], TemplateVariant],
    socket_07_promotions: dict[tuple[int, float, int, bool], list[TargetedPromotionRule]],
    plan_hsv: np.ndarray | None,
    postprocess_workers: int,
    progress_callback: Callable[[str, float, str], None],
    gray_evidence_mask: np.ndarray | None = None,
    gray_relaxed_evidence_mask: np.ndarray | None = None,
) -> ValidationResult:
    """Validate raw template hits and apply cheap targeted promotions."""

    phase_start = time.perf_counter()

    def _validate_and_promote_hit(
        hit: CandidateHit,
    ) -> tuple[CandidateHit, CandidateHit, CandidateHit | None, str | None]:
        plan_mask = plan_masks_by_template[hit.template_id]
        local_reasons: dict[str, int] = {}
        if _validate_template_hit(
            hit,
            plan_mask,
            plan_image,
            reasons=local_reasons,
            plan_hsv=plan_hsv,
            evidence_mask=gray_evidence_mask if hit.dominant_hsv is None else None,
            relaxed_evidence_mask=(
                gray_relaxed_evidence_mask if hit.dominant_hsv is None else None
            ),
        ):
            promoted_hit = _maybe_promote_socket_06_to_07(
                hit,
                plan_image,
                templates,
                plan_masks_by_template,
                dilated_plan_masks_by_template,
                variants_lookup,
                socket_07_promotions,
                plan_hsv=plan_hsv,
            )
            return hit, promoted_hit, None, None
        rejection_reason = next(iter(local_reasons), "other")
        return hit, hit, hit, rejection_reason

    validated_hits: list[CandidateHit] = []
    rejected_hits: list[CandidateHit] = []
    rejection_reasons: dict[str, int] = {}
    promoted_targeted_hits = 0
    validation_workers = max(1, min(len(raw_template_hits), postprocess_workers))

    if raw_template_hits:
        completed_validations = 0
        total_validations = max(1, len(raw_template_hits))
        with ThreadPoolExecutor(max_workers=validation_workers) as pool:
            for validation_result in pool.map(_validate_and_promote_hit, raw_template_hits):
                completed_validations += 1
                if (
                    completed_validations == 1
                    or completed_validations % 250 == 0
                    or completed_validations == total_validations
                ):
                    progress_callback(
                        "validation",
                        65 + 18 * completed_validations / total_validations,
                        f"Walidacja {completed_validations}/{total_validations}",
                    )
                original_hit, promoted_hit, rejected_hit, reason = validation_result
                if rejected_hit is not None:
                    rejected_hits.append(rejected_hit)
                    if reason is not None:
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    continue
                if (
                    promoted_hit.template_id != original_hit.template_id
                    or promoted_hit.bbox != original_hit.bbox
                ):
                    promoted_targeted_hits += 1
                validated_hits.append(promoted_hit)
    else:
        validation_workers = 0

    return ValidationResult(
        validated_hits=validated_hits,
        rejected_hits=rejected_hits,
        rejection_reasons=rejection_reasons,
        promoted_targeted_hits=promoted_targeted_hits,
        validation_workers=validation_workers,
        timing_seconds=time.perf_counter() - phase_start,
    )
