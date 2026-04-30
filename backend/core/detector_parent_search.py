"""Parent-symbol fallback search for gray detector results."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np

from core.detector_models import CandidateHit, TargetedPromotionRule, TemplateInfo, TemplateVariant
from core.detector_promotions import _maybe_promote_switch_parent_search


@dataclass(slots=True)
class ParentSearchResult:
    candidates: list[CandidateHit]
    workers: int
    input_hits: int
    attempted_candidates: int
    promoted_hits: int
    timing_seconds: float


def search_parent_candidates(
    *,
    pre_parent_candidates: list[CandidateHit],
    detector_profile: str,
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    plan_masks_by_template: dict[int, np.ndarray],
    dilated_plan_masks_by_template: dict[int, np.ndarray],
    variants_lookup: dict[tuple[int, float, int, bool], TemplateVariant],
    socket_07_promotions: dict[tuple[int, float, int, bool], list[TargetedPromotionRule]],
    plan_hsv: np.ndarray | None,
    postprocess_workers: int,
    progress_callback: Callable[[str, float, str], None],
) -> ParentSearchResult:
    """Run expensive parent fallback only for gray profiles."""

    if detector_profile == "color":
        return ParentSearchResult(
            candidates=pre_parent_candidates,
            workers=0,
            input_hits=0,
            attempted_candidates=0,
            promoted_hits=0,
            timing_seconds=0.0,
        )

    phase_start = time.perf_counter()
    progress_callback("parent_search", 88, "Szukanie pelniejszych symboli")

    def _search_parent_hit(hit: CandidateHit) -> tuple[CandidateHit, dict[str, int]]:
        local_stats: dict[str, int] = {}
        promoted_hit = _maybe_promote_switch_parent_search(
            hit,
            plan_image,
            templates,
            plan_masks_by_template,
            dilated_plan_masks_by_template,
            variants_lookup,
            socket_07_promotions,
            local_stats,
            plan_hsv=plan_hsv,
        )
        return promoted_hit, local_stats

    parent_search_candidates: list[CandidateHit] = []
    input_hits = 0
    attempted_candidates = 0
    promoted_hits = 0
    workers = max(1, min(len(pre_parent_candidates), postprocess_workers))

    if pre_parent_candidates:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for hit, (promoted_hit, local_stats) in zip(
                pre_parent_candidates, pool.map(_search_parent_hit, pre_parent_candidates)
            ):
                input_hits += local_stats.get("parent_search_input_hits", 0)
                attempted_candidates += local_stats.get("parent_search_candidates", 0)
                if promoted_hit.template_id != hit.template_id or promoted_hit.bbox != hit.bbox:
                    promoted_hits += 1
                parent_search_candidates.append(promoted_hit)
    else:
        workers = 0

    return ParentSearchResult(
        candidates=parent_search_candidates,
        workers=workers,
        input_hits=input_hits,
        attempted_candidates=attempted_candidates,
        promoted_hits=promoted_hits,
        timing_seconds=time.perf_counter() - phase_start,
    )
