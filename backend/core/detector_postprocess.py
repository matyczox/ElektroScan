"""Final detector post-processing helpers."""

from __future__ import annotations

from core.detector_models import CandidateHit


def dedupe_final_hits(final_hits: list[CandidateHit]) -> tuple[list[CandidateHit], int]:
    """Remove identical final hits while keeping the strongest scoring variant."""

    deduped: dict[tuple[int, tuple[int, int, int, int]], CandidateHit] = {}

    def quality(hit: CandidateHit) -> tuple[float, ...]:
        return (
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
            float(hit.context_purity),
        )

    for hit in final_hits:
        key = (hit.template_id, hit.bbox)
        previous = deduped.get(key)
        if previous is None or quality(hit) > quality(previous):
            deduped[key] = hit
    return list(deduped.values()), len(final_hits) - len(deduped)
