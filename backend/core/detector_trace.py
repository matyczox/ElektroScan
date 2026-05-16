"""Candidate trace capture for detector debugging."""

from __future__ import annotations

import numpy as np

from core.detector_models import CandidateHit, TemplateInfo


class CandidateTraceRecorder:
    def __init__(
        self,
        *,
        templates: list[TemplateInfo],
        trace_symbols: set[str],
        trace_points: list[tuple[float, float]],
        trace_radius: float,
        max_items: int,
    ) -> None:
        self.templates = templates
        self.trace_symbols = trace_symbols
        self.trace_points = trace_points
        self.trace_radius = trace_radius
        self.max_items = max_items
        self.enabled = bool(trace_symbols or trace_points)
        self.data: dict[str, dict] = {}

    def _symbol_matches(self, symbol_name: str) -> bool:
        if not self.trace_symbols:
            return True
        return any(
            symbol_name == requested
            or symbol_name.startswith(f"{requested}_")
            or symbol_name.startswith(requested)
            for requested in self.trace_symbols
        )

    def _box_distance(self, bbox: tuple[int, int, int, int]) -> float:
        if not self.trace_points:
            return 0.0
        x, y, w, h = bbox
        best = float("inf")
        for px, py in self.trace_points:
            dx = max(float(x) - px, 0.0, px - float(x + w))
            dy = max(float(y) - py, 0.0, py - float(y + h))
            best = min(best, float(np.hypot(dx, dy)))
        return best

    def record(
        self,
        stage: str,
        hits: list[CandidateHit],
        reason_by_id: dict[int, str] | None = None,
    ) -> None:
        if not self.enabled:
            return

        matched: list[tuple[float, CandidateHit]] = []
        for hit in hits:
            if not (0 <= hit.template_id < len(self.templates)):
                continue
            symbol_name = self.templates[hit.template_id].name
            if not self._symbol_matches(symbol_name):
                continue
            distance = self._box_distance(hit.bbox)
            if self.trace_points and distance > self.trace_radius:
                continue
            matched.append((distance, hit))

        matched.sort(
            key=lambda item: (
                item[0],
                -float(item[1].verification_score),
                -float(item[1].match_score),
            )
        )
        items = []
        reason_by_id = reason_by_id or {}
        for distance, hit in matched[: self.max_items]:
            x, y, w, h = hit.bbox
            item = {
                "symbolName": self.templates[hit.template_id].name,
                "templateId": int(hit.template_id),
                "bbox": [int(x), int(y), int(w), int(h)],
                "match": round(float(hit.match_score), 3),
                "verification": round(float(hit.verification_score), 3),
                "coverage": round(float(hit.coverage), 3),
                "purity": round(float(hit.purity), 3),
                "context": round(float(hit.context_purity), 3),
                "contentScore": round(float(hit.content_score), 3),
                "pixelCount": int(hit.pixel_count),
                "scale": round(float(hit.scale), 3),
                "rotation": int(hit.rotation),
                "mirrored": bool(hit.mirrored),
                "source": str(hit.source),
                "isTextLabel": bool(hit.is_text_label),
                "roiStrategy": str(hit.roi_strategy),
                "distance": round(float(distance), 3),
            }
            reason = reason_by_id.get(id(hit))
            if reason:
                item["reason"] = reason
            items.append(item)
        self.data[stage] = {
            "totalCandidates": int(len(hits)),
            "matchedCandidates": int(len(matched)),
            "items": items,
        }
