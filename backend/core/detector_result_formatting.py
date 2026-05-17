"""Detection result formatting helpers for the detector pipeline."""

from __future__ import annotations

import numpy as np

from core.detector_models import CandidateHit, Detection, DetectionResult, TemplateInfo


def build_detection_results(
    *,
    final_hits: list[CandidateHit],
    templates: list[TemplateInfo],
    detector_profile: str,
    plan_masks_by_template: dict[int, np.ndarray],
    subtract_legend: bool,
    legend_rect: tuple[int, int, int, int] | None,
) -> list[DetectionResult]:
    def _color_visual_bbox(hit: CandidateHit) -> tuple[int, int, int, int] | None:
        if detector_profile != "color" or hit.source == "pdf_text" or hit.dominant_hsv is None:
            return None
        plan_mask = plan_masks_by_template.get(hit.template_id)
        if plan_mask is None:
            return None
        x, y, w, h = [int(value) for value in hit.bbox]
        image_h, image_w = plan_mask.shape[:2]
        x0 = max(0, min(image_w, x))
        y0 = max(0, min(image_h, y))
        x1 = max(0, min(image_w, x + w))
        y1 = max(0, min(image_h, y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        roi = plan_mask[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if len(xs) < 4:
            return None
        pad = 3
        vx0 = max(x0, x0 + int(xs.min()) - pad)
        vy0 = max(y0, y0 + int(ys.min()) - pad)
        vx1 = min(x1, x0 + int(xs.max()) + 1 + pad)
        vy1 = min(y1, y0 + int(ys.max()) + 1 + pad)
        visual_w = vx1 - vx0
        visual_h = vy1 - vy0
        if visual_w <= 0 or visual_h <= 0:
            return None
        # Avoid replacing a normal symbol box with a single antialiased speck.
        if visual_w * visual_h < max(16, int(0.03 * w * h)):
            return None
        return (vx0, vy0, visual_w, visual_h)

    per_template: dict[int, list[Detection]] = {}
    for hit in final_hits:
        x, y, w, h = [int(value) for value in hit.bbox]
        detection = Detection(
            symbol_name=templates[hit.template_id].name,
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=round(hit.match_score, 3),
            source=hit.source,
            rotation=hit.rotation,
            scale=hit.scale,
            mirrored=hit.mirrored,
            coverage=round(hit.coverage, 3),
            purity=round(hit.purity, 3),
            context_purity=round(hit.context_purity, 3),
            color_similarity=round(hit.color_similarity, 3),
            verification_score=round(hit.verification_score, 3),
            is_text_label=hit.is_text_label,
            content_score=round(hit.content_score, 3),
            content_bbox=(
                (
                    x + hit.content_bbox[0],
                    y + hit.content_bbox[1],
                    hit.content_bbox[2],
                    hit.content_bbox[3],
                )
                if hit.content_bbox is not None
                else None
            ),
            content_source=hit.source if hit.is_text_label else "",
            roi_strategy=hit.roi_strategy,
            visual_bbox=_color_visual_bbox(hit),
        )
        per_template.setdefault(hit.template_id, []).append(detection)

    results: list[DetectionResult] = []
    for template_id, detections in per_template.items():
        detections.sort(key=lambda det: (det.verification_score, det.confidence), reverse=True)

        count = len(detections)
        # Gray PDFs use explicit legend/text/plan-zone exclusions.  Blindly
        # subtracting one hit per symbol here can hide valid plan detections
        # that already passed validation, especially when the legend is outside
        # the selected plan zone.
        blind_legend_subtract = (
            detector_profile != "gray" and subtract_legend and legend_rect is None
        )
        if blind_legend_subtract:
            count = max(0, count - 1)

        if count <= 0:
            continue

        results.append(
            DetectionResult(
                symbol_name=templates[template_id].name,
                count=count,
                color="#22c55e",
                detections=(
                    detections[:count] if blind_legend_subtract else detections
                ),
            )
        )

    results.sort(key=lambda result: result.symbol_name.lower())
    return results
