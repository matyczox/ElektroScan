"""PDF-text policy helpers for detector pipeline.

PDF text is only an assist layer in the color detector. These helpers keep
that policy outside the orchestration function without changing scoring rules.
"""

from __future__ import annotations

from collections.abc import Callable
import re

import numpy as np

from core.detector_models import CandidateHit


def filter_pdf_text_fallbacks(
    pdf_hits: list[CandidateHit],
    template_hits: list[CandidateHit],
    *,
    detector_profile: str,
    is_visual_pdf_text_blocked: Callable[[int], bool],
) -> tuple[list[CandidateHit], list[CandidateHit]]:
    """Keep color PDF text only where the visual detector has no local hit."""

    if detector_profile != "color" or not pdf_hits or not template_hits:
        return pdf_hits, []

    def template_overlaps_pdf_text(template_hit: CandidateHit, pdf_hit: CandidateHit) -> bool:
        tx, ty, tw, th = template_hit.bbox
        px, py, pw, ph = pdf_hit.bbox
        ix1 = max(tx, px)
        iy1 = max(ty, py)
        ix2 = min(tx + tw, px + pw)
        iy2 = min(ty + th, py + ph)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return False

        template_area = max(1, tw * th)
        pdf_area = max(1, pw * ph)
        union = template_area + pdf_area - inter
        iou = inter / max(1, union)
        iom = inter / min(template_area, pdf_area)

        tcx = tx + tw / 2.0
        tcy = ty + th / 2.0
        pcx = px + pw / 2.0
        pcy = py + ph / 2.0
        center_distance = float(np.hypot(tcx - pcx, tcy - pcy))
        ref_distance = max(1.0, min(float(np.hypot(tw, th)), float(np.hypot(pw, ph))))
        normalized_center_distance = center_distance / ref_distance
        template_center_inside_pdf = px <= tcx <= px + pw and py <= tcy <= py + ph

        pdf_center_inside_template = tx <= pcx <= tx + tw and ty <= pcy <= ty + th

        return (
            iou >= 0.12
            or (iom >= 0.32 and normalized_center_distance <= 0.80)
            or (template_center_inside_pdf and iom >= 0.22)
            or (pdf_center_inside_template and iom >= 0.22)
        )

    local_template_hits = [
        hit
        for hit in template_hits
        if hit.source != "pdf_text"
        and hit.dominant_hsv is not None
        and hit.verification_score >= 0.38
        and hit.coverage >= 0.45
        and hit.purity >= 0.45
    ]
    if not local_template_hits:
        return pdf_hits, []

    kept: list[CandidateHit] = []
    removed: list[CandidateHit] = []
    for pdf_hit in pdf_hits:
        if is_visual_pdf_text_blocked(pdf_hit.template_id):
            removed.append(pdf_hit)
            continue
        has_template_fallback = any(
            template_overlaps_pdf_text(template_hit, pdf_hit)
            for template_hit in local_template_hits
        )
        if has_template_fallback:
            removed.append(pdf_hit)
        else:
            kept.append(pdf_hit)
    return kept, removed


def apply_color_pdf_text_class_hints(
    template_hits: list[CandidateHit],
    pdf_hits: list[CandidateHit],
    *,
    detector_profile: str,
    template_tokens: Callable[[int], tuple[str, ...]],
) -> int:
    """Use PDF text only as a non-final class hint for local color ties."""

    if detector_profile != "color" or not template_hits or not pdf_hits:
        return 0

    def has_text_class_token(template_id: int) -> bool:
        return bool(template_tokens(template_id))

    def pdf_text_overlap(template_hit: CandidateHit, pdf_hit: CandidateHit) -> bool:
        tx, ty, tw, th = template_hit.bbox
        px, py, pw, ph = pdf_hit.bbox
        ix1 = max(tx, px)
        iy1 = max(ty, py)
        ix2 = min(tx + tw, px + pw)
        iy2 = min(ty + th, py + ph)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        template_area = max(1, tw * th)
        pdf_area = max(1, pw * ph)
        iom = inter / min(template_area, pdf_area) if inter > 0 else 0.0
        tcx = tx + tw / 2.0
        tcy = ty + th / 2.0
        pcx = px + pw / 2.0
        pcy = py + ph / 2.0
        center_distance = float(np.hypot(tcx - pcx, tcy - pcy))
        ref_distance = max(1.0, min(float(np.hypot(tw, th)), float(np.hypot(pw, ph))))
        hit_center_inside_pdf = px <= tcx <= px + pw and py <= tcy <= py + ph
        pdf_center_inside_hit = tx <= pcx <= tx + tw and ty <= pcy <= ty + th
        if template_hit.is_text_label and pdf_hit.is_text_label:
            return (
                inter > 0
                and (
                    hit_center_inside_pdf
                    or (pdf_center_inside_hit and iom >= 0.45)
                    or (iom >= 0.55 and center_distance / ref_distance <= 0.45)
                )
            )
        if has_text_class_token(template_hit.template_id) and not template_hit.is_text_label:
            x_gap = max(tx, px) - min(tx + tw, px + pw)
            y_gap = max(ty, py) - min(ty + th, py + ph)
            x_gap = max(0.0, float(x_gap))
            y_gap = max(0.0, float(y_gap))
            x_overlap = max(0, min(tx + tw, px + pw) - max(tx, px)) / max(1.0, min(tw, pw))
            y_overlap = max(0, min(ty + th, py + ph) - max(ty, py)) / max(1.0, min(th, ph))
            near_side_label = (
                x_gap <= max(18.0, float(tw) * 0.16)
                and y_gap <= max(8.0, float(th) * 0.16)
                and (y_overlap >= 0.25 or abs(tcy - pcy) <= max(18.0, float(th) * 0.38))
            )
            near_top_bottom_label = (
                y_gap <= max(18.0, float(th) * 0.16)
                and x_gap <= max(8.0, float(tw) * 0.12)
                and (x_overlap >= 0.25 or abs(tcx - pcx) <= max(18.0, float(tw) * 0.38))
            )
            return (
                (inter > 0 and iom >= 0.20)
                or center_distance / ref_distance <= 0.95
                or near_side_label
                or near_top_bottom_label
            )
        if inter <= 0:
            return False
        return iom >= 0.35 or center_distance / ref_distance <= 0.70

    def numbered_token_families(template_id: int) -> set[str]:
        families: set[str] = set()
        for token in template_tokens(template_id):
            match = re.fullmatch(r"([A-Z]+)\d+", token)
            if match:
                families.add(match.group(1))
        return families

    def is_same_numbered_text_family(left_id: int, right_id: int) -> bool:
        left_families = numbered_token_families(left_id)
        if not left_families:
            return False
        return bool(left_families & numbered_token_families(right_id))

    hinted = 0
    for hit in template_hits:
        if hit.source == "pdf_text" or hit.dominant_hsv is None:
            continue
        overlapping_pdf_hits = [
            pdf_hit
            for pdf_hit in pdf_hits
            if pdf_hit.dominant_hsv is not None
            and abs(int(hit.dominant_hsv[0]) - int(pdf_hit.dominant_hsv[0])) <= 18
            and pdf_text_overlap(hit, pdf_hit)
        ]
        if not overlapping_pdf_hits:
            continue

        has_same_template_hint = any(
            pdf_hit.template_id == hit.template_id for pdf_hit in overlapping_pdf_hits
        )
        if has_same_template_hint:
            visual_text_class_hint = has_text_class_token(hit.template_id) and not hit.is_text_label
            context_boost = 0.18 if visual_text_class_hint else 0.08
            verification_boost = 0.24 if visual_text_class_hint else 0.12
            hit.context_purity = round(
                min(1.0, float(hit.context_purity) + context_boost),
                4,
            )
            hit.verification_score = round(
                min(1.0, float(hit.verification_score) + verification_boost),
                4,
            )
            if hit.is_text_label:
                hit.content_score = round(
                    min(1.0, float(hit.content_score) + 0.18),
                    4,
                )
            hinted += 1
            continue

        has_conflicting_text_hint = any(
            pdf_hit.template_id != hit.template_id
            and (
                (pdf_hit.is_text_label and hit.is_text_label)
                or is_same_numbered_text_family(hit.template_id, pdf_hit.template_id)
            )
            for pdf_hit in overlapping_pdf_hits
        )
        if has_conflicting_text_hint:
            visual_text_class_hint = has_text_class_token(hit.template_id) and not hit.is_text_label
            context_penalty = 0.16 if visual_text_class_hint else 0.04
            verification_penalty = 0.28 if visual_text_class_hint else 0.10
            hit.context_purity = round(
                max(0.0, float(hit.context_purity) - context_penalty),
                4,
            )
            hit.verification_score = round(
                max(0.0, float(hit.verification_score) - verification_penalty),
                4,
            )
            hit.content_score = round(max(0.0, float(hit.content_score) - 0.18), 4)
            hinted += 1
    return hinted
