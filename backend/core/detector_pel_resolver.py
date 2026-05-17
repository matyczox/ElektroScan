"""PEL label evidence resolver for color detections."""

from __future__ import annotations

from dataclasses import replace
import re

from core.detector_models import CandidateHit, TemplateInfo
from core.detector_pdf import PdfWordBox


def _center_inside(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    cx = ix + iw / 2
    cy = iy + ih / 2
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def _expanded(
    bbox: tuple[int, int, int, int],
    *,
    pad_x: float,
    pad_y: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    px = int(round(pad_x))
    py = int(round(pad_y))
    return (x - px, y - py, w + 2 * px, h + 2 * py)


def _is_pel_visual_name(name: str) -> bool:
    return re.fullmatch(r"\d+_PEL\d+[A-Z]?", name.upper()) is not None


def _pel_visual_token(name: str) -> str | None:
    match = re.fullmatch(r"\d+_(PEL\d+[A-Z]?)", name.upper())
    return match.group(1) if match else None


def _is_floor_box_pel_name(name: str) -> bool:
    upper = name.upper()
    return "PELX-PP" in upper or "PUSZCE_PODLOGOWEJ" in upper


def _has_floor_box_pel_label(words: list[PdfWordBox], bbox: tuple[int, int, int, int]) -> bool:
    search_box = _expanded(bbox, pad_x=max(70.0, bbox[2] * 0.75), pad_y=max(34.0, bbox[3] * 0.55))
    for token, word_bbox in words:
        if not (token.startswith("PEL") and token.endswith("PP")):
            continue
        if _center_inside(word_bbox, search_box):
            return True
    return False


def _has_exact_pel_label(
    words: list[PdfWordBox],
    bbox: tuple[int, int, int, int],
    token: str,
) -> bool:
    search_box = _expanded(
        bbox,
        pad_x=max(48.0, bbox[2] * 0.70),
        pad_y=max(42.0, bbox[3] * 0.72),
    )
    return any(word_token == token and _center_inside(word_bbox, search_box) for word_token, word_bbox in words)


def _pel_visual_rescue_quality(hit: CandidateHit, templates: list[TemplateInfo]) -> bool:
    if hit.source == "pdf_text" or not (0 <= hit.template_id < len(templates)):
        return False
    if not _is_pel_visual_name(templates[hit.template_id].name):
        return False
    return (
        hit.match_score >= 0.56
        and hit.verification_score >= 0.70
        and hit.coverage >= 0.55
        and hit.purity >= 0.88
    )


def _same_visual_place(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> bool:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    lcx = lx + lw / 2
    lcy = ly + lh / 2
    rcx = rx + rw / 2
    rcy = ry + rh / 2
    distance = ((lcx - rcx) ** 2 + (lcy - rcy) ** 2) ** 0.5
    return distance <= max(20.0, min(lw, lh, rw, rh) * 0.55)


def _rescue_labeled_pel_visual_hits(
    hits: list[CandidateHit],
    candidates: list[CandidateHit],
    *,
    templates: list[TemplateInfo],
    pdf_word_boxes: list[PdfWordBox],
) -> tuple[list[CandidateHit], int]:
    output = list(hits)
    rescued = 0
    for candidate in sorted(
        candidates,
        key=lambda hit: (
            float(hit.verification_score),
            float(hit.match_score),
            float(hit.coverage),
            float(hit.purity),
        ),
        reverse=True,
    ):
        if not _pel_visual_rescue_quality(candidate, templates):
            continue
        token = _pel_visual_token(templates[candidate.template_id].name)
        if token is None or not _has_exact_pel_label(pdf_word_boxes, candidate.bbox, token):
            continue
        if any(
            existing.template_id == candidate.template_id
            and existing.source != "pdf_text"
            and _same_visual_place(existing.bbox, candidate.bbox)
            for existing in output
        ):
            continue
        output.append(candidate)
        rescued += 1

    return output, rescued


def resolve_pel_floor_box_hits(
    final_hits: list[CandidateHit],
    *,
    detector_profile: str,
    templates: list[TemplateInfo],
    pdf_word_boxes: list[PdfWordBox],
    candidates: list[CandidateHit] | None = None,
) -> tuple[list[CandidateHit], int]:
    """Use local ``PEL...PP`` text evidence to choose the floor-box PEL template."""

    if detector_profile != "color" or not pdf_word_boxes:
        return final_hits, 0

    target_id = next(
        (template_id for template_id, template in enumerate(templates) if _is_floor_box_pel_name(template.name)),
        None,
    )
    if target_id is None:
        return final_hits, 0

    output: list[CandidateHit] = []
    changed = 0
    for hit in final_hits:
        if not (0 <= hit.template_id < len(templates)):
            output.append(hit)
            continue
        if (
            hit.template_id != target_id
            and _is_pel_visual_name(templates[hit.template_id].name)
            and _has_floor_box_pel_label(pdf_word_boxes, hit.bbox)
        ):
            output.append(
                replace(
                    hit,
                    template_id=target_id,
                    source="template_label_disambiguation",
                    promoted_from_template_id=hit.template_id,
                    dominant_hsv=templates[target_id].dominant_hsv,
                    is_text_label=templates[target_id].is_text_label,
                )
            )
            changed += 1
            continue
        output.append(hit)

    output, visual_rescued = _rescue_labeled_pel_visual_hits(
        output,
        list(candidates or []),
        templates=templates,
        pdf_word_boxes=pdf_word_boxes,
    )
    return output, changed + visual_rescued
