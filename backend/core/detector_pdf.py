"""PDF text fallback and legend exclusion helpers."""

from __future__ import annotations

import os

import fitz
import numpy as np

from core.detector_clustering import _bbox_metrics
from core.detector_config import (
    DEFAULT_PDF_DPI,
    LEGEND_HEIGHT_PT,
    LEGEND_KEYWORD,
    LEGEND_WIDTH_PT,
    PDF_TEXT_MAX_TOKEN_LENGTH,
    PDF_TEXT_MIN_TOKEN_LENGTH,
)
from core.detector_models import CandidateHit, TemplateInfo
from core.detector_templates import _derive_text_tokens


def _apply_hidden_layers(doc: fitz.Document, hidden_layers: list[str] | None) -> None:
    """Disable selected PDF layers before text lookup."""

    if not hidden_layers:
        return

    try:
        ui_configs = doc.layer_ui_configs()
        if not ui_configs:
            return
        for config in ui_configs:
            if config.get("text") in hidden_layers:
                doc.set_layer_ui_config(config["number"], action=2)
    except Exception as exc:  # pragma: no cover - depends on PDF features
        print(f"Warning: could not apply hidden layers for text search: {exc}")


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Clamp a bbox to image bounds."""

    height = int(image_shape[0])
    width = int(image_shape[1])

    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2 - x1, y2 - y1)


def _overlaps_excluded_zones(
    bbox: tuple[int, int, int, int],
    exclude_rects: list[tuple[int, int, int, int]],
) -> bool:
    """Return True when bbox intersects an excluded rectangle."""

    for exclude_box in exclude_rects:
        inter_area, _, iom, _ = _bbox_metrics(bbox, exclude_box)
        if inter_area > 0 and iom > 0.10:
            return True
    return False


def _quad_rotation(quad: fitz.Quad) -> int:
    """Estimate text rotation from a quad when available."""

    try:
        dx = quad.ur.x - quad.ul.x
        dy = quad.ur.y - quad.ul.y
        angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        return int((round(angle / 90.0) * 90) % 360)
    except Exception:
        return 0


def _collect_pdf_text_hits(
    pdf_path: str,
    templates: list[TemplateInfo],
    plan_image_shape: tuple[int, int, int] | tuple[int, int],
    dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
) -> dict[int, list[CandidateHit]]:
    """Find short alphanumeric symbols directly in the PDF text layer."""

    if not pdf_path or not os.path.exists(pdf_path):
        return {}

    exclude_rects = exclude_rects or []
    hits_by_template: dict[int, list[CandidateHit]] = {}
    token_rect_cache: dict[str, list[fitz.Quad | fitz.Rect]] = {}

    doc = fitz.open(pdf_path)
    try:
        _apply_hidden_layers(doc, hidden_layers)
        page = doc.load_page(0)
        scale = dpi / 72.0
        page_words = page.get_text("words")

        def get_token_hits(token: str) -> list[fitz.Quad | fitz.Rect]:
            cache_key = token.upper()
            if cache_key in token_rect_cache:
                return token_rect_cache[cache_key]

            quads = page.search_for(token, quads=True)
            if quads:
                token_rect_cache[cache_key] = list(quads)
                return token_rect_cache[cache_key]

            matches: list[fitz.Rect] = []
            for word in page_words:
                if len(word) < 5:
                    continue
                if str(word[4]).upper() == cache_key:
                    matches.append(fitz.Rect(word[0], word[1], word[2], word[3]))

            token_rect_cache[cache_key] = matches
            return matches

        for template_id, template in enumerate(templates):
            if not template.text_tokens:
                continue

            seen_boxes: set[tuple[int, int, int, int]] = set()
            template_hits: list[CandidateHit] = []

            for token in template.text_tokens:
                for hit in get_token_hits(token):
                    rect = hit.rect if hasattr(hit, "rect") else hit
                    bbox = _clamp_bbox(
                        (
                            int(round(rect.x0 * scale)),
                            int(round(rect.y0 * scale)),
                            int(round(rect.width * scale)),
                            int(round(rect.height * scale)),
                        ),
                        plan_image_shape,
                    )
                    if bbox is None or bbox in seen_boxes:
                        continue
                    if _overlaps_excluded_zones(bbox, exclude_rects):
                        continue

                    seen_boxes.add(bbox)
                    template_hits.append(
                        CandidateHit(
                            template_id=template_id,
                            scale=1.0,
                            rotation=_quad_rotation(hit) if hasattr(hit, "ul") else 0,
                            mirrored=False,
                            transformed_mask=None,
                            pixel_count=max(1, bbox[2] * bbox[3]),
                            bbox=bbox,
                            match_score=1.0,
                            dominant_hsv=template.dominant_hsv,
                            source="pdf_text",
                            coverage=1.0,
                            purity=1.0,
                            color_similarity=1.0,
                            verification_score=1.0,
                        )
                    )

            if template_hits:
                hits_by_template[template_id] = template_hits
    finally:
        doc.close()

    return hits_by_template


def _estimate_legend_exclude_rect(
    pdf_path: str,
    image_shape: tuple[int, int, int] | tuple[int, int],
    dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
) -> tuple[int, int, int, int] | None:
    """Estimate the legend bbox on the rendered plan so it can be excluded."""

    if not pdf_path or not os.path.exists(pdf_path):
        return None

    doc = fitz.open(pdf_path)
    try:
        _apply_hidden_layers(doc, hidden_layers)
        page = doc.load_page(0)
        found = page.search_for(LEGEND_KEYWORD)
        if not found:
            return None

        anchor = found[0]
        scale = dpi / 72.0
        bbox = _clamp_bbox(
            (
                int(round((anchor.x0 - 20) * scale)),
                int(round(anchor.y1 * scale)),
                int(round(LEGEND_WIDTH_PT * scale)),
                int(round(LEGEND_HEIGHT_PT * scale)),
            ),
            image_shape,
        )
        return bbox
    except Exception as exc:  # pragma: no cover - depends on input PDF
        print(f"Warning: could not estimate legend area: {exc}")
        return None
    finally:
        doc.close()
