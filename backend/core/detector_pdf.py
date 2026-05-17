"""PDF text fallback and legend exclusion helpers."""

from __future__ import annotations

import os
import re

import fitz
import numpy as np

from core.detector_clustering import _bbox_metrics
from core.detector_config import DEFAULT_PDF_DPI, LEGEND_HEIGHT_PT, LEGEND_KEYWORD, LEGEND_WIDTH_PT
from core.detector_models import CandidateHit, TemplateInfo

PdfWordBox = tuple[str, tuple[int, int, int, int]]


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


def _word_matches_token(word_text: str, token: str) -> bool:
    """Exact PDF-token match for short labels.

    MuPDF search_for() is substring based, so searching for "L3" also finds
    "RL3" / "SL3". For detector labels we need full token semantics.
    """

    clean_word = str(word_text or "").strip().upper()
    clean_token = str(token or "").strip().upper()
    if not clean_word or not clean_token:
        return False
    if clean_word == clean_token:
        return True
    # Keep punctuation-separated spellings like "TB1.1" out of short label
    # fallback unless the normalized full token is exactly the template token.
    normalized_word = re.sub(r"[^A-Z0-9]+", "", clean_word)
    normalized_token = re.sub(r"[^A-Z0-9]+", "", clean_token)
    return bool(normalized_word and normalized_word == normalized_token)


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

            matches: list[fitz.Rect] = []
            for word in page_words:
                if len(word) < 5:
                    continue
                if _word_matches_token(str(word[4]), cache_key):
                    matches.append(fitz.Rect(word[0], word[1], word[2], word[3]))

            token_rect_cache[cache_key] = matches
            return matches

        def expand_text_bbox_for_template(
            template: TemplateInfo,
            text_bbox: tuple[int, int, int, int],
        ) -> tuple[int, int, int, int]:
            if (
                not template.is_text_label
                or template.content_bbox is None
                or template.mask.size == 0
            ):
                return text_bbox

            content_x, content_y, content_w, content_h = template.content_bbox
            if content_w <= 0 or content_h <= 0:
                return text_bbox

            scale_x = text_bbox[2] / max(1, content_w)
            scale_y = text_bbox[3] / max(1, content_h)
            scale_factor = max(0.55, min(1.45, (scale_x + scale_y) / 2.0))
            full_w = max(text_bbox[2], int(round(template.mask.shape[1] * scale_factor)))
            full_h = max(text_bbox[3], int(round(template.mask.shape[0] * scale_factor)))
            full_x = int(round(text_bbox[0] - content_x * scale_factor))
            full_y = int(round(text_bbox[1] - content_y * scale_factor))
            return (full_x, full_y, full_w, full_h)

        for template_id, template in enumerate(templates):
            if not template.text_tokens:
                continue

            seen_boxes: set[tuple[int, int, int, int]] = set()
            template_hits: list[CandidateHit] = []

            for token in template.text_tokens:
                for hit in get_token_hits(token):
                    rect = hit.rect if hasattr(hit, "rect") else hit
                    text_bbox = (
                        int(round(rect.x0 * scale)),
                        int(round(rect.y0 * scale)),
                        int(round(rect.width * scale)),
                        int(round(rect.height * scale)),
                    )
                    bbox = _clamp_bbox(
                        expand_text_bbox_for_template(template, text_bbox),
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
                            content_mask=None,
                            pixel_count=max(1, bbox[2] * bbox[3]),
                            content_pixel_count=0,
                            content_bbox=None,
                            bbox=bbox,
                            match_score=1.0,
                            dominant_hsv=template.dominant_hsv,
                            source="pdf_text",
                            is_text_label=template.is_text_label,
                            coverage=1.0,
                            purity=1.0,
                            color_similarity=1.0,
                            verification_score=1.0,
                            content_score=1.0 if template.is_text_label else 0.0,
                        )
                    )

            if template_hits:
                hits_by_template[template_id] = template_hits
    finally:
        doc.close()

    return hits_by_template


def _collect_pdf_word_boxes(
    pdf_path: str,
    image_shape: tuple[int, int, int] | tuple[int, int],
    dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
) -> list[PdfWordBox]:
    """Return exact PDF word tokens in rendered image coordinates."""

    if not pdf_path or not os.path.exists(pdf_path):
        return []

    exclude_rects = exclude_rects or []
    words: list[PdfWordBox] = []
    doc = fitz.open(pdf_path)
    try:
        _apply_hidden_layers(doc, hidden_layers)
        page = doc.load_page(0)
        scale = dpi / 72.0
        for word in page.get_text("words"):
            if len(word) < 5:
                continue
            token = re.sub(r"[^A-Z0-9]+", "", str(word[4] or "").upper())
            if not token:
                continue
            bbox = _clamp_bbox(
                (
                    int(round(float(word[0]) * scale)),
                    int(round(float(word[1]) * scale)),
                    int(round((float(word[2]) - float(word[0])) * scale)),
                    int(round((float(word[3]) - float(word[1])) * scale)),
                ),
                image_shape,
            )
            if bbox is None or _overlaps_excluded_zones(bbox, exclude_rects):
                continue
            words.append((token, bbox))
    finally:
        doc.close()

    return words


def _collect_pdf_text_exclude_rects(
    pdf_path: str,
    image_shape: tuple[int, int, int] | tuple[int, int],
    dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Return rendered-image rectangles for descriptive PDF text blocks.

    Gray/black plans put notes, title blocks, walls and symbols into one ink
    mask. This helper removes only longer text blocks from the gray search
    mask. Short labels such as "ZG" are deliberately kept because they can be
    part of actual symbols.
    """

    if not pdf_path or not os.path.exists(pdf_path):
        return []

    rects: list[tuple[int, int, int, int]] = []
    doc = fitz.open(pdf_path)
    try:
        _apply_hidden_layers(doc, hidden_layers)
        page = doc.load_page(0)
        scale = dpi / 72.0

        for block in page.get_text("blocks"):
            if len(block) < 7 or block[6] != 0:
                continue

            raw_text = str(block[4] or "")
            clean_text = " ".join(raw_text.split())
            alnum_len = sum(1 for char in clean_text if char.isalnum())
            if alnum_len < 6 and " " not in clean_text:
                continue

            x0, y0, x1, y1 = [float(value) for value in block[:4]]
            width_px = (x1 - x0) * scale
            height_px = (y1 - y0) * scale
            aspect = max(width_px / max(1.0, height_px), height_px / max(1.0, width_px))

            long_edge = max(width_px, height_px)
            looks_like_description = (
                alnum_len >= 10
                or (" " in clean_text and alnum_len >= 8)
                or (alnum_len >= 6 and aspect >= 2.2 and long_edge >= 120)
                or width_px >= 120
            )
            if not looks_like_description:
                continue

            padding = max(2, int(round(1.5 * scale)))
            bbox = _clamp_bbox(
                (
                    int(round(x0 * scale)) - padding,
                    int(round(y0 * scale)) - padding,
                    int(round(width_px)) + 2 * padding,
                    int(round(height_px)) + 2 * padding,
                ),
                image_shape,
            )
            if bbox is not None:
                rects.append(bbox)
    except Exception as exc:  # pragma: no cover - depends on PDF structure
        print(f"Warning: could not collect PDF text exclude rects: {exc}")
    finally:
        doc.close()

    return rects


def _estimate_title_block_exclude_rects(
    pdf_path: str,
    image_shape: tuple[int, int, int] | tuple[int, int],
    dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Estimate dense annotation/title-block regions to skip in gray mode."""

    if not pdf_path or not os.path.exists(pdf_path):
        return []

    image_h, image_w = int(image_shape[0]), int(image_shape[1])
    rects: list[tuple[int, int, int, int]] = []
    doc = fitz.open(pdf_path)
    try:
        _apply_hidden_layers(doc, hidden_layers)
        page = doc.load_page(0)
        scale = dpi / 72.0
        text_rects: list[tuple[int, int, int, int]] = []

        for block in page.get_text("blocks"):
            if len(block) < 7 or block[6] != 0:
                continue
            text = " ".join(str(block[4] or "").split())
            if len(text) < 3:
                continue
            bbox = _clamp_bbox(
                (
                    int(round(float(block[0]) * scale)),
                    int(round(float(block[1]) * scale)),
                    int(round((float(block[2]) - float(block[0])) * scale)),
                    int(round((float(block[3]) - float(block[1])) * scale)),
                ),
                image_shape,
            )
            if bbox is not None:
                text_rects.append(bbox)

        if not text_rects:
            return []

        # Right-side vertical title/notes column.
        right_boundary = int(image_w * 0.72)
        right_rects = [rect for rect in text_rects if rect[0] + rect[2] / 2 >= right_boundary]
        if len(right_rects) >= 12:
            min_x = max(0, min(x for x, _y, _w, _h in right_rects) - 16)
            min_y = max(0, min(y for _x, y, _w, _h in right_rects) - 16)
            max_x = image_w
            max_y = min(image_h, max(y + h for _x, y, _w, h in right_rects) + 16)
            if (max_y - min_y) > image_h * 0.20:
                rects.append((min_x, min_y, max_x - min_x, max_y - min_y))

        # Bottom title strip. Keep this conservative: on dense floor plans,
        # normal room labels and symbols can occupy the lower quarter of the
        # drawing, so treating that as a title block hides real detections.
        bottom_boundary = int(image_h * 0.93)
        bottom_rects = [rect for rect in text_rects if rect[1] + rect[3] / 2 >= bottom_boundary]
        if len(bottom_rects) >= 10:
            min_x = max(0, min(x for x, _y, _w, _h in bottom_rects) - 16)
            min_y = max(0, min(y for _x, y, _w, _h in bottom_rects) - 16)
            max_x = min(image_w, max(x + w for x, _y, w, _h in bottom_rects) + 16)
            max_y = image_h
            if min_y >= image_h * 0.90 and (max_x - min_x) > image_w * 0.22:
                rects.append((min_x, min_y, max_x - min_x, max_y - min_y))

    except Exception as exc:  # pragma: no cover - depends on PDF structure
        print(f"Warning: could not estimate title-block exclude rects: {exc}")
    finally:
        doc.close()

    return rects


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
