"""PDF rendering, preview, and diagnostics helpers for the API."""

from __future__ import annotations

import base64
import math
import os
from collections import OrderedDict
from threading import Lock

import cv2
import fitz
import numpy as np

from core.detector_masks import _hsv_mask, _ink_mask
from core.legend_extractor import _normalize_layer_name, get_pdf_layers, pdf_to_png


ANALYSIS_DPI = 300
PREVIEW_DPI = int(os.getenv("ELEKTROSCAN_PREVIEW_DPI", "180"))
RENDER_CACHE_MAX_ENTRIES = int(os.getenv("ELEKTROSCAN_RENDER_CACHE_MAX_ENTRIES", "4"))
RENDER_CACHE: OrderedDict[tuple[str, int, int, tuple[str, ...]], np.ndarray] = OrderedDict()
RENDER_CACHE_LOCK = Lock()


def _normalized_hidden_layer_key(hidden_layers: list[str] | None = None) -> tuple[str, ...]:
    return tuple(
        sorted(
            _normalize_layer_name(layer)
            for layer in (hidden_layers or [])
            if str(layer).strip()
        )
    )


def _clear_render_cache(session_id: str | None = None) -> None:
    with RENDER_CACHE_LOCK:
        if session_id is None:
            RENDER_CACHE.clear()
            return
        for key in list(RENDER_CACHE.keys()):
            if key[0] == session_id:
                RENDER_CACHE.pop(key, None)


def _pdf_page_size_at_dpi(pdf_path: str, *, page: int = 0, dpi: int = ANALYSIS_DPI) -> dict:
    doc = fitz.open(pdf_path)
    try:
        pg = doc.load_page(page)
        zoom = dpi / 72.0
        return {
            "width": int(math.ceil(pg.rect.width * zoom)),
            "height": int(math.ceil(pg.rect.height * zoom)),
        }
    finally:
        doc.close()


def _render_pdf_for_session(
    session_id: str,
    pdf_path: str,
    *,
    page: int = 0,
    dpi: int = ANALYSIS_DPI,
    hidden_layers: list[str] | None = None,
    copy_image: bool = False,
) -> tuple[np.ndarray, bool]:
    key = (session_id, int(page), int(dpi), _normalized_hidden_layer_key(hidden_layers))
    with RENDER_CACHE_LOCK:
        cached = RENDER_CACHE.get(key)
        if cached is not None:
            RENDER_CACHE.move_to_end(key)
            return (cached.copy() if copy_image else cached), True

    rendered = pdf_to_png(pdf_path, page=page, dpi=dpi, hidden_layers=hidden_layers or [])

    with RENDER_CACHE_LOCK:
        RENDER_CACHE[key] = rendered
        RENDER_CACHE.move_to_end(key)
        while len(RENDER_CACHE) > max(1, RENDER_CACHE_MAX_ENTRIES):
            RENDER_CACHE.popitem(last=False)

    return (rendered.copy() if copy_image else rendered), False


def _preview_response_meta(
    plan_img: np.ndarray,
    *,
    preview_dpi: int,
    analysis_size: dict,
    cache_hit: bool,
) -> dict:
    return {
        "previewDpi": int(preview_dpi),
        "analysisDpi": int(ANALYSIS_DPI),
        "analysisSize": analysis_size,
        "isFullResolution": int(preview_dpi) == int(ANALYSIS_DPI),
        "renderCacheHit": bool(cache_hit),
        "imageWidth": int(plan_img.shape[1]),
        "imageHeight": int(plan_img.shape[0]),
    }


def _ink_profile_stats(plan_img: np.ndarray) -> dict:
    total_pixels = int(plan_img.shape[0] * plan_img.shape[1])
    if total_pixels <= 0:
        return {
            "inkPct": 0.0,
            "colorfulInkPct": 0.0,
            "grayInkPct": 0.0,
            "recommendedProfile": "color",
        }

    ink = _ink_mask(plan_img, dilate=False, ignore_color=False)
    color_ink = _hsv_mask(plan_img, dilate=False)
    if cv2.countNonZero(color_ink):
        ink = cv2.bitwise_or(ink, color_ink)
    ink_pixels = int(np.count_nonzero(ink))
    if ink_pixels <= 0:
        return {
            "inkPct": 0.0,
            "colorfulInkPct": 0.0,
            "grayInkPct": 0.0,
            "recommendedProfile": "color",
        }

    colorful = cv2.bitwise_and(ink, color_ink)
    colorful_pixels = int(cv2.countNonZero(colorful))
    gray_pixels = ink_pixels - colorful_pixels
    colorful_ink_pct = (colorful_pixels / ink_pixels) * 100.0

    return {
        "inkPct": round((ink_pixels / total_pixels) * 100.0, 3),
        "colorfulInkPct": round(colorful_ink_pct, 3),
        "grayInkPct": round((gray_pixels / ink_pixels) * 100.0, 3),
        "recommendedProfile": "gray" if colorful_ink_pct < 1.0 else "color",
    }


def _build_pdf_diagnostics(pdf_path: str, plan_img: np.ndarray | None = None) -> dict:
    diagnostics = {
        "pages": 0,
        "layers": 0,
        "textCharsPage1": 0,
        "textBlocksPage1": 0,
        "drawingsPage1": 0,
        "imagesPage1": 0,
        "inkPct": 0.0,
        "colorfulInkPct": 0.0,
        "grayInkPct": 0.0,
        "recommendedProfile": "color",
    }

    try:
        diagnostics["layers"] = len(get_pdf_layers(pdf_path))
    except Exception:
        pass

    doc = fitz.open(pdf_path)
    try:
        diagnostics["pages"] = int(doc.page_count)
        if doc.page_count:
            page = doc.load_page(0)
            text_blocks = page.get_text("blocks")
            diagnostics["textBlocksPage1"] = int(
                sum(1 for block in text_blocks if len(block) > 6 and block[6] == 0)
            )
            diagnostics["textCharsPage1"] = int(len(page.get_text("text") or ""))
            try:
                diagnostics["drawingsPage1"] = int(len(page.get_drawings()))
            except Exception:
                diagnostics["drawingsPage1"] = 0
            try:
                diagnostics["imagesPage1"] = int(len(page.get_images(full=True)))
            except Exception:
                diagnostics["imagesPage1"] = 0
    finally:
        doc.close()

    if plan_img is None:
        try:
            plan_img = pdf_to_png(pdf_path, dpi=100)
        except Exception:
            plan_img = None

    if plan_img is not None:
        diagnostics.update(_ink_profile_stats(plan_img))

    return diagnostics


def _render_preview_response_for_session(
    session_id: str,
    file_path,
    body=None,
) -> dict:
    hidden_layers = body.hidden_layers if body else []
    render_dpi = PREVIEW_DPI if body and body.preview else ANALYSIS_DPI
    plan_img, cache_hit = _render_pdf_for_session(
        session_id,
        str(file_path),
        dpi=render_dpi,
        hidden_layers=hidden_layers,
    )
    analysis_size = _pdf_page_size_at_dpi(str(file_path), dpi=ANALYSIS_DPI)
    pdf_diagnostics = _build_pdf_diagnostics(str(file_path), plan_img)
    _, buffer_plan = cv2.imencode(".jpg", plan_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    plan_base64 = base64.b64encode(buffer_plan).decode("utf-8")
    return {
        "planPreview": f"data:image/jpeg;base64,{plan_base64}",
        "pdfDiagnostics": pdf_diagnostics,
        **_preview_response_meta(
            plan_img,
            preview_dpi=render_dpi,
            analysis_size=analysis_size,
            cache_hit=cache_hit,
        ),
    }
