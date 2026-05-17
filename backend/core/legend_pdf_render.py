"""PDF rendering and optional-content-layer helpers for legend extraction."""

from __future__ import annotations

import re
import unicodedata

import cv2
import fitz
import numpy as np


def get_pdf_layers(pdf_path: str) -> list[dict]:
    """Return Optional Content Group layer metadata exposed by a PDF."""
    doc = fitz.open(pdf_path)
    layers = []

    try:
        ui_configs = doc.layer_ui_configs()
        if ui_configs:
            for conf in ui_configs:
                if "text" in conf:
                    layers.append({"name": conf["text"], "visible": conf.get("on", True)})
    except Exception as exc:
        print(f"Blad odczytu warstw: {exc}")
    finally:
        doc.close()

    return layers


def _render_doc_to_bgr(doc: fitz.Document, page: int = 0, dpi: int = 300) -> np.ndarray:
    """Render a PDF page from an already prepared document."""
    pg = doc.load_page(page)
    zoom = dpi / 72.0
    pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _normalize_layer_name(name: str) -> str:
    """Normalize layer names to make matching resilient to PDF text encoding quirks."""
    text = str(name).strip()

    try:
        repaired = text.encode("latin1").decode("utf-8")
        if "\ufffd" not in repaired:
            text = repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    text = text.casefold().translate(str.maketrans({"ł": "l", "Ł": "l"}))
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _get_ocg_xrefs_in_catalog_order(doc: fitz.Document) -> list[int]:
    """Read OCG xrefs from the PDF catalog in their declared order."""
    catalog_object = doc.xref_object(doc.pdf_catalog())
    match = re.search(r"/OCGs\s*\[(.*?)\]", catalog_object, re.S)
    if not match:
        return []

    return [int(value) for value in re.findall(r"(\d+)\s+0\s+R", match.group(1))]


def _prepare_doc_with_hidden_layers(
    pdf_path: str,
    hidden_layers: list[str] | None = None,
) -> fitz.Document:
    """
    Open a PDF and apply hidden OCG layers in a way that affects rendering.

    Some PDFs expose layer UI state, but PyMuPDF's direct `set_layer_ui_config()`
    does not change `get_pixmap()` output for them. Updating the OCG defaults and
    reopening the modified in-memory PDF does.
    """
    doc = fitz.open(pdf_path)
    hidden_layers = [name for name in (hidden_layers or []) if name]
    if not hidden_layers:
        return doc

    try:
        ui_configs = doc.layer_ui_configs() or []
        ocg_xrefs = _get_ocg_xrefs_in_catalog_order(doc)
        if not ui_configs or not ocg_xrefs:
            return doc

        hidden_set = {_normalize_layer_name(name) for name in hidden_layers}
        off_refs: list[int] = []

        for config in ui_configs:
            layer_name = str(config.get("text", "")).strip()
            if not layer_name:
                continue

            number = config.get("number")
            if not isinstance(number, int):
                continue
            if number < 0 or number >= len(ocg_xrefs):
                continue

            if _normalize_layer_name(layer_name) in hidden_set:
                off_refs.append(ocg_xrefs[number])
        if not off_refs:
            return doc

        catalog_xref = doc.pdf_catalog()
        off_value = "[" + " ".join(f"{ref} 0 R" for ref in off_refs) + "]"

        doc.xref_set_key(catalog_xref, "OCProperties/D/OFF", off_value)

        mutated_pdf = doc.write()
        doc.close()
        return fitz.open(stream=mutated_pdf, filetype="pdf")
    except Exception as exc:
        print(f"Blad ukrywania warstw: {exc}")
        return doc


def pdf_to_png(
    pdf_path: str,
    page: int = 0,
    dpi: int = 300,
    hidden_layers: list[str] | None = None,
) -> np.ndarray:
    """Convert a PDF page to an OpenCV BGR image."""
    doc = _prepare_doc_with_hidden_layers(pdf_path, hidden_layers=hidden_layers)
    try:
        return _render_doc_to_bgr(doc, page=page, dpi=dpi)
    finally:
        doc.close()
