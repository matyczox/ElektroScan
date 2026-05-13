"""Coordinate helpers for legend extraction spike.

All backend geometry in the vector-first spike uses MuPDF page coordinates:
points, top-left origin, after PyMuPDF has applied page crop/rotation metadata to
``page.rect``. Raster coordinates are treated as a render cache at the chosen DPI.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz

PT_PER_INCH = 72.0

RectPt = tuple[float, float, float, float]
RectPx = tuple[int, int, int, int]


@dataclass(slots=True)
class SceneTransform:
    page_index: int
    dpi: int
    page_rect_pt: RectPt
    cropbox_pt: RectPt
    rotation_deg: int
    raster_size_px_300: tuple[int, int]
    hidden_layers: list[str]
    source_pdf_sha256: str

    def to_json(self) -> dict:
        return asdict(self)


def hash_pdf_file(pdf_path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(pdf_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rect_to_tuple(rect: fitz.Rect) -> RectPt:
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def build_scene_transform(
    page: fitz.Page,
    *,
    dpi: int = 300,
    hidden_layers: list[str] | None = None,
    source_pdf_sha256: str = "",
) -> SceneTransform:
    scale = dpi / PT_PER_INCH
    raster_w = int(round(float(page.rect.width) * scale))
    raster_h = int(round(float(page.rect.height) * scale))
    return SceneTransform(
        page_index=int(page.number),
        dpi=int(dpi),
        page_rect_pt=_rect_to_tuple(page.rect),
        cropbox_pt=_rect_to_tuple(page.cropbox),
        rotation_deg=int(page.rotation),
        raster_size_px_300=(raster_w, raster_h),
        hidden_layers=list(hidden_layers or []),
        source_pdf_sha256=source_pdf_sha256,
    )


def rect_pt_to_px300(rect_pt: RectPt, transform: SceneTransform) -> RectPx:
    """Convert a MuPDF point-space rect to raster pixels at ``transform.dpi``."""

    x0, y0, x1, y1 = rect_pt
    page_x0, page_y0, _, _ = transform.page_rect_pt
    scale = transform.dpi / PT_PER_INCH
    return (
        int(round((x0 - page_x0) * scale)),
        int(round((y0 - page_y0) * scale)),
        int(round((x1 - x0) * scale)),
        int(round((y1 - y0) * scale)),
    )


def rect_px300_to_pt(rect_px: RectPx, transform: SceneTransform) -> RectPt:
    """Convert raster pixels at ``transform.dpi`` to MuPDF point-space rect."""

    x, y, w, h = rect_px
    page_x0, page_y0, _, _ = transform.page_rect_pt
    scale = transform.dpi / PT_PER_INCH
    return (
        float(x) / scale + page_x0,
        float(y) / scale + page_y0,
        float(x + w) / scale + page_x0,
        float(y + h) / scale + page_y0,
    )


def rect_px300_to_canvas(
    rect_px: RectPx,
    transform: SceneTransform,
    canvas_size_px: tuple[int, int] | None,
) -> tuple[float, float, float, float] | None:
    if canvas_size_px is None:
        return None

    raster_w, raster_h = transform.raster_size_px_300
    canvas_w, canvas_h = canvas_size_px
    if raster_w <= 0 or raster_h <= 0:
        return None

    x, y, w, h = rect_px
    return (
        x * canvas_w / raster_w,
        y * canvas_h / raster_h,
        w * canvas_w / raster_w,
        h * canvas_h / raster_h,
    )
