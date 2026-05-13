"""Vector-readiness profiling for legend regions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import fitz

try:
    from .legend_scene_transform import SceneTransform, rect_px300_to_pt
except ImportError:  # pragma: no cover - supports direct module execution.
    from legend_scene_transform import SceneTransform, rect_px300_to_pt

PageKind = Literal["vector_rich", "mixed", "image_only", "unknown"]
LegendKind = Literal["table", "rows", "image_only", "unknown"]


@dataclass(slots=True)
class PageProfile:
    page_kind: PageKind
    legend_kind_hint: LegendKind
    vector_path_count: int
    text_span_count: int
    image_block_count: int
    image_coverage_ratio: float
    closed_or_filled_ratio: float
    table_count: int
    row_anchor_count: int
    attempt_vector: bool
    fallback_reason: str | None

    def to_json(self) -> dict:
        return asdict(self)


def _rect_intersection_area(a: fitz.Rect, b: fitz.Rect) -> float:
    x0 = max(float(a.x0), float(b.x0))
    y0 = max(float(a.y0), float(b.y0))
    x1 = min(float(a.x1), float(b.x1))
    y1 = min(float(a.y1), float(b.y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    return _rect_intersection_area(a, b) > 0


def _get_words_in_clip(page: fitz.Page, clip: fitz.Rect) -> list[tuple]:
    try:
        return list(page.get_text("words", clip=clip) or [])
    except TypeError:
        return [
            word
            for word in (page.get_text("words") or [])
            if _intersects(fitz.Rect(word[:4]), clip)
        ]


def _get_drawings(page: fitz.Page) -> list[dict]:
    getter = getattr(page, "get_cdrawings", None)
    if callable(getter):
        try:
            return list(getter() or [])
        except Exception:
            pass
    return list(page.get_drawings() or [])


def _get_tables_count(page: fitz.Page, clip: fitz.Rect) -> int:
    finder = getattr(page, "find_tables", None)
    if not callable(finder):
        return 0

    for kwargs in ({"clip": clip}, {}):
        try:
            table_finder = finder(**kwargs)
            tables = getattr(table_finder, "tables", table_finder)
            if kwargs:
                return len(list(tables or []))
            return sum(1 for table in (tables or []) if _table_intersects_clip(table, clip))
        except Exception:
            continue
    return 0


def _table_intersects_clip(table: object, clip: fitz.Rect) -> bool:
    bbox = getattr(table, "bbox", None)
    if bbox is None:
        return False
    try:
        return _intersects(fitz.Rect(bbox), clip)
    except Exception:
        return False


def _row_anchor_count(words: list[tuple]) -> int:
    centers: list[float] = []
    for word in words:
        text = str(word[4]).strip() if len(word) > 4 else ""
        if text:
            centers.append((float(word[1]) + float(word[3])) / 2.0)
    if not centers:
        return 0

    centers.sort()
    groups: list[list[float]] = []
    tolerance = 6.0
    for center in centers:
        if not groups or abs(center - (sum(groups[-1]) / len(groups[-1]))) > tolerance:
            groups.append([center])
        else:
            groups[-1].append(center)
    return len(groups)


def profile_legend_region(
    page: fitz.Page,
    legend_rect_px: tuple[int, int, int, int],
    transform: SceneTransform,
) -> PageProfile:
    legend_rect_pt = rect_px300_to_pt(legend_rect_px, transform)
    clip = fitz.Rect(legend_rect_pt)
    clip_area = max(float(clip.width * clip.height), 1.0)

    words = _get_words_in_clip(page, clip)
    drawings = []
    filled_or_closed = 0
    for drawing in _get_drawings(page):
        rect = drawing.get("rect")
        if rect is None:
            continue
        try:
            drawing_rect = fitz.Rect(rect)
        except Exception:
            continue
        if not _intersects(drawing_rect, clip):
            continue
        drawings.append(drawing)
        drawing_type = str(drawing.get("type", ""))
        if drawing.get("fill") is not None or "f" in drawing_type or drawing.get("closePath"):
            filled_or_closed += 1

    image_block_count = 0
    image_area = 0.0
    try:
        text_dict = page.get_text("dict", clip=clip)
    except TypeError:
        text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []) if isinstance(text_dict, dict) else []:
        if block.get("type") != 1:
            continue
        try:
            block_rect = fitz.Rect(block.get("bbox"))
        except Exception:
            continue
        overlap_area = _rect_intersection_area(block_rect, clip)
        if overlap_area <= 0:
            continue
        image_block_count += 1
        image_area += overlap_area

    vector_path_count = len(drawings)
    text_span_count = len([word for word in words if str(word[4]).strip()])
    image_coverage_ratio = min(1.0, image_area / clip_area)
    closed_or_filled_ratio = (
        filled_or_closed / vector_path_count if vector_path_count > 0 else 0.0
    )
    table_count = _get_tables_count(page, clip)
    row_anchor_count = _row_anchor_count(words)

    if image_coverage_ratio >= 0.65 and vector_path_count < 8:
        page_kind: PageKind = "image_only"
    elif vector_path_count >= 8 and text_span_count >= 2:
        page_kind = "vector_rich"
    elif image_block_count > 0 and (vector_path_count >= 3 or text_span_count >= 2):
        page_kind = "mixed"
    else:
        page_kind = "unknown"

    if page_kind == "image_only":
        legend_kind_hint: LegendKind = "image_only"
    elif table_count > 0:
        legend_kind_hint = "table"
    elif row_anchor_count >= 2:
        legend_kind_hint = "rows"
    else:
        legend_kind_hint = "unknown"

    fallback_reason = None
    if page_kind == "image_only":
        fallback_reason = "legend_image_dominant"
    elif vector_path_count < 8:
        fallback_reason = "insufficient_vector_primitives"
    elif text_span_count < 2:
        fallback_reason = "insufficient_text_primitives"
    elif row_anchor_count < 2:
        fallback_reason = "insufficient_row_anchors"

    attempt_vector = fallback_reason is None and page_kind in {"vector_rich", "mixed"}
    return PageProfile(
        page_kind=page_kind,
        legend_kind_hint=legend_kind_hint,
        vector_path_count=vector_path_count,
        text_span_count=text_span_count,
        image_block_count=image_block_count,
        image_coverage_ratio=round(float(image_coverage_ratio), 4),
        closed_or_filled_ratio=round(float(closed_or_filled_ratio), 4),
        table_count=table_count,
        row_anchor_count=row_anchor_count,
        attempt_vector=attempt_vector,
        fallback_reason=fallback_reason,
    )
