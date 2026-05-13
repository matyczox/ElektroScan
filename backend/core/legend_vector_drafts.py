"""Build vector-first legend drafts while keeping PNG templates as output."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Literal

import cv2
import fitz
import numpy as np

try:
    from .legend_scene_transform import SceneTransform, rect_pt_to_px300, rect_px300_to_pt
    from .legend_vector_profile import PageProfile
except ImportError:  # pragma: no cover - supports direct module execution.
    from legend_scene_transform import SceneTransform, rect_pt_to_px300, rect_px300_to_pt
    from legend_vector_profile import PageProfile

RectPt = tuple[float, float, float, float]
RectPx = tuple[int, int, int, int]


@dataclass(slots=True)
class PrimitiveRef:
    primitive_id: str
    kind: Literal["path", "text", "image"]
    bbox_pt: RectPt
    text: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class VectorLegendDraft:
    draft_id: str
    bbox_pt: RectPt
    bbox_px_300: RectPx
    row_bbox_pt: RectPt
    name_draft: str
    symbol_code: str | None
    confidence: float
    primitive_refs: list[str]
    review_required: bool
    label_source: Literal["right_text", "symbol_text", "synthetic"]
    structure_source: Literal["table_cell", "row_anchor"]
    fallback_eligible: bool
    image_bgr: np.ndarray | None = field(default=None, repr=False)

    def to_json(self) -> dict:
        payload = asdict(self)
        payload.pop("image_bgr", None)
        return payload


def _intersects(a: fitz.Rect, b: fitz.Rect) -> bool:
    return min(a.x1, b.x1) > max(a.x0, b.x0) and min(a.y1, b.y1) > max(a.y0, b.y0)


def _union_rect(rects: list[fitz.Rect]) -> fitz.Rect | None:
    if not rects:
        return None
    out = fitz.Rect(rects[0])
    for rect in rects[1:]:
        out.include_rect(rect)
    return out


def _get_words(page: fitz.Page, clip: fitz.Rect) -> list[tuple]:
    try:
        words = list(page.get_text("words", clip=clip) or [])
    except TypeError:
        words = [
            word
            for word in (page.get_text("words") or [])
            if _intersects(fitz.Rect(word[:4]), clip)
        ]
    return [word for word in words if len(word) > 4 and str(word[4]).strip()]


def _get_drawings(page: fitz.Page) -> list[dict]:
    getter = getattr(page, "get_cdrawings", None)
    if callable(getter):
        try:
            return list(getter() or [])
        except Exception:
            pass
    return list(page.get_drawings() or [])


def _drawing_refs(page: fitz.Page, clip: fitz.Rect) -> list[tuple[str, fitz.Rect]]:
    refs: list[tuple[str, fitz.Rect]] = []
    for idx, drawing in enumerate(_get_drawings(page)):
        rect = drawing.get("rect")
        if rect is None:
            continue
        try:
            drawing_rect = fitz.Rect(rect)
        except Exception:
            continue
        if _intersects(drawing_rect, clip):
            refs.append((f"p:{idx}", drawing_rect))
    return refs


def _group_words_into_rows(words: list[tuple]) -> list[list[tuple]]:
    if not words:
        return []
    heights = [max(1.0, float(word[3]) - float(word[1])) for word in words]
    tolerance = max(5.0, float(np.median(heights)) * 0.7)
    rows: list[list[tuple]] = []
    for word in sorted(
        words,
        key=lambda item: ((float(item[1]) + float(item[3])) / 2.0, float(item[0])),
    ):
        center_y = (float(word[1]) + float(word[3])) / 2.0
        if not rows:
            rows.append([word])
            continue
        last_center = np.mean([(float(item[1]) + float(item[3])) / 2.0 for item in rows[-1]])
        if abs(center_y - float(last_center)) <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return [sorted(row, key=lambda item: float(item[0])) for row in rows if len(row) >= 1]


def _clean_label(text: str) -> str:
    label = re.sub(r"\s+", " ", text).strip()
    return label[:120]


def _is_short_symbol_token(text: str) -> bool:
    token = re.sub(r"[^A-Za-z0-9]+", "", text)
    return 1 <= len(token) <= 8


def _row_bbox(row_words: list[tuple], clip: fitz.Rect) -> fitz.Rect:
    word_rects = [fitz.Rect(word[:4]) for word in row_words]
    rect = _union_rect(word_rects) or fitz.Rect(clip)
    pad_y = max(3.0, rect.height * 0.6)
    return fitz.Rect(clip.x0, max(clip.y0, rect.y0 - pad_y), clip.x1, min(clip.y1, rect.y1 + pad_y))


def _crop_symbol_image(plan_image: np.ndarray, bbox_px: RectPx) -> np.ndarray | None:
    x, y, w, h = bbox_px
    if w <= 1 or h <= 1:
        return None
    img_h, img_w = plan_image.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = plan_image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ink_mask = ((gray < 245) | (hsv[:, :, 1] > 25)).astype(np.uint8) * 255
    if int(cv2.countNonZero(ink_mask)) < 8:
        return None
    out = np.zeros_like(crop)
    out[ink_mask > 0] = crop[ink_mask > 0]
    return out


def build_vector_legend_drafts(
    page: fitz.Page,
    plan_image: np.ndarray,
    legend_rect_px: RectPx,
    transform: SceneTransform,
    profile: PageProfile,
) -> tuple[list[VectorLegendDraft], list[PrimitiveRef]]:
    legend_rect_pt = rect_px300_to_pt(legend_rect_px, transform)
    clip = fitz.Rect(legend_rect_pt)
    words = _get_words(page, clip)
    drawing_refs = _drawing_refs(page, clip)
    primitive_refs: list[PrimitiveRef] = [
        PrimitiveRef(primitive_id=primitive_id, kind="path", bbox_pt=_rect_tuple(rect))
        for primitive_id, rect in drawing_refs
    ]
    primitive_refs.extend(
        PrimitiveRef(
            primitive_id=f"t:{idx}",
            kind="text",
            bbox_pt=_rect_tuple(fitz.Rect(word[:4])),
            text=str(word[4]).strip(),
        )
        for idx, word in enumerate(words)
    )

    drafts: list[VectorLegendDraft] = []
    rows = _group_words_into_rows(words)
    if not rows or not drawing_refs:
        return drafts, primitive_refs

    for row_index, row_words in enumerate(rows, start=1):
        row_rect = _row_bbox(row_words, clip)
        row_drawing_refs = [
            (primitive_id, rect)
            for primitive_id, rect in drawing_refs
            if rect.y1 >= row_rect.y0 and rect.y0 <= row_rect.y1
        ]
        if not row_drawing_refs:
            continue

        drawing_rects = [rect for _, rect in row_drawing_refs]
        drawings_union = _union_rect(drawing_rects)
        if drawings_union is None:
            continue

        symbol_right = float(drawings_union.x1) + max(12.0, drawings_union.width * 0.55)
        code_words = [
            word
            for word in row_words
            if fitz.Rect(word[:4]).x1 <= symbol_right and _is_short_symbol_token(str(word[4]))
        ]
        label_words = [
            word
            for word in row_words
            if fitz.Rect(word[:4]).x0 > symbol_right or word not in code_words
        ]
        if not label_words and row_words:
            label_words = row_words

        label_text = _clean_label(" ".join(str(word[4]).strip() for word in label_words))
        symbol_code = _clean_label(" ".join(str(word[4]).strip() for word in code_words)) or None
        if not label_text and symbol_code:
            label_text = symbol_code
        if not label_text:
            label_text = f"symbol_{row_index:02d}"

        symbol_rects = drawing_rects + [
            fitz.Rect(word[:4]) for word in code_words if fitz.Rect(word[:4]).x0 <= symbol_right
        ]
        symbol_rect = _union_rect(symbol_rects)
        if symbol_rect is None:
            continue
        symbol_rect = fitz.Rect(
            max(clip.x0, symbol_rect.x0 - 1.5),
            max(clip.y0, symbol_rect.y0 - 1.5),
            min(clip.x1, symbol_rect.x1 + 1.5),
            min(clip.y1, symbol_rect.y1 + 1.5),
        )
        bbox_px = rect_pt_to_px300(_rect_tuple(symbol_rect), transform)
        padding = 4
        bbox_px = (
            bbox_px[0] - padding,
            bbox_px[1] - padding,
            bbox_px[2] + 2 * padding,
            bbox_px[3] + 2 * padding,
        )
        image_bgr = _crop_symbol_image(plan_image, bbox_px)
        if image_bgr is None:
            continue

        confidence = 0.55
        if label_text:
            confidence += 0.18
        if len(row_drawing_refs) >= 2:
            confidence += 0.12
        if profile.legend_kind_hint == "table":
            confidence += 0.06
        confidence = min(0.92, confidence)

        primitive_ids = [primitive_id for primitive_id, _ in row_drawing_refs]
        primitive_ids.extend(f"t:{words.index(word)}" for word in code_words if word in words)
        drafts.append(
            VectorLegendDraft(
                draft_id=f"vlegend:{row_index}",
                bbox_pt=_rect_tuple(symbol_rect),
                bbox_px_300=bbox_px,
                row_bbox_pt=_rect_tuple(row_rect),
                name_draft=label_text,
                symbol_code=symbol_code,
                confidence=round(float(confidence), 4),
                primitive_refs=primitive_ids,
                review_required=True,
                label_source="right_text" if label_text != symbol_code else "symbol_text",
                structure_source=(
                    "table_cell" if profile.legend_kind_hint == "table" else "row_anchor"
                ),
                fallback_eligible=True,
                image_bgr=image_bgr,
            )
        )

    return drafts, primitive_refs


def _rect_tuple(rect: fitz.Rect) -> RectPt:
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
