"""Classic-row and gray-row symbol bbox extraction helpers."""

from __future__ import annotations

import re

import cv2
import numpy as np

try:
    from .legend_constants import GLUE_KERNEL, MIN_PIXEL_DENSITY, MIN_SYMBOL_SIZE, SYMBOL_PADDING
    from .legend_label_extractor import (
        _get_classic_row_label_text,
        _is_row_label_prefix,
        _ocr_text_from_image,
    )
    from .legend_text import _clean_ocr_label_text, _sanitize_filename
except ImportError:  # pragma: no cover
    from legend_constants import GLUE_KERNEL, MIN_PIXEL_DENSITY, MIN_SYMBOL_SIZE, SYMBOL_PADDING
    from legend_label_extractor import (
        _get_classic_row_label_text,
        _is_row_label_prefix,
        _ocr_text_from_image,
    )
    from legend_text import _clean_ocr_label_text, _sanitize_filename

def _filter_gray_legend_symbol_contours(
    contours: list[np.ndarray],
    legend_shape: tuple[int, int, int],
) -> list[np.ndarray]:
    """
    For gray legends, descriptions use the same dark ink as symbols. Detect the
    wide description column visually and keep only row-paired shapes in the
    symbol column. This avoids learning whole text descriptions as templates.
    """
    if not contours:
        return contours

    legend_h, legend_w = legend_shape[:2]
    rects = [(contour, cv2.boundingRect(contour)) for contour in contours]
    min_text_width = max(60, int(legend_w * 0.16))
    max_text_height = max(8, int(legend_h * 0.12))
    long_text_rects = [
        rect
        for _contour, rect in rects
        if rect[2] >= min_text_width
        and rect[3] <= max_text_height
        and rect[0] >= int(legend_w * 0.05)
    ]

    if not long_text_rects:
        return contours

    text_start_x = min(x for x, _y, _w, _h in long_text_rects)
    symbol_column_right = max(0, text_start_x - 3)
    filtered: list[np.ndarray] = []

    for contour, (x, y, w, h) in rects:
        if x >= symbol_column_right:
            continue

        center_y = y + h / 2
        has_row_description = any(
            tx > x + w and abs((ty + th / 2) - center_y) <= max(24, h * 1.6, th * 1.6)
            for tx, ty, tw, th in long_text_rects
        )
        if not has_row_description:
            continue

        filtered.append(contour)

    return filtered or contours


def _strip_gray_legend_descriptions(symbol_mask: np.ndarray) -> np.ndarray:
    """
    Remove the right-side description text column from gray legends.

    This is intentionally geometry-based: long dark components in the right
    part of the legend are treated as descriptions, while the left column is
    kept as the symbol source.
    """
    if symbol_mask.size == 0:
        return symbol_mask

    cut_x = _detect_gray_description_cut(symbol_mask)
    if cut_x is None:
        return symbol_mask

    stripped = symbol_mask.copy()
    stripped[:, cut_x:] = 0
    return stripped


def _detect_gray_description_cut(symbol_mask: np.ndarray) -> int | None:
    """Find the vertical gap between the symbol column and descriptions."""

    if symbol_mask.size == 0:
        return None

    legend_h, legend_w = symbol_mask.shape[:2]
    if legend_w < 80 or legend_h < 80:
        return None

    # First try to find the vertical whitespace gap between the symbol column
    # and the description column. This is more stable than recognizing words.
    column_ink = np.count_nonzero(symbol_mask > 0, axis=0)
    has_ink = column_ink > 0
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(has_ink):
        if not value and start is None:
            start = idx
        elif value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(has_ink)))

    min_gap_width = max(14, int(legend_w * 0.025))
    usable_runs = [
        (start, end)
        for start, end in runs
        if end - start >= min_gap_width and int(legend_w * 0.08) <= start <= int(legend_w * 0.55)
    ]
    if usable_runs:
        start, end = max(usable_runs, key=lambda item: item[1] - item[0])
        cut_x = max(0, start - 2)
        if cv2.countNonZero(symbol_mask[:, :cut_x]) > 0:
            return cut_x

    # Fallback: connect letters into word/description components, without merging rows.
    text_kernel = np.ones((3, 70), np.uint8)
    text_probe = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, text_kernel)
    contours, _ = cv2.findContours(text_probe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_description_x = int(legend_w * 0.12)
    min_description_width = max(45, int(legend_w * 0.08))
    max_description_height = max(8, int(legend_h * 0.12))

    description_starts: list[int] = []
    for contour in contours:
        x, _y, w, h = cv2.boundingRect(contour)
        if x < min_description_x:
            continue
        if w < min_description_width or h > max_description_height:
            continue
        description_starts.append(x)

    if not description_starts:
        return None

    return max(0, min(description_starts) - 8)


def _rect_to_contour(rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    x2 = x + max(1, w) - 1
    y2 = y + max(1, h) - 1
    return np.array([[[x, y]], [[x2, y]], [[x2, y2]], [[x, y2]]], dtype=np.int32)


def _gray_row_symbol_bboxes(
    raw_symbol_mask: np.ndarray,
    text_blocks: list,
    *,
    x_start: int,
    y_start: int,
    scale: float,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Segment a gray legend by rows instead of connected fragments.

    In gray PDFs the frame, arcs and letters are all dark ink. Contour-based
    extraction often learns tiny pieces of a symbol. Row segmentation uses the
    description text as row anchors, but still crops the symbol from the image.
    """

    cut_x = _detect_gray_description_cut(raw_symbol_mask)
    if cut_x is None:
        return _strip_gray_legend_descriptions(raw_symbol_mask), []

    symbol_mask = raw_symbol_mask.copy()
    symbol_mask[:, cut_x:] = 0
    legend_h, legend_w = raw_symbol_mask.shape[:2]

    row_spans: list[tuple[float, float, float]] = []
    for block in text_blocks:
        if len(block) < 7 or block[6] != 0:
            continue

        text = " ".join(str(block[4] or "").split())
        if sum(1 for char in text if char.isalnum()) < 3:
            continue

        bx0 = float(block[0]) * scale - x_start
        by0 = float(block[1]) * scale - y_start
        bx1 = float(block[2]) * scale - x_start
        by1 = float(block[3]) * scale - y_start
        if bx1 <= 0 or by1 <= 0 or bx0 >= legend_w or by0 >= legend_h:
            continue

        center_x = (bx0 + bx1) / 2.0
        if center_x < cut_x + 6:
            continue

        y0 = max(0.0, by0)
        y1 = min(float(legend_h), by1)
        if y1 <= y0:
            continue
        row_spans.append((y0, y1, (y0 + y1) / 2.0))

    visual_row_spans = False
    if not row_spans:
        row_spans = _visual_gray_description_row_spans(raw_symbol_mask, cut_x)
        visual_row_spans = bool(row_spans)
    if not row_spans:
        return symbol_mask, []

    grouped_rows = (
        _group_visual_gray_row_spans(row_spans)
        if visual_row_spans
        else _group_gray_row_spans(row_spans)
    )
    centers = [row[2] for row in grouped_rows]
    bands: list[tuple[int, int]] = []
    outer_padding = 18.0 if visual_row_spans else 34.0
    for idx, row in enumerate(grouped_rows):
        row_top, row_bottom, _row_center = row
        if idx == 0:
            band_top = max(0.0, row_top - outer_padding)
        else:
            band_top = (centers[idx - 1] + centers[idx]) / 2.0

        if idx == len(grouped_rows) - 1:
            band_bottom = min(float(legend_h), row_bottom + outer_padding)
        else:
            band_bottom = (centers[idx] + centers[idx + 1]) / 2.0

        bands.append((max(0, int(np.floor(band_top))), min(legend_h, int(np.ceil(band_bottom)))))

    bboxes: list[tuple[int, int, int, int]] = []
    for y0, y1 in bands:
        if y1 <= y0:
            continue
        row_mask = symbol_mask[y0:y1, :cut_x]
        pixels = cv2.findNonZero(row_mask)
        if pixels is None:
            continue

        rx, ry, rw, rh = cv2.boundingRect(pixels)
        x1 = max(0, rx - SYMBOL_PADDING)
        y1_abs = max(0, y0 + ry - SYMBOL_PADDING)
        x2 = min(cut_x, rx + rw + SYMBOL_PADDING)
        y2_abs = min(legend_h, y0 + ry + rh + SYMBOL_PADDING)
        if x2 <= x1 or y2_abs <= y1_abs:
            continue

        bbox_w = x2 - x1
        bbox_h = y2_abs - y1_abs
        if bbox_w < 8 or bbox_h < 8:
            continue
        bboxes.append((x1, y1_abs, bbox_w, bbox_h))

    return symbol_mask, bboxes


def _visual_gray_description_row_spans(
    raw_symbol_mask: np.ndarray,
    cut_x: int,
) -> list[tuple[float, float, float]]:
    """Use raster description text as row anchors when PDF text is unavailable."""

    if raw_symbol_mask.size == 0:
        return []

    legend_h, legend_w = raw_symbol_mask.shape[:2]
    if cut_x >= legend_w - 10:
        return []

    desc_mask = raw_symbol_mask[:, cut_x:]
    if cv2.countNonZero(desc_mask) == 0:
        return []

    kernel_w = max(24, min(120, int(legend_w * 0.09)))
    text_kernel = np.ones((3, kernel_w), np.uint8)
    text_probe = cv2.morphologyEx(desc_mask, cv2.MORPH_CLOSE, text_kernel)
    contours, _ = cv2.findContours(text_probe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_description_width = max(24, int(legend_w * 0.06))
    max_description_height = max(10, int(legend_h * 0.16))
    spans: list[tuple[float, float, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_description_width or h > max_description_height:
            continue
        if y <= 1 or y + h >= legend_h - 1:
            continue

        row_top = float(max(0, y - 2))
        row_bottom = float(min(legend_h, y + h + 2))
        spans.append((row_top, row_bottom, (row_top + row_bottom) / 2.0))

    spans.sort(key=lambda item: item[2])
    return spans


def _group_gray_row_spans(
    row_spans: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Group text-line spans into legend rows."""

    if not row_spans:
        return []

    ordered = sorted(row_spans, key=lambda item: item[2])
    grouped: list[list[tuple[float, float, float]]] = []
    for span in ordered:
        if not grouped:
            grouped.append([span])
            continue
        previous = grouped[-1]
        previous_center = sum(item[2] for item in previous) / len(previous)
        previous_height = max(item[1] - item[0] for item in previous)
        if abs(span[2] - previous_center) <= max(26.0, previous_height * 1.8):
            previous.append(span)
        else:
            grouped.append([span])

    rows: list[tuple[float, float, float]] = []
    for group in grouped:
        top = min(item[0] for item in group)
        bottom = max(item[1] for item in group)
        center = sum(item[2] for item in group) / len(group)
        rows.append((top, bottom, center))
    return rows


def _group_visual_gray_row_spans(
    row_spans: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Group raster-detected description fragments without merging adjacent legend rows."""

    if not row_spans:
        return []

    ordered = sorted(row_spans, key=lambda item: item[2])
    grouped: list[list[tuple[float, float, float]]] = []
    for span in ordered:
        if not grouped:
            grouped.append([span])
            continue
        previous = grouped[-1]
        previous_center = sum(item[2] for item in previous) / len(previous)
        previous_height = max(item[1] - item[0] for item in previous)
        current_height = span[1] - span[0]
        max_gap = max(18.0, min(previous_height, current_height) * 1.25)
        if abs(span[2] - previous_center) <= max_gap:
            previous.append(span)
        else:
            grouped.append([span])

    rows: list[tuple[float, float, float]] = []
    for group in grouped:
        top = min(item[0] for item in group)
        bottom = max(item[1] for item in group)
        center = sum(item[2] for item in group) / len(group)
        rows.append((top, bottom, center))
    return rows


def _color_classic_row_symbol_bboxes(
    raw_symbol_mask: np.ndarray,
    text_words: list,
    *,
    x_start: int,
    y_start: int,
    scale: float,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], dict[tuple[int, int, int, int], str]]:
    """
    Segment a color classic legend by text rows.

    Colored legends often have multi-part symbols (for example a letter above
    a square). Contour extraction can split those parts. Row segmentation keeps
    every colored component from the symbol column in one description row.
    """

    if raw_symbol_mask.size == 0 or not text_words:
        return raw_symbol_mask, [], {}

    legend_h, legend_w = raw_symbol_mask.shape[:2]
    pixels = cv2.findNonZero(raw_symbol_mask)
    if pixels is None:
        return raw_symbol_mask, [], {}

    color_x, _color_y, color_w, _color_h = cv2.boundingRect(pixels)
    symbol_column_right = min(
        legend_w,
        color_x + color_w + max(12, int(legend_w * 0.025)),
    )
    if symbol_column_right >= legend_w - 20:
        return raw_symbol_mask, [], {}

    ignored = {"legenda", "legend", "symbol", "opis", "nazwa", "oznaczenia"}
    line_items: dict[tuple[int, int] | int, list[tuple[float, float, float, float, str]]] = {}

    for word in text_words:
        if len(word) < 5:
            continue

        wx0 = float(word[0]) * scale - x_start
        wy0 = float(word[1]) * scale - y_start
        wx1 = float(word[2]) * scale - x_start
        wy1 = float(word[3]) * scale - y_start
        if wx1 <= 0 or wy1 <= 0 or wx0 >= legend_w or wy0 >= legend_h:
            continue

        center_x = (wx0 + wx1) / 2.0
        if center_x < symbol_column_right + 4:
            continue

        text = " ".join(str(word[4] or "").split())
        if sum(1 for char in text if char.isalnum()) < 2:
            continue

        compact = _sanitize_filename(text).casefold().strip("_")
        if compact in ignored:
            continue

        if len(word) >= 7:
            key: tuple[int, int] | int = (int(word[5]), int(word[6]))
        else:
            key = int(round(((wy0 + wy1) / 2.0) * 2))
        line_items.setdefault(key, []).append((wx0, wy0, wx1, wy1, text))

    line_spans: list[tuple[float, float, float, str]] = []
    for items in line_items.values():
        items.sort(key=lambda item: item[0])
        tokens = [text for _x0, _y0, _x1, _y1, text in items]
        while len(tokens) > 1 and _is_row_label_prefix(tokens[0]):
            tokens.pop(0)
        label = " ".join(tokens).strip()
        safe_label = _sanitize_filename(label)
        if len(safe_label) < 2:
            continue
        top = max(0.0, min(item[1] for item in items))
        bottom = min(float(legend_h), max(item[3] for item in items))
        line_spans.append((top, bottom, (top + bottom) / 2.0, safe_label))

    if not line_spans:
        return raw_symbol_mask, [], {}

    line_spans.sort(key=lambda item: item[2])
    grouped: list[list[tuple[float, float, float, str]]] = []

    def should_continue_previous_row(
        previous_group: list[tuple[float, float, float, str]],
        next_span: tuple[float, float, float, str],
    ) -> bool:
        previous_label = previous_group[-1][3].casefold()
        previous_bottom = previous_group[-1][1]
        gap = next_span[0] - previous_bottom
        if gap > 18.0:
            return False
        # Product/specification continuations in these legends are commonly
        # introduced by "np." on the first line. Do not group ordinary adjacent
        # one-line rows such as TM/TAB/TSM or GSW/MSW.
        return bool(re.search(r"(?:^|_)np(?:_|$)", previous_label))

    for span in line_spans:
        if not grouped:
            grouped.append([span])
            continue
        previous = grouped[-1]
        if should_continue_previous_row(previous, span):
            previous.append(span)
        else:
            grouped.append([span])

    rows: list[tuple[float, float, float, str]] = []
    for group in grouped:
        top = min(item[0] for item in group)
        bottom = max(item[1] for item in group)
        center = sum(item[2] for item in group) / len(group)
        label = "_".join(item[3] for item in sorted(group, key=lambda item: item[2]))
        rows.append((top, bottom, center, _sanitize_filename(label)))

    centers = [row[2] for row in rows]
    row_bands: list[tuple[int, int]] = []
    for idx, row in enumerate(rows):
        row_top, row_bottom, _row_center, _label = row
        if idx == 0:
            band_top = max(0.0, row_top - 14.0)
        else:
            band_top = (centers[idx - 1] + centers[idx]) / 2.0

        if idx == len(rows) - 1:
            band_bottom = min(float(legend_h), row_bottom + 14.0)
        else:
            band_bottom = (centers[idx] + centers[idx + 1]) / 2.0

        row_bands.append(
            (
                max(0, int(np.floor(band_top))),
                min(legend_h, int(np.ceil(band_bottom))),
            )
        )

    symbol_region = raw_symbol_mask[:, :symbol_column_right]
    component_contours, _ = cv2.findContours(
        symbol_region,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    assigned_components: list[list[tuple[int, int, int, int]]] = [[] for _row in rows]

    for contour in component_contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw <= 0 or ch <= 0:
            continue
        component_mask = symbol_region[cy : cy + ch, cx : cx + cw]
        if cv2.countNonZero(component_mask) < 6:
            continue

        center_y = cy + ch / 2.0
        row_index = next(
            (
                idx
                for idx, (band_top, band_bottom) in enumerate(row_bands)
                if band_top <= center_y < band_bottom
            ),
            None,
        )
        if row_index is None:
            row_index = min(range(len(rows)), key=lambda idx: abs(centers[idx] - center_y))
            if abs(centers[row_index] - center_y) > max(42.0, ch * 1.4):
                continue

        assigned_components[row_index].append((cx, cy, cw, ch))

    bboxes: list[tuple[int, int, int, int]] = []
    label_by_bbox: dict[tuple[int, int, int, int], str] = {}
    for row, components in zip(rows, assigned_components):
        _row_top, _row_bottom, _row_center, label = row
        if not components:
            continue

        x1 = max(0, min(cx for cx, _cy, _cw, _ch in components) - SYMBOL_PADDING)
        y1_abs = max(0, min(cy for _cx, cy, _cw, _ch in components) - SYMBOL_PADDING)
        x2 = min(
            symbol_column_right,
            max(cx + cw for cx, _cy, cw, _ch in components) + SYMBOL_PADDING,
        )
        y2_abs = min(
            legend_h,
            max(cy + ch for _cx, cy, _cw, ch in components) + SYMBOL_PADDING,
        )
        if x2 <= x1 or y2_abs <= y1_abs:
            continue

        bbox = (x1, y1_abs, x2 - x1, y2_abs - y1_abs)
        if bbox[2] < 8 or bbox[3] < 8:
            continue

        bboxes.append(bbox)
        if label:
            label_by_bbox[bbox] = label

    return raw_symbol_mask, bboxes, label_by_bbox


def _get_classic_row_label_ocr(
    legend_area: np.ndarray,
    *,
    local_bbox: tuple[int, int, int, int],
    symbol_mask: np.ndarray | None = None,
) -> str | None:
    """OCR the raster description line to the right of a classic legend symbol."""

    x, y, w, h = local_bbox
    legend_h, legend_w = legend_area.shape[:2]
    if legend_h <= 0 or legend_w <= 0:
        return None

    label_left = min(legend_w, x + w + max(8, int(w * 0.25)))
    if label_left >= legend_w - 8:
        return None

    row_top = max(0, y - max(8, int(h * 0.65)))
    row_bottom = min(legend_h, y + h + max(8, int(h * 0.65)))

    if symbol_mask is not None and symbol_mask.size:
        cut_x = _detect_gray_description_cut(symbol_mask)
        if cut_x is not None:
            label_left = max(label_left, min(legend_w - 1, cut_x + 2))

            rows = _group_gray_row_spans(_visual_gray_description_row_spans(symbol_mask, cut_x))
            if rows:
                symbol_center_y = y + h / 2.0
                best_row = min(rows, key=lambda row: abs(row[2] - symbol_center_y))
                if abs(best_row[2] - symbol_center_y) <= max(28.0, h * 1.35):
                    row_top = max(0, int(np.floor(best_row[0] - 6)))
                    row_bottom = min(legend_h, int(np.ceil(best_row[1] + 6)))

        desc_mask = symbol_mask[row_top:row_bottom, label_left:legend_w]
        pixels = cv2.findNonZero(desc_mask)
        if pixels is not None:
            dx, dy, dw, dh = cv2.boundingRect(pixels)
            label_left = max(label_left, label_left + dx - 6)
            row_top = max(0, row_top + dy - 10)
            row_bottom = min(legend_h, row_top + dh + 24)

    label_crop = legend_area[row_top:row_bottom, label_left:legend_w]
    if label_crop.size == 0:
        return None

    text = _ocr_text_from_image(label_crop, psm=6)
    if not text:
        return None

    clean_text = _clean_ocr_label_text(text)
    if not clean_text:
        return None

    safe_label = _sanitize_filename(clean_text)
    if len(safe_label) < 2:
        return None
    return safe_label
