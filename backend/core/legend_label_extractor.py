"""PDF text and OCR label lookup helpers for legend extraction."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

try:
    from .legend_constants import (
        CELL_BORDER_TRIM,
        TEXT_MAX_DISTANCE_X,
        TEXT_MIN_OVERLAP_X,
        TEXT_TOLERANCE_Y,
    )
    from .legend_text import (
        _clean_ocr_label_text,
        _clean_table_description_ocr_text,
        _sanitize_filename,
        _symbol_text_token,
    )
    from .legend_visual_code import _read_visual_symbol_code
except ImportError:  # pragma: no cover
    from legend_constants import (
        CELL_BORDER_TRIM,
        TEXT_MAX_DISTANCE_X,
        TEXT_MIN_OVERLAP_X,
        TEXT_TOLERANCE_Y,
    )
    from legend_text import (
        _clean_ocr_label_text,
        _clean_table_description_ocr_text,
        _sanitize_filename,
        _symbol_text_token,
    )
    from legend_visual_code import _read_visual_symbol_code

def _get_row_index_text(
    text_blocks: list,
    x_start: int,
    y_start: int,
    scale: float,
    row_top_px: int,
    row_bottom_px: int,
    col_right_px: int,
) -> str | None:
    """Szuka krótkiego kodu indeksu (np. A1, AW2) w drugiej kolumnie wiersza tabeli."""
    row_top_pt = (y_start + row_top_px) / scale
    row_bottom_pt = (y_start + row_bottom_px) / scale
    col2_left_pt = (x_start + col_right_px) / scale
    col2_right_pt = col2_left_pt + 120
    for block in text_blocks:
        if len(block) < 5:
            continue
        if len(block) >= 7 and block[6] != 0:
            continue
        bx0 = float(block[0])
        by0 = float(block[1])
        bx1 = float(block[2])
        by1 = float(block[3])
        text = str(block[4]).strip().replace("\n", " ")
        block_center_y = (by0 + by1) / 2.0
        if not (row_top_pt - 2 <= block_center_y <= row_bottom_pt + 2):
            continue
        if not (bx0 >= col2_left_pt - 10 and bx1 <= col2_right_pt):
            continue
        if 1 <= len(text) <= 10 and not any(c in text for c in ["/", "(", "+", "="]):
            return _sanitize_filename(text)
    return None


def _is_row_label_prefix(text: str) -> bool:
    """Return True for leading row counters/codes that should not become symbol labels."""

    compact = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
    if not compact:
        return True
    if compact.isdigit():
        return True
    if re.fullmatch(r"[A-Za-z]{0,4}\d{1,4}[A-Za-z]?", compact):
        return True
    return compact.casefold() in {"lp", "l.p", "nr", "no", "poz", "pos"}


def _is_generic_table_symbol_token(token: str) -> bool:
    """Return True for quantities and electrical parameters, not legend symbol codes."""

    compact = re.sub(r"[^A-Z0-9]+", "", str(token or "").upper())
    if not compact:
        return True
    if re.fullmatch(r"\d+X", compact):
        return True
    if re.fullmatch(r"\d+(?:V|A|M|MM|CM|MB|GB)", compact):
        return True
    if re.fullmatch(r"IP\d{0,2}", compact):
        return True
    if compact in {
        "DATA",
        "HDMI",
        "HTTP",
        "KAT",
        "KAT5",
        "KAT5E",
        "KAT6",
        "KAT6A",
        "LAN",
        "POE",
        "RJ11",
        "RJ12",
        "RJ45",
        "USB",
    }:
        return True
    return False


def _table_symbol_code_token(text: str) -> str | None:
    """Return a compact legend code from a symbol cell while skipping generic specs."""

    raw = str(text or "").strip()
    direct = _symbol_text_token(raw)
    if direct and not _is_generic_table_symbol_token(direct):
        return direct

    token = _sanitize_filename(raw).upper()
    token = re.sub(r"[-_/]+", "", token)
    token = re.sub(r"[^A-Z0-9]+", "", token)
    if not (2 <= len(token) <= 12):
        return None
    if not re.search(r"[A-Z]", token):
        return None
    if not (re.search(r"\d", token) or len(token) <= 6):
        return None
    if _is_generic_table_symbol_token(token):
        return None
    return token


def _get_row_label_text(
    text_items: list,
    x_start: int,
    y_start: int,
    scale: float,
    row_top_px: int,
    row_bottom_px: int,
    col_right_px: int,
    legend_width_px: int,
) -> str | None:
    """Find the human-readable label/description text in the same table row."""

    row_top_pt = (y_start + row_top_px) / scale
    row_bottom_pt = (y_start + row_bottom_px) / scale
    label_left_pt = (x_start + col_right_px) / scale
    label_right_pt = (x_start + legend_width_px) / scale
    ignored = {"lp", "l.p", "l.p.", "symbol", "opis", "nazwa", "oznaczenie"}
    candidates: list[tuple[float, str]] = []

    for item in text_items:
        if len(item) < 5:
            continue
        if len(item) == 7 and item[6] != 0:
            continue

        bx0 = float(item[0])
        by0 = float(item[1])
        bx1 = float(item[2])
        by1 = float(item[3])
        if bx1 < label_left_pt - 8 or bx0 > label_right_pt + 8:
            continue

        overlap_y = min(by1, row_bottom_pt + 2) - max(by0, row_top_pt - 2)
        if overlap_y <= 0:
            continue

        text = " ".join(str(item[4] or "").split())
        if sum(1 for char in text if char.isalnum()) < 2:
            continue
        if text.strip().casefold().strip(":") in ignored:
            continue

        candidates.append((bx0, text))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    tokens = [text for _x, text in candidates]
    while len(tokens) > 1 and _is_row_label_prefix(tokens[0]):
        tokens.pop(0)
    label = " ".join(tokens)
    safe_label = _sanitize_filename(label)
    return safe_label if len(safe_label) >= 2 else None


def _get_table_description_label_text(
    text_items: list,
    x_start: int,
    y_start: int,
    scale: float,
    row_top_px: int,
    row_bottom_px: int,
    col_boundaries: list[int],
    legend_width_px: int,
) -> str | None:
    """Read the final description/product column for the same table row."""

    if not text_items:
        return None

    boundaries = sorted({int(col) for col in col_boundaries if 0 <= int(col) <= legend_width_px})
    if len(boundaries) < 3:
        return _get_row_label_text(
            text_items,
            x_start,
            y_start,
            scale,
            row_top_px,
            row_bottom_px,
            boundaries[0] if boundaries else max(20, int(legend_width_px * 0.12)),
            legend_width_px,
        )

    description_left_px = boundaries[-2] + CELL_BORDER_TRIM
    description_right_px = boundaries[-1] - CELL_BORDER_TRIM
    if description_right_px <= description_left_px:
        description_right_px = legend_width_px

    row_top_pt = (y_start + row_top_px) / scale
    row_bottom_pt = (y_start + row_bottom_px) / scale
    description_left_pt = (x_start + description_left_px) / scale
    description_right_pt = (x_start + description_right_px) / scale
    ignored = {
        "legenda",
        "legend",
        "symbol",
        "opis",
        "nazwa",
        "nazwa_artykulu",
        "nazwaartykułu",
        "indeks",
        "producent",
    }

    candidates: list[tuple[float, float, str]] = []
    for item in text_items:
        if len(item) < 5:
            continue
        if len(item) == 7 and item[6] != 0:
            continue

        bx0 = float(item[0])
        by0 = float(item[1])
        bx1 = float(item[2])
        by1 = float(item[3])
        if bx1 < description_left_pt - 4 or bx0 > description_right_pt + 8:
            continue

        overlap_y = min(by1, row_bottom_pt + 2) - max(by0, row_top_pt - 2)
        if overlap_y <= 0:
            continue

        text = " ".join(str(item[4] or "").split())
        if sum(1 for char in text if char.isalnum()) < 2:
            continue

        compact = _sanitize_filename(text).casefold().strip("_")
        if compact in ignored:
            continue

        line_key = float(item[6]) if len(item) >= 7 else (by0 + by1) / 2.0
        candidates.append((line_key, bx0, text))

    if not candidates:
        return _get_row_label_text(
            text_items,
            x_start,
            y_start,
            scale,
            row_top_px,
            row_bottom_px,
            boundaries[-2],
            legend_width_px,
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    label = " ".join(text for _line, _x, text in candidates)
    safe_label = _sanitize_filename(label)
    return safe_label if len(safe_label) >= 2 else None


def _get_symbol_text_inside_region(
    text_words: list,
    *,
    x_start: int,
    y_start: int,
    scale: float,
    local_bbox: tuple[int, int, int, int],
) -> str | None:
    """Find a short PDF text token inside the cropped legend symbol bbox."""

    x, y, w, h = local_bbox
    pad = max(4, int(round(max(w, h) * 0.12)))
    left = (x_start + x - pad) / scale
    top = (y_start + y - pad) / scale
    right = (x_start + x + w + pad) / scale
    bottom = (y_start + y + h + pad) / scale

    candidates: list[tuple[float, str]] = []
    for word in text_words:
        if len(word) < 5:
            continue

        token = _symbol_text_token(str(word[4]))
        if token is None:
            continue

        wx0 = float(word[0])
        wy0 = float(word[1])
        wx1 = float(word[2])
        wy1 = float(word[3])
        center_x = (wx0 + wx1) / 2.0
        center_y = (wy0 + wy1) / 2.0
        if not (left <= center_x <= right and top <= center_y <= bottom):
            continue

        symbol_center_x = (left + right) / 2.0
        symbol_center_y = (top + bottom) / 2.0
        distance = abs(center_x - symbol_center_x) + abs(center_y - symbol_center_y)
        candidates.append((distance, token))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _get_classic_row_label_text(
    text_words: list,
    *,
    x_start: int,
    y_start: int,
    scale: float,
    local_bbox: tuple[int, int, int, int],
    legend_width_px: int,
) -> str | None:
    """Find a human-readable description to the right of a classic legend symbol."""

    if not text_words:
        return None

    x, y, w, h = local_bbox
    row_center_pt = (y_start + y + h / 2.0) / scale
    symbol_right_pt = (x_start + x + w) / scale
    legend_right_pt = (x_start + legend_width_px) / scale
    tolerance_y = max(TEXT_TOLERANCE_Y, min(30.0, h / scale + 8.0))
    ignored = {
        "legenda",
        "oznaczenia",
        "oznaczenie",
        "symbol",
        "opis",
        "nazwa",
        "indeks",
        "producent",
    }
    lines: dict[tuple[int, int] | int, list[tuple[float, float, str]]] = {}

    for word in text_words:
        if len(word) < 5:
            continue

        wx0 = float(word[0])
        wy0 = float(word[1])
        wx1 = float(word[2])
        wy1 = float(word[3])
        center_x = (wx0 + wx1) / 2.0
        center_y = (wy0 + wy1) / 2.0
        if center_x < symbol_right_pt + TEXT_MIN_OVERLAP_X:
            continue
        if center_x > min(legend_right_pt + 8.0, symbol_right_pt + TEXT_MAX_DISTANCE_X):
            continue
        if abs(center_y - row_center_pt) > tolerance_y:
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
            key = int(round(center_y * 2))
        lines.setdefault(key, []).append((wx0, center_y, text))

    if not lines:
        return None

    rendered_lines: list[tuple[float, str]] = []
    for items in lines.values():
        items.sort(key=lambda item: item[0])
        tokens = [text for _x, _y, text in items]
        while len(tokens) > 1 and _is_row_label_prefix(tokens[0]):
            tokens.pop(0)
        line_text = " ".join(tokens).strip()
        if line_text:
            rendered_lines.append(
                (sum(item[1] for item in items) / len(items), line_text)
            )

    if not rendered_lines:
        return None

    rendered_lines.sort(key=lambda item: item[0])
    label = " ".join(line for _center_y, line in rendered_lines)
    safe_label = _sanitize_filename(label)
    return safe_label if len(safe_label) >= 2 else None


def _ocr_text_from_image(image_bgr: np.ndarray, *, psm: int = 6) -> str | None:
    """Read raster legend text with optional Tesseract, if available."""

    if image_bgr.size == 0 or shutil.which("tesseract") is None:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale = 2
    if min(gray.shape[:2]) < 70:
        scale = 3
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dark_ratio = float(np.mean(binary == 0))
    if dark_ratio > 0.55:
        binary = cv2.bitwise_not(binary)

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
        if not cv2.imwrite(tmp_file.name, binary):
            return None

        cmd = [
            "tesseract",
            tmp_file.name,
            "stdout",
            "--psm",
            str(psm),
            "-l",
            "pol+eng",
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None

    text = " ".join(completed.stdout.split())
    if completed.returncode != 0 or sum(1 for char in text if char.isalnum()) < 2:
        return None
    return text


def _get_table_description_label_ocr(
    legend_area: np.ndarray,
    row_top_px: int,
    row_bottom_px: int,
    col_boundaries: list[int],
) -> str | None:
    """OCR the final table description cell in one legend row."""

    if legend_area.size == 0:
        return None

    h, w = legend_area.shape[:2]
    boundaries = sorted({int(col) for col in col_boundaries if 0 <= int(col) <= w})
    if len(boundaries) < 3:
        return None

    x1 = min(w, max(0, boundaries[-2] + CELL_BORDER_TRIM + 2))
    x2 = min(w, max(x1, boundaries[-1] - CELL_BORDER_TRIM - 2))
    y1 = min(h, max(0, row_top_px + CELL_BORDER_TRIM + 1))
    y2 = min(h, max(y1, row_bottom_px - CELL_BORDER_TRIM - 1))
    if x2 - x1 < 12 or y2 - y1 < 6:
        return None

    crop = legend_area[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    text = _ocr_text_from_image(crop, psm=6)
    if not text:
        return None

    clean_text = _clean_table_description_ocr_text(text)
    if not clean_text:
        return None

    return clean_text if sum(1 for char in clean_text if char.isalnum()) >= 2 else None


def _get_row_symbol_code_text(
    text_words: list | None,
    x_start: int,
    y_start: int,
    scale: float,
    row_top_px: int,
    row_bottom_px: int,
    col_right_px: int,
) -> str | None:
    """Find a short symbol code printed in the table symbol column."""

    if not text_words:
        return None

    row_top_pt = (y_start + row_top_px) / scale
    row_bottom_pt = (y_start + row_bottom_px) / scale
    row_center_pt = (row_top_pt + row_bottom_pt) / 2.0
    col_left_pt = x_start / scale
    col_right_pt = (x_start + col_right_px) / scale
    candidates: list[tuple[float, float, str]] = []

    for word in text_words:
        if len(word) < 5:
            continue

        token = _table_symbol_code_token(str(word[4]))
        if token is None:
            continue

        wx0 = float(word[0])
        wy0 = float(word[1])
        wx1 = float(word[2])
        wy1 = float(word[3])
        center_x = (wx0 + wx1) / 2.0
        center_y = (wy0 + wy1) / 2.0
        if not (col_left_pt - 8 <= center_x <= col_right_pt + 12):
            continue

        overlap_y = min(wy1, row_bottom_pt + 2) - max(wy0, row_top_pt - 2)
        if overlap_y <= 0:
            continue

        candidates.append((abs(center_y - row_center_pt), center_x, token))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _get_row_symbol_code_boxes_px(
    text_words: list | None,
    x_start: int,
    y_start: int,
    scale: float,
    row_top_px: int,
    row_bottom_px: int,
    col_right_px: int,
) -> list[tuple[int, int, int, int, str]]:
    """Return symbol-column code word boxes in legend-local pixel coordinates."""

    if not text_words:
        return []

    row_top_pt = (y_start + row_top_px) / scale
    row_bottom_pt = (y_start + row_bottom_px) / scale
    col_left_pt = x_start / scale
    col_right_pt = (x_start + col_right_px) / scale
    boxes: list[tuple[int, int, int, int, str]] = []

    for word in text_words:
        if len(word) < 5:
            continue

        token = _table_symbol_code_token(str(word[4]))
        if token is None:
            continue

        wx0 = float(word[0])
        wy0 = float(word[1])
        wx1 = float(word[2])
        wy1 = float(word[3])
        center_x = (wx0 + wx1) / 2.0
        if not (col_left_pt - 8 <= center_x <= col_right_pt + 12):
            continue

        overlap_y = min(wy1, row_bottom_pt + 2) - max(wy0, row_top_pt - 2)
        if overlap_y <= 0:
            continue

        x0 = int(round(wx0 * scale - x_start))
        y0 = int(round(wy0 * scale - y_start))
        x1 = int(round(wx1 * scale - x_start))
        y1 = int(round(wy1 * scale - y_start))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append((x0, y0, x1, y1, token))

    return boxes


def _get_visual_row_index_text(
    legend_area: np.ndarray,
    row_top_px: int,
    row_bottom_px: int,
    col_boundaries: list[int],
    first_col_right_px: int,
) -> str | None:
    """Read the visual code from the table index column right of an external symbol gutter."""

    next_columns = [col for col in col_boundaries if col > first_col_right_px + 8]
    if not next_columns:
        return None

    _height, width = legend_area.shape[:2]
    row_inner_top = min(row_bottom_px, row_top_px + CELL_BORDER_TRIM + 1)
    row_inner_bottom = max(row_inner_top, row_bottom_px - CELL_BORDER_TRIM - 1)
    col_left = min(width, first_col_right_px + CELL_BORDER_TRIM + 2)
    col_right = max(col_left, min(width, next_columns[0] - CELL_BORDER_TRIM - 2))
    if row_inner_bottom <= row_inner_top or col_right <= col_left:
        return None

    cell = legend_area[row_inner_top:row_inner_bottom, col_left:col_right]
    return _read_visual_symbol_code(cell)
