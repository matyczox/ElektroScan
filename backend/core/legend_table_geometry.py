"""Table-grid geometry and quality helpers for legend extraction."""

from __future__ import annotations

import re

import cv2
import numpy as np

try:
    from .legend_constants import CELL_BORDER_TRIM
    from .legend_mask_utils import _hsv_mask, _visible_ink_mask
except ImportError:  # pragma: no cover
    from legend_constants import CELL_BORDER_TRIM
    from legend_mask_utils import _hsv_mask, _visible_ink_mask

def _detect_legend_format(legend_area: np.ndarray) -> str:
    """Zwraca 'table' jeśli w obszarze legendy wykryto siatkę tabelaryczną, inaczej 'classic'."""
    h, w = legend_area.shape[:2]
    if h < 40 or w < 40:
        return "classic"
    gray = cv2.cvtColor(legend_area, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 2), 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    row_sums = horiz_lines.sum(axis=1)
    line_threshold = w * 0.5 * 255
    in_line = False
    line_count = 0
    for s in row_sums:
        if s >= line_threshold:
            if not in_line:
                line_count += 1
                in_line = True
        else:
            in_line = False
    return "table" if line_count >= 3 else "classic"


def _merge_close_indices(indices: np.ndarray, gap: int = 5) -> list[int]:
    """Grupuje bliskie indeksy (odległość <= gap) i zwraca medianę każdej grupy."""
    if len(indices) == 0:
        return []
    groups: list[list[int]] = []
    current: list[int] = [int(indices[0])]
    for idx in indices[1:]:
        if int(idx) - current[-1] <= gap:
            current.append(int(idx))
        else:
            groups.append(current)
            current = [int(idx)]
    groups.append(current)
    return [int(np.median(g)) for g in groups]


def _first_table_symbol_column_right(col_boundaries: list[int], legend_width: int) -> int:
    """Pick the first real separator after the symbol column, ignoring outer borders."""

    min_x = max(3, int(legend_width * 0.015))
    max_x = max(min_x + 1, int(legend_width * 0.60))
    candidates = [x for x in col_boundaries if min_x < x < max_x]
    if candidates:
        return candidates[0]
    return max(20, int(legend_width * 0.12))


def _cell_has_content(cell: np.ndarray, min_density: float = 0.004) -> bool:
    """Sprawdza czy komórka tabeli zawiera rysunek (ciemne piksele > min_density lub >= 8 px).

    Niski próg gęstości (0.4%) bo małe, cienkie symbole (kółka, strzałki) w wysokich
    wierszach tabeli mają bardzo małą gęstość względem całej komórki.
    """
    if cell.size == 0:
        return False
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    dark_count = int(np.sum(gray < 210))
    total = cell.shape[0] * cell.shape[1]
    return total > 0 and (dark_count >= 8 or dark_count / total >= min_density)


def _row_has_table_separator(
    vert_lines: np.ndarray,
    row_top_px: int,
    row_bottom_px: int,
    separator_x_px: int,
) -> bool:
    """Return True when the symbol/description separator exists in this row."""

    if vert_lines.size == 0:
        return True

    height, width = vert_lines.shape[:2]
    y1 = max(0, row_top_px + CELL_BORDER_TRIM)
    y2 = min(height, row_bottom_px - CELL_BORDER_TRIM)
    if y2 - y1 < 4:
        return True

    x1 = max(0, separator_x_px - 2)
    x2 = min(width, separator_x_px + 3)
    if x2 <= x1:
        return True

    line_pixels = cv2.countNonZero(vert_lines[y1:y2, x1:x2])
    return line_pixels >= max(3, int((y2 - y1) * 0.3))


def _remove_bottom_neighbor_label_components(cell_mask: np.ndarray) -> np.ndarray:
    """Drop a next-row text label that falls into the bottom edge of a gutter row."""

    if cell_mask.size == 0:
        return cell_mask

    height, _width = cell_mask.shape[:2]
    if height < 32:
        return cell_mask

    contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for contour in contours:
        x, y, width, comp_height = cv2.boundingRect(contour)
        area = int(cv2.contourArea(contour))
        if width * comp_height < 5:
            continue
        components.append((x, y, width, comp_height, area, contour))

    if not components:
        return cell_mask

    bottom_band_top = height - max(12, int(height * 0.14))
    has_main_ink_above = any(
        y < bottom_band_top and area >= 16
        for _x, y, _width, _comp_height, area, _contour in components
    )
    if not has_main_ink_above:
        return cell_mask

    cleaned = cell_mask.copy()
    removed = False
    for x, y, width, comp_height, _area, contour in components:
        if y < bottom_band_top:
            continue
        if width > 70 or comp_height > 34:
            continue
        cv2.drawContours(cleaned, [contour], -1, 0, thickness=-1)
        removed = True

    if not removed:
        return cell_mask

    return cleaned


def _expand_legend_rect_to_table(
    plan_image: np.ndarray,
    rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Expand a partial table selection to the visible grid enclosing it."""

    image_h, image_w = plan_image.shape[:2]
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return None

    pad_x = max(360, int(w * 0.75))
    pad_y = max(90, int(h * 0.08))
    sx = max(0, x - pad_x)
    sy = max(0, y - pad_y)
    ex = min(image_w, x + w + pad_x)
    ey = min(image_h, y + h + pad_y)
    search = plan_image[sy:ey, sx:ex]
    if search.size == 0:
        return None

    search_h, search_w = search.shape[:2]
    if search_h < 40 or search_w < 40:
        return None

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(30, min(search_w, search_w // 2)), 1),
    )
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    row_sums = horiz_lines.sum(axis=1)
    row_threshold = max(120, int(search_w * 0.22)) * 255
    row_boundaries = _merge_close_indices(np.where(row_sums >= row_threshold)[0], gap=5)

    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(30, min(search_h, search_h // 3))),
    )
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    col_sums = vert_lines.sum(axis=0)
    col_threshold = max(80, int(search_h * 0.18)) * 255
    col_boundaries = _merge_close_indices(np.where(col_sums >= col_threshold)[0], gap=5)

    if len(row_boundaries) < 3 or len(col_boundaries) < 2:
        return None

    local_x1 = x - sx
    local_x2 = local_x1 + w
    local_y1 = y - sy
    local_y2 = local_y1 + h
    col_candidates = [col for col in col_boundaries if col <= local_x2 + 40 and col >= local_x1 - pad_x]
    row_candidates = [row for row in row_boundaries if row <= local_y2 + 40 and row >= local_y1 - pad_y]
    if len(row_candidates) < 3 or len(col_candidates) < 2:
        return None

    table_x1 = min(col_candidates)
    table_x2 = max(col_candidates)
    table_y1 = min(row_candidates)
    table_y2 = max(row_candidates)
    if table_x2 - table_x1 < max(40, min(w, 160)):
        return None
    if table_y2 - table_y1 < max(40, min(h, 160)):
        return None

    original_center_x = local_x1 + w / 2.0
    original_center_y = local_y1 + h / 2.0
    if not (table_x1 - 30 <= original_center_x <= table_x2 + 30):
        return None
    if not (table_y1 - 30 <= original_center_y <= table_y2 + 30):
        return None

    pad = 4
    expanded_x1 = max(0, sx + table_x1 - pad)
    expanded_y1 = max(0, sy + table_y1 - pad)
    expanded_x2 = min(image_w, sx + table_x2 + pad)
    expanded_y2 = min(image_h, sy + table_y2 + pad)
    if expanded_x2 <= expanded_x1 or expanded_y2 <= expanded_y1:
        return None

    return (
        int(expanded_x1),
        int(expanded_y1),
        int(expanded_x2 - expanded_x1),
        int(expanded_y2 - expanded_y1),
    )


def _table_symbol_quality(raw_symbols: list[tuple[np.ndarray, str]]) -> float:
    """Score whether table extraction looks like symbol rows, not description text."""

    if not raw_symbols:
        return 0.0

    names = [name for _image, name in raw_symbols]
    code_names = sum(
        1
        for name in names
        if re.fullmatch(r"[A-Z]{1,4}\d{1,4}[A-Z]?", str(name).upper())
    )
    fallback_names = sum(
        1
        for name in names
        if str(name).startswith("sym_") or "LEGENDA" in str(name).upper()
    )
    areas = [image.shape[0] * image.shape[1] for image, _name in raw_symbols if image.size]
    median_area = float(np.median(areas)) if areas else 0.0
    area_bonus = min(8.0, median_area / 1200.0)

    return len(raw_symbols) * 1.5 + code_names * 5.0 + area_bonus - fallback_names * 3.0


def _table_symbols_need_expansion(raw_symbols: list[tuple[np.ndarray, str]]) -> bool:
    """Return True when a table crop looks partial or text-column driven."""

    if len(raw_symbols) < 3:
        return True
    if len(raw_symbols) > 35:
        return True

    names = [str(name) for _image, name in raw_symbols]
    fallback_names = sum(
        1 for name in names if name.startswith("sym_") or "LEGENDA" in name.upper()
    )
    code_names = sum(
        1
        for name in names
        if re.fullmatch(r"[A-Z]{1,4}\d{1,4}[A-Z]?", name.upper())
    )

    if fallback_names > 0 or code_names < max(3, int(len(raw_symbols) * 0.6)):
        if _table_symbol_images_look_valid(raw_symbols) and _table_symbol_images_have_color(raw_symbols):
            return False
        return True

    return False


def _table_symbol_images_have_color(raw_symbols: list[tuple[np.ndarray, str]]) -> bool:
    """Return True when most table crops contain real colored legend ink."""

    if not raw_symbols:
        return False

    colored = 0
    for image, _name in raw_symbols:
        if image.size == 0:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ink_pixels = int(np.sum(gray < 230))
        if ink_pixels < 12:
            continue
        color_pixels = int(cv2.countNonZero(_hsv_mask(image)))
        if color_pixels >= max(8, int(ink_pixels * 0.12)):
            colored += 1

    return colored >= max(3, int(len(raw_symbols) * 0.6))


def _find_left_symbol_gutter_x(
    legend_area: np.ndarray,
    table_trim: tuple[int, int, int, int],
) -> int | None:
    """Find the left edge of symbols drawn outside the table grid."""

    tx, ty, _tw, th = table_trim
    if tx < 16 or th < 40:
        return None

    gutter_area = legend_area[ty : ty + th, :tx]
    if gutter_area.size == 0:
        return None

    mask = _visible_ink_mask(gutter_area)
    if mask.shape[0] > CELL_BORDER_TRIM * 2:
        mask[:CELL_BORDER_TRIM, :] = 0
        mask[-CELL_BORDER_TRIM:, :] = 0
    if mask.shape[1] > CELL_BORDER_TRIM * 2:
        mask[:, :CELL_BORDER_TRIM] = 0
        mask[:, -CELL_BORDER_TRIM:] = 0

    ink_count = int(cv2.countNonZero(mask))
    min_ink = max(24, int(mask.shape[0] * mask.shape[1] * 0.0005))
    if ink_count < min_ink:
        return None

    bands = np.array_split(mask, min(12, max(1, mask.shape[0] // 20)), axis=0)
    active_bands = sum(1 for band in bands if cv2.countNonZero(band) >= 8)
    if active_bands < 3:
        return None

    pixels = cv2.findNonZero(mask)
    if pixels is None:
        return None

    x, _y, width, height = cv2.boundingRect(pixels)
    if width < 6 or height < max(20, int(th * 0.12)):
        return None

    return max(0, x - 6)


def _table_symbol_images_look_valid(raw_symbols: list[tuple[np.ndarray, str]]) -> bool:
    """Accept table rows whose image crops look like compact legend symbols."""

    if len(raw_symbols) < 3 or len(raw_symbols) > 35:
        return False

    valid = 0
    areas: list[int] = []
    for image, _name in raw_symbols:
        if image.size == 0:
            continue
        height, width = image.shape[:2]
        if height < 3 or width < 3:
            continue
        if height > 220 or width > 260:
            continue
        area = height * width
        if area > 32000:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ink_pixels = int(np.sum(gray < 220))
        if ink_pixels < 18:
            continue
        valid += 1
        areas.append(area)

    if valid < max(3, int(len(raw_symbols) * 0.6)):
        return False

    median_area = float(np.median(areas)) if areas else 0.0
    return median_area <= 18000


def _tighten_gray_legend_symbol_crop(symbol_image: np.ndarray) -> np.ndarray:
    """
    Remove empty cell/row padding introduced by gray legend extraction.

    Accepted gray fixtures are learned from tight symbol crops. Later row/table
    extraction paths kept an extra blank row/column from the selected region,
    which changes template geometry enough to create misses and ghosts.
    """

    if symbol_image.size == 0:
        return symbol_image

    result = symbol_image

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    dark = gray < 220
    if result.shape[0] > 4 and int(dark[0, :].sum()) == 0 and int(dark[1, :].sum()) == 0:
        result = result[1:, :]

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    dark = gray < 220
    if (
        result.shape[1] > 4
        and (result.shape[1] < 120 or result.shape[0] <= 60)
        and int(dark[:, 0].sum()) == 0
        and int(dark[:, 1].sum()) == 0
    ):
        result = result[:, 1:]

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    dark = gray < 220
    if (
        result.shape[1] > 60
        and int(dark[:, -1].sum()) == 0
        and int(dark[:, -2].sum()) == 0
        and int(dark[:, -3].sum()) == 0
    ):
        result = result[:, :-2]

    return result


def _trim_selection_to_table_grid(legend_area: np.ndarray) -> tuple[int, int, int, int] | None:
    """Find a table grid inside a user selection that includes surrounding plan margin."""

    h, w = legend_area.shape[:2]
    if h < 40 or w < 40:
        return None

    gray = cv2.cvtColor(legend_area, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, min(w, w // 3)), 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    contours, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_width = max(120, int(w * 0.35))
    line_rects: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, line_w, line_h = cv2.boundingRect(contour)
        if line_w < min_width:
            continue
        if line_h > max(8, int(h * 0.02)):
            continue
        line_rects.append((x, y, line_w, line_h))

    if len(line_rects) < 3:
        return None

    line_rects.sort(key=lambda rect: rect[1])
    x_left = int(np.median([rect[0] for rect in line_rects]))
    x_right = int(np.median([rect[0] + rect[2] for rect in line_rects]))
    y_top = min(rect[1] for rect in line_rects)
    y_bottom = max(rect[1] + rect[3] for rect in line_rects)

    if x_right - x_left < min_width:
        return None
    if y_bottom - y_top < max(80, int(h * 0.35)):
        return None

    pad = 4
    x1 = max(0, x_left - pad)
    y1 = max(0, y_top - pad)
    x2 = min(w, x_right + pad)
    y2 = min(h, y_bottom + pad)

    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _table_grid_region_candidates(legend_area: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return likely table grids inside a broad legend selection."""

    h, w = legend_area.shape[:2]
    if h < 40 or w < 40:
        return []

    candidates: list[tuple[int, int, int, int]] = []

    def add_candidate(rect: tuple[int, int, int, int] | None) -> None:
        if rect is None:
            return
        x, y, rect_w, rect_h = rect
        x1 = max(0, min(w, int(x)))
        y1 = max(0, min(h, int(y)))
        x2 = max(x1, min(w, int(x + rect_w)))
        y2 = max(y1, min(h, int(y + rect_h)))
        if x2 - x1 < 40 or y2 - y1 < 40:
            return
        normalized = (x1, y1, x2 - x1, y2 - y1)
        for existing in list(candidates):
            ex, ey, ew, eh = existing
            overlap_x = max(0, min(x2, ex + ew) - max(x1, ex))
            overlap_y = max(0, min(y2, ey + eh) - max(y1, ey))
            overlap_area = overlap_x * overlap_y
            current_area = (x2 - x1) * (y2 - y1)
            existing_area = ew * eh
            if overlap_area >= min(current_area, existing_area) * 0.82:
                if current_area < existing_area * 0.72:
                    candidates.remove(existing)
                    continue
                if existing_area < current_area * 0.72:
                    return
                return
        candidates.append(normalized)

    add_candidate(_trim_selection_to_table_grid(legend_area))
    if _detect_legend_format(legend_area) == "table":
        add_candidate((0, 0, w, h))

    gray = cv2.cvtColor(legend_area, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel_width = max(30, min(520, max(30, w // 4)))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    contours, _ = cv2.findContours(horiz_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_line_width = max(120, int(w * 0.10))
    endpoint_bucket = max(24, int(round(w * 0.025)))
    grouped_lines: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for contour in contours:
        x, y, line_w, line_h = cv2.boundingRect(contour)
        if line_w < min_line_width:
            continue
        if line_h > max(8, int(h * 0.02)):
            continue
        key = (round(x / endpoint_bucket), round((x + line_w) / endpoint_bucket))
        grouped_lines.setdefault(key, []).append((x, y, line_w, line_h))

    for line_rects in grouped_lines.values():
        if len(line_rects) < 3:
            continue

        line_rects.sort(key=lambda rect: rect[1])
        centers_y = [rect[1] + rect[3] / 2.0 for rect in line_rects]
        gaps = [
            centers_y[index + 1] - centers_y[index]
            for index in range(len(centers_y) - 1)
            if centers_y[index + 1] > centers_y[index]
        ]
        median_gap = float(np.median(gaps)) if gaps else 0.0
        split_gap = max(90.0, min(360.0, median_gap * 1.8 if median_gap else 140.0))

        segments: list[list[tuple[int, int, int, int]]] = []
        current_segment: list[tuple[int, int, int, int]] = []
        previous_center: float | None = None
        for rect, center_y in zip(line_rects, centers_y):
            if (
                previous_center is not None
                and center_y - previous_center > split_gap
                and len(current_segment) >= 3
            ):
                segments.append(current_segment)
                current_segment = []
            current_segment.append(rect)
            previous_center = center_y
        if len(current_segment) >= 3:
            segments.append(current_segment)

        for segment in segments:
            x_left = int(np.median([rect[0] for rect in segment]))
            x_right = int(np.median([rect[0] + rect[2] for rect in segment]))
            y_top = min(rect[1] for rect in segment)
            y_bottom = max(rect[1] + rect[3] for rect in segment)
            if x_right - x_left < min_line_width:
                continue
            if y_bottom - y_top < max(80, int(h * 0.08)):
                continue
            pad = 4
            add_candidate(
                (
                    max(0, x_left - pad),
                    max(0, y_top - pad),
                    min(w, x_right + pad) - max(0, x_left - pad),
                    min(h, y_bottom + pad) - max(0, y_top - pad),
                )
            )

    candidates.sort(key=lambda rect: (rect[1], rect[0], -(rect[2] * rect[3])))
    return candidates
