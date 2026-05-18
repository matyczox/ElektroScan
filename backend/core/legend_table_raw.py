"""Raw table legend symbol crop extraction."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .legend_constants import CELL_BORDER_TRIM
    from .legend_label_extractor import (
        _get_row_index_text,
        _get_row_label_text,
        _get_row_symbol_code_boxes_px,
        _get_row_symbol_code_text,
        _get_visual_row_index_text,
    )
    from .legend_mask_utils import _visible_ink_mask
    from .legend_table_geometry import (
        _find_left_symbol_gutter_x,
        _first_table_symbol_column_right,
        _merge_close_indices,
        _remove_bottom_neighbor_label_components,
        _row_has_table_separator,
        _table_symbol_images_look_valid,
        _tighten_gray_legend_symbol_crop,
    )
except ImportError:  # pragma: no cover
    from legend_constants import CELL_BORDER_TRIM
    from legend_label_extractor import (
        _get_row_index_text,
        _get_row_label_text,
        _get_row_symbol_code_boxes_px,
        _get_row_symbol_code_text,
        _get_visual_row_index_text,
    )
    from legend_mask_utils import _visible_ink_mask
    from legend_table_geometry import (
        _find_left_symbol_gutter_x,
        _first_table_symbol_column_right,
        _merge_close_indices,
        _remove_bottom_neighbor_label_components,
        _row_has_table_separator,
        _table_symbol_images_look_valid,
        _tighten_gray_legend_symbol_crop,
    )


def _table_symbol_mask(image_bgr: np.ndarray, *, use_visible_symbol_mask: bool) -> np.ndarray:
    """Return symbol ink for a table cell, including bright colored CAD strokes."""

    if use_visible_symbol_mask:
        return _visible_ink_mask(image_bgr, gray_threshold=235)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return (gray < 200).astype(np.uint8) * 255


def _erase_symbol_cell_code_text(
    mask: np.ndarray,
    code_boxes_px: list[tuple[int, int, int, int, str]],
    *,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    """Remove printed legend codes from color symbol masks without using class rules."""

    if not code_boxes_px or mask.size == 0:
        return mask

    stripped = mask.copy()
    original_pixels = int(cv2.countNonZero(mask))
    for x0, y0, x1, y1, token in code_boxes_px:
        # A single numeric marker can be a real part of an electrical symbol.
        # Alphanumeric row codes such as G5-20, PEL2 or FB1 are labels/evidence,
        # not the visual geometry to be matched later on the plan.
        if not any(char.isalpha() for char in token):
            continue
        lx0 = max(0, x0 - offset_x - 2)
        ly0 = max(0, y0 - offset_y - 2)
        lx1 = min(mask.shape[1], x1 - offset_x + 2)
        ly1 = min(mask.shape[0], y1 - offset_y + 2)
        if lx1 > lx0 and ly1 > ly0:
            stripped[ly0:ly1, lx0:lx1] = 0

    stripped_pixels = int(cv2.countNonZero(stripped))
    if stripped_pixels < max(20, int(original_pixels * 0.15)):
        return mask
    return stripped


def _extract_table_legend_raw(
    legend_area: np.ndarray,
    text_blocks: list,
    text_words: list | None,
    x_start: int,
    y_start: int,
    scale: float,
    allow_description_labels: bool = True,
    allow_visual_index_labels: bool = False,
    tighten_gray_table_crops: bool = False,
    use_visible_symbol_mask: bool = True,
) -> list[tuple[np.ndarray, str]]:
    """Wyciąga symbole z legendy w formacie tabelarycznym (siatka z wierszami)."""
    h, w = legend_area.shape[:2]
    gray = cv2.cvtColor(legend_area, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 2), 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    row_sums = horiz_lines.sum(axis=1)
    line_threshold = w * 0.4 * 255
    line_row_indices = np.where(row_sums >= line_threshold)[0]
    row_boundaries = _merge_close_indices(line_row_indices, gap=5)

    if len(row_boundaries) < 2:
        return []

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 3)))
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    col_sums = vert_lines.sum(axis=0)
    col_threshold = h * 0.3 * 255
    col_indices = np.where(col_sums >= col_threshold)[0]
    col_boundaries = _merge_close_indices(col_indices, gap=5)

    first_col_right = _first_table_symbol_column_right(col_boundaries, w)
    has_detected_separator = any(abs(col - first_col_right) <= 3 for col in col_boundaries)

    row_spans = [
        (row_boundaries[i], row_boundaries[i + 1])
        for i in range(len(row_boundaries) - 1)
        if row_boundaries[i + 1] - row_boundaries[i] >= 8
    ]
    if not row_spans:
        return []

    cell_right = max(5, first_col_right - 2)

    results: list[tuple[np.ndarray, str]] = []
    counter = 1

    for row_top, row_bottom in row_spans:
        if has_detected_separator and not _row_has_table_separator(
            vert_lines, row_top, row_bottom, first_col_right
        ):
            continue

        row_inner_top = min(row_bottom, row_top + CELL_BORDER_TRIM + 1)
        row_inner_bottom = max(row_inner_top, row_bottom - CELL_BORDER_TRIM - 1)
        cell_left = CELL_BORDER_TRIM
        cell_right_inner = max(cell_left + 1, cell_right)
        cell = legend_area[row_inner_top:row_inner_bottom, cell_left:cell_right_inner]
        if cell.size == 0:
            continue

        code_boxes_px = (
            _get_row_symbol_code_boxes_px(
                text_words,
                x_start,
                y_start,
                scale,
                row_top,
                row_bottom,
                first_col_right,
            )
            if use_visible_symbol_mask
            else []
        )
        cell_mask = _table_symbol_mask(
            cell,
            use_visible_symbol_mask=use_visible_symbol_mask,
        )
        grid_mask = cv2.bitwise_or(
            horiz_lines[row_inner_top:row_inner_bottom, cell_left:cell_right_inner],
            vert_lines[row_inner_top:row_inner_bottom, cell_left:cell_right_inner],
        )
        if grid_mask.size:
            grid_mask = cv2.dilate(grid_mask, np.ones((3, 3), np.uint8), iterations=1)
            cell_mask[grid_mask > 0] = 0
        if cell_mask.shape[0] > CELL_BORDER_TRIM * 2:
            cell_mask[:CELL_BORDER_TRIM, :] = 0
            cell_mask[-CELL_BORDER_TRIM:, :] = 0
        if cell_mask.shape[1] > CELL_BORDER_TRIM * 2:
            cell_mask[:, :CELL_BORDER_TRIM] = 0
            cell_mask[:, -CELL_BORDER_TRIM:] = 0
        if allow_visual_index_labels and not allow_description_labels:
            cell_mask = _remove_bottom_neighbor_label_components(cell_mask)
        if use_visible_symbol_mask:
            cell_mask = _erase_symbol_cell_code_text(
                cell_mask,
                code_boxes_px,
                offset_x=cell_left,
                offset_y=row_inner_top,
            )

        pixels = cv2.findNonZero(cell_mask)
        if pixels is None:
            continue

        rx, ry, rw, rh = cv2.boundingRect(pixels)
        if rw < 2 or rh < 2:
            continue

        pad = 4
        top_pad = 10 if allow_visual_index_labels and not allow_description_labels else pad
        sx1 = max(0, cell_left + rx - pad)
        sy1 = max(0, row_inner_top + ry - top_pad)
        sx2 = min(cell_right_inner, cell_left + rx + rw + pad)
        sy2 = min(h, row_inner_top + ry + rh + pad)
        if sx2 - sx1 < 3 or sy2 - sy1 < 3:
            continue

        symbol_crop = legend_area[sy1:sy2, sx1:sx2]
        symbol_image = np.full_like(symbol_crop, 255)
        symbol_mask = _table_symbol_mask(
            symbol_crop,
            use_visible_symbol_mask=use_visible_symbol_mask,
        )
        symbol_grid_mask = cv2.bitwise_or(
            horiz_lines[sy1:sy2, sx1:sx2],
            vert_lines[sy1:sy2, sx1:sx2],
        )
        if symbol_grid_mask.size:
            symbol_grid_mask = cv2.dilate(
                symbol_grid_mask,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
            symbol_mask[symbol_grid_mask > 0] = 0
        if use_visible_symbol_mask:
            symbol_mask = _erase_symbol_cell_code_text(
                symbol_mask,
                code_boxes_px,
                offset_x=sx1,
                offset_y=sy1,
            )
        symbol_image[symbol_mask > 0] = symbol_crop[symbol_mask > 0]
        if tighten_gray_table_crops:
            symbol_image = _tighten_gray_legend_symbol_crop(symbol_image)

        label_source = text_words if text_words else text_blocks
        name = _get_row_symbol_code_text(
            text_words,
            x_start,
            y_start,
            scale,
            row_top,
            row_bottom,
            first_col_right,
        )
        if allow_visual_index_labels and not name:
            name = _get_visual_row_index_text(
                legend_area,
                row_top,
                row_bottom,
                col_boundaries,
                first_col_right,
            )
        if allow_description_labels and not name:
            name = _get_row_label_text(
                label_source,
                x_start,
                y_start,
                scale,
                row_top,
                row_bottom,
                first_col_right,
                w,
            )
        if allow_description_labels and not name:
            name = _get_row_index_text(
                text_blocks,
                x_start,
                y_start,
                scale,
                row_top,
                row_bottom,
                first_col_right,
            )
        if not name:
            name = f"sym_{counter:02d}"

        results.append((symbol_image, name))
        counter += 1

    return results


def _extract_gray_table_legend_raw(
    legend_area: np.ndarray,
    text_blocks: list,
    x_start: int,
    y_start: int,
    scale: float,
) -> list[tuple[np.ndarray, str]]:
    """Extract gray table legend symbols using the accepted legacy geometry."""

    h, w = legend_area.shape[:2]
    gray = cv2.cvtColor(legend_area, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 2), 1))
    horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    row_sums = horiz_lines.sum(axis=1)
    line_threshold = w * 0.4 * 255
    line_row_indices = np.where(row_sums >= line_threshold)[0]
    row_boundaries = _merge_close_indices(line_row_indices, gap=5)

    if len(row_boundaries) < 2:
        return []

    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 3)))
    vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    col_sums = vert_lines.sum(axis=0)
    col_threshold = h * 0.3 * 255
    col_indices = np.where(col_sums >= col_threshold)[0]
    col_boundaries = _merge_close_indices(col_indices, gap=5)

    first_col_right = col_boundaries[0] if col_boundaries else max(20, int(w * 0.12))

    row_spans = [
        (row_boundaries[i], row_boundaries[i + 1])
        for i in range(len(row_boundaries) - 1)
        if row_boundaries[i + 1] - row_boundaries[i] >= 8
    ]
    if not row_spans:
        return []

    row_centers = [(top + bottom) / 2.0 for top, bottom in row_spans]
    assigned_components: list[list[tuple[int, int, int, int]]] = [[] for _ in row_spans]
    cell_right = max(5, first_col_right - 2)

    symbol_strip = legend_area[:, 0:cell_right]
    strip_gray = cv2.cvtColor(symbol_strip, cv2.COLOR_BGR2GRAY)
    strip_mask = (strip_gray < 200).astype(np.uint8) * 255
    if symbol_strip.shape[1] > CELL_BORDER_TRIM * 4:
        strip_mask[:, :CELL_BORDER_TRIM] = 0
        strip_mask[:, -CELL_BORDER_TRIM:] = 0

    component_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(strip_mask, 8)
    for component_index in range(1, component_count):
        cx = int(stats[component_index, cv2.CC_STAT_LEFT])
        cy = int(stats[component_index, cv2.CC_STAT_TOP])
        cw = int(stats[component_index, cv2.CC_STAT_WIDTH])
        ch = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < 4 or cw < 2 or ch < 2:
            continue

        component_center_y = float(centroids[component_index][1])
        nearest_row = min(
            range(len(row_centers)),
            key=lambda index: abs(component_center_y - row_centers[index]),
        )
        row_top, row_bottom = row_spans[nearest_row]
        row_height = row_bottom - row_top
        if not (
            row_top - row_height * 0.45 <= component_center_y <= row_bottom + row_height * 0.45
        ):
            continue

        assigned_components[nearest_row].append((cx, cy, cw, ch))

    results: list[tuple[np.ndarray, str]] = []
    counter = 1

    for (row_top, row_bottom), components in zip(row_spans, assigned_components):
        if not components:
            continue

        x1 = min(cx for cx, _cy, _cw, _ch in components)
        y1 = min(cy for _cx, cy, _cw, _ch in components)
        x2 = max(cx + cw for cx, _cy, cw, _ch in components)
        y2 = max(cy + ch for _cx, cy, _cw, ch in components)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        pad = 4
        sx1 = max(0, x1 - pad)
        sy1 = max(0, y1 - pad)
        sx2 = min(symbol_strip.shape[1], x2 + pad)
        sy2 = min(symbol_strip.shape[0], y2 + pad)

        symbol_crop = symbol_strip[sy1:sy2, sx1:sx2]
        if symbol_crop.size == 0:
            continue

        symbol_image = np.full_like(symbol_crop, 255)
        crop_gray = cv2.cvtColor(symbol_crop, cv2.COLOR_BGR2GRAY)
        dark_px = crop_gray < 200
        symbol_image[dark_px] = symbol_crop[dark_px]

        name = _get_row_index_text(
            text_blocks,
            x_start,
            y_start,
            scale,
            row_top,
            row_bottom,
            first_col_right,
        )
        if not name:
            name = f"sym_{counter:02d}"

        results.append((symbol_image, name))
        counter += 1

    return results


def _extract_left_gutter_table_legend_raw(
    legend_area: np.ndarray,
    table_trim: tuple[int, int, int, int],
    text_blocks: list,
    text_words: list | None,
    x_start: int,
    y_start: int,
    scale: float,
    tighten_gray_table_crops: bool = False,
    use_visible_symbol_mask: bool = True,
) -> tuple[list[tuple[np.ndarray, str]], tuple[int, int, int, int]] | None:
    """Extract table legend symbols when the symbol column sits left of the grid."""

    tx, ty, tw, th = table_trim
    gutter_x = _find_left_symbol_gutter_x(legend_area, table_trim)
    if gutter_x is None or gutter_x >= tx:
        return None

    crop_x1 = gutter_x
    crop_y1 = ty
    crop_x2 = tx + tw
    crop_y2 = ty + th
    if crop_x2 - crop_x1 < 40 or crop_y2 - crop_y1 < 40:
        return None

    crop = legend_area[crop_y1:crop_y2, crop_x1:crop_x2]
    raw_symbols = _extract_table_legend_raw(
        crop,
        text_blocks,
        text_words,
        x_start + crop_x1,
        y_start + crop_y1,
        scale,
        allow_description_labels=False,
        allow_visual_index_labels=True,
        tighten_gray_table_crops=tighten_gray_table_crops,
        use_visible_symbol_mask=use_visible_symbol_mask,
    )
    if not _table_symbol_images_look_valid(raw_symbols):
        return None

    return raw_symbols, (crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)
