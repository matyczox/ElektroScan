"""Display-only vector draft builders for table legends."""

from __future__ import annotations

import re

import cv2
import fitz
import numpy as np

try:
    from .legend_label_extractor import (
        _get_row_index_text,
        _get_row_label_text,
        _get_table_description_label_ocr,
        _get_table_description_label_text,
    )
    from .legend_mask_utils import _hsv_mask, _legend_symbol_mask
    from .legend_scene_transform import rect_px300_to_pt
    from .legend_table_geometry import (
        _detect_legend_format,
        _merge_close_indices,
        _table_grid_region_candidates,
        _trim_selection_to_table_grid,
    )
    from .legend_text import _sanitize_filename
    from .legend_vector_drafts import VectorLegendDraft
except ImportError:  # pragma: no cover
    from legend_label_extractor import (
        _get_row_index_text,
        _get_row_label_text,
        _get_table_description_label_ocr,
        _get_table_description_label_text,
    )
    from legend_mask_utils import _hsv_mask, _legend_symbol_mask
    from legend_scene_transform import rect_px300_to_pt
    from legend_table_geometry import (
        _detect_legend_format,
        _merge_close_indices,
        _table_grid_region_candidates,
        _trim_selection_to_table_grid,
    )
    from legend_text import _sanitize_filename
    from legend_vector_drafts import VectorLegendDraft

def _build_table_grid_description_drafts(
    text_items: list,
    legend_area: np.ndarray,
    x_start: int,
    y_start: int,
    scale: float,
    transform,
    *,
    draft_prefix: str,
) -> list[VectorLegendDraft]:
    """Build display labels directly from table rows and the final description column."""

    h, w = legend_area.shape[:2]
    if h < 40 or w < 40:
        return []

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
    if col_boundaries and col_boundaries[0] > max(4, int(w * 0.02)):
        col_boundaries.insert(0, 0)
    if col_boundaries and col_boundaries[-1] < w - max(8, int(w * 0.04)):
        col_boundaries.append(w)
    if len(col_boundaries) < 3:
        return []

    row_spans = [
        (row_boundaries[i], row_boundaries[i + 1])
        for i in range(len(row_boundaries) - 1)
        if row_boundaries[i + 1] - row_boundaries[i] >= 8
    ]
    if not row_spans:
        return []

    drafts: list[VectorLegendDraft] = []
    for row_top, row_bottom in row_spans:
        pdf_label = _get_table_description_label_text(
            text_items,
            x_start,
            y_start,
            scale,
            row_top,
            row_bottom,
            col_boundaries,
            w,
        )
        ocr_label = _get_table_description_label_ocr(
            legend_area,
            row_top,
            row_bottom,
            col_boundaries,
        )
        label = ocr_label or pdf_label
        if not label:
            continue

        row_bbox_px = (x_start, y_start + row_top, w, max(1, row_bottom - row_top))
        draft_index = len(drafts) + 1
        drafts.append(
            VectorLegendDraft(
                draft_id=f"{draft_prefix}:{draft_index}",
                bbox_pt=rect_px300_to_pt(row_bbox_px, transform),
                bbox_px_300=row_bbox_px,
                row_bbox_pt=rect_px300_to_pt(row_bbox_px, transform),
                name_draft=label,
                symbol_code=None,
                confidence=0.88,
                primitive_refs=[],
                review_required=True,
                label_source="right_text",
                structure_source="table_cell",
                fallback_eligible=True,
                image_bgr=None,
            )
        )

    return drafts


def _is_descriptive_table_label(label: str | None) -> bool:
    """Reject row/index codes when looking for user-facing legend names."""

    text = " ".join(str(label or "").replace("_", " ").split())
    if sum(1 for char in text if char.isalnum()) < 5:
        return False

    safe = _sanitize_filename(text)
    compact = safe.replace("_", "")
    if not compact:
        return False
    if safe.casefold().startswith("sym_"):
        return False
    if compact.casefold().startswith(("legend", "legenda", "genda", "oznaczenia")):
        return False
    if re.fullmatch(r"[A-Z]{0,4}\d{1,4}[A-Z]?", compact.upper()):
        return False
    if compact.casefold() in {
        "legenda",
        "oznaczenia",
        "symbol",
        "opis",
        "indeks",
        "producent",
    }:
        return False

    tokens = text.split()
    return len(tokens) >= 2 or len(compact) >= 10


def _table_region_color_score(table_area: np.ndarray) -> int:
    """Count colored ink in the likely symbol/index side of a table."""

    if table_area.size == 0:
        return 0

    h, w = table_area.shape[:2]
    if h <= 0 or w <= 0:
        return 0

    left_band = table_area[:, : max(1, int(w * 0.42))]
    return int(cv2.countNonZero(_hsv_mask(left_band)))


def _build_color_table_display_drafts(
    page: fitz.Page,
    plan_image: np.ndarray,
    legend_rect_px: tuple[int, int, int, int],
    transform,
    *,
    expected_count: int,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
) -> list[VectorLegendDraft]:
    """Build display-only labels from the final description column of a color table."""

    if legend_rect_px is None or expected_count <= 0:
        return []

    if exclude_rects:
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(plan_image, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), -1)

    try:
        text_blocks = page.get_text("blocks")
        text_words = page.get_text("words")
    except Exception:
        return []

    scale = transform.dpi / 72.0
    x_start = max(0, min(int(legend_rect_px[0]), plan_image.shape[1] - 1))
    y_start = max(0, min(int(legend_rect_px[1]), plan_image.shape[0] - 1))
    width = int(legend_rect_px[2])
    height = int(legend_rect_px[3])
    x_end = min(x_start + width, plan_image.shape[1])
    y_end = min(y_start + height, plan_image.shape[0])
    legend_area = plan_image[y_start:y_end, x_start:x_end]
    if legend_area.size == 0:
        return []

    label_source = text_words if text_words else text_blocks
    scored: list[tuple[int, float, list[VectorLegendDraft]]] = []
    for candidate_index, (tx, ty, tw, th) in enumerate(
        _table_grid_region_candidates(legend_area),
        start=1,
    ):
        table_area = legend_area[ty : ty + th, tx : tx + tw]
        if table_area.size == 0 or _detect_legend_format(table_area) != "table":
            continue

        color_score = _table_region_color_score(table_area)
        min_color = max(18, int(table_area.shape[0] * table_area.shape[1] * 0.00004))
        if color_score < min_color:
            continue

        drafts = _build_table_grid_description_drafts(
            label_source,
            table_area,
            x_start + tx,
            y_start + ty,
            scale,
            transform,
            draft_prefix=f"vlegend:color-table-grid:{candidate_index}",
        )
        drafts = [
            draft for draft in drafts if _is_descriptive_table_label(draft.name_draft)
        ]
        if len(drafts) != expected_count:
            continue

        descriptive_count = len(drafts)
        if descriptive_count < max(2, int(expected_count * 0.72)):
            continue

        mean_label_len = sum(
            sum(1 for char in str(draft.name_draft) if char.isalnum()) for draft in drafts
        ) / max(1, len(drafts))
        quality = descriptive_count * 12.0 + mean_label_len + min(color_score / 100.0, 80.0)
        scored.append((candidate_index, quality, drafts))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[0][2]


def _build_gray_table_display_drafts(
    page: fitz.Page,
    plan_image: np.ndarray,
    legend_rect_px: tuple[int, int, int, int],
    transform,
    *,
    mask_mode: str,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
) -> list[VectorLegendDraft]:
    """Build display-only labels for gray table legends without changing template crops."""

    if legend_rect_px is None:
        return []

    if exclude_rects:
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(plan_image, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), -1)

    try:
        text_blocks = page.get_text("blocks")
        text_words = page.get_text("words")
    except Exception:
        return []

    scale = transform.dpi / 72.0
    x_start = max(0, min(int(legend_rect_px[0]), plan_image.shape[1] - 1))
    y_start = max(0, min(int(legend_rect_px[1]), plan_image.shape[0] - 1))
    width = int(legend_rect_px[2])
    height = int(legend_rect_px[3])
    x_end = min(x_start + width, plan_image.shape[1])
    y_end = min(y_start + height, plan_image.shape[0])
    legend_area = plan_image[y_start:y_end, x_start:x_end]
    if legend_area.size == 0:
        return []

    legend_format = _detect_legend_format(legend_area)
    table_trim = _trim_selection_to_table_grid(legend_area)
    if legend_format == "classic" and table_trim is not None:
        tx, ty, tw, th = table_trim
        trimmed_area = legend_area[ty : ty + th, tx : tx + tw]
        if _detect_legend_format(trimmed_area) == "table":
            legend_format = "table"
    if legend_format != "table":
        return []

    label_source = text_words if text_words else text_blocks
    if table_trim is not None:
        tx, ty, tw, th = table_trim
        grid_drafts = _build_table_grid_description_drafts(
            label_source,
            legend_area[ty : ty + th, tx : tx + tw],
            x_start + tx,
            y_start + ty,
            scale,
            transform,
            draft_prefix="vlegend:gray-table-grid",
        )
        if grid_drafts:
            return grid_drafts

    _probe_mask, table_mask_used = _legend_symbol_mask(legend_area, mask_mode=mask_mode)
    if table_mask_used != "gray":
        return []

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

    drafts: list[VectorLegendDraft] = []
    for draft_index, ((row_top, row_bottom), components) in enumerate(
        ((span, comps) for span, comps in zip(row_spans, assigned_components) if comps),
        start=1,
    ):
        pdf_label = _get_table_description_label_text(
            label_source,
            x_start,
            y_start,
            scale,
            row_top,
            row_bottom,
            col_boundaries,
            w,
        )
        ocr_label = _get_table_description_label_ocr(
            legend_area,
            row_top,
            row_bottom,
            col_boundaries,
        )
        label = ocr_label or pdf_label
        if not label:
            label = _get_row_label_text(
                label_source,
                x_start,
                y_start,
                scale,
                row_top,
                row_bottom,
                first_col_right,
                w,
            )
        if not label:
            label = _get_row_index_text(
                text_blocks,
                x_start,
                y_start,
                scale,
                row_top,
                row_bottom,
                first_col_right,
            )
        if not label:
            label = f"symbol_{draft_index:02d}"

        x1 = min(cx for cx, _cy, _cw, _ch in components)
        y1 = min(cy for _cx, cy, _cw, _ch in components)
        x2 = max(cx + cw for cx, _cy, cw, _ch in components)
        y2 = max(cy + ch for _cx, cy, _cw, ch in components)
        bbox_px = (x_start + x1, y_start + y1, max(1, x2 - x1), max(1, y2 - y1))
        row_bbox_px = (x_start, y_start + row_top, w, max(1, row_bottom - row_top))

        drafts.append(
            VectorLegendDraft(
                draft_id=f"vlegend:gray-table:{draft_index}",
                bbox_pt=rect_px300_to_pt(bbox_px, transform),
                bbox_px_300=bbox_px,
                row_bbox_pt=rect_px300_to_pt(row_bbox_px, transform),
                name_draft=label,
                symbol_code=None,
                confidence=0.82,
                primitive_refs=[],
                review_required=True,
                label_source="right_text",
                structure_source="table_cell",
                fallback_eligible=True,
                image_bgr=None,
            )
        )

    return drafts
