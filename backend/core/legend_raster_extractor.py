"""Raster fallback legend extraction path."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

try:
    from .legend_common import _next_template_index
    from .legend_constants import GLUE_KERNEL, MIN_PIXEL_DENSITY, MIN_SYMBOL_SIZE, SYMBOL_PADDING
    from .legend_constants import TEXT_MAX_DISTANCE_X, TEXT_MIN_OVERLAP_X, TEXT_TOLERANCE_Y
    from .legend_label_extractor import _get_classic_row_label_text, _get_symbol_text_inside_region
    from .legend_mask_utils import _hsv_mask, _legend_symbol_mask
    from .legend_models import ExtractedSymbol
    from .legend_pdf_render import _prepare_doc_with_hidden_layers
    from .legend_row_extractors import (
        _color_classic_row_symbol_bboxes,
        _filter_gray_legend_symbol_contours,
        _get_classic_row_label_ocr,
        _gray_row_symbol_bboxes,
        _rect_to_contour,
    )
    from .legend_table_geometry import (
        _detect_legend_format,
        _table_symbols_need_expansion,
        _tighten_gray_legend_symbol_crop,
        _trim_selection_to_table_grid,
    )
    from .legend_table_raw import (
        _extract_gray_table_legend_raw,
        _extract_left_gutter_table_legend_raw,
        _extract_table_legend_raw,
    )
    from .legend_text import _sanitize_filename
except ImportError:  # pragma: no cover
    from legend_common import _next_template_index
    from legend_constants import GLUE_KERNEL, MIN_PIXEL_DENSITY, MIN_SYMBOL_SIZE, SYMBOL_PADDING
    from legend_constants import TEXT_MAX_DISTANCE_X, TEXT_MIN_OVERLAP_X, TEXT_TOLERANCE_Y
    from legend_label_extractor import _get_classic_row_label_text, _get_symbol_text_inside_region
    from legend_mask_utils import _hsv_mask, _legend_symbol_mask
    from legend_models import ExtractedSymbol
    from legend_pdf_render import _prepare_doc_with_hidden_layers
    from legend_row_extractors import (
        _color_classic_row_symbol_bboxes,
        _filter_gray_legend_symbol_contours,
        _get_classic_row_label_ocr,
        _gray_row_symbol_bboxes,
        _rect_to_contour,
    )
    from legend_table_geometry import (
        _detect_legend_format,
        _table_symbols_need_expansion,
        _tighten_gray_legend_symbol_crop,
        _trim_selection_to_table_grid,
    )
    from legend_table_raw import (
        _extract_gray_table_legend_raw,
        _extract_left_gutter_table_legend_raw,
        _extract_table_legend_raw,
    )
    from legend_text import _sanitize_filename

def _is_spurious_color_legend_fragment(
    *,
    safe_name: str,
    symbol_image: np.ndarray,
    pixel_count: int,
    previous: ExtractedSymbol | None,
) -> bool:
    """Drop tiny duplicate color crops while preserving the legend index gap."""

    if previous is None:
        return False
    if _sanitize_filename(previous.name).casefold() != _sanitize_filename(safe_name).casefold():
        return False
    prev_h, prev_w = previous.image.shape[:2]
    cur_h, cur_w = symbol_image.shape[:2]
    if prev_w <= 0 or prev_h <= 0 or cur_w <= 0 or cur_h <= 0:
        return False

    return (
        pixel_count < max(260, int(previous.pixel_count * 0.36))
        and cur_w <= max(24, int(prev_w * 0.72))
        and cur_h <= max(28, int(prev_h * 0.72))
    )


def _extract_legend_raster_current(
    pdf_path: str,
    plan_image: np.ndarray,
    output_dir: str = "templates",
    dpi: int = 300,
    exclude_rects: list[tuple[int, int, int, int]] = None,
    legend_rect_px: tuple[int, int, int, int] | None = None,
    mask_mode: str = "auto",
    return_used_rect: bool = False,
    hidden_layers: list[str] | None = None,
) -> list[ExtractedSymbol] | tuple[list[ExtractedSymbol], tuple[int, int, int, int]]:
    """
    Wyciąga wzorce symboli z legendy planu elektrycznego.

    Args:
        pdf_path:        Ścieżka do pliku PDF.
        plan_image:      Obraz planu jako BGR np.ndarray (ten sam DPI co poniżej).
        output_dir:      Folder docelowy na wzorce (tworzony automatycznie).
        dpi:             DPI użyte przy konwersji PDF → PNG.
        exclude_rects:   Strefy do zignorowania.
        legend_rect_px:  Obszar legendy w pikselach (x, y, w, h) — wymagane.
        mask_mode:       Tryb maskowania: 'auto', 'color', 'gray'.
    """
    if legend_rect_px is None:
        raise ValueError(
            "legend_rect_px jest wymagane. Zaznacz strefę legendy na planie przed ekstrakcją."
        )
    doc = _prepare_doc_with_hidden_layers(pdf_path, hidden_layers=hidden_layers)
    page = doc.load_page(0)
    text_blocks = page.get_text("blocks")
    text_words = page.get_text("words")

    # Aplikujemy strefy wykluczone do obrazu planu (żeby zamazać niechciane fragmenty legendy)
    if exclude_rects:
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(plan_image, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), -1)

    # 1. Lokalizacja legendy — wyłącznie z ręcznie zaznaczonego obszaru
    scale = dpi / 72.0
    x_start = int(legend_rect_px[0])
    y_start = int(legend_rect_px[1])
    width = int(legend_rect_px[2])
    height = int(legend_rect_px[3])

    # Zabezpieczenie przed wyjściem poza obraz
    x_start = max(0, min(x_start, plan_image.shape[1] - 1))
    y_start = max(0, min(y_start, plan_image.shape[0] - 1))
    legend_rect_px = (x_start, y_start, width, height)
    used_legend_rect_px = legend_rect_px
    y_end = min(y_start + height, plan_image.shape[0])
    x_end = min(x_start + width, plan_image.shape[1])
    legend_area = plan_image[y_start:y_end, x_start:x_end]
    if legend_area.size == 0:
        raise ValueError("Zaznaczona strefa legendy jest pusta albo poza obrazem.")

    legend_format = _detect_legend_format(legend_area)
    table_trim = _trim_selection_to_table_grid(legend_area)
    if legend_format == "classic" and table_trim is not None:
        tx, ty, tw, th = table_trim
        trimmed_area = legend_area[ty : ty + th, tx : tx + tw]
        if _detect_legend_format(trimmed_area) == "table":
            legend_format = "table"

    if legend_format == "table":
        _probe_mask, table_mask_used = _legend_symbol_mask(legend_area, mask_mode=mask_mode)
        if table_mask_used == "gray":
            raw_symbols = _extract_gray_table_legend_raw(
                legend_area,
                text_blocks,
                x_start,
                y_start,
                scale,
            )
            if raw_symbols:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                results: list[ExtractedSymbol] = []
                start_index = _next_template_index(output_path)
                for counter, (symbol_image, name) in enumerate(raw_symbols, start=start_index):
                    filename = f"{counter:02d}_{name}.png"
                    ok, buf = cv2.imencode(".png", symbol_image)
                    if ok:
                        (output_path / filename).write_bytes(buf.tobytes())
                    px_count = int(np.sum(cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY) < 180))
                    results.append(
                        ExtractedSymbol(
                            name=name,
                            image=symbol_image,
                            index=counter,
                            pixel_count=px_count,
                        )
                    )
                return (results, used_legend_rect_px) if return_used_rect else results

        original_legend_area = legend_area
        original_x_start = x_start
        original_y_start = y_start
        if table_trim is not None:
            tx, ty, tw, th = table_trim
            legend_area = legend_area[ty : ty + th, tx : tx + tw]
            x_start += tx
            y_start += ty
            used_legend_rect_px = (x_start, y_start, tw, th)

        raw_symbols = _extract_table_legend_raw(
            legend_area,
            text_blocks,
            text_words,
            x_start,
            y_start,
            scale,
            tighten_gray_table_crops=mask_mode == "gray",
        )
        table_needs_expansion = _table_symbols_need_expansion(raw_symbols)
        if table_needs_expansion:
            raw_symbols = []
            if table_trim is not None:
                gutter_result = _extract_left_gutter_table_legend_raw(
                    original_legend_area,
                    table_trim,
                    text_blocks,
                    text_words,
                    original_x_start,
                    original_y_start,
                    scale,
                    tighten_gray_table_crops=mask_mode == "gray",
                )
                if gutter_result is not None:
                    raw_symbols, gutter_trim = gutter_result
                    gx, gy, gw, gh = gutter_trim
                    used_legend_rect_px = (
                        original_x_start + gx,
                        original_y_start + gy,
                        gw,
                        gh,
                    )
        if raw_symbols:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            results: list[ExtractedSymbol] = []
            start_index = _next_template_index(output_path)
            for counter, (symbol_image, name) in enumerate(raw_symbols, start=start_index):
                filename = f"{counter:02d}_{name}.png"
                ok, buf = cv2.imencode(".png", symbol_image)
                if ok:
                    (output_path / filename).write_bytes(buf.tobytes())
                px_count = int(np.sum(cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY) < 180))
                results.append(
                    ExtractedSymbol(
                        name=name,
                        image=symbol_image,
                        index=counter,
                        pixel_count=px_count,
                    )
                )
            return (results, used_legend_rect_px) if return_used_rect else results
        if table_needs_expansion:
            empty_results: list[ExtractedSymbol] = []
            return (empty_results, used_legend_rect_px) if return_used_rect else empty_results

    if legend_format == "classic" and legend_area.shape[1] > 1400 and legend_area.shape[0] > 800:
        empty_results: list[ExtractedSymbol] = []
        return (empty_results, used_legend_rect_px) if return_used_rect else empty_results

    # 2. Maska kolorowa + morphological CLOSE (klejenie symboli)
    raw_symbol_mask, _mask_used = _legend_symbol_mask(legend_area, mask_mode=mask_mode)
    row_label_hints: dict[tuple[int, int, int, int], str] = {}
    tighten_gray_row_crops = False
    if _mask_used == "gray":
        symbol_mask, row_bboxes = _gray_row_symbol_bboxes(
            raw_symbol_mask,
            text_blocks,
            x_start=x_start,
            y_start=y_start,
            scale=scale,
        )
        if row_bboxes:
            contours = [_rect_to_contour(rect) for rect in row_bboxes]
            tighten_gray_row_crops = True
        else:
            glued_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, GLUE_KERNEL)
            contours, _ = cv2.findContours(glued_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = _filter_gray_legend_symbol_contours(contours, legend_area.shape)
    else:
        symbol_mask = raw_symbol_mask
        glued_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, GLUE_KERNEL)
        contours, _ = cv2.findContours(glued_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    # 4. Ekstrakcja symboli
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[ExtractedSymbol] = []
    counter = _next_template_index(output_path)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < MIN_SYMBOL_SIZE or h < MIN_SYMBOL_SIZE:
            continue

        # ── Ciasne wycinanie (tight-cut) ──
        # Operujemy na ORYGINALNEJ masce (nie sklejonej) żeby znaleźć
        # dokładne granice kolorowych pikseli — sklejona maska jest zbyt
        # 'napompowana' przez MORPH_CLOSE i dałaby za duży bbox.
        roi_mask = symbol_mask[y : y + h, x : x + w]
        colored_pixels = cv2.findNonZero(roi_mask)

        if colored_pixels is None:
            continue

        # Sprawdzamy gęstość kolorowych pikseli
        pixel_count = len(colored_pixels)
        density = pixel_count / (w * h)
        if density < MIN_PIXEL_DENSITY:
            continue

        # Minimalna ramka wokół samych kolorowych pikseli
        tx, ty, tw, th = cv2.boundingRect(colored_pixels)

        # Dodajemy margines (clampowany do granic obszaru legendy)
        x1 = max(0, x + tx - SYMBOL_PADDING)
        y1 = max(0, y + ty - SYMBOL_PADDING)
        x2 = min(legend_area.shape[1], x + tx + tw + SYMBOL_PADDING)
        y2 = min(legend_area.shape[0], y + ty + th + SYMBOL_PADDING)

        # ── KLUCZOWA NAPRAWA: Czarne tło zamiast białego ──
        # Tworzymy pusty (czarny) obraz o rozmiarze wycinanego symbolu
        out_w = x2 - x1
        out_h = y2 - y1

        if out_w <= 0 or out_h <= 0:
            continue

        # Wycinamy fragment oryginalnej kolorowej legendy
        color_roi = legend_area[y1:y2, x1:x2]
        # Wycinamy odpowiadający fragment maski
        mask_roi = symbol_mask[y1:y2, x1:x2]

        # Color templates keep black background for HSV; gray templates need
        # white background so dark-ink matching does not treat the background
        # itself as part of the symbol.
        symbol_image = (
            np.full_like(color_roi, 255) if _mask_used == "gray" else np.zeros_like(color_roi)
        )
        # Kopiujemy TYLKO kolorowe piksele — reszta zostaje czarna
        symbol_image[mask_roi > 0] = color_roi[mask_roi > 0]
        if tighten_gray_row_crops:
            symbol_image = _tighten_gray_legend_symbol_crop(symbol_image)

        if symbol_image.size == 0:
            continue

        symbol_text = _get_symbol_text_inside_region(
            text_words,
            x_start=x_start,
            y_start=y_start,
            scale=scale,
            local_bbox=(x1, y1, out_w, out_h),
        )
        row_label_text = _get_classic_row_label_text(
            text_words,
            x_start=x_start,
            y_start=y_start,
            scale=scale,
            local_bbox=(x, y, w, h),
            legend_width_px=legend_area.shape[1],
        )
        row_label_hint = row_label_hints.get((x, y, w, h))
        row_label_ocr = _get_classic_row_label_ocr(
            legend_area,
            local_bbox=(x, y, w, h),
            symbol_mask=raw_symbol_mask,
        )

        # ── Dopasowanie tekstu z PDF ──
        # Używamy współrzędnych ORYGINALNEGO konturu (ze sklejonej maski)
        # bo ona daje lepszy "środek" grupy symboli złożonych
        center_y_pdf = (y_start + y + h / 2) / scale
        right_edge_pdf = (x_start + x + w) / scale

        found_texts: list[tuple[float, str]] = []

        for block in text_blocks:
            if block[6] != 0:  # pomijamy nie-teksty
                continue

            block_center_y = (block[1] + block[3]) / 2
            block_left_x = block[0]

            dy = abs(block_center_y - center_y_pdf)
            dx = block_left_x - right_edge_pdf

            if dy < TEXT_TOLERANCE_Y and TEXT_MIN_OVERLAP_X < dx < TEXT_MAX_DISTANCE_X:
                found_texts.append((dx, block[4].strip()))

        if found_texts:
            found_texts.sort(key=lambda t: t[0])
            full_name = "_".join(t[1] for t in found_texts)
            found_safe_name = _sanitize_filename(full_name)
        else:
            found_safe_name = ""

        # Łączymy wszystkie fragmenty tekstu (posortowane od lewej do prawej)
        if symbol_text:
            safe_name = symbol_text
            filename = f"{counter:02d}_{safe_name}.png"
        elif _mask_used != "gray" and found_safe_name:
            safe_name = found_safe_name
            filename = f"{counter:02d}_{safe_name}.png"
        elif row_label_hint:
            safe_name = row_label_hint
            filename = f"{counter:02d}_{safe_name}.png"
        elif row_label_text:
            safe_name = row_label_text
            filename = f"{counter:02d}_{safe_name}.png"
        elif row_label_ocr:
            safe_name = row_label_ocr
            filename = f"{counter:02d}_{safe_name}.png"
        elif found_safe_name:
            safe_name = found_safe_name
            filename = f"{counter:02d}_{safe_name}.png"
        else:
            safe_name = f"symbol_{counter:02d}"
            filename = f"{counter:02d}_{safe_name}.png"

        if _mask_used != "gray" and _is_spurious_color_legend_fragment(
            safe_name=safe_name,
            symbol_image=symbol_image,
            pixel_count=pixel_count,
            previous=results[-1] if results else None,
        ):
            counter += 1
            continue

        # ── Zapis (cv2.imencode + write_bytes zamiast imwrite dla Unicode) ──
        file_path = output_path / filename
        ok, buf = cv2.imencode(".png", symbol_image)
        if ok:
            file_path.write_bytes(buf.tobytes())

        results.append(
            ExtractedSymbol(
                name=safe_name,
                image=symbol_image,
                index=counter,
                pixel_count=pixel_count,
            )
        )
        counter += 1

    return (results, used_legend_rect_px) if return_used_rect else results
