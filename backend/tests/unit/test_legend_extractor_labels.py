from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from core.legend_extractor import (
    _clean_ocr_label_text,
    _color_classic_row_symbol_bboxes,
    _expand_legend_rect_to_table,
    _extract_left_gutter_table_legend_raw,
    _extract_table_legend_raw,
    _get_classic_row_label_text,
    _gray_row_symbol_bboxes,
    _get_row_label_text,
    extract_legend,
    pdf_to_png,
    _read_visual_symbol_code,
    _table_symbols_need_expansion,
    _trim_selection_to_table_grid,
)


def test_clean_ocr_label_text_canonicalizes_noisy_electrical_labels():
    assert _clean_ocr_label_text("ROZDZIELNICA") == "ROZDZIELNICA"
    assert (
        _clean_ocr_label_text("GNIAZDO 1-F Z BOLCEM OCHRONNYM, 16A, IP20")
        == "GNIAZDO 1-F Z BOLCEM OCHRONNYM 16A IP20"
    )
    assert (
        _clean_ocr_label_text("WYIFUSI ZL OUIANT ZJUV WYPUST ZE SCIANY 400V GNIATDO ZLE")
        == "WYPUST ZE SCIANY 400V"
    )
    assert (
        _clean_ocr_label_text("ZESTAW GNIAZD 2x16A Sf 2x16A if SOCKET KIT")
        == "ZESTAW GNIAZD 2x16A 3f 2x16A 1f"
    )
    assert (
        _clean_ocr_label_text("E 400V wypust 400V zasilanie kuchenki")
        == "E 400V wypust 400V zasilanie kuchenki"
    )


def test_row_label_text_uses_description_from_same_table_row():
    text_words = [
        (130.0, 25.0, 190.0, 38.0, "Opis", 0, 0, 0),
        (122.0, 61.0, 134.0, 75.0, "A1", 0, 1, 0),
        (140.0, 61.0, 180.0, 75.0, "Oprawa", 0, 1, 1),
        (184.0, 61.0, 215.0, 75.0, "LED", 0, 1, 2),
        (132.0, 101.0, 218.0, 115.0, "Gniazdo", 0, 2, 0),
    ]

    label = _get_row_label_text(
        text_words,
        x_start=0,
        y_start=0,
        scale=1.0,
        row_top_px=50,
        row_bottom_px=86,
        col_right_px=120,
        legend_width_px=260,
    )

    assert label == "Oprawa_LED"


def test_row_label_text_returns_none_without_same_row_text():
    text_blocks = [
        (132.0, 101.0, 218.0, 115.0, "Gniazdo 230V", 0, 0),
    ]

    label = _get_row_label_text(
        text_blocks,
        x_start=0,
        y_start=0,
        scale=1.0,
        row_top_px=50,
        row_bottom_px=86,
        col_right_px=120,
        legend_width_px=260,
    )

    assert label is None


def test_classic_row_label_text_uses_description_to_right_of_symbol():
    text_words = [
        (110.0, 34.0, 122.0, 45.0, "02", 0, 0, 0),
        (138.0, 34.0, 196.0, 45.0, "ROZDZIELNICA", 0, 0, 1),
        (136.0, 74.0, 180.0, 85.0, "GNIAZDO", 0, 1, 0),
    ]

    label = _get_classic_row_label_text(
        text_words,
        x_start=0,
        y_start=0,
        scale=1.0,
        local_bbox=(80, 30, 40, 20),
        legend_width_px=260,
    )

    assert label == "ROZDZIELNICA"


def test_gray_classic_rows_use_raster_descriptions_and_skip_header():
    mask = np.zeros((150, 340), dtype=np.uint8)
    cv2.putText(mask, "OZNACZENIA:", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
    cv2.rectangle(mask, (36, 45), (66, 58), 255, 2)
    cv2.putText(mask, "ROZDZIELNICA", (115, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)
    cv2.circle(mask, (52, 92), 12, 255, 2)
    cv2.line(mask, (52, 92), (52, 118), 255, 2)
    cv2.putText(mask, "GNIAZDO 1F IP20", (115, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1)

    _symbol_mask, bboxes = _gray_row_symbol_bboxes(
        mask,
        text_blocks=[],
        x_start=0,
        y_start=0,
        scale=1.0,
    )

    assert len(bboxes) == 2
    assert all(y > 30 for _x, y, _w, _h in bboxes)
    assert bboxes[0][0] < 75
    assert bboxes[1][0] < 75


def test_color_classic_rows_keep_letter_and_symbol_together():
    mask = np.zeros((180, 360), dtype=np.uint8)
    cv2.putText(mask, "A", (36, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.circle(mask, (46, 88), 13, 255, -1)
    cv2.putText(mask, "B", (35, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    cv2.rectangle(mask, (31, 136), (61, 166), 255, -1)

    text_words = [
        (120.0, 50.0, 164.0, 64.0, "oprawa", 0, 0, 0),
        (168.0, 50.0, 230.0, 64.0, "oswietleniowa", 0, 0, 1),
        (234.0, 50.0, 250.0, 64.0, "np", 0, 0, 2),
        (120.0, 76.0, 178.0, 90.0, "LOTOS", 0, 1, 0),
        (120.0, 110.0, 164.0, 124.0, "oprawa", 0, 2, 0),
        (168.0, 110.0, 230.0, 124.0, "oswietleniowa", 0, 2, 1),
        (234.0, 110.0, 250.0, 124.0, "np", 0, 2, 2),
        (120.0, 136.0, 190.0, 150.0, "PLAFOND", 0, 3, 0),
    ]

    _symbol_mask, bboxes, labels = _color_classic_row_symbol_bboxes(
        mask,
        text_words,
        x_start=0,
        y_start=0,
        scale=1.0,
    )

    assert len(bboxes) == 2
    assert bboxes[0][1] + bboxes[0][3] <= bboxes[1][1]
    assert bboxes[1][1] < 122
    assert bboxes[1][1] + bboxes[1][3] >= 166
    assert labels[bboxes[1]] == "oprawa_oswietleniowa_np_PLAFOND"


def test_color_classic_rows_do_not_bleed_blue_labels_between_rows():
    mask = np.zeros((160, 360), dtype=np.uint8)
    cv2.rectangle(mask, (25, 20), (88, 45), 255, 2)
    cv2.putText(mask, "TSM", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2)
    cv2.rectangle(mask, (25, 66), (88, 91), 255, 2)
    cv2.putText(mask, "GSw", (30, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2)
    cv2.rectangle(mask, (25, 112), (88, 137), 255, 2)
    cv2.putText(mask, "MSw", (30, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2)

    text_words = [
        (120.0, 24.0, 210.0, 38.0, "teletechniczna", 0, 0, 0),
        (214.0, 24.0, 288.0, 38.0, "skrzynka", 0, 0, 1),
        (120.0, 70.0, 170.0, 84.0, "glowna", 0, 1, 0),
        (174.0, 70.0, 246.0, 84.0, "szyna", 0, 1, 1),
        (120.0, 116.0, 176.0, 130.0, "lokalna", 0, 2, 0),
        (180.0, 116.0, 246.0, 130.0, "szyna", 0, 2, 1),
    ]

    _symbol_mask, bboxes, labels = _color_classic_row_symbol_bboxes(
        mask,
        text_words,
        x_start=0,
        y_start=0,
        scale=1.0,
    )

    assert len(bboxes) == 3
    assert labels[bboxes[1]] == "glowna_szyna"
    assert labels[bboxes[2]] == "lokalna_szyna"


def test_table_legend_uses_text_label_and_ignores_outer_border():
    legend_area = np.full((90, 260, 3), 255, dtype=np.uint8)
    for y in (0, 30, 60, 89):
        cv2.line(legend_area, (0, y), (259, y), (0, 0, 0), 1)
    for x in (0, 70, 259):
        cv2.line(legend_area, (x, 0), (x, 89), (0, 0, 0), 1)
    cv2.circle(legend_area, (35, 45), 8, (0, 0, 0), 2)
    cv2.line(legend_area, (30, 45), (40, 45), (0, 0, 0), 2)

    text_words = [
        (82.0, 37.0, 96.0, 50.0, "A1", 0, 1, 0),
        (108.0, 37.0, 154.0, 50.0, "Oprawa", 0, 1, 1),
        (160.0, 37.0, 184.0, 50.0, "LED", 0, 1, 2),
    ]

    symbols = _extract_table_legend_raw(
        legend_area,
        text_blocks=[],
        text_words=text_words,
        x_start=0,
        y_start=0,
        scale=1.0,
    )

    assert len(symbols) == 1
    assert symbols[0][1] == "Oprawa_LED"


def test_table_legend_prefers_left_symbol_code_and_skips_title_row():
    legend_area = np.full((150, 260, 3), 255, dtype=np.uint8)
    for y in (0, 30, 70, 110, 149):
        cv2.line(legend_area, (0, y), (259, y), (0, 0, 0), 1)
    cv2.line(legend_area, (0, 0), (0, 149), (0, 0, 0), 1)
    cv2.line(legend_area, (259, 0), (259, 149), (0, 0, 0), 1)
    cv2.line(legend_area, (80, 30), (80, 149), (0, 0, 0), 1)

    cv2.putText(legend_area, "LEGENDA", (55, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.rectangle(legend_area, (18, 43), (46, 58), (0, 0, 255), 2)
    cv2.putText(legend_area, "L1", (51, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
    cv2.circle(legend_area, (31, 90), 9, (0, 180, 0), 2)
    cv2.putText(legend_area, "AW1", (47, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 180, 0), 1)

    text_words = [
        (54.0, 48.0, 64.0, 60.0, "L1", 0, 1, 0),
        (92.0, 45.0, 140.0, 60.0, "Oprawa", 0, 1, 1),
        (48.0, 87.0, 70.0, 99.0, "AW1", 0, 2, 0),
        (92.0, 85.0, 160.0, 100.0, "Awaryjna", 0, 2, 1),
    ]

    symbols = _extract_table_legend_raw(
        legend_area,
        text_blocks=[],
        text_words=text_words,
        x_start=0,
        y_start=0,
        scale=1.0,
    )

    assert [name for _image, name in symbols] == ["L1", "AW1"]


def test_partial_table_selection_expands_to_full_grid():
    image = np.full((180, 280, 3), 255, dtype=np.uint8)
    for y in (10, 45, 90, 135, 170):
        cv2.line(image, (20, y), (250, y), (0, 0, 0), 1)
    for x in (20, 90, 250):
        cv2.line(image, (x, 10), (x, 170), (0, 0, 0), 1)

    expanded = _expand_legend_rect_to_table(image, (112, 35, 110, 115))

    assert expanded is not None
    x, y, w, h = expanded
    assert x <= 24
    assert y <= 14
    assert x + w >= 246
    assert y + h >= 166


def test_table_selection_trims_surrounding_plan_margin():
    image = np.full((180, 320, 3), 255, dtype=np.uint8)
    cv2.line(image, (20, 15), (20, 165), (120, 120, 120), 1)
    for y in (10, 45, 90, 135, 170):
        cv2.line(image, (80, y), (300, y), (0, 0, 0), 1)
    for x in (80, 150, 300):
        cv2.line(image, (x, 10), (x, 170), (0, 0, 0), 1)

    trimmed = _trim_selection_to_table_grid(image)

    assert trimmed is not None
    x, y, w, h = trimmed
    assert 70 <= x <= 85
    assert y <= 14
    assert x + w >= 296
    assert y + h >= 166


def test_visual_symbol_code_reads_short_cad_like_label():
    cell = np.full((64, 140, 3), 255, dtype=np.uint8)
    cv2.putText(cell, "AW2", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)

    assert _read_visual_symbol_code(cell) == "AW2"


def test_table_extraction_uses_external_symbol_gutter_and_visual_index():
    image = np.full((180, 340, 3), 255, dtype=np.uint8)
    for y in (10, 45, 90, 135, 170):
        cv2.line(image, (90, y), (330, y), (0, 0, 0), 1)
    for x in (90, 150, 330):
        cv2.line(image, (x, 10), (x, 170), (0, 0, 0), 1)

    cv2.line(image, (24, 58), (70, 58), (0, 0, 0), 2)
    cv2.putText(image, "A1", (106, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.rectangle(image, (35, 105), (63, 120), (0, 0, 0), 2)
    cv2.putText(image, "AW2", (100, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(image, "E1", (42, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.circle(image, (50, 152), 8, (0, 0, 0), 2)
    cv2.putText(image, "E1", (106, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    table_trim = _trim_selection_to_table_grid(image)
    assert table_trim is not None
    assert table_trim[0] >= 84

    result = _extract_left_gutter_table_legend_raw(
        image,
        table_trim,
        text_blocks=[],
        text_words=[],
        x_start=0,
        y_start=0,
        scale=1.0,
    )

    assert result is not None
    symbols, used_rect = result
    assert used_rect[0] <= 24
    assert [name for _image, name in symbols] == ["A1", "AW2", "E1"]
    assert symbols[1][0].shape[0] < 42


def test_table_extraction_quality_flags_text_column_results():
    symbol = np.full((30, 40, 3), 255, dtype=np.uint8)
    good_symbols = [(symbol, "L1"), (symbol, "L2"), (symbol, "AW1"), (symbol, "EW1")]
    bad_symbols = [(symbol, "LEGENDA_oprawy"), (symbol, "sym_02"), (symbol, "sym_03")]
    tiny_single_text = [(symbol, "L7")]
    overgrown_plan_selection = [(symbol, f"L{index}") for index in range(1, 37)]

    assert _table_symbols_need_expansion(good_symbols) is False
    assert _table_symbols_need_expansion(bad_symbols) is True
    assert _table_symbols_need_expansion(tiny_single_text) is True
    assert _table_symbols_need_expansion(overgrown_plan_selection) is True


def test_gray_legend_extraction_matches_e8_fixture_geometry(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "test_pdfs" / "VIKING-BRONISZE-ELE-Rzuty-E8.pdf"
    fixture_dir = repo_root / "backend" / "tests" / "fixtures" / "viking_bronisze_e8_gray" / "templates"

    plan_image = pdf_to_png(str(pdf_path), dpi=300)
    extract_legend(
        str(pdf_path),
        plan_image,
        output_dir=str(tmp_path),
        dpi=300,
        legend_rect_px=(8625, 275, 1221, 1177),
        mask_mode="gray",
    )

    generated = sorted(tmp_path.glob("*.png"))
    fixtures = sorted(fixture_dir.glob("*.png"))
    assert len(generated) == len(fixtures) == 7

    exact_hash_indexes = {0, 1, 2, 3, 4, 6}
    for index, (generated_path, fixture_path) in enumerate(zip(generated, fixtures)):
        generated_image = cv2.imread(str(generated_path))
        fixture_image = cv2.imread(str(fixture_path))
        assert generated_image.shape[:2] == fixture_image.shape[:2]

        if index in exact_hash_indexes:
            assert hashlib.md5(generated_path.read_bytes()).hexdigest() == hashlib.md5(
                fixture_path.read_bytes()
            ).hexdigest()


def test_gray_table_legend_extraction_matches_e9_fixture_geometry(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "test_pdfs" / "VIKING-BRONISZE-ELE-Rzuty-E9.pdf"
    fixture_dir = repo_root / "backend" / "tests" / "fixtures" / "viking_bronisze_e9_gray" / "templates"

    plan_image = pdf_to_png(str(pdf_path), dpi=300)
    extract_legend(
        str(pdf_path),
        plan_image,
        output_dir=str(tmp_path),
        dpi=300,
        legend_rect_px=(383, 205, 3137, 2524),
        mask_mode="gray",
    )

    generated = sorted(tmp_path.glob("*.png"))
    fixtures = sorted(fixture_dir.glob("*.png"))
    assert [path.name for path in generated] == [path.name for path in fixtures]

    for generated_path, fixture_path in zip(generated, fixtures):
        generated_image = cv2.imread(str(generated_path))
        fixture_image = cv2.imread(str(fixture_path))
        assert generated_image.shape[:2] == fixture_image.shape[:2]
        assert hashlib.md5(generated_path.read_bytes()).hexdigest() == hashlib.md5(
            fixture_path.read_bytes()
        ).hexdigest()


def test_gray_classic_legend_extraction_matches_e10_fixture_geometry(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    pdf_path = repo_root / "test_pdfs" / "VIKING-BRONISZE-ELE-Rzuty-E10.pdf"
    fixture_dir = repo_root / "backend" / "tests" / "fixtures" / "viking_bronisze_e10_gray" / "templates"

    plan_image = pdf_to_png(str(pdf_path), dpi=300)
    extract_legend(
        str(pdf_path),
        plan_image,
        output_dir=str(tmp_path),
        dpi=300,
        legend_rect_px=(8800, 500, 1100, 950),
        mask_mode="gray",
    )

    generated = sorted(tmp_path.glob("*.png"))
    fixtures = sorted(fixture_dir.glob("*.png"))
    assert len(generated) == len(fixtures) == 11

    for generated_path, fixture_path in zip(generated, fixtures):
        generated_image = cv2.imread(str(generated_path))
        fixture_image = cv2.imread(str(fixture_path))
        assert generated_image.shape[:2] == fixture_image.shape[:2]
        assert hashlib.md5(generated_path.read_bytes()).hexdigest() == hashlib.md5(
            fixture_path.read_bytes()
        ).hexdigest()
