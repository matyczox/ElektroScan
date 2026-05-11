from __future__ import annotations

import cv2
import numpy as np

from core.legend_extractor import _extract_table_legend_raw, _get_row_label_text


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
