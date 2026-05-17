"""Visual short-code recognition helpers for raster legend cells."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from .legend_text import _symbol_text_token
except ImportError:  # pragma: no cover - keeps direct script execution working.
    from legend_text import _symbol_text_token

HSV_LOWER = np.array([0, 30, 50])
HSV_UPPER = np.array([180, 255, 255])
CELL_BORDER_TRIM = 2
VISUAL_CODE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_VISUAL_CODE_TEMPLATES: list[tuple[str, np.ndarray]] | None = None


def _hsv_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)


def _visible_ink_mask(image_bgr: np.ndarray, gray_threshold: int = 235) -> np.ndarray:
    """Mask visible dark or colored drawing pixels."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_pixels = gray < gray_threshold
    color_pixels = _hsv_mask(image_bgr) > 0
    return np.where(np.logical_or(dark_pixels, color_pixels), 255, 0).astype(np.uint8)


def _normalize_visual_char_mask(mask: np.ndarray, size: tuple[int, int] = (32, 32)) -> np.ndarray:
    """Normalize one isolated character mask for lightweight template matching."""

    pixels = cv2.findNonZero(mask)
    if pixels is None:
        return np.zeros(size, dtype=np.uint8)

    x, y, width, height = cv2.boundingRect(pixels)
    crop = mask[y : y + height, x : x + width]
    target_w, target_h = size
    scale = min((target_w - 6) / max(1, width), (target_h - 6) / max(1, height))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return canvas


def _get_visual_code_templates() -> list[tuple[str, np.ndarray]]:
    """Build small OCR templates from OpenCV fonts for short CAD-like legend codes."""

    global _VISUAL_CODE_TEMPLATES
    if _VISUAL_CODE_TEMPLATES is not None:
        return _VISUAL_CODE_TEMPLATES

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
    ]
    templates: list[tuple[str, np.ndarray]] = []
    for char in VISUAL_CODE_CHARS:
        for font in fonts:
            for scale in (0.7, 0.9, 1.1, 1.3, 1.5):
                for thickness in (1, 2):
                    image = np.zeros((80, 80), dtype=np.uint8)
                    cv2.putText(
                        image,
                        char,
                        (8, 58),
                        font,
                        scale,
                        255,
                        thickness,
                        cv2.LINE_AA,
                    )
                    templates.append((char, _normalize_visual_char_mask(image)))

    _VISUAL_CODE_TEMPLATES = templates
    return templates


def _classify_visual_code_char(char_mask: np.ndarray) -> tuple[str, float]:
    normalized = _normalize_visual_char_mask(char_mask)
    best_char = ""
    best_score = float("inf")

    for candidate, template in _get_visual_code_templates():
        score = float(np.mean(cv2.absdiff(normalized, template)) / 255.0)
        if score < best_score:
            best_char = candidate
            best_score = score

    return best_char, best_score


def _read_visual_symbol_code(cell_image: np.ndarray) -> str | None:
    """Read a short printed code from a simple table cell without external OCR."""

    if cell_image.size == 0:
        return None

    mask = _visible_ink_mask(cell_image, gray_threshold=190)
    if mask.shape[0] > CELL_BORDER_TRIM * 2:
        mask[:CELL_BORDER_TRIM, :] = 0
        mask[-CELL_BORDER_TRIM:, :] = 0
    if mask.shape[1] > CELL_BORDER_TRIM * 2:
        mask[:, :CELL_BORDER_TRIM] = 0
        mask[:, -CELL_BORDER_TRIM:] = 0

    pixels = cv2.findNonZero(mask)
    if pixels is None:
        return None

    x, y, width, height = cv2.boundingRect(pixels)
    if height < 8 or width < 4:
        return None
    mask = mask[max(0, y - 2) : y + height + 2, max(0, x - 2) : x + width + 2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if cw * ch < 5 or ch < 6:
            continue
        boxes.append((cx, cy, cw, ch))

    if not 1 <= len(boxes) <= 6:
        return None

    token = ""
    for cx, cy, cw, ch in sorted(boxes, key=lambda item: item[0]):
        char, score = _classify_visual_code_char(mask[cy : cy + ch, cx : cx + cw])
        if not char or score > 0.22:
            return None
        token += char

    return _symbol_text_token(token)
