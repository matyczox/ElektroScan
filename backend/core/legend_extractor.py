"""
legend_extractor.py — Ekstrakcja wzorców symboli z ręcznie zaznaczonej strefy legendy PDF.

Algorytm:
  1. Przyjmuje współrzędne legend_rect_px zaznaczone przez użytkownika w UI.
  2. Wycina obszar legendy z planu PNG (300 DPI).
  3. Filtruje HSV → maska kolorowych pikseli.
  4. Morphological CLOSE (kernel 4×40) skleja rozbite symbole (np. -[INT).
  5. Kontury na sklejonej masce → lista potencjalnych symboli.
  6. Dla każdego konturu:
     a. Ciasne wycinanie (findNonZero na ORYGINALNEJ masce — nie sklejonej).
     b. Ekstrakcja TYLKO kolorowych pikseli → czarne tło (kluczowa naprawa!).
     c. Margines 2px żeby nie ucinać krawędzi symbolu.
  7. Dopasowanie tekstu z PDF.
  8. Zapis z czarnym tłem → matchTemplate porównuje tylko kształt, nie tło.

Kluczowa zmiana vs poprzednia wersja:
  STARY KOD: zapisywał wycinek BGR z białym/szarym tłem planszy.
  NOWY KOD: tworzy nowy obraz (czarne tło) i wkleja tylko kolorowe piksele.
  Dzięki temu detector widzi sam symbol, a nie 'symbol + otoczenie'.
"""

import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

# ── Stałe ──────────────────────────────────────────────────────────────────

# Granice HSV izolujące kolory (S>30 odrzuca biel/szarość, V>50 odrzuca czerń)
HSV_LOWER = np.array([0, 30, 50])
HSV_UPPER = np.array([180, 255, 255])

# Kernel MORPH_CLOSE — "klei" rozbite symbole w poziomie,
# nie łącząc wierszy (4px pion, 40px poziom).
GLUE_KERNEL = np.ones((4, 40), np.uint8)

# Minimalny rozmiar konturu żeby nie łapać śmieci
MIN_SYMBOL_SIZE = 15

# Margines wokół wyciętego symbolu (px) — zapobiega ucinaniu krawędzi
SYMBOL_PADDING = 2

# Liczba pikseli przycinanych z każdej krawędzi komórki TABLE przed findNonZero —
# usuwa ciemne linie ramki tabeli, które rozszerzałyby boundingRect do pełnej szerokości.
CELL_BORDER_TRIM = 2

# Minimalna gęstość pikseli — symbol musi mieć przynajmniej X% kolorowych
# pikseli w swoim bboxie, żeby nie złapać jednej kolorowej kreski jako symbolu
MIN_PIXEL_DENSITY = 0.05  # 5% — celowo nisko, żeby nie odrzucać cienkich symboli

# Tolerancje dopasowania tekstu z PDF (w punktach PDF)
TEXT_TOLERANCE_Y = 15  # ±15 pt w pionie
TEXT_MAX_DISTANCE_X = 250  # max 250 pt w prawo
TEXT_MIN_OVERLAP_X = -15  # lekkie najście tekstu na symbol dozwolone

# Maksymalna długość nazwy pliku
MAX_FILENAME_LENGTH = 80

VISUAL_CODE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_VISUAL_CODE_TEMPLATES: list[tuple[str, np.ndarray]] | None = None


# ── Pomocnicze ─────────────────────────────────────────────────────────────


def _sanitize_filename(text: str) -> str:
    """Czyści tekst do bezpiecznej nazwy pliku (ASCII, underscory)."""
    text = text.strip().replace("\n", "_")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    # Transliteracja polskich znaków (cv2.imwrite nie radzi z Unicode na Windows)
    _PL = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")
    text = text.translate(_PL)
    return text[:MAX_FILENAME_LENGTH].strip("_")


def _clean_ocr_label_text(text: str) -> str | None:
    """Turn noisy OCR from a legend description into a stable human label."""

    raw = " ".join(str(text or "").replace("_", " ").split())
    if sum(1 for char in raw if char.isalnum()) < 2:
        return None

    upper = unicodedata.normalize("NFKC", raw).upper()
    upper = upper.translate(str.maketrans("ĄĆĘŁŃÓŚŹŻ", "ACELNOSZZ"))
    upper = re.sub(r"[^A-Z0-9+\-/ ]+", " ", upper)
    upper = re.sub(r"\s+", " ", upper).strip()
    compact = re.sub(r"[^A-Z0-9]+", " ", upper)

    def contains_any(*needles: str) -> bool:
        return any(needle in compact for needle in needles)

    voltage_match = re.search(r"\b(230|400)\s*V\b", compact)
    voltage = f"{voltage_match.group(1)}V" if voltage_match else None
    ip_match = re.search(r"\bIP\s*(20|44|54|65)\b", compact)
    ip = f"IP{ip_match.group(1)}" if ip_match else None
    phase_match = re.search(r"\b([135])\s*[-]?\s*F\b", compact)
    phase = phase_match.group(1) if phase_match else None
    if phase == "5":
        # Tesseract commonly misreads the 3-phase row as 5-F on this CAD font.
        phase = "3"

    if "ROZDZ" in compact:
        return "ROZDZIELNICA"

    if contains_any("WYPUST", "WYPUS", "WYFUS", "WYIFUS") and contains_any(
        "SCIANY", "SCLANY", "SC1ANY"
    ):
        suffix = f" {voltage}" if voltage else ""
        return f"WYPUST ZE SCIANY{suffix}".strip()

    if contains_any("ZESTAW", "SOCKET KIT") and ("2X16" in compact or "SOCKET KIT" in compact):
        return "ZESTAW GNIAZD 2x16A 3f 2x16A 1f"

    if contains_any("BOLCEM", "ROICEM", "OCHRONNYM", "OCHRONNY"):
        parts = ["GNIAZDO"]
        if phase:
            parts.append(f"{phase}-F")
        parts.extend(["Z", "BOLCEM", "OCHRONNYM"])
        if "16A" in compact or "I6A" in compact:
            parts.append("16A")
        if ip:
            parts.append(ip)
        return " ".join(parts)

    # Generic fallback for other projects: keep readable OCR, but trim very long
    # endings that usually contain accidental neighboring rows.
    readable = re.sub(r"[^0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż+\-/ ]+", " ", raw)
    readable = re.sub(r"\s+", " ", readable).strip()
    tokens = readable.split()
    if len(tokens) > 12:
        tokens = tokens[:12]
    cleaned = " ".join(tokens)
    if sum(1 for char in cleaned if char.isalnum()) < 2:
        return None
    return cleaned


def _symbol_text_token(text: str) -> str | None:
    """Return a short alphanumeric symbol token from PDF text, if it looks like one."""

    token = _sanitize_filename(str(text or "")).upper()
    if not re.fullmatch(r"[A-Z0-9_]{2,12}", token):
        return None

    compact = token.replace("_", "")
    if not (2 <= len(compact) <= 10):
        return None
    if not re.search(r"[A-Z]", compact):
        return None
    if not (re.search(r"\d", compact) or len(compact) <= 4):
        return None
    if re.fullmatch(r"\d+(?:X\d+)+", compact):
        return None

    return compact


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


def _next_template_index(output_path: Path) -> int:
    """Return next numeric template prefix so repeated legend crops append."""
    max_index = 0
    if output_path.exists():
        for existing in output_path.glob("*.png"):
            match = re.match(r"^(\d+)_", existing.stem)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _hsv_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Tworzy binarną maskę kolorowych pikseli (HSV)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)


def _ink_mask(image_bgr: np.ndarray, threshold: int = 238) -> np.ndarray:
    """Create a binary mask for dark ink in gray/black PDFs."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ink_pixels = gray < threshold
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_pixels = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER) > 0
    ink_pixels = np.logical_and(ink_pixels, np.logical_not(color_pixels))
    return np.where(ink_pixels, 255, 0).astype(np.uint8)


def _legend_symbol_mask(image_bgr: np.ndarray, mask_mode: str = "auto") -> tuple[np.ndarray, str]:
    """Pick HSV color masking or dark-ink masking for legend segmentation."""
    requested = (mask_mode or "auto").lower()
    if requested not in {"auto", "color", "gray"}:
        requested = "auto"

    color_mask = _hsv_mask(image_bgr)
    if requested == "color":
        return color_mask, "color"

    ink_mask = _ink_mask(image_bgr)
    if requested == "gray":
        return ink_mask, "gray"

    color_pixels = int(cv2.countNonZero(color_mask))
    ink_pixels = int(cv2.countNonZero(ink_mask))
    if ink_pixels == 0:
        return color_mask, "color"

    color_ratio = color_pixels / max(ink_pixels, 1)
    if color_pixels < 100 or color_ratio < 0.08:
        return ink_mask, "gray"
    return color_mask, "color"


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

        token = _symbol_text_token(str(word[4]))
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

    return fallback_names > 0 or code_names < max(3, int(len(raw_symbols) * 0.6))


def _visible_ink_mask(image_bgr: np.ndarray, gray_threshold: int = 235) -> np.ndarray:
    """Mask visible dark or colored drawing pixels."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_pixels = gray < gray_threshold
    color_pixels = _hsv_mask(image_bgr) > 0
    return np.where(np.logical_or(dark_pixels, color_pixels), 255, 0).astype(np.uint8)


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

        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        cell_mask = (cell_gray < 200).astype(np.uint8) * 255
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
        crop_gray = cv2.cvtColor(symbol_crop, cv2.COLOR_BGR2GRAY)
        dark_px = crop_gray < 200
        symbol_image[dark_px] = symbol_crop[dark_px]
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
    )
    if not _table_symbol_images_look_valid(raw_symbols):
        return None

    return raw_symbols, (crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)


@dataclass
class ExtractedSymbol:
    """Wynik ekstrakcji jednego symbolu z legendy."""

    name: str
    image: np.ndarray  # BGR z CZARNYM tłem (tylko kolorowe piksele symbolu)
    index: int
    pixel_count: int = 0  # liczba kolorowych pikseli — przydatna do sortowania


def get_pdf_layers(pdf_path: str) -> list[dict]:
    """
    Zwraca listę warstw (Optional Content Groups - OCG) dostępnych w pliku PDF.
    """
    doc = fitz.open(pdf_path)
    layers = []

    # Próbujemy pobrać konfigurację warstw
    try:
        ui_configs = doc.layer_ui_configs()
        if ui_configs:
            for conf in ui_configs:
                # conf to dict, np. {'text': 'Warstwa 1', 'depth': 0, 'on': True, ...}
                if "text" in conf:
                    layers.append({"name": conf["text"], "visible": conf.get("on", True)})
    except Exception as e:
        print(f"Błąd odczytu warstw: {e}")

    return layers


def _render_doc_to_bgr(doc: fitz.Document, page: int = 0, dpi: int = 300) -> np.ndarray:
    """Render a PDF page from an already prepared document."""
    pg = doc.load_page(page)
    zoom = dpi / 72.0
    pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _normalize_layer_name(name: str) -> str:
    """Normalize layer names to make matching resilient to PDF text encoding quirks."""
    text = str(name).strip()

    # When layer names pass through different shells / encodings we sometimes
    # get mojibake like "UKLAD" instead of "UKŁAD". Repair that first when
    # possible, then normalize everything to the same ASCII-ish form.
    try:
        repaired = text.encode("latin1").decode("utf-8")
        if "\ufffd" not in repaired:
            text = repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    text = text.casefold().translate(str.maketrans({"ł": "l", "Ł": "l"}))
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _get_ocg_xrefs_in_catalog_order(doc: fitz.Document) -> list[int]:
    """Read OCG xrefs from the PDF catalog in their declared order."""
    catalog_object = doc.xref_object(doc.pdf_catalog())
    match = re.search(r"/OCGs\s*\[(.*?)\]", catalog_object, re.S)
    if not match:
        return []

    return [int(value) for value in re.findall(r"(\d+)\s+0\s+R", match.group(1))]


def _prepare_doc_with_hidden_layers(
    pdf_path: str,
    hidden_layers: list[str] | None = None,
) -> fitz.Document:
    """
    Open a PDF and apply hidden OCG layers in a way that affects rendering.

    Some PDFs expose layer UI state, but PyMuPDF's direct `set_layer_ui_config()`
    does not change `get_pixmap()` output for them. Updating the OCG defaults and
    reopening the modified in-memory PDF does.
    """
    doc = fitz.open(pdf_path)
    hidden_layers = [name for name in (hidden_layers or []) if name]
    if not hidden_layers:
        return doc

    try:
        ui_configs = doc.layer_ui_configs() or []
        ocg_xrefs = _get_ocg_xrefs_in_catalog_order(doc)
        if not ui_configs or not ocg_xrefs:
            return doc

        hidden_set = {_normalize_layer_name(name) for name in hidden_layers}
        off_refs: list[int] = []

        for config in ui_configs:
            layer_name = str(config.get("text", "")).strip()
            if not layer_name:
                continue

            number = config.get("number")
            if not isinstance(number, int):
                continue
            if number < 0 or number >= len(ocg_xrefs):
                continue

            if _normalize_layer_name(layer_name) in hidden_set:
                off_refs.append(ocg_xrefs[number])
        if not off_refs:
            return doc

        catalog_xref = doc.pdf_catalog()
        off_value = "[" + " ".join(f"{ref} 0 R" for ref in off_refs) + "]"

        doc.xref_set_key(catalog_xref, "OCProperties/D/OFF", off_value)

        mutated_pdf = doc.write()
        doc.close()
        return fitz.open(stream=mutated_pdf, filetype="pdf")
    except Exception as e:
        print(f"Błąd ukrywania warstw: {e}")
        return doc


def pdf_to_png(
    pdf_path: str, page: int = 0, dpi: int = 300, hidden_layers: list[str] = None
) -> np.ndarray:
    """
    Konwertuje stronę PDF do obrazu OpenCV (BGR).
    Pozwala na wyłączenie wybranych warstw (hidden_layers) przed renderowaniem.
    """
    doc = _prepare_doc_with_hidden_layers(pdf_path, hidden_layers=hidden_layers)
    try:
        return _render_doc_to_bgr(doc, page=page, dpi=dpi)
    finally:
        doc.close()


def extract_legend(
    pdf_path: str,
    plan_image: np.ndarray,
    output_dir: str = "templates",
    dpi: int = 300,
    exclude_rects: list[tuple[int, int, int, int]] = None,
    legend_rect_px: tuple[int, int, int, int] | None = None,
    mask_mode: str = "auto",
    return_used_rect: bool = False,
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
    doc = fitz.open(pdf_path)
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
        table_trim = _trim_selection_to_table_grid(legend_area)
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
        symbol_mask, row_bboxes, row_label_hints = _color_classic_row_symbol_bboxes(
            raw_symbol_mask,
            text_words,
            x_start=x_start,
            y_start=y_start,
            scale=scale,
        )
        if row_bboxes:
            contours = [_rect_to_contour(rect) for rect in row_bboxes]
        else:
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

        # Łączymy wszystkie fragmenty tekstu (posortowane od lewej do prawej)
        if symbol_text:
            safe_name = symbol_text
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
        elif found_texts:
            found_texts.sort(key=lambda t: t[0])
            full_name = "_".join(t[1] for t in found_texts)
            safe_name = _sanitize_filename(full_name)
            filename = f"{counter:02d}_{safe_name}.png"
        else:
            safe_name = f"symbol_{counter:02d}"
            filename = f"{counter:02d}_{safe_name}.png"

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


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Ekstrakcja legendy wymaga ręcznego zaznaczenia strefy przez UI.")
    print("Uruchom frontend i użyj trybu 'Legenda' do zaznaczenia obszaru.")
