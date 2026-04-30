"""
legend_extractor.py — Automatyczna ekstrakcja wzorców symboli z legendy PDF.

Algorytm:
  1. Znajduje słowo "LEGENDA" w PDF (PyMuPDF).
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

# Minimalna gęstość pikseli — symbol musi mieć przynajmniej X% kolorowych
# pikseli w swoim bboxie, żeby nie złapać jednej kolorowej kreski jako symbolu
MIN_PIXEL_DENSITY = 0.05  # 5% — celowo nisko, żeby nie odrzucać cienkich symboli

# Tolerancje dopasowania tekstu z PDF (w punktach PDF)
TEXT_TOLERANCE_Y = 15  # ±15 pt w pionie
TEXT_MAX_DISTANCE_X = 250  # max 250 pt w prawo
TEXT_MIN_OVERLAP_X = -15  # lekkie najście tekstu na symbol dozwolone

# Maksymalna długość nazwy pliku
MAX_FILENAME_LENGTH = 80


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


def _hsv_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Tworzy binarną maskę kolorowych pikseli (HSV)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)


def _ink_mask(image_bgr: np.ndarray, threshold: int = 238) -> np.ndarray:
    """Create a binary mask for dark ink in gray/black PDFs."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


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
            tx > x + w
            and abs((ty + th / 2) - center_y) <= max(24, h * 1.6, th * 1.6)
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
        if end - start >= min_gap_width
        and int(legend_w * 0.08) <= start <= int(legend_w * 0.55)
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
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)


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

    if not row_spans:
        return symbol_mask, []

    row_spans.sort(key=lambda item: item[2])
    grouped: list[list[tuple[float, float, float]]] = []
    for span in row_spans:
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

    centers = [sum(item[2] for item in group) / len(group) for group in grouped]
    bands: list[tuple[int, int]] = []
    for idx, group in enumerate(grouped):
        if idx == 0:
            band_top = max(0.0, min(item[0] for item in group) - 34.0)
        else:
            band_top = (centers[idx - 1] + centers[idx]) / 2.0

        if idx == len(grouped) - 1:
            band_bottom = min(float(legend_h), max(item[1] for item in group) + 34.0)
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
    # get mojibake like "UKĹAD" instead of "UKŁAD". Repair that first when
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
    legend_keyword: str = "LEGENDA",
    legend_width_pt: float = 300,
    legend_height_pt: float = 550,
    exclude_rects: list[tuple[int, int, int, int]] = None,
    legend_rect_px: tuple[int, int, int, int] | None = None,
    mask_mode: str = "auto",
) -> list[ExtractedSymbol]:
    """
    Wyciąga wzorce symboli z legendy planu elektrycznego.

    Args:
        pdf_path:         Ścieżka do pliku PDF.
        plan_image:       Obraz planu jako BGR np.ndarray (ten sam DPI co poniżej).
        output_dir:       Folder docelowy na wzorce (tworzony automatycznie).
        dpi:              DPI użyte przy konwersji PDF → PNG.
        legend_keyword:   Słowo kluczowe do zlokalizowania legendy.
        legend_width_pt:  Szacowana szerokość legendy w punktach PDF.
        legend_height_pt: Szacowana wysokość legendy w punktach PDF.
        exclude_rects:    Strefy do zignorowania.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    text_blocks = page.get_text("blocks")

    # Aplikujemy strefy wykluczone do obrazu planu (żeby zamazać niechciane fragmenty legendy)
    if exclude_rects:
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(plan_image, (ex, ey), (ex + ew, ey + eh), (255, 255, 255), -1)

    # 1. Lokalizacja legendy
    if legend_rect_px is not None:
        zone_scale = dpi / 72.0
        found = [
            fitz.Rect(
                (legend_rect_px[0] / zone_scale) + 20,
                0,
                0,
                legend_rect_px[1] / zone_scale,
            )
        ]
    else:
        found = page.search_for(legend_keyword)
    if found is not None and not found:
        raise ValueError(f"Nie znaleziono słowa '{legend_keyword}' w PDF.")

    anchor = found[0]
    scale = dpi / 72.0

    # Wyliczamy piksele obszaru legendy z kotwicy
    x_start = int((anchor.x0 - 20) * scale)
    y_start = int(anchor.y1 * scale)
    width = int(legend_width_pt * scale)
    height = int(legend_height_pt * scale)
    if legend_rect_px is not None:
        width = int(round(legend_rect_px[2]))
        height = int(round(legend_rect_px[3]))

    # Zabezpieczenie przed wyjściem poza obraz
    x_start = max(0, min(x_start, plan_image.shape[1] - 1))
    y_start = max(0, min(y_start, plan_image.shape[0] - 1))
    y_end = min(y_start + height, plan_image.shape[0])
    x_end = min(x_start + width, plan_image.shape[1])
    legend_area = plan_image[y_start:y_end, x_start:x_end]
    if legend_area.size == 0:
        raise ValueError("Zaznaczona strefa legendy jest pusta albo poza obrazem.")

    # 2. Maska kolorowa + morphological CLOSE (klejenie symboli)
    raw_symbol_mask, _mask_used = _legend_symbol_mask(legend_area, mask_mode=mask_mode)
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
    counter = 1

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

        if symbol_image.size == 0:
            continue

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
        if found_texts:
            found_texts.sort(key=lambda t: t[0])
            full_name = "_".join(t[1] for t in found_texts)
            safe_name = _sanitize_filename(full_name)
            filename = f"{counter:02d}_{safe_name}.png"
        else:
            filename = f"{counter:02d}_nieznany_symbol.png"
            safe_name = "nieznany_symbol"

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

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "plan.pdf"

    print(f"Konwersja {pdf} → PNG (300 DPI)...")
    plan = pdf_to_png(pdf, dpi=300)

    print("Ekstrakcja legendy...")
    symbols = extract_legend(pdf, plan, output_dir="templates")

    print(f"\n{'NR':>3} | {'NAZWA':<50}")
    print("-" * 58)
    for s in symbols:
        print(f"{s.index:>3} | {s.name:<50}")
    print(f"\nZapisano {len(symbols)} wzorców do folderu 'templates/'.")
