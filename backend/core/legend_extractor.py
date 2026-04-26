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

import cv2
import numpy as np
import fitz  # PyMuPDF
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


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
TEXT_TOLERANCE_Y = 15   # ±15 pt w pionie
TEXT_MAX_DISTANCE_X = 250  # max 250 pt w prawo
TEXT_MIN_OVERLAP_X = -15   # lekkie najście tekstu na symbol dozwolone

# Maksymalna długość nazwy pliku
MAX_FILENAME_LENGTH = 80


# ── Pomocnicze ─────────────────────────────────────────────────────────────

def _sanitize_filename(text: str) -> str:
    """Czyści tekst do bezpiecznej nazwy pliku (ASCII, underscory)."""
    text = text.strip().replace('\n', '_')
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '_', text)
    # Transliteracja polskich znaków (cv2.imwrite nie radzi z Unicode na Windows)
    _PL = str.maketrans('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ', 'acelnoszzACELNOSZZ')
    text = text.translate(_PL)
    return text[:MAX_FILENAME_LENGTH].strip('_')


def _hsv_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Tworzy binarną maskę kolorowych pikseli (HSV)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)


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
                if 'text' in conf:
                    layers.append({
                        "name": conf["text"],
                        "visible": conf.get("on", True)
                    })
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

def pdf_to_png(pdf_path: str, page: int = 0, dpi: int = 300, hidden_layers: list[str] = None) -> np.ndarray:
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
    exclude_rects: list[tuple[int, int, int, int]] = None
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
    found = page.search_for(legend_keyword)
    if not found:
        raise ValueError(f"Nie znaleziono słowa '{legend_keyword}' w PDF.")

    anchor = found[0]
    scale = dpi / 72.0

    # Wyliczamy piksele obszaru legendy z kotwicy
    x_start = int((anchor.x0 - 20) * scale)
    y_start = int(anchor.y1 * scale)
    width = int(legend_width_pt * scale)
    height = int(legend_height_pt * scale)

    # Zabezpieczenie przed wyjściem poza obraz
    y_end = min(y_start + height, plan_image.shape[0])
    x_end = min(x_start + width, plan_image.shape[1])
    legend_area = plan_image[y_start:y_end, x_start:x_end]

    # 2. Maska kolorowa + morphological CLOSE (klejenie symboli)
    color_mask = _hsv_mask(legend_area)
    glued_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, GLUE_KERNEL)

    # 3. Kontury na sklejonej masce (posortowane od góry do dołu)
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
        roi_mask = color_mask[y:y+h, x:x+w]
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
        mask_roi = color_mask[y1:y2, x1:x2]

        # Tworzymy czarny obraz wynikowy
        symbol_image = np.zeros_like(color_roi)
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
        ok, buf = cv2.imencode('.png', symbol_image)
        if ok:
            file_path.write_bytes(buf.tobytes())

        results.append(ExtractedSymbol(
            name=safe_name,
            image=symbol_image,
            index=counter,
            pixel_count=pixel_count,
        ))
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
