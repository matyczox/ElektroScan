"""
detector.py — Silnik detekcji symboli elektrycznych na planie.

Czysta wersja policz_v4.py (hybrydowy) z poprawkami:
  - Dual mask (precyzyjna + pogrubiona) z automatyczną decyzją per-wzorzec
  - Complexity sorting (największe symbole najpierw)
  - Destructive masking na OBU maskach
  - Bezpiecznik >1500 trafień
  - Walidacja gęstości pikseli (40%)
  - Odejmowanie 1 (legenda matchuje siebie)

Nowe w stosunku do oryginału:
  - Typowanie (dataclasses, type hints)
  - Konfiguracja przez stałe (nie hardcoded w kodzie)
  - Zwraca struktury danych zamiast drukowania do konsoli
  - Rozdzielone IO od logiki
"""

import cv2
import numpy as np
import os
import glob
from dataclasses import dataclass, field
from pathlib import Path


# ── Stałe konfiguracyjne ──────────────────────────────────────────────────

# Granice HSV (identyczne jak w legend_extractor — to fundament systemu)
HSV_LOWER = np.array([0, 30, 50])
HSV_UPPER = np.array([180, 255, 255])

# Kernel do pogrubiania cienkich linii (dilate)
DILATE_KERNEL = np.ones((3, 3), np.uint8)

# Rotacje — schematy elektryczne mają symbole tylko co 90°
ROTATIONS = [
    None,                           # 0° (oryginał)
    cv2.ROTATE_90_CLOCKWISE,        # 90°
    cv2.ROTATE_180,                 # 180°
    cv2.ROTATE_90_COUNTERCLOCKWISE, # 270°
]

# Próg dopasowania — osobny dla precyzyjnych i pogrubionych symboli
THRESHOLD_PRECISE = 0.65  # gniazda, wypusty (delikatne kształty)
THRESHOLD_DILATED = 0.60  # łączniki (cienkie linie, potrzebują luzu)

# Bezpiecznik: jeśli jedna rotacja generuje > tyle trafień, to wzorzec jest
# zbyt generyczny i zawiesi CPU. Pomijamy.
MAX_HITS_PER_ROTATION = 1500

# Walidacja: trafienie jest akceptowane tylko jeśli wycinek maski planu
# zawiera przynajmniej tyle % pikseli co wzorzec.
MIN_PIXEL_DENSITY_RATIO = 0.4

# Minimalny rozmiar wzorca po filtracji HSV (odrzuca śmieci)
MIN_TEMPLATE_PIXELS = 20

# Słowa kluczowe w nazwie pliku decydujące o trybie precyzyjnym
PRECISE_KEYWORDS = ["gniazdo", "wypust"]


# ── Struktury danych ──────────────────────────────────────────────────────

@dataclass
class TemplateInfo:
    """Załadowany wzorzec z metadanymi."""
    path: str
    name: str
    pixel_count: int        # ilość kolorowych pikseli po HSV
    mask: np.ndarray        # binarna maska HSV wzorca
    requires_precision: bool  # True = gniazdo/wypust → maska precyzyjna


@dataclass
class Detection:
    """Pojedyncze wykrycie symbolu na planie."""
    symbol_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0


@dataclass
class DetectionResult:
    """Wynik detekcji dla jednego typu symbolu."""
    symbol_name: str
    count: int  # ilość na planie (po odjęciu 1 za legendę)
    color: str = "#10b981"  # kolor hex dla frontendu
    detections: list[Detection] = field(default_factory=list)


# ── Filtracja HSV ─────────────────────────────────────────────────────────

def _hsv_mask(image_bgr: np.ndarray, dilate: bool = False) -> np.ndarray:
    """
    Tworzy binarną maskę kolorowych pikseli.

    Args:
        image_bgr: Obraz BGR.
        dilate:    Jeśli True, pogrubia linie (dla cienkich symboli).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    if dilate:
        mask = cv2.dilate(mask, DILATE_KERNEL, iterations=1)

    return mask


# ── Ładowanie wzorców ──────────────────────────────────────────────────────

def load_templates(folder: str) -> list[TemplateInfo]:
    """
    Ładuje wzorce z folderu, filtruje HSV, sortuje po złożoności (malejąco).

    Wzorce muszą być kolorowymi PNG (BGR). Puste po filtracji są pomijane.
    """
    paths = glob.glob(os.path.join(folder, "*.png"))
    templates: list[TemplateInfo] = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue

        name = Path(path).stem

        # Decyzja o trybie na podstawie nazwy
        name_lower = name.lower()
        requires_precision = any(kw in name_lower for kw in PRECISE_KEYWORDS)

        # Filtrujemy wzorzec — precyzyjne BEZ pogrubienia, reszta Z
        mask = _hsv_mask(img, dilate=not requires_precision)
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > MIN_TEMPLATE_PIXELS:
            templates.append(TemplateInfo(
                path=path,
                name=name,
                pixel_count=pixel_count,
                mask=mask,
                requires_precision=requires_precision,
            ))

    # Sortowanie malejące po złożoności — kluczowy niuans!
    templates.sort(key=lambda t: t.pixel_count, reverse=True)

    return templates


# ── Detekcja ───────────────────────────────────────────────────────────────

def detect_symbols(
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    subtract_legend: bool = True,
    exclude_rects: list[tuple[int, int, int, int]] = None
) -> list[DetectionResult]:
    """
    Wykrywa symbole na planie metodą Template Matching z HSV.

    Args:
        plan_image:       Kolorowy obraz planu (BGR).
        templates:        Lista załadowanych wzorców.
        subtract_legend:  Jeśli True, odejmuje 1 od wyniku.
        exclude_rects:    Lista stref do wyzerowania (x, y, w, h).
    """
    mask_precise = _hsv_mask(plan_image, dilate=False)
    mask_dilated = _hsv_mask(plan_image, dilate=True)

    for zone in (exclude_rects or []):
        ex, ey, ew, eh = zone
        cv2.rectangle(mask_precise, (ex, ey), (ex + ew, ey + eh), 0, -1)
        cv2.rectangle(mask_dilated, (ex, ey), (ex + ew, ey + eh), 0, -1)

    results: list[DetectionResult] = []

    SCALES = [0.90, 1.00, 1.10]  # 3 skale — łapie symbole lekko większe/mniejsze niż wzorzec

    for tpl in templates:
        # Wybór maski i progu w zależności od typu symbolu
        if tpl.requires_precision:
            plan_mask = mask_precise
            threshold = THRESHOLD_PRECISE
        else:
            plan_mask = mask_dilated
            threshold = THRESHOLD_DILATED

        # Przygotowanie 4 rotacji × 3 skale = 12 prób na wzorzec
        raw_hits: list[list[int]] = []
        hit_scores: dict[tuple, float] = {}

        for scale in SCALES:
            for rot in ROTATIONS:
                # Skalujemy maskę wzorca
                base_mask = tpl.mask
                if scale != 1.0:
                    new_w = max(1, int(base_mask.shape[1] * scale))
                    new_h = max(1, int(base_mask.shape[0] * scale))
                    scaled_mask = cv2.resize(base_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                else:
                    scaled_mask = base_mask

                # Rotacja
                if rot is not None:
                    rot_mask = cv2.rotate(scaled_mask, rot)
                else:
                    rot_mask = scaled_mask

                h, w = rot_mask.shape[:2]

                # Zabezpieczenie: wzorzec większy niż plan
                if h > plan_mask.shape[0] or w > plan_mask.shape[1]:
                    continue

                match_result = cv2.matchTemplate(plan_mask, rot_mask, cv2.TM_CCOEFF_NORMED)
                locations = np.where(match_result >= threshold)

                # Bezpiecznik przeciążeniowy
                if len(locations[0]) > MAX_HITS_PER_ROTATION:
                    continue

                for pt in zip(*locations[::-1]):
                    px, py = int(pt[0]), int(pt[1])
                    score = float(match_result[py, px])
                    # Normalizujemy rozmiar do skali 1.0 dla groupRectangles
                    norm_w = int(tpl.mask.shape[1])
                    norm_h = int(tpl.mask.shape[0])
                    raw_hits.append([px, py, norm_w, norm_h])
                    key = (px, py)
                    if key not in hit_scores or score > hit_scores[key]:
                        hit_scores[key] = score

        # Grupowanie duplikatów
        if len(raw_hits) == 0:
            continue

        grouped, _ = cv2.groupRectangles(raw_hits, groupThreshold=1, eps=0.5)
        if len(grouped) == 0 and len(raw_hits) > 0:
            grouped = raw_hits

        # Walidacja gęstości pikseli + destructive masking
        detections: list[Detection] = []

        for (x, y, w, h) in grouped:
            # Sprawdź czy w znalezionym prostokącie jest wystarczająco koloru
            roi = plan_mask[y:y+h, x:x+w]
            if cv2.countNonZero(roi) <= tpl.pixel_count * MIN_PIXEL_DENSITY_RATIO:
                continue

            # Najlepsze confidence dla tego prostokąta
            best_conf = hit_scores.get((int(x), int(y)), threshold)

            detections.append(Detection(
                symbol_name=tpl.name,
                x=int(x), y=int(y),
                width=int(w), height=int(h),
                confidence=round(best_conf, 3),
            ))

            # ── DESTRUCTIVE MASKING ──
            cv2.rectangle(mask_precise, (x, y), (x+w, y+h), 0, -1)
            cv2.rectangle(mask_dilated, (x, y), (x+w, y+h), 0, -1)

        # Odejmij 1 za legendę
        count = len(detections)
        if subtract_legend:
            count = max(0, count - 1)

        if count > 0:
            import hashlib
            h_val = int(hashlib.md5(tpl.name.encode()).hexdigest()[:6], 16)
            r = (h_val >> 16) & 0xFF | 0x40
            g = (h_val >> 8) & 0xFF | 0x40
            b = h_val & 0xFF | 0x40
            color_hex = f"#{min(r,255):02x}{min(g,255):02x}{min(b,255):02x}"

            results.append(DetectionResult(
                symbol_name=tpl.name,
                count=count,
                color=color_hex,
                detections=detections[:count] if subtract_legend else detections,
            ))

    return results



def draw_results(
    plan_image: np.ndarray,
    results: list[DetectionResult],
) -> np.ndarray:
    """
    Nanosi ramki detekcji na kopię planu (losowy kolor per symbol).
    Zwraca nowy obraz — nie modyfikuje oryginału.
    """
    output = plan_image.copy()

    for result in results:
        color = np.random.randint(0, 255, size=3).tolist()

        for det in result.detections:
            cv2.rectangle(
                output,
                (det.x, det.y),
                (det.x + det.width, det.y + det.height),
                color, 2,
            )

    return output


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    plan_path = sys.argv[1] if len(sys.argv) > 1 else "wygenerowany_plan_300dpi.png"
    templates_dir = sys.argv[2] if len(sys.argv) > 2 else "templates"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "wynik.png"

    print(f"Ładowanie planu: {plan_path}")
    plan = cv2.imread(plan_path)
    if plan is None:
        print(f"Błąd: nie można wczytać {plan_path}")
        sys.exit(1)

    print(f"Ładowanie wzorców z: {templates_dir}")
    templates = load_templates(templates_dir)
    print(f"Załadowano {len(templates)} wzorców (posortowanych po złożoności).\n")

    print(f"{'NAZWA':<45} | {'TYP':<8} | {'ILOŚĆ':>5}")
    print("-" * 65)

    results = detect_symbols(plan, templates)

    total = 0
    for r in results:
        mode = "[SNIPER]" if any(
            kw in r.symbol_name.lower() for kw in PRECISE_KEYWORDS
        ) else "[DILATE]"
        print(f"{r.symbol_name[:43]:<45} | {mode:<8} | {r.count:>5} szt.")
        total += r.count

    print("-" * 65)
    print(f"{'SUMA ELEMENTÓW':<45} | {'':8} | {total:>5} szt.")

    output_image = draw_results(plan, results)
    cv2.imwrite(output_path, output_image)
    print(f"\nZapisano wynik: {output_path}")
