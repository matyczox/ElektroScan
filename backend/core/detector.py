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

# Granice HSV (ogólne — do wykrywania czy cokolwiek jest kolorowe)
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

# Próg dopasowania
THRESHOLD_PRECISE = 0.55
THRESHOLD_DILATED = 0.45

# Bezpiecznik: jeśli jedna rotacja generuje > tyle trafień, to wzorzec jest
# zbyt generyczny i zawiesi CPU. Pomijamy.
MAX_HITS_PER_ROTATION = 1500

# Walidacja gęstości — niższa wartość przepuści cienkie symbole (MSW, ramki)
MIN_PIXEL_DENSITY_RATIO = 0.15

# Minimalny rozmiar wzorca po filtracji HSV (odrzuca śmieci)
MIN_TEMPLATE_PIXELS = 20

# Słowa kluczowe w nazwie pliku decydujące o trybie precyzyjnym
PRECISE_KEYWORDS = ["gniazdo", "wypust"]

# Tolerancja koloru w HSV dla maski per-wzorzec (±hue, ±sat, ±val)
COLOR_HUE_TOLERANCE = 18
COLOR_SAT_TOLERANCE = 80
COLOR_VAL_TOLERANCE = 80


# ── Struktury danych ──────────────────────────────────────────────────────
# TemplateInfo jest teraz zdefiniowany pod load_templates (potrzebuje funkcji)

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
    count: int
    color: str = "#10b981"
    detections: list[Detection] = field(default_factory=list)


# ── Filtracja HSV ─────────────────────────────────────────────────────────

def _hsv_mask(image_bgr: np.ndarray, dilate: bool = False) -> np.ndarray:
    """
    Tworzy binarną maskę kolorowych pikseli (wszystkie kolory).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    if dilate:
        mask = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    return mask


def _dominant_hsv_color(image_bgr: np.ndarray) -> tuple[int, int, int] | None:
    """
    Zwraca dominujący kolor (H, S, V) z pikseli kolorowych w obrazie wzorca.
    Zwraca None jeśli brak kolorowych pikseli.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    colored_pixels = hsv[mask > 0]
    if len(colored_pixels) == 0:
        return None
    # Mediana jest odporna na outliers (np. antyaliasing na brzegach)
    h_med = int(np.median(colored_pixels[:, 0]))
    s_med = int(np.median(colored_pixels[:, 1]))
    v_med = int(np.median(colored_pixels[:, 2]))
    return (h_med, s_med, v_med)


def _color_mask_for_template(
    image_bgr: np.ndarray,
    dominant_hsv: tuple[int, int, int],
    dilate: bool = False
) -> np.ndarray:
    """
    Tworzy maskę binarną pasującą TYLKO do dominującego koloru wzorca.
    Dzięki temu czerwony wzorzec szuka tylko na czerwonych pikselach planu
    i NIE może przypadkowo usunąć niebieskich sąsiadów.
    """
    h, s, v = dominant_hsv
    # Zawijamy zakres Hue wokół 0/180 (czerwień)
    lower1 = np.array([max(0, h - COLOR_HUE_TOLERANCE), max(0, s - COLOR_SAT_TOLERANCE), max(0, v - COLOR_VAL_TOLERANCE)])
    upper1 = np.array([min(180, h + COLOR_HUE_TOLERANCE), min(255, s + COLOR_SAT_TOLERANCE), min(255, v + COLOR_VAL_TOLERANCE)])

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower1, upper1)

    # Obsługa zawijania Hue dla czerwieni (H blisko 0 lub 180)
    if h - COLOR_HUE_TOLERANCE < 0:
        lower2 = np.array([180 + h - COLOR_HUE_TOLERANCE, lower1[1], lower1[2]])
        upper2 = np.array([180, upper1[1], upper1[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))
    elif h + COLOR_HUE_TOLERANCE > 180:
        lower2 = np.array([0, lower1[1], lower1[2]])
        upper2 = np.array([h + COLOR_HUE_TOLERANCE - 180, upper1[1], upper1[2]])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower2, upper2))

    if dilate:
        mask = cv2.dilate(mask, DILATE_KERNEL, iterations=1)
    return mask


# ── Ładowanie wzorców ──────────────────────────────────────────────────────

@dataclass
class TemplateInfo:
    """Załadowany wzorzec z metadanymi."""
    path: str
    name: str
    pixel_count: int
    mask: np.ndarray
    requires_precision: bool
    image_bgr: np.ndarray        # oryginał do wyciągania koloru dominującego
    dominant_hsv: tuple | None   # dominujący kolor w HSV


def load_templates(folder: str) -> list[TemplateInfo]:
    """
    Ładuje wzorce z folderu, filtruje HSV, sortuje po złożoności (malejąco).
    Każdy wzorzec zapamiętuje swój dominujący kolor (H, S, V).
    """
    paths = glob.glob(os.path.join(folder, "*.png"))
    templates: list[TemplateInfo] = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue

        name = Path(path).stem
        name_lower = name.lower()
        requires_precision = any(kw in name_lower for kw in PRECISE_KEYWORDS)

        mask = _hsv_mask(img, dilate=not requires_precision)
        pixel_count = cv2.countNonZero(mask)

        if pixel_count > MIN_TEMPLATE_PIXELS:
            dominant_hsv = _dominant_hsv_color(img)
            templates.append(TemplateInfo(
                path=path,
                name=name,
                pixel_count=pixel_count,
                mask=mask,
                requires_precision=requires_precision,
                image_bgr=img,
                dominant_hsv=dominant_hsv,
            ))

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
    # ── Budowa masek kolorów per-wzorzec ──
    # Każdy kolor (czerwony, zielony, niebieski) dostaje osobną maskę.
    # Dzięki temu destructive masking czerwonego NIE zjada niebieskiego sąsiada.
    color_masks_cache: dict[str, np.ndarray] = {}

    def _get_plan_mask(tpl: TemplateInfo) -> np.ndarray:
        if tpl.dominant_hsv is not None:
            color_key = str(tpl.dominant_hsv)
            if color_key not in color_masks_cache:
                m = _color_mask_for_template(
                    plan_image, tpl.dominant_hsv,
                    dilate=not tpl.requires_precision
                )
                for zone in (exclude_rects or []):
                    ex, ey, ew, eh = zone
                    cv2.rectangle(m, (ex, ey), (ex + ew, ey + eh), 0, -1)
                color_masks_cache[color_key] = m
            return color_masks_cache[color_key]
        # Fallback: ogólna maska wszystkich kolorów
        fallback = _hsv_mask(plan_image, dilate=False)
        for zone in (exclude_rects or []):
            ex, ey, ew, eh = zone
            cv2.rectangle(fallback, (ex, ey), (ex + ew, ey + eh), 0, -1)
        return fallback

    # ── Faza 1: Zbieranie surowych trafień (RÓWNOLEGLE) ──
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    SCALES = [0.90, 1.00, 1.10]

    def _scan_template(tpl: TemplateInfo) -> tuple[TemplateInfo, list, dict]:
        """Skanuje plan jednym wzorcem. Zwraca surowe trafienia."""
        threshold = THRESHOLD_PRECISE if tpl.requires_precision else THRESHOLD_DILATED
        plan_mask = _get_plan_mask(tpl)

        raw_hits: list[list[int]] = []
        hit_scores: dict[tuple, float] = {}

        for scale in SCALES:
            for rot in ROTATIONS:
                base_mask = tpl.mask
                if scale != 1.0:
                    new_w = max(1, int(base_mask.shape[1] * scale))
                    new_h = max(1, int(base_mask.shape[0] * scale))
                    scaled_mask = cv2.resize(base_mask, (new_w, new_h),
                                             interpolation=cv2.INTER_NEAREST)
                else:
                    scaled_mask = base_mask

                if rot is not None:
                    rot_mask = cv2.rotate(scaled_mask, rot)
                else:
                    rot_mask = scaled_mask

                h, w = rot_mask.shape[:2]
                if h > plan_mask.shape[0] or w > plan_mask.shape[1]:
                    continue

                match_result = cv2.matchTemplate(
                    plan_mask, rot_mask, cv2.TM_CCOEFF_NORMED
                )
                locations = np.where(match_result >= threshold)

                if len(locations[0]) > MAX_HITS_PER_ROTATION:
                    continue

                for pt in zip(*locations[::-1]):
                    px, py = int(pt[0]), int(pt[1])
                    score = float(match_result[py, px])
                    # KLUCZOWA ZMIANA: używamy w, h z rot_mask (czyli po obrocie i skali)
                    raw_hits.append([px, py, w, h])
                    key = (px, py, w, h)
                    if key not in hit_scores or score > hit_scores[key]:
                        hit_scores[key] = score

        return (tpl, raw_hits, hit_scores)

    # Uruchamiamy skanowanie na wielu wątkach
    # (OpenCV zwalnia GIL w matchTemplate, więc threading daje realny speedup)
    import os as _os
    num_workers = min(len(templates), _os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        scan_results = list(pool.map(_scan_template, templates))

    # ── Faza 2: Globalny cross-template NMS ──
    # Zbieramy WSZYSTKIE kandydatury ze WSZYSTKICH wzorców do jednej puli,
    # a dopiero potem robimy jeden wspólny NMS. To eliminuje sytuację gdzie
    # dwa podobne wzorce (np. łącznik_1 i łącznik_2) matchują ten sam symbol
    # i obie detekcje przechodzą przez lokalne NMS — bo każdy robił NMS osobno.

    all_boxes: list[list[int]] = []
    all_scores: list[float] = []
    all_tpl_idx: list[int] = []  # który wzorzec wygenerował to trafienie

    for tpl_idx, (tpl, raw_hits, hit_scores) in enumerate(scan_results):
        if len(raw_hits) == 0:
            continue
        threshold = THRESHOLD_PRECISE if tpl.requires_precision else THRESHOLD_DILATED
        plan_mask = _get_plan_mask(tpl)

        for x, y, w, h in raw_hits:
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Walidacja gęstości już tu — żeby śmieci nie szły do globalnego NMS
            roi = plan_mask[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            if cv2.countNonZero(roi) <= templates[tpl_idx].pixel_count * MIN_PIXEL_DENSITY_RATIO:
                continue
            score = float(hit_scores.get((x, y, w, h), threshold))
            all_boxes.append([x, y, w, h])
            all_scores.append(score)
            all_tpl_idx.append(tpl_idx)

    if len(all_boxes) == 0:
        return []

    # Jeden globalny NMS na wszystkich wzorcach naraz (Standardowy IoU)
    global_indices = cv2.dnn.NMSBoxes(
        all_boxes, all_scores,
        score_threshold=min(THRESHOLD_PRECISE, THRESHOLD_DILATED),
        nms_threshold=0.30,  
    )

    # Filtr IoM (Intersection over Minimum Area) - usuwa małe ramki wewnątrz dużych
    final_indices = []
    if len(global_indices) > 0:
        flat_indices = global_indices.flatten().tolist()
        # Sortujemy po score malejąco, żeby faworyzować pewniejsze trafienia
        flat_indices.sort(key=lambda idx: all_scores[idx], reverse=True)
        
        keep_flags = [True] * len(flat_indices)
        for i in range(len(flat_indices)):
            if not keep_flags[i]: continue
            idx1 = flat_indices[i]
            x1, y1, w1, h1 = all_boxes[idx1]
            area1 = w1 * h1
            
            for j in range(i + 1, len(flat_indices)):
                if not keep_flags[j]: continue
                idx2 = flat_indices[j]
                x2, y2, w2, h2 = all_boxes[idx2]
                area2 = w2 * h2
                
                # Oblicz pole przecięcia
                inter_x = max(x1, x2)
                inter_y = max(y1, y2)
                inter_w = min(x1+w1, x2+w2) - inter_x
                inter_h = min(y1+h1, y2+h2) - inter_y
                
                if inter_w > 0 and inter_h > 0:
                    inter_area = inter_w * inter_h
                    min_area = min(area1, area2)
                    # Jeśli pokrywają się w 40% mniejszego pudełka
                    if inter_area / min_area > 0.40:
                        keep_flags[j] = False
                        
        final_indices = [flat_indices[i] for i in range(len(flat_indices)) if keep_flags[i]]

    # Grupujemy zwycięskie detekcje po wzorcu
    per_template: dict[int, list[Detection]] = {}
    for idx in final_indices:
        x, y, w, h = [int(v) for v in all_boxes[idx]]
        score = all_scores[idx]
        tpl_idx = all_tpl_idx[idx]
        det = Detection(
            symbol_name=templates[tpl_idx].name,
            x=x, y=y, width=w, height=h,
            confidence=round(score, 3),
        )
        per_template.setdefault(tpl_idx, []).append(det)

    results: list[DetectionResult] = []
    for tpl_idx, detections in per_template.items():
        tpl = templates[tpl_idx]
        count = len(detections)
        if subtract_legend:
            count = max(0, count - 1)
        if count > 0:
            results.append(DetectionResult(
                symbol_name=tpl.name,
                count=count,
                color="#22c55e",  # jeden kolor dla wszystkich — łatwiej debugować
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
