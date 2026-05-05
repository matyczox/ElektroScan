# Plan: Legenda Manualna + Ekstrakcja z Tabelki

## Kontekst i Motywacja

Obecny flow ma dwa problemy:

1. **Auto-legenda jest zawodna.** Szuka słowa "LEGENDA" w PDF, zakłada stałe `legend_width_pt` /
   `legend_height_pt` i działa tylko dla legendy w klasycznym układzie (symbol po lewej, tekst po
   prawej). Na PDFach bez tekstowego słowa "LEGENDA" lub z niestandardowym układem — odpada.

2. **Legendy tabelaryczne nie są obsługiwane.** Format:

   ```
   [symbol] | Indeks | Producent | Nazwa artykułu
   [symbol] | A1     | BEE LIGHT | ASTER CC N PC …
   [symbol] | A2     | BEE LIGHT | ASTER CC N PC …
   …
   ```

   Symbole są w pierwszej kolumnie siatki (niekiedy wystawione poza lewą krawędź tabeli).
   Obecny ekstraktor HSV/gray nie radzi z tą strukturą — łapie fragmenty siatki, tekst
   kolumn, albo nie wyciąga nic.

**Cel tego planu:**

- Wymagać ręcznego zaznaczenia strefy legendy (rysowanie prostokąta na canvasie).
- Po zaznaczeniu i kliknięciu przycisku — analiza tej strefy i budowa bazy wzorców.
- Działać poprawnie zarówno dla klasycznych legend (symbol + tekst obok), jak i
  tabelarycznych (symbol w komórce siatki).

---

## Zasady Bezpieczeństwa (nie łamać)

- Bez hardkodowanych koordynat.
- Zmiany tylko w `legend_extractor.py` i warstwie UI (`Sidebar.tsx`, `App.tsx`).
  Pipeline detekcji (`detector*.py`) zostaje nienaruszony.
- Gray PDF / color PDF rozróżnienie zostaje — ekstraktor tabelaryczny musi działać
  dla obu typów (symbole w tabelce są zwykle czarne/szare, ale PDF może być kolorowy).

---

## Zakres Zmian

### 1. Frontend — wymagaj ręcznego zaznaczenia

**Plik:** `frontend/src/components/Sidebar.tsx`

Zmiana zachowania przycisku "1. Legenda":

| Stan | Obecne zachowanie | Nowe zachowanie |
|---|---|---|
| Brak strefy | Przycisk aktywny, label "Auto-Legenda" | Przycisk **nieaktywny**, tooltip "Zaznacz strefę legendy na planie" |
| Strefa zaznaczona | Przycisk aktywny, label "Legenda z zaznaczenia" | Przycisk aktywny, label "Wyciągnij legendę z zaznaczenia" |

Dodać pod przyciskiem uploadu krótki hint: `"Aby wyciągnąć wzorce, zaznacz strefę legendy na planie (tryb Legenda)"` — widoczny gdy brak strefy.

Nie usuwać istniejących propsów (`hasLegendZone`, `onClearLegendZone`) — zmiana tylko
w warunku aktywności przycisku.

**Plik:** `frontend/src/App.tsx`

Upewnić się, że przy braku `legendZone` wywołanie `onExtractLegend` jest zablokowane
(guard w handlerze, nie tylko w UI).

---

### 2. Backend — usuń auto-detekcję, wymagaj `legend_rect_px`

**Plik:** `backend/core/legend_extractor.py`

**Auto-detekcja (szukanie słowa "LEGENDA") zostaje całkowicie usunięta.**
Funkcja `extract_legend` bez `legend_rect_px` rzuca `ValueError` natychmiast — brak
żadnego fallbacku na keyword search.

Parametry `legend_keyword`, `legend_width_pt`, `legend_height_pt` są usuwane z
sygnatury (nie są już potrzebne). `legend_rect_px` staje się **wymaganym** argumentem
(lub co najmniej rzuca błąd gdy `None`).

Nowa logika początku funkcji:

```python
if legend_rect_px is None:
    raise ValueError(
        "legend_rect_px jest wymagane. Zaznacz strefę legendy na planie przed ekstrakcją."
    )

scale = dpi / 72.0
x_start = int(legend_rect_px[0])
y_start = int(legend_rect_px[1])
width   = int(legend_rect_px[2])
height  = int(legend_rect_px[3])
# anchor_rect_pt — do dopasowania text_blocks z fitz (w pt PDF)
anchor_rect_pt = fitz.Rect(
    x_start / scale,
    y_start / scale,
    (x_start + width) / scale,
    (y_start + height) / scale,
)
```

Reszta pipeline'u (wycinanie `legend_area`, maska, kontury, text matching) używa
`x_start`, `y_start`, `width`, `height` i `anchor_rect_pt` — bez dalszych zmian.

**Miejsca do sprawdzenia po usunięciu parametrów:**
- `main.py` — wywołanie `extract_legend(...)`: upewnić się że przekazuje `legend_rect_px`
  i nie przekazuje już `legend_keyword` / `legend_width_pt` / `legend_height_pt`.
- Testy jednostkowe (jeśli istnieją dla `legend_extractor`) — zaktualizować sygnatury.

---

### 3. Backend — wykrywanie i ekstrakcja legendy tabelarycznej

**Plik:** `backend/core/legend_extractor.py`

#### 3a. Detekcja formatu legendy

Nowa funkcja `_detect_legend_format(legend_area_bgr) -> Literal["table", "classic"]`.

Algorytm:
1. Konwertuj do skali szarości, binaryzuj (Otsu lub stały próg ~180).
2. Eroduj horyzontalnie (kernel `1×(width*0.6)`): zostają tylko długie poziome linie.
3. Policz linie wynikowe (`cv2.findContours` lub sumy wierszy).
4. Jeśli `≥ 3` linii o długości `> 50% szerokości strefy` → format `"table"`.
5. Inaczej → format `"classic"` (obecna ścieżka).

Parametr wyjście: `"table"` lub `"classic"`.

#### 3b. Ekstrakcja tabelaryczna — `_extract_table_legend`

Nowa funkcja `_extract_table_legend(legend_area_bgr, text_blocks, x_start, y_start, scale) -> list[ExtractedSymbol]`.

**Krok 1 — znajdź wiersze tabeli**

```python
gray = cv2.cvtColor(legend_area_bgr, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

h, w = binary.shape
horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)

# Sumy wierszy → szczyty = poziome linie tabeli
row_sums = horiz_lines.sum(axis=1)
threshold = w * 0.5 * 255
line_rows = np.where(row_sums > threshold)[0]
# Grupuj bliskie wiersze w jedną linię (tolerancja 5px)
row_boundaries = _merge_close_indices(line_rows, gap=5)
```

`_merge_close_indices` — pomocnicza: grupuje indeksy odległe o `< gap` w jedną medianę.

**Krok 2 — znajdź kolumny (pionowe linie)**

Analogicznie z pionowym kernelem `(1 × h//2)`.
Wynikowe `col_boundaries` to lista `x` podziałów kolumn.

Jeśli brak wyraźnych pionowych linii — użyj heurystyki: pierwsza kolumna to
lewe `15%` szerokości strefy (symbole są wąskie).

**Krok 3 — wytnij komórkę symbolu z każdego wiersza**

```python
for i in range(len(row_boundaries) - 1):
    row_top    = row_boundaries[i]
    row_bottom = row_boundaries[i + 1]
    # Pierwsza kolumna (symbol)
    col_left  = 0
    col_right = col_boundaries[0] if col_boundaries else int(w * 0.15)

    cell = legend_area_bgr[row_top:row_bottom, col_left:col_right]
    # Usuń piksele siatki (skrajna ramka 2px)
    cell_inner = cell[2:-2, 2:-2]
    # Sprawdź czy komórka zawiera coś poza białym tłem
    if _cell_has_content(cell_inner):
        symbol_img = _tight_crop_symbol(cell_inner)
        name = _get_row_index_text(text_blocks, x_start, y_start, scale,
                                   row_top, row_bottom, col_right)
        yield ExtractedSymbol(image=symbol_img, name=name or f"sym_{i+1:02d}")
```

**Krok 4 — `_cell_has_content`**

Szybki test: konwertuj do szarości, policz piksele `< 200` (ciemne).
Minimalna gęstość: `2%` komórki. Odrzuć puste lub prawie puste komórki (nagłówek
tabeli, separator).

**Krok 5 — `_tight_crop_symbol`**

Istniejąca logika ciasnego wycinania (findNonZero na binarnej masce ciemnych
pikseli) — ta sama co w ścieżce klasycznej, tylko bez HSV (symbole tabelaryczne
są czarne, nie kolorowe).

Maska dla symboli tabelarycznych: piksele `< 160` w szarości (czarne linie symbolu).
Margines `SYMBOL_PADDING = 4px` (nieco więcej niż klasyczny 2px, bo komórki bywają
ciaśniejsze).

**Krok 6 — `_get_row_index_text`**

Używa `text_blocks` z fitz (PDF text layer) do znalezienia krótkiego tekstu
(kodu indeksu, np. `A1`, `AW2`) w komórce drugiej kolumny tego samego wiersza.

```python
# Konwertuj px→pt
row_top_pt    = (y_start + row_top)    / scale
row_bottom_pt = (y_start + row_bottom) / scale
col2_left_pt  = (x_start + col_right)  / scale
col2_right_pt = col2_left_pt + 100  # max 100pt na kod indeksu

for block in text_blocks:
    bx0, by0, bx1, by1, text, *_ = block
    if by0 >= row_top_pt and by1 <= row_bottom_pt + 5:
        if bx0 >= col2_left_pt and bx1 <= col2_right_pt:
            candidate = text.strip().replace("\n", " ")
            if 1 <= len(candidate) <= 8:  # krótki kod, nie długa nazwa
                return _sanitize_filename(candidate)
return None
```

Fallback: jeśli text layer jest pusty lub brak match → `sym_01`, `sym_02`, …

#### 3c. Integracja w `extract_legend`

```python
legend_format = _detect_legend_format(legend_area)

if legend_format == "table":
    results = list(_extract_table_legend(
        legend_area, text_blocks, x_start, y_start, scale
    ))
else:
    # istniejąca ścieżka color / gray
    ...
```

Parametr `mask_mode` (color/gray/auto) zostaje — klasyczna ścieżka nadal go używa.
Ścieżka tabelaryczna ignoruje `mask_mode` (działa na ciemnych pikselach zawsze).

---

## Przypadki Brzegowe

| Sytuacja | Obsługa |
|---|---|
| Symbol wystaje poza lewą krawędź tabeli | Poszerzamy `col_left` o `10px` w lewo od lewego marginesu strefy (clamp do 0) |
| Wiersz nagłówka (tekst "Indeks", "Producent") | `_cell_has_content` zwraca False (tylko tekst, brak ciemnego symbolu graficznego) |
| Pusty wiersz / separator | `_cell_has_content` zwraca False |
| Brak text layer (skan) | `_get_row_index_text` zwraca None → fallback `sym_01`, `sym_02` |
| Kilka symboli w jednej komórce (stacked) | Rare; ciasne wycinanie zwraca cały content — użytkownik może ręcznie usunąć/podzielić przez PatternModal |
| Format mieszany (część klasyczna, część tabelaryczna) | Nie obsługujemy w pierwszej iteracji. Użytkownik zaznacza osobno każdą strefę. |

---

## Zachowanie Niezmienione

- `mask_mode` ("color" / "gray" / "auto") — klasyczna ścieżka bez zmian.
- `exclude_rects` — zamaływanie stref przed ekstrakcją, bez zmian.
- Zapis do `backend/templates/` i response API `/api/extract-legend` — bez zmian.
- Pipeline detekcji — całkowicie nienaruszony.
- Golden snapshoty — niezmienione, bo ekstrakcja legendy nie wchodzi w zakres testu goldenów.

---

## Kolejność Implementacji

1. **Krok 1** — Frontend: zablokuj przycisk gdy brak `legendZone`. Mały, natychmiastowy fix UX.
2. **Krok 2** — Backend: napraw istniejącą obsługę `legend_rect_px`. Konieczne zanim cokolwiek
   innego będzie działać dla ręcznych stref.
3. **Krok 3a** — Detekcja formatu (`_detect_legend_format`). Sprawdź na przykładowych PDFach
   klasycznych i tabelarycznych.
4. **Krok 3b** — Ekstrakcja tabelaryczna. Testuj na screenshocie legendy z pliku pokazanego
   przez użytkownika (BEE LIGHT / AWEX tabelka).
5. **Krok 3c** — Integracja. Połącz obie ścieżki w `extract_legend`.
6. **Test ręczny** — Wgraj PDF z legendą tabelaryczną, zaznacz strefę, sprawdź wyekstrahowane
   wzorce w Bazie Wzorców. Porównaj z oczekiwanymi symbolami.

---

## Pliki do Zmiany

| Plik | Rodzaj zmiany |
|---|---|
| `frontend/src/components/Sidebar.tsx` | Disabled button gdy brak strefy, hint tekstowy |
| `frontend/src/App.tsx` | Guard w handlerze `onExtractLegend` |
| `backend/core/legend_extractor.py` | Naprawa `legend_rect_px`, dodanie `_detect_legend_format`, `_extract_table_legend`, `_get_row_index_text`, `_cell_has_content`, `_merge_close_indices` |

Brak zmian w: `detector.py`, `detector_*.py`, `main.py`, `CanvasView.tsx`, `ResultsPanel.tsx`.
