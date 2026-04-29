# Architektura ElektroScan

## Cel Projektu

ElektroScan wykrywa symbole instalacji elektrycznej na planach PDF/obrazach na podstawie wzorców z legendy. Nie jest detektorem pod jeden PDF — ma działać możliwie uniwersalnie na podobnych schematach, z trybem HITL dla granicznych przypadków.

Aktualny cel jakościowy:
- Automatycznie `90-95%+` poprawnych detekcji na znanych planach.
- HITL domyka pozostałe przypadki zamiast dopisywania wyjątków po koordynatach.
- Nie dążymy na siłę do 100% przez overfit pod dwa testowe PDF-y.

## Struktura Katalogów

```text
backend/
  main.py                        # FastAPI, upload, render, ekstrakcja legendy, analiza
  requirements.txt
  templates/                     # wzorce symboli (PNG/JPG), zarządzane przez API i UI
  uploads/                       # uploadowane PDF-y (tymczasowe, per session)
  analysis_debug/                # snapshoty JSON — NIE commitować
  core/
    __init__.py
    detector.py                  # główny pipeline detekcji (~988 linii)
    detector_config.py           # progi, skale, env vars (~171 linii)
    detector_models.py           # dataclassy: TemplateInfo, CandidateHit, Detection itd.
    detector_masks.py            # maski HSV, walidacja kandydata, ROI, content mask
    detector_templates.py        # ładowanie wzorców, warianty scale/rotation/mirror
    detector_clustering.py       # prefiltering, clustering, metryki overlap
    detector_promotions.py       # promocje rodzinne: core -> fuller parent
    detector_pdf.py              # pomocnicze PDF text i legend exclude (nie OCR prod)
    legend_extractor.py          # render PDF do obrazu, obsługa warstw, ekstrakcja legendy
  tools/
    compare_analysis_snapshot.py     # porównanie dwóch snapshotów JSON
    summarize_analysis_performance.py # profil wydajnościowy snapshotów

frontend/
  src/
    App.tsx                      # stan, requesty API, HITL state, ręczne boxy
    components/
      CanvasView.tsx             # render planu, finalne boxy, debug boxy, ręczny box
      ResultsPanel.tsx           # lista wyników, zmiana klasy, debug lista
      Sidebar.tsx                # upload, legenda, analiza, lista wzorców
      PatternModal.tsx           # modal edycji/usunięcia pojedynczego wzorca
      CostPanel.tsx              # kosztorys wykonawczy (ilość × cena PLN)
```

## Opis Modułów Backendu

### main.py
FastAPI. Obsługuje: upload PDF, render preview, ekstrakcję legendy, analizę, zarządzanie wzorcami (templates), snapshoty debug. Odpowiada za formatowanie odpowiedzi dla frontendu i asynchroniczny zapis snapshotów przez `SNAPSHOT_EXECUTOR`.

### core/legend_extractor.py
Renderowanie PDF do obrazu 300 DPI przez pymupdf/fitz. Obsługa warstw PDF (ukrywanie przed renderem). Ekstrakcja legendy z obrazu lub bezpośrednio z PDF.

### core/detector.py
Główna funkcja `detect_symbols()`. Ładuje template'y, buduje warianty, maskuje HSV, skanuje ROI przez `cv2.matchTemplate`, waliduje kandydatów, klastruje, generuje finalne `Detection`. Przy `include_debug=True` generuje `debugCandidates` dla HITL.

### core/detector_config.py
Wszystkie progi, skale, rotacje, limity. Konfigurowalny przez zmienne środowiskowe:
- `ELEKTROSCAN_DETECTOR_SCAN_WORKERS` — workerzy przy skanowaniu (domyślnie: liczba CPU)
- `ELEKTROSCAN_DETECTOR_POSTPROCESS_WORKERS` — workerzy postprocessingu (domyślnie: liczba CPU)
- `ELEKTROSCAN_OPENCV_THREADS` — wątki OpenCV (domyślnie: 1, unikamy konfliktu z Python MP)

### core/detector_models.py
Dataclassy: `TemplateInfo`, `TemplateVariant`, `CandidateHit`, `Detection`, `DetectionResult`. Czysta warstwa danych — nie zawiera logiki.

### core/detector_masks.py
Maski HSV, maski kolorów z template'u, walidacja kandydata (coverage, purity, context_purity, color_similarity), ROI na komponentach kolorowych, `content_mask` dla labeli tekstowych.

### core/detector_templates.py
Ładowanie plików z `backend/templates/`. Budowanie wariantów: scale × rotation × mirror. Rozpoznawanie label-like template'ów po geometrii i treści.

### core/detector_clustering.py
Prefiltering raw peaków, clustering kandydatów przez IoU/overlap, metryki do deduplicacji.

### core/detector_promotions.py
Rodzinne promocje: mniejszy rdzeń (`child/core`) może zostać zastąpiony pełniejszym symbolem (`parent`). Aktualnie ręczne reguły dla rodziny `06/07` i `10/11/12`. Nie usuwać bez gotowego mechanizmu ogólnego.

### core/detector_pdf.py
Pomocnicze: wyciąganie tekstu z warstwy PDF, wykluczanie strefy legendy. Nie używać jako głównego OCR produkcyjnego — produkcyjnie wejście może być skanem/zdjęciem.

## Opis Komponentów Frontendu

### App.tsx
Zarządza stanem: wyniki, HITL boxy, ręczne boxy, debugCandidates, wzorce. Komunikacja z API. Przekazuje props do wszystkich komponentów.

### CanvasView.tsx
Renderuje obraz planu (base64) na canvas. Rysuje zielone finalne boxy, czerwone/pomarańczowe debug boxy. Kliknięcie boxa kopiuje debug payload do schowka. Tryb ręcznego rysowania boxa.

### ResultsPanel.tsx
Lista finalnych wyników (pogrupowanych). Zmiana klasy boxa, usuwanie boxa. Lista HITL/debug kandydatów z przyciskami "Dodaj" / "Ukryj".

### Sidebar.tsx
Upload PDF, ekstrakcja legendy, uruchomienie analizy, wybór warstw. Lista załadowanych wzorców z miniaturą i przyciskiem edycji (otwiera PatternModal). Przycisk czyszczenia całej bazy wzorców.

### PatternModal.tsx
Modal do edycji nazwy wzorca lub jego usunięcia. Otwierany z Sidebar przy kliknięciu ikony edycji przy wzorcu.

### CostPanel.tsx
Panel kosztorysu. Dla każdego symbolu z wyników: ilość (readonly) + pole ceny netto PLN. Suma na dole. Stan cen żyje tylko w React — nie jest persystowany ani wysyłany do backendu.

## Pipeline Detekcji (Krok po Kroku)

1. PDF renderowany do obrazu 300 DPI.
2. Warstwy PDF mogą być ukryte przed renderem; projekt musi działać też bez idealnych warstw.
3. Template'y ładowane z `backend/templates/`.
4. Budowanie wariantów dla każdego template'u:
   - skale: `0.90`, `1.00`, `1.10`
   - rotacje: `0°`, `90°`, `180°`, `270°`
   - mirror: tylko dla wybranych rodzin (`06`, `07`, `09`, `10`, `11`, `12`) i labeli tekstowych
5. Obraz planu maskowany kolorem HSV.
6. Budowanie ROI na komponentach kolorowych — unikamy skanowania całego obrazu.
7. `cv2.matchTemplate` generuje raw peaki.
8. Raw peaki filtrowane (prefiltering).
9. Walidacja kandydatów:
   - `match_score`, `coverage`, `purity`, `context_purity`, `color_similarity`, `verification_score`
10. Promocje rodzinne (patrz `detection.md`).
11. Clustering → finalne `Detection`.
12. Przy `include_debug=True`: generowanie `debugCandidates` dla HITL.

## Warstwy PDF

Lokalna reprodukcja z warstwami może się różnić przez polskie znaki, normalizację Unicode, inne session_id. Do debug payload dodano: `analysis_session`, `source_pdf`, `hidden_layers_used`, `hidden_layers_unmatched`, `hidden_layers_repr` — żeby diagnozować rozbieżności między środowiskami.
