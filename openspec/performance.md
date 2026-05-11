# Wydajność i Narzędzia Diagnostyczne

## Wydajność Analizy

Historyczne: `54-58 s`. Po optymalizacjach: ~kilkanaście sekund na i7-13700KF.

### Najcięższe Etapy

- `scan` — template matching po całym obrazie
- `validation_targeted` — walidacja kandydatów z metrykami
- `parent_search` — szukanie parentów dla znalezionych rdzeni

CPU nie zawsze pokazuje 100% bo:
- część etapów jest I/O/serializacyjna,
- OpenCV puszcza native code,
- Python GIL blokuje łączenie wyników.

### Zasady Wydajnościowe

- Świeży run dla każdego wejścia — brak cache globalnego schematu.
- Można tworzyć lokalne struktury w pamięci w ramach jednego requestu.
- Nie optymalizować kosztem jakości bez benchmarku na znanych PDF-ach.
- Nie obniżać DPI poniżej 300 bez twardego testu jakości.

## Konfiguracja Wielowątkowości (Env Vars)

Zmienne środowiskowe dla `backend/core/detector_config.py`:

| Zmienna | Domyślna | Opis |
| --- | --- | --- |
| `ELEKTROSCAN_DETECTOR_SCAN_WORKERS` | liczba CPU | Workerzy przy skanowaniu template'ów |
| `ELEKTROSCAN_DETECTOR_POSTPROCESS_WORKERS` | liczba CPU | Workerzy postprocessingu |
| `ELEKTROSCAN_OPENCV_THREADS` | `1` | Wątki OpenCV (niska wartość unika konfliktu z Python MP) |

Przykład lokalny:

```bash
ELEKTROSCAN_DETECTOR_SCAN_WORKERS=8 ELEKTROSCAN_OPENCV_THREADS=2 \
  backend/venv/bin/python backend/main.py
```

## Narzędzia Diagnostyczne (backend/tools/)

Skrypty CLI uruchamiane ręcznie, poza normalnym flow backendu.

### compare_analysis_snapshot.py

Porównuje dwa snapshoty JSON z `analysis_debug/` — golden (wzorcowy) vs candidate (nowy run).

```bash
backend/venv/bin/python backend/tools/compare_analysis_snapshot.py golden.json candidate.json
backend/venv/bin/python backend/tools/compare_analysis_snapshot.py golden.json candidate.json \
  --focus 06,07,10,11,12 \
  --center-tolerance 18 \
  --size-tolerance 0.35
```

Wypisuje:
- liczby boxów w każdym pliku,
- dla wskazanych prefiksów rodzinnych: golden vs candidate ilości i delta,
- brakujące boxy (są w golden, nie ma w candidate),
- nadmiarowe boxy (są w candidate, nie ma w golden),
- konflikty klas — np. golden ma `06`, candidate ma `09` w tym samym miejscu,
- zmiany source — ten sam symbol, inna ścieżka skąd pochodzi.

**Kiedy używać:** Przed zmianą progu — zapisać snapshot z dobrze działającego run jako golden. Po zmianie — porównać nowy snapshot z golden.

### summarize_analysis_performance.py

Podsumowuje czasy etapów i countery z jednego lub wielu snapshotów.

```bash
backend/venv/bin/python backend/tools/summarize_analysis_performance.py backend/analysis_debug/
backend/venv/bin/python backend/tools/summarize_analysis_performance.py backend/analysis_debug/ --latest 3 --top 8
backend/venv/bin/python backend/tools/summarize_analysis_performance.py moj_snapshot.json
```

Wypisuje:
- `analysisId`, `sourcePdf`, łączny czas,
- top wolnych etapów backendu i detektora (ms),
- countery: `templatesLoaded`, `boxes`, `raw_peaks`, `validated_template_hits`, `final_hits` itd.

**Kiedy używać:** Sprawdzenie regresji wydajnościowej po zmianie konfiguracji wielowątkowej lub progów.

## Snapshot — Format i Lokalizacja

Przy `include_debug=true` backend zapisuje snapshot JSON do katalogu debug
aktualnego trybu:

- legacy/dev: `backend/analysis_debug/`
- projekt lokalny: `backend/data/projects/{project_id}/analysis_debug/`
- projekt Docker: `/app/data/projects/{project_id}/analysis_debug/`

Plik zawiera:

- `boxes` — finalne wykrycia
- `analysisContext` z `performance.backendTimingsMs`, `performance.detector.timingsMs`, `performance.backendCounters`
- `debugCandidates`

**Nie commitować `backend/analysis_debug/` ani `backend/data/`** — to lokalna
diagnostyka i dane robocze. Są w `.gitignore`.

## Committed Golden Snapshots

Committed goldeny regresyjne trzymamy osobno:

- `backend/tests/golden/viking_bronisze_e8_gray_first_pdf_100pct.json`

Porownanie aktualnego runu z goldenem:

```bash
backend/venv/bin/python backend/tools/compare_analysis_snapshot.py \
  backend/tests/golden/viking_bronisze_e8_gray_first_pdf_100pct.json \
  backend/analysis_debug/<analysis_id>.json \
  --focus 01,02,03,04,05,06,07 \
  --center-tolerance 20 \
  --size-tolerance 0.45
```

Smoke test po zmianach:

```bash
PYTHONPATH=backend backend/venv/bin/python -m pytest backend/tests/unit
cd frontend && npm run test -- --run
cd frontend && npm run build
```
