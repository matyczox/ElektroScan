# Backend Debug Playbook

Ten plik ma byc szybkim context packiem do pracy nad silnikiem bez ponownego
czytania calego backendu.

## Start Od Regresji

Najpierw sprawdz baseline lokalnie:

```powershell
py -3.11 backend\tools\run_quality_gate.py
```

Oczekiwane wyniki:

- PZU EL_01 GNIAZDA color: `204/204`.
- PZU EL_02 color: `318/318`, manual sentinels fixed.
- PW-E-01 color: `151/151`.
- PW-E-02 color: `134/134`.

Po wiekszym checkpointcie:

```powershell
py -3.11 -m pytest backend\tests -q
cd frontend
npm run build
```

## Gdzie Patrzec

- PDF text i exact tokeny: `backend/core/detector_pdf.py`,
  `backend/core/detector_pdf_policy.py`.
- Glowny przeplyw detektora: `backend/core/detector_pipeline.py`.
- Maski planu: `backend/core/detector_plan_masks.py`,
  `backend/core/detector_mask_builders.py`.
- Color family postprocess: `backend/core/detector_color_resolvers.py`.
- Clustering i wybor winnera: `backend/core/detector_clustering.py`,
  `backend/core/detector_candidate_selection.py`.
- Legenda: `backend/core/LEGEND_MAP.md`, potem `legend_extractor.py` i
  wyspecjalizowane `legend_*`.
- API/export/template store: `backend/main.py`, `backend/api_preview_service.py`,
  `backend/api_legend_service.py`, `backend/api_debug_service.py`,
  `backend/analysis_export.py`, `backend/template_store.py`.

## Zasady Detektora

- Nie dodawaj reguly po koordynatach konkretnego PDF w silniku.
- Koordynaty sa dozwolone w sentinelach, goldenach i narzedziach diagnostycznych.
- PDF text jest evidence/resolverem, nie samodzielna prawda wizualna.
- Kolor maski sluzy do wyboru pikseli skanu; sam kolor nie wybiera klasy.
- PZU jest caution baseline. Nie aktualizuj go automatycznie po kazdym runie.
- Refactor nie moze zmieniac finalnych boxow.

## ROI / Manual Review

Gdy pojawi sie podejrzany diff:

1. Otworz kandydacki JSON z `backend/tests/output/quality_gate/local_regression/`
   albo `backend/analysis_debug/local_regression/`.
2. Sprawdz `source`, `bbox`, `raw_bbox`, `frontend_nearby_boxes`.
3. Uzyj Inspektora ROI w UI albo cropa z debug output.
4. Decyzja manualna ma trafic do sentinela tylko jako test, nie jako logika.

Dobry sentinel opisuje objaw:

- `require_near` dla oczekiwanego symbolu,
- `forbid_near` dla falszywego symbolu,
- `allow_any_near` tylko dla realnie niejednoznacznej rodziny.

Case report dla aktywnego PDF:

```powershell
py -3.11 backend\tools\build_pdf_case_report.py `
  backend\tests\fixtures\pzu_bydgoszcz_el01_gniazda_color\case_pack.json `
  --output-dir backend\tests\output\pzu_el01_case_report
```

Jesli chcesz zobaczyc, w ktorej fazie pipeline zgubil hit, najpierw wygeneruj
candidate JSON z trace:

```powershell
py -3.11 backend\tools\run_local_golden_regression.py `
  --fixture pzu_bydgoszcz_el01_gniazda_color `
  --trace-point 1570,700 `
  --trace-radius 90 `
  --output-dir backend\tests\output\trace_probe
```

## Gdy Refactor Zmienia Wynik

1. Nie aktualizuj goldena.
2. Porownaj diff z ostatnim zielonym commitem.
3. Jesli to import/cache/order effect, napraw refactor.
4. Jesli to realny bug silnika, zatrzymaj refactor i zrob osobny fix commit.
