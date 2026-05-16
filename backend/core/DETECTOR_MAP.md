# Detector Map

Ten plik jest mapa kontekstu dla zmian w silniku. Ma pomoc szybko wejsc w
backend bez czytania kilku tysiecy linii naraz.

## Glowny przeplyw

- `detector_pipeline.py` jest orkiestratorem: przygotowuje maski, odpala scan,
  walidacje, clustering, postprocess i buduje `DetectionResult`.
- `detector_scanning.py` generuje kandydatow template matching.
- `detector_validation.py` sprawdza kandydatow po masce, coverage, purity i
  verification.
- `detector_clustering.py` wybiera lokalnych winnerow i tlumi konflikty.
- `detector_masks.py` buduje maski koloru/szarego tuszu i metryki fragmentow.
- `detector_pdf.py` czyta pomocniczy tekst PDF; tekst nie jest prawda wizualna.
- `detector_context.py` trzyma czyste helpery tokenow, bboxow i trace input.
- `detector_postprocess.py` trzyma finalne operacje po wyborze hitow.

## Invariants

- Nie dodajemy reguly detektora po koordynatach konkretnego PDF. Koordynaty moga
  byc tylko w goldenach, sentinelach i narzedziach diagnostycznych.
- PDF text w color path jest evidence/resolverem. Nie moze sam tworzyc finalnych
  kolorowych pictogramow rodzin `L`, `AW`, `EW`, `TB`.
- Exact token chroni przed substringami typu `RL3`/`PL3`, ale sam token nie
  wystarcza do zmiany klasy bez zgodnosci shape.
- Kolor maski sluzy do skanu i odciecia tla. Sam hue nie wybiera klasy symbolu.
- PZU jest caution/sentinel baseline. PW-E color i zaakceptowane gray goldeny sa
  release-gating.
- Refactor mechaniczny nie moze zmieniac snapshotow. Jesli wynik sie zmienia,
  najpierw naprawiamy refactor albo cofamy fragment.

## Refactor Order

1. `detector_pipeline.py`: wyciagac czyste helpery i postprocess, bez zmian progow.
2. `detector_clustering.py`: oddzielic geometrie, ranking winnera i clustering.
3. `detector_masks.py`: oddzielic mask building od metryk fragmentow/wave/rail.
4. `legend_extractor.py`: rozdzielic raster/table/OCR/vector.
5. `main.py`: rozbic routery i serwisy dopiero po ustabilizowaniu core.

## Required Smoke Before Commit

```powershell
py -3.11 backend\tools\run_local_golden_regression.py --fixture pzu_bydgoszcz_el02_color --fixture pw_e_01_rev2_color --fixture pw_e_02_rev2_color
py -3.11 -m pytest backend\tests -q
npm run build
```
