# Aktualny Kontekst Pracy

Ten plik jest szybkim startem dla Codexa, Claude i innych agentow. Jesli inne
pliki OpenSpec sa sprzeczne z tym plikiem, ten plik opisuje aktualniejszy stan.

## Workspace

- Aktywny repo root: `C:\Users\Admin\Desktop\elektroskanNOWYSWIEZYSTART\ElektroScan`.
- Glowna galaz robocza: `main`.
- Przed zmianami uruchom `git status --short --branch`.
- Backend zwykle dziala na `http://127.0.0.1:8000`, frontend na
  `http://127.0.0.1:5173`.
- Preferowane uruchomienie calego stacku: `docker compose up -d --build`.
- Nie commituj `backend/analysis_debug/`, `backend/tests/output/`,
  `backend/data/`, cache ani lokalnych PDF.

## Aktualny Stan Goldenow

Lokalny gate ma pozostac zielony przed commitem:

```powershell
py -3.11 backend\tools\run_quality_gate.py
```

Oczekiwane wyniki detektora:

- `pzu_bydgoszcz_el01_gniazda_color`: `204/204`
- `pzu_bydgoszcz_el02_color`: `318/318`
- `pw_e_01_rev2_color`: `151/151`
- `pw_e_02_rev2_color`: `134/134`

PZU EL01/EL02 sa caution baseline: chronia lokalnie przed regresja i pomagaja
debugowac, ale nie sa dowodem uniwersalnej prawdy dla wszystkich kolorowych PDF.
PW-E i zaakceptowane gray goldeny dalej pilnuja szerokiego zachowania.

## Architektura Detektora

Glowne wejscia:

- `backend/core/detector.py` - publiczny router `detect_symbols(...)`.
- `backend/core/detector_color_engine.py` - wejscie dla kolorowych PDF.
- `backend/core/detector_gray_engine.py` - wejscie dla szarych PDF.

Wspolny przeplyw:

- `backend/core/detector_pipeline.py` - orkiestracja faz.
- `backend/core/detector_scanning.py` - `cv2.matchTemplate`, skale, rotacje,
  raw candidates.
- `backend/core/detector_validation.py` i `detector_hit_validation.py` -
  walidacja kandydatow i powody odrzucen.
- `backend/core/detector_clustering.py` - lokalni winnerzy, NMS i konflikty.
- `backend/core/detector_color_resolvers.py` oraz wyspecjalizowane resolvery -
  guarded color-family postprocess.
- `backend/core/roi_inspector.py` - lokalna diagnostyka ROI.
- `backend/core/DETECTOR_MAP.md` - bardziej szczegolowa mapa faz.

## Invariants

- Detektor nie moze miec regul po koordynatach konkretnego PDF.
- Kolor maski sluzy do skanu i odciecia tla. Sam hue nie wybiera klasy.
- PDF text w color path jest evidence/resolverem, nie samodzielnym detektorem
  kolorowych pictogramow rodzin `L`, `AW`, `EW`, `TB`.
- Exact token chroni przed substringami typu `RL3`/`PL3`, ale token nie zmienia
  klasy bez zgodnosci shape.
- Nie koduj mapowan typu `L9 => L7` albo `magenta => 21`. Klasa ma wynikac z
  geometrii template-vs-ROI.
- Refactor mechaniczny nie aktualizuje goldenow. Jesli boxy sie zmieniaja,
  refactor jest bledny do czasu wyjasnienia.

## Debug Workflow

1. Dla nowego PDF najpierw zrob case pack, nie zmiane detektora.
2. Zbierz ROI jako `expected`, `wrong` albo `manual_check`.
3. Uruchom case report:

```powershell
py -3.11 backend\tools\build_pdf_case_report.py <case_pack.json> --analysis <candidate.json> --templates-dir <templates>
```

4. Jesli final nie pokazuje poprawnego symbolu, sprawdz trace etapami:
   `raw_scan -> raw_budget -> raw_prefilter -> validation -> clustering ->
   color/gray postprocess -> final`.
5. Jesli diff poza znanym ROI wyglada podejrzanie, wygeneruj crop i popros o
   manual check zamiast zgadywac prog.

## Frontend / Manual Review

- Aktualny flow jest human-in-the-loop: legenda musi byc sprawdzona przed
  analiza.
- Wyniki mozna edytowac w UI, dodawac manualne boxy, dopisac note i eksportowac
  review JSON obok XLSX.
- Inspektor ROI jest glownym narzedziem diagnostycznym: pokazuje raw mask,
  scan mask, color/dark scan mask, top candidates i reject reasons.
- `reviewStatus` i `note` w boxach sluza do budowania goldenow i case packow,
  nie do produkcyjnych heurystyk detektora.

## Gdzie Szukac

- Komendy i workflow: `backend/CONTEXT_PACK.md`, `backend/DEBUG_PLAYBOOK.md`.
- Golden policy: `backend/tests/golden/README.md`.
- Active EL01 case pack: `backend/tests/fixtures/pzu_bydgoszcz_el01_gniazda_color/case_pack.json`.
- Active IE.05.01 legend extraction case pack:
  `backend/tests/fixtures/ie0501_salon_color/case_pack.json`.
- Narzedzia: `backend/tools/run_quality_gate.py`,
  `backend/tools/run_local_golden_regression.py`,
  `backend/tools/build_pdf_case_report.py`,
  `backend/tools/check_manual_sentinels.py`.
