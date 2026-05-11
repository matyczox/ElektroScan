# Changelog — Decyzje i Zmiany Architektoniczne

Ten plik służy do logowania ważnych zmian i decyzji projektowych. Nie zastępuje git log — tu trafia kontekst, który nie jest oczywisty z kodu ani historii commitów.

## 2026-05-11 — Projekty stabilne po powrocie i kompletna legenda

**Dotyczy:** `backend/main.py`, `backend/core/legend_extractor.py`,
`backend/Dockerfile`, `frontend/src/App.tsx`,
`frontend/src/components/CanvasView.tsx`,
`frontend/src/components/LegendReviewPanel.tsx`,
`frontend/src/components/ResultsPanel.tsx`,
`frontend/src/symbolLabels.ts`, OpenSpec

- Powrót do istniejącego projektu przywraca podgląd PDF, warstwy, zaznaczoną
  legendę, wzorce i ostatnią zakończoną analizę. Wyjście z projektu nie kasuje
  ani nie anuluje trwającej analizy.
- Projekt z już zaznaczoną i sprawdzoną legendą może od razu uruchomić analizę
  planu po ponownym wejściu.
- Usunięto wymóg weryfikacji e-mail z flow auth.
- Poprawiono ekstrakcję legend gray/raster: opisy mogą być czytane przez
  Tesseract OCR, a nazwy są normalizowane do czytelnych etykiet zamiast
  `nieznany_symbol`.
- Poprawiono legendy tabelaryczne i klasyczne: `C1`/`D1` trzymają właściwy
  kwadrat, `GSW`/`MSW` nie przejmują nazw z sąsiednich wierszy, a pary typu
  `A + kółko` oraz `B + kwadrat` nie są mieszane.
- Dodano przyjazne etykiety i ręczną zmianę nazw w wynikach oraz panelu
  review; stare zapisane złe wzorce mogą wymagać ponownego wyciągnięcia legendy.
- Nie dodano żadnych reguł po koordynatach konkretnego PDF.
- Weryfikacja po zmianach: backend unit `74 passed`, frontend vitest
  `18 passed`, frontend build OK.

## 2026-05-10 — Nazwy wzorców z tekstu legendy i poprawka edycji projektów

**Dotyczy:** `backend/core/legend_extractor.py`,
`frontend/src/components/ProjectDashboard.tsx`, `frontend/src/index.css`, OpenSpec

- Formularz edycji projektu w dashboardzie nie trzyma już pól i przycisków w
  jednym rzędzie; akcje zapisu/anulowania przechodzą pod pola i zawijają się,
  więc długi opis nie rozlewa layoutu poza kartę projektu.
- Ekstrakcja legend tabelarycznych próbuje nazwać wzorzec opisem z tego samego
  wiersza tabeli na podstawie `page.get_text("words")`.
- Wiodące liczniki/kody typu `01` albo `A1` są pomijane, jeżeli za nimi jest
  właściwy opis. Fallback pozostaje: `_get_row_index_text`, potem `sym_XX`.
- Dodano unit testy dla mapowania opisu wiersza i tabeli z lewą ramką.

## 2026-05-09 — Rozszerzenie auth, sesji i dashboardu projektów

**Dotyczy:** `backend/auth_store.py`, `backend/main.py`, frontend dashboard, OpenSpec

- Dodano tokeny jednorazowe `password_reset`.
- Dodano endpointy profilu, resetu hasła, listy sesji, usuwania sesji i
  wylogowania ze wszystkich sesji.
- Reset hasła usuwa aktywne sesje użytkownika.
- Dodano historię analiz projektu przez `/api/projects/{project_id}/analysis-runs`.
- Dashboard projektów dostał wyszukiwanie, sortowanie, edycję/archiwizację,
  panel konta, aktywne sesje i historię analiz.
- Role/współdzielenie projektów pozostają następnym modułem; obecny model jest
  owner-only.

## 2026-05-09 — Logowanie i projekty MVP

**Dotyczy:** backend auth/storage, frontend flow, API

- Dodano `backend/auth_store.py`: SQLite, użytkownicy, PBKDF2 password hash,
  sesje `HttpOnly` cookie, projekty, sesje uploadu i rejestr analiz.
- Dodano endpointy `/api/auth/*` oraz `/api/projects/*`.
- Dodano projektowe odpowiedniki workflow: upload, layers, render-preview,
  extract-legend, analyze, inspect-roi, gray-debug-zones i templates.
- Frontend startuje od logowania/rejestracji, potem pokazuje dashboard
  projektów. Workspace detekcji działa po wyborze projektu.
- Dane robocze projektu są izolowane w `backend/data/projects/{project_id}/`.
  Stare endpointy bez `project_id` zostają jako legacy/dev fallback.

## Format Wpisu

```markdown
### YYYY-MM-DD — Krótki opis zmiany

**Commit(y):** `hash — opis`
**Dotyczy:** plik/moduł
**Zmiana:** Co konkretnie zostało zmienione.
**Dlaczego:** Motywacja — jaki problem to rozwiązuje.
**Ryzyko:** Co może się popsuć; jak sprawdzić regresję.
**Złote przypadki:** Które boxy regresyjne trzeba sprawdzić.
```

---

## 2026-04-29 — Reorganizacja OPEN_SPEC do katalogu openspec/

**Dotyczy:** dokumentacja projektu
**Zmiana:** OPEN_SPEC.md rozbity na tematyczne pliki w `openspec/`. Root `OPEN_SPEC.md` stał się krótkim indeksem.
**Dlaczego:** Jeden plik rósł do ~700 linii, trudno było go aktualizować punktowo przy zmianach. Podział na moduły pozwala edytować `known-issues.md` bez ruszania `performance.md` itd.
**Ryzyko:** Brak — to tylko dokumentacja.

---

## 2026-04-29 — Warstwa HITL i debug candidates

**Commit:** `6fb831a — Niepewne bledy HITL debug`
**Dotyczy:** `backend/core/detector.py`, `frontend/src/App.tsx`, `frontend/src/components/CanvasView.tsx`, `ResultsPanel.tsx`
**Zmiana:** Dodano typy debug kandidatów: `accepted_uncertain`, `rejected_candidate`, `rejected_low_content`, `unexplained_component`, `overlap_conflict`, `partial_ghost`. Payload debug boxa zawiera teraz `frontend_nearby_boxes` i `frontend_nearby_debug_candidates`.
**Dlaczego:** Wcześniej kliknięcie finalnego boxa nie informowało o tym, że w sąsiedztwie jest `Brak?`. HITL wymaga kontekstu sąsiedztwa.
**Ryzyko:** Debug kandydaci mogą zalewać UI szumem przy złych progach.
**Złote przypadki:** `bbox=2293,1548` musi nadal być `12`. Brakujący `08` obok `06@1363,737` musi przynajmniej dać `Brak?`.

---

## 2026-04-XX — Text label pipeline

**Commit:** `3186d5d — Progres tekstowy`
**Dotyczy:** `backend/core/detector_masks.py`, `detector_templates.py`
**Zmiana:** Dodano `content_mask` dla symboli tekstowych (litery po odjęciu ramek/linii). Kandydat labela oceniany po treści, nie tylko po ramce.
**Dlaczego:** `MSW` i `GSW` mają podobne ramki — samo matching po ramce myliło `04/05`. `TSM` z przesuniętą kreską też przestał mylić.
**Ryzyko:** Agresywna zmiana `content_mask` może psuje rozróżnienie MSW/GSW.
**Złote przypadki:** MSW/GSW przy `bbox~2293,1856`. TM/TSM blisko siebie. INT/TV odwrócone.

---

## 2026-04-XX — Optymalizacja wydajności

**Commit:** `7d45d22 — Mega Dobra optymalizacja-OBECNA`
**Dotyczy:** `backend/core/detector.py`, `detector_config.py`
**Zmiana:** Wielowątkowość przez `ELEKTROSCAN_DETECTOR_SCAN_WORKERS` / `ELEKTROSCAN_DETECTOR_POSTPROCESS_WORKERS`. ROI na komponentach kolorowych zamiast full-scan. Ograniczony `parent_search` po klastrowaniu.
**Dlaczego:** Czas analizy wynosił ~54-58 s, po optymalizacji ~kilkanaście sekund.
**Ryzyko:** Agresywne obcięcie ROI może pominąć symbole poza głównymi komponentami kolorowymi.
**Złote przypadki:** Wszystkie z `known-issues.md`.

---

<!-- Dodawaj nowe wpisy na górze listy, po tej linii -->

## 2026-05-05 - Review legendy, reczny crop wzorcow i reset bazy na nowy PDF

**Dotyczy:** `backend/main.py`, `backend/core/legend_extractor.py`,
`frontend/src/App.tsx`, `frontend/src/components/CanvasView.tsx`,
`frontend/src/components/Sidebar.tsx`,
`frontend/src/components/LegendReviewPanel.tsx`, OpenSpec
**Zmiana:** Dodano obowiazkowy przeglad wzorcow legendy po ekstrakcji. Wzorce
maja statusy `pending`, `accepted`, `fixed`, `rejected`; analiza jest
zablokowana, dopoki sa pozycje `pending`. Uzytkownik moze zaakceptowac wzorzec,
odrzucic go, zmienic nazwe, poprawic crop rysujac prostokat na legendzie albo
dodac brakujacy wzorzec. Backend dostal `POST /api/templates/{template_name}/crop`
i `PATCH /api/templates/{template_name}`. Nowy upload PDF przez `/api/preview`
czysci baze wzorcow, a frontend nie laduje juz starych template'ow przy starcie.
**Dlaczego:** Stare wzorce z poprzedniego PDF-a mogly pozwolic na analize
nowego planu bez analizy legendy. Zly crop legendy jest single point of failure
dla calej detekcji, wiec uzytkownik musi potwierdzic albo poprawic kazdy znak.
**Ryzyko:** Review metadata jest na razie trzymane glownie po stronie frontendu;
reload strony w trakcie review moze wymagac ponownej ekstrakcji. Dodawanie
brakujacego wzorca uzywa prostego promptu.
**Testy:** `npm run build`, `npm run test` (`17` testow), `compileall`
backendowych plikow. Backendowe unity nadal maja znany niezalezny fail:
brak `DEBUG_CANDIDATES_LIMIT` w `detector_config`.

## 2026-05-05 - Legenda tabelaryczna: poprawa cropow C1/D1

**Dotyczy:** `backend/core/legend_extractor.py`
**Zmiana:** Ekstrakcja symboli z legend tabelarycznych przypisuje ciemne
komponenty w kolumnie symboli do najblizszego srodka wiersza, zamiast wycinac
sztywno granice miedzy liniami tabeli. Poprawiono przypadki, gdzie `C1` bylo
uciete od gory, a `D1` dostawalo fragment sasiadujacego napisu.
**Dlaczego:** W legendach tabelarycznych etykieta i symbol moga wychodzic poza
idealne granice komorki, a linie tabeli nie sa wystarczajaca prawda dla cropa.
**Ryzyko:** Nietypowe tabele bez wyraznych komponentow moga wymagac recznej
korekty w nowym panelu review.
**Testy:** `compileall`, lokalne testy smoke helperow ekstraktora oraz review
w UI.

## 2026-05-05 - UI: polskie znaki i stabilna baza wzorcow

**Dotyczy:** `frontend/src/components/CanvasView.tsx`,
`frontend/src/components/ResultsPanel.tsx`, `frontend/src/components/Sidebar.tsx`,
`frontend/src/index.css`
**Zmiana:** Poprawiono teksty zapisane jako mojibake (`Brak podglÄ...du` itd.)
na normalne UTF-8. Karta "Baza Wzorcow" ma teraz staly naglowek i przewijana
liste wzorcow, zeby licznik i ikony nie ucinaly sie przy wielu template'ach.
**Dlaczego:** UI bylo trudne do czytania, a panel bazy wzorcow rozjezdzal sie
po ekstrakcji wielu symboli.
**Ryzyko:** Brak znanego ryzyka poza zwyklym layoutem przy bardzo malych
wysokosciach okna.
**Testy:** `npm run build`, `npm run test`.

## 2026-05-01 - Golden dla pierwszego szarego PDF Viking

**Dotyczy:** `backend/tests/golden/`, `openspec/`
**Zmiana:** Dodano committed golden snapshot
`viking_bronisze_e8_gray_first_pdf_100pct.json` dla
`VIKING-BRONISZE-ELE-Rzuty-E8.pdf`. Snapshot ma `81` boxow i rozklad
`01:7, 02:8, 03:11, 04:12, 05:13, 06:14, 07:16`.
**Dlaczego:** Po pracy z Inspektorem ROI pierwszy szary PDF jest zaakceptowany
jako 100% aktualnego celu i trzeba go chronic przed regresjami.
**Ryzyko:** Golden obejmuje tylko pierwszy szary Viking. Inne szare PDF nadal
moga wymagac strojenia. Nie wolno podmieniac goldena bez sprawdzenia ROI.
**Zlote przypadki:** Porownywac przez
`backend/tools/compare_analysis_snapshot.py` z fokusem `01,02,03,04,05,06,07`.

## 2026-04-30 - Gray Viking: dark ink zones i trace faz

**Dotyczy:** `backend/core/detector_gray.py`, `backend/core/detector_masks.py`,
`backend/core/detector_pipeline.py`, Inspektor ROI, OpenSpec
**Zmiana:** Gray PDF dostal ciemne strefy tuszu kalibrowane z legendy, skanowanie
po `zone_raw` / `zone_suppressed`, fair peak budget per ROI dla wydluzonych
symboli, naprawe slepego odejmowania legendy w `format_results` oraz lagodniejsza
walidacje strong geometry przy pelnym `coverage`.
**Dlaczego:** Inspektor ROI pokazywal `PASS`, ale finalna analiza gubila trafienia
w pozniejszych fazach. Problemem nie byl jeden magiczny threshold, tylko fazy:
budget, globalny limit peakow, final formatting i zbyt twarde purity dla gray.
**Ryzyko:** Nadal mozliwe false-positive albo missy na innych szarych PDF. Nie
uogolniac wyniku Vikinga bez kolejnych testow.
**Zlote przypadki:** Viking gray: brakujace `04` ok. `6490,710`, `7469,4316`,
`3742,5948`; `05` ok. `8019,3324`, `4516,5850`, `2255,5851`; `07` ok.
`5789,1121`; `06` ok. `4520,4372`.

## 2026-04-29 — Wdrożenie Docker, linting, testów i GitHub Actions CI

**Dotyczy:** infrastruktura, `backend/`, `frontend/`, `.github/`
**Co zostało zrobione:**

Docker:
- `backend/Dockerfile` (Python 3.11-slim), `frontend/Dockerfile` (Node 24-alpine), `docker-compose.yml` z named volumes
- `opencv-python` → `opencv-python-headless` w `requirements.txt` (brak potrzeby GUI w kontenerze)
- `backend/.dockerignore`, `frontend/.dockerignore`

Linting Python — 0 błędów po wdrożeniu:
- `backend/pyproject.toml` (black line-length=100, isort profile=black, mypy ignore_missing_imports)
- `backend/.flake8` (max-line-length=100, exclude tools/)
- `backend/requirements-dev.txt` (black, isort, flake8, mypy, pytest, pytest-cov)
- Naprawione: usunięto nieużywane importy (JSONResponse, PDF_TEXT_MIN/MAX_TOKEN_LENGTH, _derive_text_tokens, os), zduplikowane importy w main.py, nieużywana zmienna `y` w detector_masks.py, `# noqa: E501` na niemożliwych do skrócenia liniach diagnostycznych

Linting Frontend:
- `frontend/.prettierrc`, `eslint-config-prettier` dodany do eslint.config.js jako ostatni wpis
- Skrypty: `npm run format`, `npm run format:check`

Testy backend — 43 testy, 0 błędów, Python 3.12 (venv w `backend/venv/`):
- `backend/pytest.ini` (pythonpath=., testpaths=tests)
- `tests/unit/test_detector_config.py` — 13 testów (stałe, env vars przez monkeypatch)
- `tests/unit/test_detector_models.py` — 11 testów (Detection, DetectionResult, CandidateHit, TemplateVariant)
- `tests/unit/test_detector_clustering.py` — 19 testów (_bbox_metrics, _box_center, _center_inside_box, _candidate_rank_key)

Testy frontend — 14 testów, 0 błędów:
- vitest + @testing-library/react + jsdom
- `frontend/src/tests/setup.ts`, `CostPanel.test.tsx` (7 testów), `PatternModal.test.tsx` (7 testów)
- `vite.config.mjs` rozszerzony o sekcję `test: { environment: 'jsdom', globals: true }`

GitHub Actions — `.github/workflows/ci.yml`:
- 5 jobów: lint-backend, test-backend, lint-frontend, test-frontend, docker-build
- Odpala się na push do main i codex-test-niewiadoma-optymalizacja, oraz na PR do main
- **Aktywne dopiero po push do GitHub** — lokalnie plik tylko leży

**Ryzyko:** mypy celowo pominięty w CI (patrz devops.md). Backend wymaga `venv` z Python 3.12 (system Python 3.9 nie obsługuje `@dataclass(slots=True)`).

## 2026-04-29 — Plan wdrożenia Docker, linting, testów i CI/CD

**Dotyczy:** infrastruktura projektu
**Zmiana:** Dodano `openspec/devops.md` z pełnym planem: Dockerfile dla backendu i frontendu, docker-compose.yml, konfiguracja black/isort/flake8/mypy, struktura testów pytest + vitest, workflow GitHub Actions.
**Dlaczego:** Projekt wymaga dockeryzacji (`docker compose up -d`), linterów i testów jako must-have przed dalszym rozwojem i ewentualnym wdrożeniem produkcyjnym.
**Kluczowe decyzje:**
- `opencv-python` → `opencv-python-headless` w kontenerze (brak potrzeby GUI/X11)
- `requirements-dev.txt` osobny od prod — linting i pytest nie trafiają do obrazu Docker
- Testy zaczynamy od `test_detector_clustering.py` — czysta logika, zero I/O, najłatwiejszy start
- Golden snapshot test opiera się na `compare_analysis_snapshot.py` który już istnieje
**Ryzyko:** Brak — to tylko plan i dokumentacja. Implementacja w osobnych krokach (patrz sekcja "Kolejność Wdrożenia" w devops.md).
