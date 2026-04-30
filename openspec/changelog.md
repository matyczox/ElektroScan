# Changelog — Decyzje i Zmiany Architektoniczne

Ten plik służy do logowania ważnych zmian i decyzji projektowych. Nie zastępuje git log — tu trafia kontekst, który nie jest oczywisty z kodu ani historii commitów.

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
