# DevOps — Docker, Linting, Testy, CI/CD

## Stan Obecny

| Obszar | Status | Szczegóły |
| --- | --- | --- |
| Docker | ✅ Wdrożone | `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` |
| Linting Python | ✅ Wdrożone | black + isort + flake8, 0 błędów. mypy skonfigurowany, nie w CI (patrz niżej) |
| Linting Frontend | ✅ Wdrożone | ESLint + TS + Prettier, `eslint-config-prettier` |
| Testy backend | ✅ Wdrożone | 74 testy w `backend/tests/unit/`, pytest + pytest-cov |
| Testy frontend | ✅ Wdrożone | 18 testów w `frontend/src/tests/`, vitest + testing-library |
| GitHub Actions | ✅ Wdrożone | `.github/workflows/ci.yml`, 5 jobów, aktywne po push na GitHub |

### Uwaga o mypy

mypy jest zainstalowany (`requirements-dev.txt`) i skonfigurowany (`pyproject.toml`) ale **celowo pominięty w CI** na tym etapie. Powód: pierwsze uruchomienie na całym kodzie może generować dziesiątki błędów typowania wymagających ręcznych poprawek. Dodać do CI po uzupełnieniu adnotacji typów w krytycznych modułach (`detector.py`, `legend_extractor.py`).

Uruchomienie lokalne:
```bash
cd backend && venv/bin/mypy core/ main.py
```

## Kolejność Wdrożenia

1. **Docker** — bo wszystko inne (CI, testy) może go używać.
2. **Linting** — szybkie wygrane, fundament CI.
3. **Testy** — największy nakład pracy, zaczynamy od backendu.
4. **GitHub Actions** — spina wszystko razem.

---

## 1. Docker

### Problem z zależnościami systemowymi

`opencv-python` i `pymupdf` wymagają bibliotek systemowych. Zamiast `opencv-python` używamy `opencv-python-headless` (bez GUI) — mniejszy obraz, bez potrzeby libGL/X11.

Aktualizacja `backend/requirements.txt`: zamień `opencv-python` na `opencv-python-headless`.

### backend/Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libfontconfig1 \
    libfreetype6 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-pol \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads templates analysis_debug

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### frontend/Dockerfile

```dockerfile
FROM node:24-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
```

### docker-compose.yml (root projektu)

```yaml
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - backend_uploads:/app/uploads
      - backend_templates:/app/templates
      - backend_debug:/app/analysis_debug
      - backend_data:/app/data
    environment:
      - ELEKTROSCAN_OPENCV_THREADS=1
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "5173:5173"
    volumes:
      - ./frontend/src:/app/src
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  backend_uploads:
  backend_templates:
  backend_debug:
  backend_data:
```

### Uruchomienie

```bash
docker compose up -d          # uruchom w tle
docker compose logs -f        # podgląd logów
docker compose down           # zatrzymaj
docker compose build --no-cache  # przebuduj od zera
```

### .dockerignore — dodać w backend/ i frontend/

`backend/.dockerignore`:
```
__pycache__/
*.pyc
.venv/
venv/
analysis_debug/
uploads/
*.pdf
*.log
```

`frontend/.dockerignore`:
```
node_modules/
dist/
*.log
```

### Uwaga o volumes

`templates/`, `uploads/`, `analysis_debug/` i `data/` są w `.gitignore`
(słusznie), ale Docker potrzebuje ich persystencji między restartami — stąd
named volumes. Nowy flow projektów zapisuje najważniejsze dane w
`/app/data/projects/{project_id}/`.

Przy pierwszym `docker compose up` legacy globalne volumes będą puste; wzorce
najlepiej załadować przez UI. Ręczne kopiowanie jest tylko awaryjne:
```bash
docker compose cp ./backend/templates/. backend:/app/templates/
```

---

## 2. Linting

### Backend — Python

**Narzędzia:**
- `black` — formatter (nie wymaga decyzji stylistycznych)
- `isort` — sortowanie importów
- `flake8` — sprawdzanie stylu (E/W/F kody)
- `mypy` — statyczne typowanie

**Instalacja (dev only, nie w requirements.txt):**
```bash
pip install black isort flake8 mypy
```

Dodać `backend/requirements-dev.txt`:
```
black==24.4.2
isort==5.13.2
flake8==7.1.0
mypy==1.10.0
pytest==8.2.2
pytest-cov==5.0.0
```

**backend/pyproject.toml** (nowy plik):
```toml
[tool.black]
line-length = 100
target-version = ["py311"]

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_unused_ignores = true
# Na start nie strict — wprowadzamy stopniowo
# strict = true
```

**backend/.flake8** (nowy plik):
```ini
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude =
    __pycache__,
    .venv,
    venv,
    analysis_debug
```

**Komendy:**
```bash
cd backend
black --check .          # sprawdź bez zmian
black .                  # formatuj
isort --check-only .     # sprawdź
isort .                  # posortuj importy
flake8 .                 # sprawdź styl
mypy core/ main.py       # sprawdź typy
```

**Ważne:** Na początku `black .` i `isort .` zmodyfikują wiele plików — to jednorazowa operacja. Commit osobny: `style: apply black + isort formatting`.

### Frontend — TypeScript/ESLint

ESLint i TypeScript są już skonfigurowane. Dodać Prettier:

```bash
cd frontend
npm install -D prettier eslint-config-prettier
```

**frontend/.prettierrc** (nowy plik):
```json
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100
}
```

**Komendy:**
```bash
cd frontend
npx tsc -p tsconfig.app.json --noEmit   # sprawdź typy
npx eslint src/                          # sprawdź lint
npx prettier --check src/               # sprawdź format
npx prettier --write src/               # formatuj
```

---

## 3. Testy

### Backend — pytest

**Struktura:**
```
backend/
  tests/
    __init__.py
    unit/
      __init__.py
      test_detector_config.py
      test_detector_models.py
      test_detector_clustering.py
      test_detector_masks.py
      test_detector_promotions.py
    integration/
      __init__.py
      test_api_health.py
      test_api_templates.py
    golden/
      __init__.py
      test_golden_snapshots.py
      snapshots/
        pw_e_02_rev2_golden.json   # zapisać po dobrym runie
        pw_e_01_rev2_golden.json
```

**backend/pytest.ini** (nowy plik):
```ini
[pytest]
testpaths = tests
addopts = --tb=short -q
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

**Priorytety testów (od najważniejszego):**

1. `test_detector_config.py` — czy env vary są poprawnie czytane, defaults działają
2. `test_detector_models.py` — czy dataclassy tworzone poprawnie, pola są oczekiwane
3. `test_detector_clustering.py` — IoU, overlap, clustering (czysta logika, łatwe do testowania)
4. `test_detector_masks.py` — walidacja coverage/purity przy znanych danych
5. `test_api_health.py` — `GET /` zwraca 200, `GET /api/templates` zwraca listę
6. `test_golden_snapshots.py` — porównanie snapshotów z golden JSON (integracja z `compare_analysis_snapshot.py`)

**Przykład testu (test_detector_clustering.py):**
```python
import numpy as np
from core.detector_clustering import compute_iou  # lub odpowiednia funkcja

def test_iou_full_overlap():
    # dwa identyczne boxy -> IoU = 1.0
    box = (10, 10, 50, 50)
    assert compute_iou(box, box) == pytest.approx(1.0)

def test_iou_no_overlap():
    box_a = (0, 0, 10, 10)
    box_b = (20, 20, 10, 10)
    assert compute_iou(box_a, box_b) == pytest.approx(0.0)
```

**Komendy:**
```bash
cd backend
pytest                           # wszystkie testy
pytest tests/unit/               # tylko jednostkowe
pytest -v --cov=core --cov-report=term-missing  # z pokryciem
```

**Golden snapshot test** — po zapisaniu `pw_e_02_rev2_golden.json`:
```python
from backend.tools.compare_analysis_snapshot import compare_snapshots
from pathlib import Path

def test_golden_pw_e_02():
    result = compare_snapshots(
        Path("tests/golden/snapshots/pw_e_02_rev2_golden.json"),
        Path("backend/analysis_debug/latest.json"),  # najnowszy run
        focus_prefixes=("06", "07", "10", "11", "12"),
        center_tolerance=18.0,
        size_tolerance=0.35,
    )
    assert "Missing focus boxes: 0" in result
    assert "Extra focus boxes: 0" in result
```

### Frontend — vitest

```bash
cd frontend
npm install -D vitest @testing-library/react @testing-library/jest-dom @testing-library/user-event jsdom
```

**frontend/vite.config.mjs** — dodać sekcję test:
```js
export default defineConfig({
  // ... istniejąca konfiguracja ...
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/tests/setup.ts',
  },
})
```

**frontend/src/tests/setup.ts** (nowy):
```ts
import '@testing-library/jest-dom';
```

**Struktura:**
```
frontend/src/tests/
  setup.ts
  CostPanel.test.tsx
  PatternModal.test.tsx
  ResultsPanel.test.tsx
```

**Priorytety:**

1. `CostPanel.test.tsx` — render, zmiana ceny, suma całkowita (czysta logika, zero deps zewnętrznych)
2. `PatternModal.test.tsx` — render, kliknięcie Usuń, kliknięcie Zapisz
3. `ResultsPanel.test.tsx` — render listy wyników, zmiana klasy

**Komendy:**
```bash
cd frontend
npx vitest run          # jednorazowo
npx vitest              # watch mode
npx vitest --coverage   # z pokryciem
```

### E2E — Playwright (opcjonalnie, po podstawowych testach)

```bash
cd frontend
npm install -D @playwright/test
npx playwright install
```

**Minimalny e2e test:**
```ts
// frontend/e2e/basic.spec.ts
import { test, expect } from '@playwright/test';

test('strona główna ładuje się', async ({ page }) => {
  await page.goto('http://localhost:5173');
  await expect(page.locator('h1')).toBeVisible();
});
```

---

## 4. GitHub Actions

**Plik:** `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, codex-test-niewiadoma-optymalizacja]
  pull_request:
    branches: [main]

jobs:
  lint-backend:
    name: Lint — Python
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install lint deps
        run: pip install black isort flake8 mypy

      - name: black
        run: black --check backend/

      - name: isort
        run: isort --check-only backend/

      - name: flake8
        run: flake8 backend/

      - name: mypy
        run: mypy backend/core/ backend/main.py

  test-backend:
    name: Testy — Python
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install deps
        run: |
          pip install opencv-python-headless
          pip install -r backend/requirements.txt
          pip install -r backend/requirements-dev.txt

      - name: pytest
        run: |
          cd backend
          pytest tests/unit/ --tb=short -q

  lint-frontend:
    name: Lint — TypeScript
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: "24"
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: npm ci
        run: npm ci --prefix frontend

      - name: tsc
        run: npx --prefix frontend tsc -p tsconfig.app.json --noEmit

      - name: eslint
        run: npx --prefix frontend eslint src/

      - name: prettier
        run: npx --prefix frontend prettier --check src/

  test-frontend:
    name: Testy — Frontend
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: "24"
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: npm ci
        run: npm ci --prefix frontend

      - name: vitest
        run: npx --prefix frontend vitest run

  docker-build:
    name: Docker Build Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build backend image
        run: docker build -t elektroscan-backend ./backend

      - name: Build frontend image
        run: docker build -t elektroscan-frontend ./frontend
```

### Kiedy CI odpala się

- Push na `main` lub `codex-test-niewiadoma-optymalizacja`
- Każdy Pull Request do `main`

### Badges (opcjonalnie, do README.md)

```markdown
![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)
```

---

## Kolejność Wdrożenia (Tygodniowo)

### Krok 1 — Docker (zrób najpierw, sprawdź lokalnie)

1. Dodaj `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`.
2. Dodaj `.dockerignore` w obu katalogach.
3. Zmień `opencv-python` → `opencv-python-headless` w `requirements.txt`.
4. `docker compose up -d` — sprawdź czy backend na 8000 i frontend na 5173.

### Krok 2 — Linting (jeden commit na formatowanie)

1. Dodaj `backend/requirements-dev.txt`.
2. Dodaj `backend/pyproject.toml` i `backend/.flake8`.
3. Uruchom `black . && isort .` w backend — jeden osobny commit `style: initial formatting`.
4. Dodaj `frontend/.prettierrc` i `eslint-config-prettier`.
5. Uruchom `prettier --write src/` — osobny commit `style: initial prettier`.

### Krok 3 — Testy backend (zaczynaj od unit)

1. Utwórz `backend/tests/__init__.py`, `backend/pytest.ini`.
2. Napisz `test_detector_config.py` i `test_detector_models.py` — prosty start.
3. Napisz `test_detector_clustering.py` — czysta logika, najłatwiejsze.
4. Zapisz golden snapshot z dobrego runu (`PW-E-02 Rev2.pdf` na `7d45d22`).

### Krok 4 — Testy frontend

1. Dodaj vitest, testing-library, `setup.ts`.
2. Zacznij od `CostPanel.test.tsx` — zero deps zewnętrznych.

### Krok 5 — GitHub Actions

1. Utwórz `.github/workflows/ci.yml`.
2. Zrób push i sprawdź czy CI przechodzi.
3. Napraw pierwsze błędy (mypy zwykle wymaga kilku poprawek).

---

## Słownik DevOps (dla tego projektu)

- `requirements.txt` — zależności produkcyjne (trafiają do Dockera).
- `requirements-dev.txt` — narzędzia deweloperskie (linting, testy) — nie trafiają do Dockera.
- `pyproject.toml` — konfiguracja black, isort, mypy.
- `.flake8` — konfiguracja flake8 (ma swój format, nie używa pyproject.toml).
- named volume — Docker przechowuje dane poza kontenerem (templates, uploads).
- `docker compose cp` — kopiowanie plików do/z kontenera.
