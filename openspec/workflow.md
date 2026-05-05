# Workflow — Uruchamianie, Praca z AI, Debugowanie

## Jak Uruchomić

### Backend

```powershell
cd C:\Users\Admin\Desktop\elektroskanNOWYSWIEZYSTART\ElektroScan\backend
py -3 main.py
```

Backend: `http://127.0.0.1:8000`

### Frontend

```powershell
cd C:\Users\Admin\Desktop\elektroskanNOWYSWIEZYSTART\ElektroScan\frontend
npm run dev -- --host 127.0.0.1
```

Frontend: `http://127.0.0.1:5173`

## Aktualny Flow Legendy

1. Wgraj PDF. Nowy upload czysci baze wzorcow, wiec analiza nie uzyje
   template'ow z poprzedniego planu.
2. Zaznacz strefe legendy na canvasie trybem `Legenda`.
3. Kliknij `Wyciagnij legende z zaznaczenia`.
4. Przejdz przez panel `Sprawdz wzorce legendy`.
5. Dla kazdego wzorca wybierz jedna z akcji:
   - zaakceptuj,
   - popraw zaznaczenie na legendzie,
   - zmien nazwe,
   - odrzuc,
   - dodaj brakujacy wzorzec.
6. Analiza planu odblokuje sie dopiero wtedy, gdy nie ma juz wzorcow
   `pending`.

Wazne: reczny crop wzorca operuje na wspolrzednych podgladu PDF 300 DPI i
zapisuje wynik w `backend/templates/` dla aktualnej sesji.

### Restart Backendu (Windows)

Jeżeli wyniki wyglądają jak stary stan mimo zmian w kodzie — stary backend wisi na porcie 8000:

```powershell
$listeners = netstat -ano | Select-String ':8000' | ForEach-Object { ($_ -split '\s+')[-1] } | Where-Object { $_ -match '^\d+$' } | Sort-Object -Unique
foreach ($processId in $listeners) {
    if ([int]$processId -ne 0) {
        Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
    }
}
Start-Process -FilePath py -ArgumentList @('-3', 'main.py') -WorkingDirectory 'C:\Users\Admin\Desktop\elektroskanNOWYSWIEZYSTART\ElektroScan\backend' -WindowStyle Hidden
```

## Komendy Kontrolne (Przed i Po Zmianach)

**Przed zmianami:**
```powershell
git status --short
git log --oneline -5
py -3 -m compileall backend\core backend\main.py
cd frontend && npx tsc -p tsconfig.app.json --noEmit
```

**Po zmianach:**
```powershell
py -3 -m compileall backend\core backend\main.py
cd frontend
npx tsc -p tsconfig.app.json --noEmit
npx vite build
```

## Testy i Linting — Komendy

**Backend (uruchamiać z `backend/` przez venv):**
```bash
# Testy jednostkowe
venv/bin/python -m pytest tests/unit/ -v

# Testy z pokryciem
venv/bin/python -m pytest tests/unit/ --cov=core --cov-report=term-missing

# Linting
venv/bin/black --check core/ main.py tools/
venv/bin/isort --check-only core/ main.py tools/
venv/bin/flake8 core/ main.py tools/

# Formatowanie (jednorazowe po zmianach)
venv/bin/black core/ main.py tools/
venv/bin/isort core/ main.py tools/
```

**Frontend (uruchamiać z `frontend/`):**
```bash
npm test                    # vitest run (jednorazowo)
npm run test:watch          # vitest tryb watch (dev)
npm run test:coverage       # z raportem pokrycia
npm run lint                # eslint
npm run format:check        # prettier check
npm run format              # prettier write (formatowanie)
```

**Docker:**
```bash
docker compose up -d        # uruchom w tle
docker compose logs -f      # podgląd logów
docker compose down         # zatrzymaj
docker compose build --no-cache  # przebuduj od zera

# Skopiuj wzorce do kontenera po pierwszym uruchomieniu
docker compose cp ./backend/templates/. backend:/app/templates/
```

## GitHub Actions — Co To Jest

GitHub Actions to automat wbudowany w GitHub. Po każdym `git push` GitHub sam uruchamia pipeline na swoich serwerach — bez żadnej dodatkowej konfiguracji poza plikiem `.github/workflows/ci.yml`.

```
git push → GitHub widzi commit → uruchamia ci.yml → 5 jobów równolegle:
                                                      lint-backend (black, isort, flake8)
                                                      test-backend (pytest)
                                                      lint-frontend (tsc, eslint, prettier)
                                                      test-frontend (vitest)
                                                      docker-build (docker build check)
```

Widzisz ✅ lub ❌ przy każdym commicie i PR-ze na GitHub.

**Żeby zadziałało:** repo musi być na GitHubie. Plik `.github/workflows/ci.yml` już istnieje — wystarczy `git push`. Bezpłatne dla publicznych repozytoriów, dla prywatnych 2000 minut/miesiąc gratis.

## Commity Referencyjne

```text
6fb831a  Niepewne bledy HITL debug           ← aktualny, dodaje warstwę HITL
3186d5d  Progres tekstowy                    ← dobry punkt text-label
7d45d22  Mega Dobra optymalizacja-OBECNA     ← dobry punkt wydajnościowy
901c5b9  Bardzo dobra optymalizacja 2 bledy trzeba testow
b9b06cd  ogranicz parent search po klastrowaniu
97fc492  dodaj profil wydajnosci analizy
```

Jeżeli trzeba wrócić:
- `3186d5d` — dobry punkt text-label.
- `7d45d22` — dobry punkt optymalizacyjny.
- `6fb831a` — dodaje HITL, ale nie rozwiązuje jeszcze wszystkich braków.

## Jak Pracować z Codexem / AI

**Dobre polecenie startowe:**
```text
Przeczytaj openspec/README.md i openspec/architecture.md, sprawdź git log i aktualny branch.
Nie usuwaj reguł rodzinnych 06/07 i 10/11/12 bez testów.
Nie dodawaj reguł po koordynatach. Nie commituj backend/analysis_debug.
Najpierw reprodukuj na PW-E-01 Rev2 i PW-E-02 Rev2, potem rób małe zmiany.
```

**Minimalny prompt (mały kontekst agenta):**
```text
Najważniejsze: nie psuj 12 przy bbox 2293,1548 na PW-E-02; nie usuwaj reguł 06/07 i 10/11/12;
text labels bez mapy MSW=05; debug/HITL pokazuje braki, ale nie zamienia ich automatycznie
w final bez bezpiecznych progów.
```

**Jeżeli UI pokazuje stary stan:**
1. Sprawdź, czy backend na `8000` jest aktualny.
2. Ubij stary proces po PID.
3. Uruchom backend ponownie.
4. Zrób nową analizę i patrz na nowe `analysis_id`.

## Rytuał Debugowania Nowego Błędu

Nie zaczynać od pisania reguły. Najpierw zebrać dane:

1. Kliknij problematyczny box w UI.
2. Skopiuj debug payload.
3. Sprawdź `analysis_id`, `analysis_session`, `source_pdf`, `hidden_layers_used`.
4. Sprawdź `frontend_nearby_boxes`.
5. Sprawdź `frontend_nearby_debug_candidates`.
6. Porównaj z podobnym poprawnym przypadkiem z tego samego PDF-a.
7. Zadaj pytanie: różnica wynika z treści / ramki / koloru / overlapu / rotacji / skali / walidacji?
8. Dopiero potem zmieniaj próg albo logikę.

### Klasy Problemów

| Objaw | Przyczyna | Kierunek |
| --- | --- | --- |
| Template widzi, ale wybiera złą klasę | Konflikt klas | Poprawić ranking/verification |
| Template w ogóle nie widzi | Brak wariantu, zbyt niski match, zbyt mały ROI, zniekształcony obraz | Rozszerzyć warianty lub ROI |
| Finalny box jest, ale powinien być inny | Graniczny przypadek | `accepted_uncertain` + panel korekty |
| Kolorowy fragment bez boxa | `unexplained_component` | HITL, ewentualnie rescue parent |
| Mały fałszywy box w środku większego | `partial_ghost` | Overlap tłumiony przez parent |

### Format Zgłoszenia Błędu

```text
PDF:
Czy warstwy ukryte:
Oczekiwany symbol:
Aktualny symbol:
bbox:
debug payload:
screen/crop:
Poprawny podobny przykład:
```
