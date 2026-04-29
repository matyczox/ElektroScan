# ElektroScan — Indeks Dokumentacji

Katalog `openspec/` zawiera pełną dokumentację projektu podzieloną tematycznie. Każdy plik można czytać niezależnie.

## Pliki

| Plik | Zawartość |
| --- | --- |
| [architecture.md](architecture.md) | Struktura projektu, pipeline detektora, opis modułów |
| [detection.md](detection.md) | Metryki detekcji, text labels, rodzinne promocje, HITL i debug candidates |
| [api.md](api.md) | Endpointy API, zarządzanie wzorcami, CostPanel, frontend HITL |
| [performance.md](performance.md) | Wydajność, env vars, narzędzia diagnostyczne |
| [known-issues.md](known-issues.md) | Znane problemy, złote przypadki regresyjne |
| [workflow.md](workflow.md) | Jak uruchomić, jak pracować z AI/Codexem, rytuał debugowania |
| [decisions.md](decisions.md) | Inwarianty, czego nie robić, mapa ryzyk, plan prac |
| [devops.md](devops.md) | Docker, linting, testy, GitHub Actions — plan wdrożenia |
| [changelog.md](changelog.md) | Log zmian i decyzji architektonicznych |

## Stan Referencyjny

- Branch: `codex-test-niewiadoma-optymalizacja`
- Ostatni commit: `6fb831a Niepewne bledy HITL debug`
- Dobry punkt optymalizacyjny: `7d45d22 Mega Dobra optymalizacja-OBECNA`
- Lokalnie może istnieć `backend/analysis_debug/` — nie commitować.

## Pakiet Startowy

1. Przeczytaj ten indeks i przejrzyj `architecture.md`.
2. Uruchom `git status --short` — upewnij się że nie ma przypadkowych zmian.
3. Sprawdź `git log --oneline -8` — upewnij się że jesteś na właściwym branchu.
4. Odpal backend i frontend (instrukcja w `workflow.md`).
5. Wrzuć `PW-E-02 Rev2.pdf` → sprawdź `bbox=2293,1548` (ma być `12`).
6. Wrzuć `PW-E-01 Rev2 (1).pdf` → sprawdź labels `TM/MSW/TSM/INT/TV`.
7. Dopiero potem zmieniaj kod.

Minimalny prompt dla nowego Codexa/AI:

```text
Pracujesz nad ElektroScan. Przeczytaj openspec/README.md i openspec/architecture.md.
Detektor symboli elektrycznych z PDF/obrazu oparty o OpenCV/template matching i HITL.
Nie wolno hardcodować koordynat, nie opierać produkcyjnej logiki na PDF text layer,
nie commituj backend/analysis_debug. Przed zmianą sprawdź branch, status i commity.
Nie usuwaj reguł rodzinnych 06/07 i 10/11/12 bez testów regresji.
```
