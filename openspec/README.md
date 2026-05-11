# ElektroScan - OpenSpec

Katalog `openspec/` trzyma dokumentację projektu dla ludzi i AI. Jeżeli dwa
pliki mówią co innego, najpierw ufaj `current-context.md`, potem temu plikowi,
a dopiero potem historycznym planom.

## Najpierw Czytaj

1. [current-context.md](current-context.md) - aktualny stan pracy, zasady
   bezpieczeństwa, auth/projekty, legenda i detektor.
2. [architecture.md](architecture.md) - szersza architektura backendu,
   frontendu i pipeline detekcji.
3. [workflow.md](workflow.md) - jak uruchamiać, testować i debugować projekt.
4. [api.md](api.md) - endpointy i kontrakty odpowiedzi.

## Pliki

| Plik | Zawartość |
| --- | --- |
| [current-context.md](current-context.md) | Aktualny kontekst roboczy dla AI |
| [architecture.md](architecture.md) | Struktura projektu i pipeline detektora |
| [api.md](api.md) | Endpointy API i formaty odpowiedzi |
| [workflow.md](workflow.md) | Uruchamianie, testowanie, praca z AI |
| [detection.md](detection.md) | Metryki detekcji, walidacja, promocje |
| [known-issues.md](known-issues.md) | Znane problemy i przypadki regresyjne |
| [performance.md](performance.md) | Wydajność, env vars, diagnostyka |
| [decisions.md](decisions.md) | Decyzje architektoniczne i inwarianty |
| [devops.md](devops.md) | Docker, CI, lint, testy |
| [changelog.md](changelog.md) | Historia zmian |
| [gray-dark-ink-plan.md](gray-dark-ink-plan.md) | Plan historyczny dla szarych PDF: dark ink zones |
| [legend-manual-table-plan.md](legend-manual-table-plan.md) | Plan ręcznego zaznaczania legendy i tabel |
| [legend-review-plan.md](legend-review-plan.md) | Plan przeglądu i korekty wzorców legendy |

## Aktualne Środowisko

- Workspace: `/Users/jakublewosz/Code/matiprojekt`
- Gałąź demo/produkcyjna: `main`
- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`
- Zalecane uruchomienie: `docker compose up -d --build`

## Goldeny

Committed snapshoty regresyjne są w `backend/tests/golden/`.

- `viking_bronisze_e8_gray_first_pdf_100pct.json` - pierwszy szary PDF
  zaakceptowany jako 100% dla aktualnego celu.

`backend/analysis_debug/` oraz `backend/data/projects/*/analysis_debug/` są
lokalną diagnostyką i nie powinny trafiać do commita.

## Minimalny Prompt Dla Nowego AI

```text
Pracujesz w /Users/jakublewosz/Code/matiprojekt. Najpierw przeczytaj
openspec/current-context.md. Projekt ma auth, dashboard projektów i endpointy
projektowe /api/projects/{project_id}/...; legacy endpointy bez project_id są
tylko fallbackiem dev. Nie hardcoduj koordynat ani nazw symboli pod jeden PDF.
Legenda działa przez zaznaczenie, review wzorców, OCR opisów i ręczną korektę.
Detektor ma rozdzielone wejścia color/gray i wspólny pipeline. Do diagnozy
używaj Inspektora ROI oraz testów, nie przywracaj starego panelu debugCandidates
jako domyślnej funkcji UI.
```
