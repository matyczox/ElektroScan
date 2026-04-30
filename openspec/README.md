# ElektroScan - OpenSpec

Katalog `openspec/` trzyma dokumentacje projektu dla ludzi i AI.

## Najpierw Czytaj

1. [current-context.md](current-context.md) - aktualny stan pracy, rozdzial
   color/gray, zasady bezpieczenstwa i kierunek dla szarych PDF.
2. [architecture.md](architecture.md) - szersza architektura projektu.
3. [workflow.md](workflow.md) - jak uruchamiac i testowac projekt.

Jesli starsze pliki sa sprzeczne z `current-context.md`, traktuj
`current-context.md` jako aktualniejsze zrodlo prawdy.

## Pliki

| Plik | Zawartosc |
| --- | --- |
| [current-context.md](current-context.md) | Aktualny kontekst roboczy dla AI |
| [gray-dark-ink-plan.md](gray-dark-ink-plan.md) | Plan zmiany dla szarych PDF: dark ink zones |
| [architecture.md](architecture.md) | Struktura projektu i pipeline detektora |
| [detection.md](detection.md) | Metryki detekcji, walidacja, promocje |
| [api.md](api.md) | Endpointy API i formaty odpowiedzi |
| [performance.md](performance.md) | Wydajnosc, env vars, diagnostyka |
| [known-issues.md](known-issues.md) | Znane problemy i przypadki regresyjne |
| [workflow.md](workflow.md) | Uruchamianie, testowanie, praca z AI |
| [decisions.md](decisions.md) | Decyzje architektoniczne i inwarianty |
| [devops.md](devops.md) | Docker, CI, lint, testy |
| [changelog.md](changelog.md) | Historia zmian |

## Goldeny

Committed snapshoty regresyjne sa w `backend/tests/golden/`.

- `viking_bronisze_e8_gray_first_pdf_100pct.json` - pierwszy szary PDF
  zaakceptowany jako 100% dla aktualnego celu.

`backend/analysis_debug/` jest tylko lokalna diagnostyka i nie powinien trafic
do commita.

## Minimalny Prompt Dla Nowego AI

```text
Pracujesz w C:\Users\Admin\Desktop\elektroskan_claude. Najpierw przeczytaj
openspec/current-context.md. Detektor ma rozdzielone wejscia color/gray, ale
wciaz ma wspolny pipeline. Nie hardcoduj koordynat ani nazw symboli. Gray PDF
tunuj tylko w gray-only sciezce. Kolorowy silnik ma zostac szybki i nietkniety.
Do diagnozy brakow uzywaj Inspektora ROI, nie przywracaj starego panelu
"Pokaz niepewne/brakujace".
```
