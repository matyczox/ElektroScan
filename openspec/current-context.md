# Aktualny Kontekst Pracy

Ten plik jest szybkim startem dla Codexa, Claude i innych AI. Czytaj go przed
dlugim grzebaniem w detektorze. Jesli inne pliki OpenSpec sa sprzeczne z tym
plikiem, ten plik opisuje aktualniejszy stan.

## Workspace

- Aktywny folder projektu: `C:\Users\Admin\Desktop\elektroskan_claude`
- Nie pracowac w `C:\Users\Admin\Desktop\elektroskanNOWYSWIEZYSTART`, chyba ze
  uzytkownik jawnie o to poprosi.
- Przed zmianami uruchomic `git status --short`.
- Nie ruszac przypadkowych zmian w `.claude/skills`, `.agents/skills`,
  `skills-lock.json` bez wyraznej prosby.

## Aktualna Architektura Detektora

Glowne wejscia:

- `backend/core/detector.py` - publiczny router `detect_symbols(...)`.
- `backend/core/detector_color_engine.py` - wejscie dla kolorowych PDF.
- `backend/core/detector_gray_engine.py` - wejscie dla szarych PDF.

Wspolny pipeline:

- `backend/core/detector_pipeline.py` - orkiestrator faz.
- `backend/core/detector_scanning.py` - `cv2.matchTemplate`, skale, rotacje,
  raw candidates.
- `backend/core/detector_validation.py` - walidacja kandydatow i tanie promocje.
- `backend/core/detector_parent_search.py` - drozsze szukanie pelniejszych
  symboli; kolorowy profil ma to omijac.

Strategie i pomocnicy:

- `backend/core/detector_color.py` - logika/progi specyficzne dla kolorowych PDF.
- `backend/core/detector_gray.py` - logika/progi/budzety dla szarych PDF.
- `backend/core/detector_masks.py` - maski, walidacja geometryczna,
  `coverage/purity/context_purity`, ink/dark ink.
- `backend/core/detector_templates.py` - template loading, warianty skali,
  rotacji i mirror.
- `backend/core/detector_clustering.py` - prefilter, NMS, clustering.
- `backend/core/detector_promotions.py` - rodzinne promocje, np. `06 -> 07`.
- `backend/core/legend_extractor.py` - ekstrakcja legendy.
- `backend/core/roi_inspector.py` - Inspektor ROI.

## Zasady Bezpieczenstwa

- Nie dodawac regul po koordynatach.
- Nie kodowac zasad typu "09 zawsze znaczy 07".
- Kolorowy silnik musi pozostac szybki. Gray-only heurystyki nie moga
  spowalniac kolorowych PDF.
- Zmiany dla gray powinny isc przez `detector_gray.py`, `detector_gray_engine.py`
  albo jasno oznaczony warunek `detector_profile == "gray"`.
- `parent_search` ma pozostac w praktyce gray-only.
- Po zmianach uruchomic:

```powershell
py -3 -m compileall -q backend
cd frontend
npm run build
```

## Status UI

- Usunieto stary panel `Pokaz niepewne/brakujace` i debugCandidates z glownego
  UI. Nie przywracac tego jako domyslnej funkcji.
- Narzedziem diagnostycznym jest teraz Inspektor ROI.
- Inspektor ROI pokazuje lokalnie, co silnik widzi w zaznaczonym boxie:
  raw mask, scan mask, dark scan mask, peaki per scale, PASS/odrzuty.

## Szare PDF - Aktualny Kierunek

Plik roboczy: `VIKING-BRONISZE-ELE-Rzuty-E8.pdf`.

Co juz wiadomo:

- Szary Viking zaczal lapac sporo symboli po rozszerzeniu skali do okolic `0.50`.
- PlanZone pomaga odciac legendy, ramki i marginesy.
- Inspektor ROI czesto pokazuje `PASS` dla brakujacych symboli, wiec problem
  bywa po budzecie, prefilterze, clusterze albo konkurencji kandydatow.
- Symbole na szarych PDF sa zwykle czarniejsze niz linie architektoniczne.
  To sugeruje kierunek: skanowanie po ciemniejszym tuszu / dark ink zones,
  zamiast mielenia calej szarej maski planu.

Obecne problemy:

- False positive `02/03` budowane z jasnych szarych linii planu.
- `01` lapie fragmenty tekstu lub pojedyncze kreski.
- Niektore `04/05/06` odpadaja, bo przez symbol przechodzi jasniejsza linia
  architektoniczna i psuje shape/match.
- Gray jest nadal zbyt compute-heavy na duzym PDF.

Preferowany nastepny eksperyment:

- Dla gray budowac kandydackie strefy z bardzo ciemnego tuszu (`dark_zone`).
- `dark_zone` ma decydowac nie tylko gdzie skanowac, ale tez byc maska
  `matchTemplate` przez `zone_raw` / `zone_suppressed`.
- Szersze `dark_raw` / `dark_suppressed` zostaja diagnostyka/pomoca, ale nie
  powinny same produkowac trafien z jasnoszarych linii planu.
- Kandydat gray musi przejsc `dark_evidence`: bardzo czarny, niedylatowany tusz
  musi pokrywac wystarczajaca czesc template'u wewnatrz bboxa.
- Duzy tile fallback dla gray jest domyslnie wylaczony, bo robil nachodzace
  okna `512x512` i spowalnial skanowanie.
- Nie obnizac globalnie progow dla wszystkich symboli, bo to zwieksza false
  positives.
- Najpierw logowac i porownywac w Inspektorze ROI, potem dopiero zmieniac final.

Szczegolowy plan tej zmiany jest w:

- `openspec/gray-dark-ink-plan.md`

## Kolorowe PDF - Inwariant

Kolorowe PDF dzialaly dobrze i szybko przed praca nad gray. Przy zmianach gray:

- Nie zmieniac globalnych `SCALES` dla color.
- Nie wlaczac gray raw budget, dark ink, text suppression ani parent fallback dla color.
- Po wiekszej zmianie przetestowac przynajmniej jeden stary kolorowy PDF i
  sprawdzic czas oraz liczbe boxow.

## Minimalny Prompt Dla Nowego AI

```text
Pracujesz w C:\Users\Admin\Desktop\elektroskan_claude. Najpierw przeczytaj
openspec/current-context.md. Detektor ma rozdzielone wejscia color/gray, ale
wciaz ma wspolny pipeline. Nie hardcoduj koordynat ani nazw symboli. Gray PDF
tunuj tylko w gray-only sciezce. Kolorowy silnik ma zostac szybki i nietkniety.
Do diagnozy brakow uzywaj Inspektora ROI, nie przywracaj starego panelu
"Pokaz niepewne/brakujace".
```
