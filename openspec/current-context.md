# Aktualny Kontekst Pracy

Ten plik jest szybkim startem dla Codexa, Claude i innych AI. Czytaj go przed
dlugim grzebaniem w detektorze. Jesli inne pliki OpenSpec sa sprzeczne z tym
plikiem, ten plik opisuje aktualniejszy stan.

## Workspace

- Aktywny folder projektu: `/Users/jakublewosz/Code/matiprojekt`.
- Gałąź demo/produkcyjna: `main`.
- Backend zwykle działa na `http://127.0.0.1:8000`, frontend na
  `http://127.0.0.1:5173`.
- Preferowane uruchomienie całości: `docker compose up -d --build`.
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
- Po zmianach funkcjonalnych uruchomic:

```bash
PYTHONPATH=backend backend/venv/bin/python -m pytest backend/tests/unit
cd frontend && npm run test -- --run
cd frontend && npm run build
```

## Status UI

- Dodano fundament logowania i projektów. Frontend startuje od logowania,
  potem pokazuje listę projektów; właściwy workspace detekcji działa dopiero po
  wyborze projektu. Nowy flow używa endpointów `/api/projects/{project_id}/...`.
- Aktualny dashboard projektów ma wyszukiwarkę, sortowanie, edycję nazwy/opisu,
  archiwizację projektu, historię analiz, ustawienia profilu oraz listę
  aktywnych sesji.
- Auth ma MVP resetu hasła. Lokalny/dev backend może zwrócić token resetu w
  odpowiedzi API, żeby dało się testować bez wysyłki maili; produkcja musi
  wyłączyć `ELEKTROSCAN_AUTH_DEV_TOKENS` i podpiąć dostawcę e-mail.
- Rejestracja nie wymaga weryfikacji e-mail. Nie przywracać email verification
  bez wyraźnej decyzji produktowej.
- Dane projektu są izolowane w `backend/data/projects/{project_id}/`:
  `uploads/`, `templates/`, `analysis_debug/`. Globalne endpointy bez
  `project_id` są legacy/dev fallbackiem.
- Lista projektów zwraca `latestSessionId`. Frontend przy ponownym wejściu do
  projektu odtwarza ostatni podgląd PDF i warstwy z tej sesji zamiast pokazywać
  pusty workspace.
- Wyjście z projektu do listy projektów nie czyści już lokalnego workspace.
  Dzięki temu trwająca analiza i jej progress nie są gubione przy powrocie do
  tego samego projektu. Reset następuje dopiero przy wyborze innego projektu,
  wylogowaniu albo czyszczeniu projektu.
- Analizy projektowe są rejestrowane w SQLite i widoczne przez
  `/api/projects/{project_id}/analysis-runs`.
- Po powrocie do projektu po pracy w innym projekcie frontend odtwarza ostatnią
  zakończoną analizę dla aktualnej sesji PDF ze snapshotu
  `/api/projects/{project_id}/analysis-runs/{analysis_id}`.
- Powrót do projektu z zaznaczoną legendą ma pozwalać od razu analizować plan,
  jeżeli wzorce legendy są już sprawdzone. Czarny canvas po powrocie do projektu
  był regresją i powinien być traktowany jako blocker.
- Role, zaproszenia i współdzielenie projektów nie są jeszcze wdrożone. Obecny
  model uprawnień to owner-only; przyszły moduł powinien dodać membership table
  zamiast rozluźniać `owner_user_id` w istniejących query.
- Usunieto stary panel `Pokaz niepewne/brakujace` i debugCandidates z glownego
  UI. Nie przywracac tego jako domyslnej funkcji.
- Narzedziem diagnostycznym jest teraz Inspektor ROI.
- Inspektor ROI pokazuje lokalnie, co silnik widzi w zaznaczonym boxie:
  raw mask, scan mask, dark scan mask, peaki per scale, PASS/odrzuty.
- Aktualny flow legendy jest human-in-the-loop:
  - nowy PDF w projekcie startuje z pustą bazą wzorców tego projektu; legacy
    `POST /api/preview` czyści `backend/templates/`,
  - uzytkownik musi zaznaczyc strefe legendy i wyciagnac wzorce,
  - po ekstrakcji otwiera sie `LegendReviewPanel`,
  - analiza jest zablokowana, dopoki kazdy wzorzec nie ma statusu innego niz
    `pending`,
  - wzorzec mozna zaakceptowac, odrzucic, zmienic nazwe, poprawic crop na
    canvasie albo dodac brakujacy wzorzec recznie.
- Ekstrakcja legendy nie może bazować na twardych współrzędnych konkretnych
  PDF. Kluczowe są ogólne heurystyki: geometria tabeli, komponenty symboli,
  tekst z PDF/OCR i relacje wierszy.
- Dla legend tabelarycznych poprawiono wycinanie znakow typu `C1`/`D1`: znaki
  sa przypisywane do najblizszego srodka wiersza na podstawie ciemnych
  komponentow w kolumnie symboli, zamiast prostego cropa miedzy liniami tabeli.
  To chroni przed ucieciem etykiety z gory i dobraniem fragmentu kolejnego
  wiersza z dolu.
- Dla legend tabelarycznych nazwa wzorca jest teraz brana z tekstu/opisu w tym
  samym wierszu tabeli, a nie tylko z fallbacku `sym_XX`. Ekstraktor uzywa
  `page.get_text("words")`, pomija wiodace kody/liczniki typu `A1`/`01`, a gdy
  nie znajdzie opisu, wraca do starego `_get_row_index_text` i dopiero potem do
  `sym_XX`.
- Dla szarych/rastrowych legend opis może być czytany przez OCR Tesseract, jeżeli
  tekst PDF nie wystarcza. Docker backend instaluje `tesseract-ocr`,
  `tesseract-ocr-eng` i `tesseract-ocr-pol`.
- Dla kolorowych klasycznych legend ekstraktor rozbija symbole po wierszach i
  komponentach, żeby indeks tekstowy i grafika tego samego symbolu zostały razem,
  a sąsiednie wiersze nie podkradały sobie nazw. Przypadki kontrolne:
  `GSW`/`MSW` mają poprawne opisy, `A + kółko` jest osobno od `B + kwadrat`.
- Nazwy typu `nieznany_symbol` są traktowane jako fallback/legacy. Backend i
  frontend próbują je humanizować, ale stare zapisane wzorce mogą wymagać
  ponownego wyciągnięcia legendy albo ręcznej zmiany nazwy.
- Widoczne teksty UI po ostatnich zmianach powinny byc zapisane jako UTF-8.
  Nie zostawiac mojibake typu `Brak podglÄ...du`.

## Szare PDF - Aktualny Kierunek

Plik roboczy: `VIKING-BRONISZE-ELE-Rzuty-E8.pdf`.

Aktualny stan po pracy 2026-05-01:

- Pierwszy szary PDF Viking E8 jest zaakceptowany przez uzytkownika jako
  dzialajacy w 100% dla obecnego celu.
- Golden snapshot: `backend/tests/golden/viking_bronisze_e8_gray_first_pdf_100pct.json`.
- Golden ma `81` finalnych boxow: `01:7, 02:8, 03:11, 04:12, 05:13,
  06:14, 07:16`.
- Lokalny smoke run generujacy goldena trwal okolo `22.6s`.
- To jest baseline dla jednego szarego PDF. Inne szare PDF nadal trzeba
  przejsc osobno Inspektorem ROI.
- Nie znaleziono committed golden JSON dla dwoch pierwszych kolorowych PDF.
  One sa opisane jako dzialajace okolo 95%, ale formalny kolorowy golden
  trzeba jeszcze dopisac, gdy beda znane stabilne snapshoty.

Aktualny stan po pracy 2026-04-30 wieczorem:

- Viking gray jest blisko uzywalnego stanu dla pierwszego PDF. Ostatni lokalny
  run po poprawkach dal okolo `77` finalnych detekcji.
- Ostatni lokalny test diagnostyczny: okolo `23.7s`, `77` finalnych detekcji,
  rozklad `01:1, 02:8, 03:15, 04:12, 05:13, 06:12, 07:16`.
- Najwazniejsza lekcja: Inspektor ROI nie jest tylko podgladem. Jest lokalna
  prawda diagnostyczna. Jesli Inspektor pokazuje mocny `PASS`, a final go nie
  pokazuje, trzeba sledzic hit przez fazy: `scan_raw -> raw_budget ->
  raw_prefilter -> validation -> clustering -> format_results`.
- Nie zgadywac progow w ciemno. Najpierw ustalic, w ktorej fazie hit ginie.
- Dwa trywialne, ale krytyczne bledy juz znalezione:
  - `format_results` robil slepe `count - 1` za legende, gdy `legend_rect`
    bylo puste. Dla gray to chowalo poprawne detekcje, mimo ze walidacja je
    przepuscila. Gray ma uzywac jawnych wykluczen legendy/tekstu/plan zone, a
    nie slepego odejmowania.
  - wydluzone symbole `04/05` potrafily nie dojsc do pozniejszych ROI, bo
    wariant wypelnial globalny limit peakow na wczesniejszych miejscach.
    Dla gray elongated nalezy stosowac fair limit per ROI.
  - prawdziwe `06` z pelna geometria potrafilo odpasc na `low_match_strict`,
    bo tekst/sciana zanizaly `purity` do okolic `0.45`. Dla gray strong
    geometry uzywac progow rescue, nie twardego progu `0.50`.
- Poprawki sa gray-only albo pilnowane przez `detector_profile == "gray"`.

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

Najwazniejsze lekcje z dopinania Viking gray:

- Jesli Inspektor ROI pokazuje `PASS`, a final nie, to zwykle nie jest kwestia
  jednego progu. Sprawdz faze, w ktorej kandydat ginie.
- Gdy final wykrywa cos z jasnoszarych linii, problemem nie jest nazwa symbolu,
  tylko brak twardego dowodu ciemnego tuszu w bboxie.
- `01` wymaga osobnego myslenia, bo na planie bywa wiekszy niz w legendzie i
  moze skladac sie z dwoch przesunietych polow. Rozwiazanie: geometryczny rescue
  i merge nachodzacych ramek, nie reguly po koordynatach.
- `03` w srodku prawdziwego `06` trzeba eliminowac relacja containment/overlap:
  pelniejszy symbol z dobra geometria wygrywa z mniejszym rdzeniem.
- `04/05` sa cienkie i wydluzone. Potrzebuja fair ROI/peak handling, inaczej
  globalny budzet zjada je przed walidacja.
- `06/07` sa dobre testy parent/child: `07` nie moze chowac `06`, ale mniejsze
  rdzenie nie moga zostawac jako osobne false-positive, jesli pelny symbol jest
  juz zaakceptowany.

Aktualne heurystyki, ktore dzialaja dobrze na Viking gray:

- Progi czerni sa kalibrowane z ciemnych pikseli legendy.
- `zone_raw` / `zone_suppressed` sa uzywane do skanowania gray zamiast calej
  jasnoszarej maski planu.
- `gray_dark_evidence` ma blokowac hity zbudowane z jasnych linii planu.
- Mocny gray hit to nie tylko wysoki `match`, ale przede wszystkim pelna
  geometria: wysokie `coverage`, sensowne `purity`, sensowny `context`.
- Dla symboli sklejonych z tekstem/sciana `purity` moze spasc w okolice `0.40`
  mimo prawdziwego symbolu; jesli `coverage` jest pelne, nie odrzucac tego zbyt
  agresywnie.

Szczegolowy plan tej zmiany jest w:

- `openspec/gray-dark-ink-plan.md`

Szczegolowy plan review legendy jest w:

- `openspec/legend-review-plan.md`
- `openspec/legend-manual-table-plan.md`

## Kolorowe PDF - Inwariant

Kolorowe PDF dzialaly dobrze i szybko przed praca nad gray. Przy zmianach gray
oraz legend:

- Nie zmieniac globalnych `SCALES` dla color.
- Nie wlaczac gray raw budget, dark ink, text suppression ani parent fallback dla color.
- Zmiany w `legend_extractor.py` dla kolorowych legend muszą pozostać ogólne:
  row/component grouping, OCR/PDF text i normalizacja nazw, bez współrzędnych
  pod konkretny rysunek.
- Po wiekszej zmianie przetestowac przynajmniej jeden stary kolorowy PDF i
  sprawdzic czas oraz liczbe boxow.

## Minimalny Prompt Dla Nowego AI

```text
Pracujesz w /Users/jakublewosz/Code/matiprojekt. Najpierw przeczytaj
openspec/current-context.md. Projekt ma auth, dashboard projektów i endpointy
projektowe. Detektor ma rozdzielone wejścia color/gray, ale wciąż ma wspólny
pipeline. Nie hardcoduj koordynat ani nazw symboli. Gray PDF tunuj tylko w
gray-only ścieżce. Legendę poprawiaj przez ogólne reguły tabel/wierszy/OCR i
review wzorców. Do diagnozy braków używaj Inspektora ROI, nie przywracaj starego
panelu "Pokaż niepewne/brakujące".
```
