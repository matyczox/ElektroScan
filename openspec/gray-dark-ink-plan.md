# Plan Zmiany: Gray Dark Ink Zones

## Cel

Przyspieszyc i ustabilizowac detekcje szarych PDF przez prowadzenie skanowania
po ciemnym tuszu symboli, a nie po calej szarej masce planu.

Po ludzku: symbole elektryczne na szarych PDF sa zwykle czarniejsze niz linie
architektury, wymiarowania i tlo. Silnik ma najpierw znalezc czarne strefy, a
dopiero potem szukac w nich symboli.

## Zakres

Dotyczy tylko profilu `gray`.

Nie wolno zmieniac zachowania profilu `color`, poza ewentualnym refaktorem bez
zmiany logiki.

## Problem

Obecny gray pipeline nadal bywa za ciezki i za czuly:

- `02/03` potrafia skladac sie z jasnych szarych kresek planu.
- `01` lapie fragmenty tekstu albo pojedyncze kreski.
- `04/05/06` moga odpasc, gdy przez symbol przechodzi jasna linia
  architektoniczna.
- Inspektor ROI pokazuje, ze `dark scan mask` czesto lepiej separuje symbole
  od tla niz zwykla szara/ink maska.

## Zasady

- Nie hardcodowac koordynat.
- Nie robic map typu `09 = 07`.
- Nie obnizac globalnie progow dla wszystkich symboli.
- Nie przywracac starego panelu `Pokaz niepewne/brakujace`.
- Inspektor ROI jest glownym narzedziem weryfikacji.
- Kolorowy silnik ma zostac szybki.

## Proponowana Implementacja

### 1. Dark Ink Mask Dla Gray

Dodac gray-only funkcje budowania maski ciemnego tuszu:

- wejscie: `plan_image`
- wyjscie: maska pikseli ciemniejszych niz prog
- sa dwa progi:
  `GRAY_DARK_INK_THRESHOLD` do diagnostyki/szerszej maski oraz
  `GRAY_DARK_ZONE_THRESHOLD` do faktycznego ROI i skanowania
- oba progi maja byc konfigurowalne przez env
- docelowo prog moze byc adaptacyjny z legendy

Nie usuwac jeszcze zwyklej `raw ink mask`, bo walidacja nadal potrzebuje
pelnego kontekstu.

### 2. Dark Ink Zones / ROI

Dla gray budowac dodatkowe kandydackie ROI z komponentow dark ink:

- komponenty z ciemnego tuszu sa miejscami, gdzie warto skanowac
- ROI powinno miec margines, bo symbol moze byc przeciety jasna linia lub miec
  element poza samym czarnym komponentem
- szare cienkie linie architektoniczne nie powinny same tworzyc mocnych ROI
- duzy kafelkowy fallback ma byc domyslnie wylaczony, bo robi ogromne,
  zachodzace okna i jest sprzeczny z zasada skanowania wokol czarnego tuszu

Implementacja powinna rozroznic:

- `dark_zone` - ostrzejsza maska czarnego tuszu do decyzji gdzie wolno
  skanowac oraz jako zrodlo maski dla `matchTemplate`.
- `zone_raw` / `zone_suppressed` - maski skanowania budowane z `dark_zone`.
- `dark_evidence` - najostrzejsza, niedylatowana maska czerni do walidacji
  bboxa; ma blokowac trafienia zbudowane z pelnoszarych linii.
- `dark_raw` / `dark_suppressed` - szersze maski diagnostyczne, przydatne do
  porownan w Inspektorze, ale nie powinny same generowac trafien.
- `raw ink` - pelna maska pomocnicza do walidacji geometrii.

To daje zasade: czarny tusz wybiera strefe i piksele skanowania. Szersza/raw
maska moze pomagac w walidacji, ale nie moze sama zbudowac kandydata z
jasnoszarych linii planu.

### 3. Scan Mask Policy

Rozdzielic jawnie wybor maski do skanowania:

- male/ciemne symbole moga uzywac `zone_suppressed`
- wieksze ramki typu `07/ZG` moga uzywac `zone_raw`, jesli suppression ucina
  litery albo ramke
- decyzja ma byc w `detector_gray.py`, nie w losowych miejscach pipeline

### 4. Walidacja Na Dwoch Prawdach

Walidacja gray powinna miec dostep do:

- `raw ink mask` - zeby nie zgubic prawdziwego symbolu
- `dark ink mask` - zeby karac dopasowania zbudowane z jasnych linii planu
- `dark_evidence` - kandydat musi miec wystarczajaca czesc template'u
  podparta czarnym tuszem wewnatrz wlasnego bboxa, a nie tylko czarny piksel
  gdzies obok w strefie ROI

Nie robic brutalnie: dark mask nie moze byc jedyna prawda, bo przez symbole
czasem przechodza jasniejsze linie planu.

### 5. Diagnostyka

Dopisac do logow i Inspektora ROI:

- dark threshold
- dark pixels raw/scan
- ile ROI powstalo z dark ink
- ile kandydatow pochodzi z dark mask vs raw mask
- per-symbol: `mask=zone_raw|zone_suppressed|dark_raw|dark_suppressed`

Inspektor ROI juz pokazuje czesc tych danych. Finalna analiza powinna miec
podobne liczniki w `Detection diagnostics`.

## Plan Prac

1. Przeczytac `openspec/current-context.md`.
2. Sprawdzic `git status --short`.
3. Zlokalizowac obecne dark mask helpers w `detector_gray.py`,
   `detector_masks.py`, `roi_inspector.py`.
4. Uporzadkowac gray mask policy w `detector_gray.py`.
5. Dodac dark ink ROI/scanning tylko dla `profile=gray`.
6. Zostawic color path nietkniety.
7. Uruchomic `py -3 -m compileall -q backend`.
8. Uruchomic `npm run build`.
9. Przetestowac Viking gray z PlanZone.
10. Porownac z Inspektorem ROI miejsca, ktore uzytkownik wskazal jako:
    dobre `07`, dobre `03`, brakujace `04/06`, false `02/03/01`.

## Test Plan

### Gray Viking

PDF: `VIKING-BRONISZE-ELE-Rzuty-E8.pdf`

Sprawdzic:

- `07/ZG` w okolicach `3424,4321`, `4510,4321`, `4952,4321`.
- `07/ZG` obrocone w okolicach `5790,1619`, `5789,2172`, `6209,2930`.
- poprawne `03` w okolicach `6139,4050`, `6139,3821`, `5975,4307`,
  `5834,4329`.
- poprawne `02` w okolicach `5802,3070`, `6105,3070`.
- poprawne `04` w okolicach cienkich paleczek, np. `6034,3249`.
- brak false-positive `02` z jasnych linii planu w grupach typu
  `5541,2471`, `5603,2471`, `5592,3824`.
- ograniczyc false-positive `01` na tekstach i pojedynczych kreskach.

### Nowa Obserwacja: Pale-Line False Positives

Rozne symbole moga przejsc walidacje shape-only, gdy jasnoszare linie planu
uloza sie podobnie do template'u. Wtedy `coverage` i `purity` wygladaja dobrze,
ale piksele nie sa wystarczajaco czarne jak symbole z legendy.

Kierunek: nie dodawac filtrow per symbol ani per koordynata. Gray pipeline ma
najpierw usunac jasnoszare tlo i budowac kandydackie strefy z czarnego tuszu.
Dopiero wokol tych stref skanowac symbole. Raw/szara maska moze pomagac w
walidacji geometrii, ale nie powinna sama generowac miejsc skanowania.

### Color Regression

Na co najmniej jednym starym kolorowym PDF:

- czas nie moze wrocic do wielominutowego mielenia
- liczba boxow i glowne klasy maja zostac stabilne
- `parent_search` nie powinien dzialac dla `profile=color`

## Kryteria Sukcesu

- Gray Viking wykrywa wiekszosc prawdziwych symboli, ale nie zalewa planu
  false-positive z jasnych linii.
- `Detection diagnostics` pokazuje mniej raw candidates / mniej skanowanych
  obszarow niz obecnie.
- Inspektor ROI i finalna analiza zgadzaja sie czesciej: jesli ROI pokazuje
  mocny `PASS`, final nie powinien go gubic przez budzet/prefilter.
- Color PDF nie ma regresji wydajnosciowej.

## Kryteria Stopu

Przerwac i nie brnac w progi, jesli:

- dark mask usuwa prawdziwe symbole czesciej niz usuwa tlo
- trzeba zaczac dopisywac reguly po koordynatach
- poprawa gray powoduje spowolnienie color
- liczba false-positive rosnie szybciej niz liczba poprawnych trafien

## Notatka Dla AI

To nie jest rewrite silnika. To eksperyment zmieniajacy prowadzenie skanowania
gray PDF przez ciemny tusz. Najpierw poprawic maski i ROI, potem dopiero progi.
