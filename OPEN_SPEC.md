# ElektroScan Open Spec

Pełna dokumentacja modułowa znajduje się też w katalogu [`openspec/`](openspec/). Ten plik zostaje jako duży handoff end-to-end dla człowieka i agenta AI, a `openspec/` jako wersja podzielona tematycznie.

Ten dokument jest przekazaniem projektu dla kolejnej osoby i jej agenta AI/Codexa. Ma dać pełny obraz tego, co projekt robi, jak jest zbudowany, jakie decyzje już podjęto, czego nie psuć oraz gdzie są obecne graniczne problemy.

Stan referencyjny na moment spisania:

- Branch: `codex-test-niewiadoma-optymalizacja`
- Najnowszy commit: `6fb831a Niepewne bledy HITL debug`
- Poprzedni dobry punkt odniesienia: `3186d5d Progres tekstowy`
- Bardzo dobry punkt optymalizacyjny: `7d45d22 Mega Dobra optymalizacja-OBECNA`
- Lokalnie może istnieć `backend/analysis_debug/`; to są snapshoty diagnostyczne i nie powinny być commitowane.

## 1. Cel Projektu

ElektroScan wykrywa symbole instalacji elektrycznej na planach PDF/obrazach na podstawie wzorców z legendy. Projekt nie ma być tylko detektorem pod jeden PDF. Docelowo ma działać możliwie uniwersalnie na podobnych schematach, z dopuszczalnym trybem HITL, czyli człowiek poprawia ostatnie graniczne przypadki.

Aktualny cel jakościowy:

- Automatycznie około `90-95%+` poprawnych detekcji na znanych planach.
- HITL domyka pozostałe przypadki zamiast dopisywania wyjątków po koordynatach.
- Nie dążymy na siłę do 100% przez overfit pod dwa testowe PDF-y.

Kluczowa filozofia:

- Najpierw jakość i uniwersalność.
- Nie dodawać reguł typu `jeżeli x=1299,y=722 to symbol 08`.
- Nie czytać tekstu z warstwy PDF jako głównego mechanizmu, bo produkcyjnie wejście może być zdjęciem/skanem.
- Jeżeli coś da się rozwiązać obrazowo, rozwiązujemy obrazowo.
- Jeżeli przypadek jest bardzo graniczny, pokazujemy go w HITL jako `Sprawdź`, `Może` albo `Brak?`.

## 2. Jak Uruchomić

Backend:

```powershell
cd C:\Users\Admin\Desktop\elektroskanNOWYSWIEZYSTART\ElektroScan\backend
py -3 main.py
```

Backend działa na:

```text
http://127.0.0.1:8000
```

Frontend:

```powershell
cd C:\Users\Admin\Desktop\elektroskanNOWYSWIEZYSTART\ElektroScan\frontend
npm run dev -- --host 127.0.0.1
```

Frontend zwykle działa na:

```text
http://127.0.0.1:5173
```

Ważne: jeżeli wyniki wyglądają jak stary stan mimo zmian w kodzie, sprawdzić proces na porcie `8000`. Kilka razy problemem był stary backend wiszący w tle.

Przykład restartu backendu na Windows:

```powershell
$listeners = netstat -ano | Select-String ':8000' | ForEach-Object { ($_ -split '\s+')[-1] } | Where-Object { $_ -match '^\d+$' } | Sort-Object -Unique
foreach ($processId in $listeners) {
    if ([int]$processId -ne 0) {
        Stop-Process -Id ([int]$processId) -Force -ErrorAction SilentlyContinue
    }
}
Start-Process -FilePath py -ArgumentList @('-3', 'main.py') -WorkingDirectory 'C:\Users\Admin\Desktop\elektroskanNOWYSWIEZYSTART\ElektroScan\backend' -WindowStyle Hidden
```

## 3. Struktura Projektu

Najważniejsze katalogi:

```text
backend/
  main.py
  templates/
  uploads/
  analysis_debug/
  core/
    detector.py
    detector_config.py
    detector_models.py
    detector_masks.py
    detector_templates.py
    detector_clustering.py
    detector_promotions.py
    detector_pdf.py
    legend_extractor.py

frontend/
  src/
    App.tsx
    components/
      CanvasView.tsx
      ResultsPanel.tsx
      Sidebar.tsx
```

Rola plików backendu:

- `backend/main.py`: FastAPI, upload PDF, render preview, ekstrakcja legendy, analiza, snapshot debug, formatowanie odpowiedzi dla frontendu.
- `backend/core/legend_extractor.py`: renderowanie PDF do obrazu, obsługa warstw PDF, ekstrakcja legendy z obrazu/PDF.
- `backend/core/detector.py`: główny pipeline detekcji, przygotowanie wariantów, skanowanie, walidacja, promocje rodzinne, debug/HITL candidates.
- `backend/core/detector_config.py`: progi, skale, rotacje, limity workerów, limity debug/HITL.
- `backend/core/detector_models.py`: dataclassy `TemplateInfo`, `TemplateVariant`, `CandidateHit`, `Detection`, `DetectionResult`.
- `backend/core/detector_masks.py`: maski HSV, maski kolorów, walidacja kandydata, ROI, content mask dla labeli.
- `backend/core/detector_templates.py`: ładowanie template’ów, warianty scale/rotation/mirror, rozpoznawanie label-like template’ów.
- `backend/core/detector_clustering.py`: prefiltering, clustering, metryki overlap.
- `backend/core/detector_promotions.py`: rodzinne promocje typu mniejszy rdzeń -> pełniejszy parent.
- `backend/core/detector_pdf.py`: pomocnicze funkcje PDF text i legend exclude; nie traktować tego jako głównego OCR produkcyjnego.

Rola plików frontendu:

- `frontend/src/App.tsx`: stan aplikacji, requesty API, wyniki, HITL state, ręczne boxy, debugCandidates.
- `frontend/src/components/CanvasView.tsx`: render planu, finalnych boxów, debug/HITL kandydatów, kopiowanie payloadu debug, dodawanie ręcznego boxa.
- `frontend/src/components/ResultsPanel.tsx`: lista wyników, zmiana klasy boxa, usuwanie boxa, lista HITL/debug kandydatów.
- `frontend/src/components/Sidebar.tsx`: upload, ekstrakcja legendy, analiza, warstwy, template’y.

## 4. Pipeline Detektora

W uproszczeniu:

1. PDF jest renderowany do obrazu 300 DPI.
2. Warstwy PDF mogą być ukrywane przed renderem, ale projekt musi działać też bez założenia idealnych warstw.
3. Template’y są ładowane z `backend/templates`.
4. Dla template’ów budowane są warianty:
   - skale: `0.90`, `1.00`, `1.10`
   - rotacje: `0`, `90`, `180`, `270`
   - mirror tylko dla wybranych rodzin i labeli tekstowych
5. Obraz planu jest maskowany kolorem HSV.
6. Dla każdego template’u budowane są ROI, żeby nie skanować całego obrazu bez potrzeby.
7. `cv2.matchTemplate` generuje raw peaki.
8. Raw peaki są filtrowane.
9. Kandydaci są walidowani metrykami:
   - `match_score`
   - `coverage`
   - `purity`
   - `context_purity`
   - `color_similarity`
   - `verification_score`
10. Działają promocje rodzinne, np. mniejszy rdzeń może zostać podniesiony do pełniejszego symbolu.
11. Kandydaci są klastrowani i zamieniani na finalne `Detection`.
12. Jeżeli `include_debug=true`, generowane są `debugCandidates` dla HITL.

## 5. Ważne Metryki

Pola w debug payload:

- `symbol`: nazwa symbolu.
- `bbox`: `x,y,width,height`.
- `match`: wynik template matching.
- `verification`: złożona ocena po walidacji.
- `coverage`: ile pikseli template’u jest wyjaśnione przez plan.
- `purity`: ile pikseli ROI pasuje do template’u.
- `context_purity`: jak bardzo symbol wyjaśnia lokalny kontekst.
- `color_similarity`: zgodność koloru.
- `rotation`: rotacja wariantu.
- `scale`: skala wariantu.
- `mirrored`: czy wariant jest lustrzany.
- `source`: np. `template`, `template_promoted_*`, `template_content`, `unexplained_component`.
- `reason`: finalne lub debugowe uzasadnienie.

Interpretacja:

- Wysoki `match` nie wystarcza, jeżeli `coverage/purity/context` są złe.
- Niskie `context_purity` często oznacza zatłoczone miejsce albo partial/ghost.
- `verification=0` przy debug kandydacie oznacza, że kandydat odpadł w walidacji, ale warto go pokazać człowiekowi.

## 6. Text Labels

W commitcie `3186d5d Progres tekstowy` dodano uniwersalną ścieżkę dla symboli tekstowych.

Przykłady:

- `TM`
- `TAB`
- `TSM`
- `MSW`
- `GSW`
- `INT`
- `TV`

Zasada:

- Nie ma mapy `MSW=05`, `GSW=04`, `TSM=03`.
- Silnik obrazowo rozpoznaje label-like template po geometrii i treści.
- Dla labeli tworzona jest maska treści `content_mask`, czyli same litery/znaki po odjęciu ramek i linii.
- Kandydat labela jest oceniany po pełnym znaku i po treści.
- Dla labeli zwycięstwo powinno zależeć mocniej od `content_score`, a nie tylko od ramki.

Dlaczego to ważne:

- `MSW` i `GSW` mają podobne ramki, więc samo template matching po ramce potrafi mylić `04/05`.
- `TSM` potrafi mieć prostokąt/kreskę przesuniętą względem legendy, a napis nadal jest czytelny.
- `INT/TV` mogą być odwrócone/lustrzane; trzeba patrzeć na obraz, nie na PDF text layer.

## 7. Rodzinne Promocje

Obecnie istnieją zabezpieczenia rodzinne dla podobnych symboli:

- rodzina `06/07`
- rodzina `10/11/12`

Nie usuwać ich gwałtownie.

Dlaczego:

- Chronią jakość na znanych PDF-ach.
- Wcześniej usuwanie lub agresywna zmiana powodowała powroty błędów.

Docelowo:

- Zastąpić je bardziej uniwersalnym mechanizmem `core -> fuller parent`.
- Relacje powinny wynikać z masek template’ów, zawierania i dodatkowych pikseli, nie z ręcznego wpisania symbolu.
- Najpierw logować taki mechanizm równolegle w debug, potem dopiero przełączać finalną decyzję.

## 8. HITL i Debug Candidates

Commit `6fb831a Niepewne bledy HITL debug` dodał warstwę debug/HITL.

Typy kandydatów:

- `accepted_uncertain`: finalny zielony box istnieje, ale metryki mówią, że człowiek powinien go sprawdzić.
- `rejected_candidate`: template coś widział, ale walidacja odrzuciła.
- `rejected_low_content`: label/text kandydat odpadł po treści.
- `unexplained_component`: kolorowy komponent planu nie został sensownie wyjaśniony przez finalne boxy.
- `overlap_conflict`: kilka klas walczy o podobny obszar.
- `partial_ghost`: częściowy duplikat/ghost w środku większego symbolu.

W UI:

- Zielone boxy to finalne detekcje.
- Czerwone/pomarańczowe debug boxy nie są finalną detekcją.
- `Sprawdź` oznacza finalny box niepewny.
- `Może` oznacza kandydat odrzucony, ale potencjalnie przydatny.
- `Brak?` oznacza niewyjaśniony komponent.

Ważne:

- `Brak?` może być szeroki i obejmować kilka połączonych kresek.
- `Brak?` nie zna klasy symbolu, tylko mówi: “tu jest kolorowy komponent, którego finalne boxy nie tłumaczą”.
- `accepted_uncertain` nie powinien mieć przycisku “Dodaj”, bo to już finalny box; użytkownik ma raczej zmienić klasę lub usunąć.
- `rejected_candidate` i `unexplained_component` mogą być dodane ręcznie.

## 9. Aktualne Znane Problemy

Znane problemy nie są powodem do panicznego overfitu.

### 9.1 Brakujące / trudne `08 E 400V`

Na `PW-E-01 Rev2 (1).pdf` są przypadki `08_E_400V...`, które wyglądają podobnie, ale jeden jest wykrywany, a inny nie.

Dobry przykład:

```text
symbol=08_E_400V_wypust_400V_zasilanie_kuchenki_zakonczony_puszka_nt
bbox=1187,1767,46,44
match=0.644
verification=0.638
coverage=0.720
purity=0.765
rotation=270
scale=1.000
mirrored=false
```

Problemowy rejon:

```text
final nearby:
06 @1363,737,32,31
09 @1299,722,34,34
```

Test diagnostyczny pokazał:

- Dla dobrego `08` max match template’u wynosi około `0.644`.
- Dla brakującego rejonu max match `08` wynosi tylko około `0.421`.
- To oznacza, że problem nie jest prostym brakiem rotacji.
- Eksperyment “mirror dla wszystkich template’ów” też nie naprawił tego przypadku.
- Problemem jest raczej zlepienie/zakłócenie kształtu i to, że pełny template `08` nie widzi tam wystarczająco podobnego układu.

Obecne rozwiązanie:

- Pokazywać lokalne `Brak?` wokół niepewnych/zatłoczonych miejsc.
- Pokazywać sąsiednie `accepted_uncertain`.
- Nie zmieniać finalnej klasy na siłę.

Potencjalny dalszy kierunek:

- Debug-only `template_probe`: pokazywać słabe próby template’u przy niepewnym miejscu, np. `08_probe match=0.42`.
- Uniwersalny rescue dla parentów: jeżeli mniejszy symbol i dodatkowy komponent obok razem pasują do pełniejszego template’u, pokazać to jako HITL albo finalnie po bezpiecznych progach.

### 9.2 Zielone układy `11/12/13`

Znane miejsca z grupami `07`, `11`, `12`, `13` potrafią generować:

- poprawne `12`
- dodatkowe `11`
- brakujące `13`
- `Brak?` jako duży komponent obejmujący kilka połączonych zielonych części

To nie zawsze oznacza błąd detektora; często kilka symboli jest fizycznie zlepionych na rysunku.

Obecne rozwiązanie:

- `accepted_uncertain` oznacza finalne zielone boxy w zatłoczonych miejscach.
- `unexplained_component` pokazuje większe niepokryte komponenty.
- Człowiek w HITL decyduje, czy dodać/usunąć/zmienić klasę.

### 9.3 `06/09` pomyłki

Na `PW-E-02 Rev2.pdf` znany błąd:

```text
powinno być 06
wykrywa jako 09
bbox=2742,975,31,31
```

Nie dopisywać reguły po tej koordynacie.

Lepszy kierunek:

- Porównać pełniejszy shape-score.
- Logować lokalne konflikty.
- Opracować ogólną regułę parent/core, jeśli powtarza się rodzinnie.

### 9.4 `MSW/GSW`

Wcześniej `MSW` bywało czytane jako `04` zamiast `05`, bo ramka była podobna do `GSW`.

Obecnie text-label pipeline poprawił sytuację, ale uważać przy zmianach content mask.

Nie wprowadzać mapy `MSW=05`.

## 10. Golden / Baseline Testy

Znane PDF-y:

- `PW-E-02 Rev2.pdf`
- `PW-E-01 Rev2.pdf`
- czasem plik jest widoczny jako `PW-E-01 Rev2 (1).pdf` po ponownym uploadzie

Znane baseline’y z rozmów:

### `PW-E-02 Rev2.pdf`

Dobry punkt po optymalizacji:

- około `134-139` boxów, zależnie od warstw i etapu.
- problematyczny dawniej punkt `11 -> 12` powinien być `12`:

```text
symbol=12_lacznik_jednobiegunowy_lacznik_swiecznikowy_wypust_oswietleniowy_sufitowy
bbox=2293,1548,48,31
source=template_promoted_16_to_23 lub podobne
```

Znany błąd do HITL:

```text
powinno być 06, bywa 09
bbox=2742,975,31,31
```

### `PW-E-01 Rev2 / PW-E-01 Rev2 (1).pdf`

Sprawdzać:

- `TM`, `TSM`, `MSW`, `GSW`, `INT`, `TV` po image-content labels.
- `E 400V` przypadki `08`.
- grupy zielonych `11/12/13`.
- fałszywe środki `21/23`, które wcześniej potrafiły pojawić się w zlepionych symbolach.

## 11. API

Najważniejsze endpointy:

- `POST /api/upload`
- `GET/POST /api/layers`
- `POST /api/render-preview`
- `POST /api/extract-legend`
- `POST /api/analyze`
- `POST /api/clear`
- `DELETE /api/templates`

`/api/analyze` przyjmuje:

```json
{
  "excluded_zones": [],
  "hidden_layers": [],
  "include_debug": true,
  "include_image": true
}
```

Normalnie UI obecnie wysyła `include_debug=true`, bo projekt jest w fazie strojenia.

Odpowiedź zawiera:

- `results`
- `boxes`
- `resultImage`
- `analysisContext`
- `debugCandidates` przy debug
- snapshot JSON w `backend/analysis_debug/` przy debug

## 12. Frontend HITL

Użytkownik może:

- Kliknąć box i skopiować debug payload.
- Usunąć fałszywy finalny box.
- Zmienić klasę finalnego boxa.
- Dodać debug-kandydata jako ręczny box.
- Dodać ręczny box z toolbaru.
- Ukryć debug-kandydata.

Debug payload zawiera też:

- `frontend_nearby_boxes`
- `frontend_debug_candidates_count`
- `frontend_nearby_debug_candidates`

To jest ważne, bo kliknięcie finalnego zielonego boxa samo w sobie nie mówiło wcześniej, że obok istnieje `Brak?`.

## 13. Performance

Były etapy, gdzie analiza trwała około `54-58 s`. Po optymalizacji spadła do około kilkunastu sekund na i7-13700KF.

Ważne obserwacje:

- Najcięższe etapy to zwykle `scan`, `validation_targeted`, `parent_search`.
- CPU nie zawsze pokazuje 100%, bo część etapów jest I/O/serializacyjna, część OpenCV puszcza native code, a część czeka na Python/GIL/łączenie wyników.
- Nie optymalizować kosztem jakości bez benchmarku.
- Na tym etapie nie używać cache PDF-a/wyników między analizami.

Obecna zasada:

- Świeży run dla każdego wejścia.
- Można tworzyć lokalne struktury w pamięci w ramach jednego requestu.
- Nie zapamiętywać schematu globalnie.

## 14. Warstwy PDF

Użytkownik testował różne ustawienia warstw. Ważne było odkrycie, że lokalna reprodukcja z warstwami może się różnić przez:

- polskie znaki,
- normalizację Unicode,
- ukryte/aktywne warstwy,
- inny session_id.

Do debug payload dodano:

- `analysis_session`
- `source_pdf`
- `hidden_layers_used`
- `hidden_layers_unmatched`
- `hidden_layers_repr`

To było potrzebne, żeby skończyć “kotka i myszkę” z innym renderem u użytkownika i u agenta.

## 15. Czego Nie Robić

Nie robić:

- Nie hardcodować koordynat.
- Nie dopisywać wyjątków typu `jeżeli symbol w tym miejscu, to popraw na X`.
- Nie usuwać obecnych reguł rodzinnych bez testów regresji.
- Nie bazować głównie na PDF text layer.
- Nie commitować `backend/analysis_debug/`.
- Nie robić agresywnego cache schematu/PDF/wyników.
- Nie obniżać DPI do `150` bez twardego testu jakości.
- Nie uznawać jednego PDF-a za całą specyfikację świata.

Można robić:

- Debug-only kandydaci.
- HITL i narzędzia korekty.
- Uniwersalne relacje obrazowe `core -> fuller parent`.
- Image-content matching dla labeli.
- Profilowanie etapów.
- Lokalne, per-request struktury przyspieszające.
- Refaktoryzację plików, jeśli zachowana jest regresja.

## 16. Najbliższy Sensowny Plan Prac

Najbardziej sensowna kolejność:

1. Ustabilizować HITL/debug, żeby prawdziwe braki były widoczne, a fałszywe debug boxy nie przeszkadzały.
2. Dodać panel/tryb listy “niepewne miejsca” z sortowaniem po powodzie i koordynatach.
3. Dodać debug-only `template_probe` dla słabych, ale lokalnie sensownych prób template’u.
4. Opracować uniwersalny mechanizm `core -> fuller parent`, równolegle logowany z obecnymi promocjami.
5. Dopiero po braku regresji wymieniać ręczne reguły rodzinne.
6. Zebrać więcej PDF-ów przed dużą zmianą architektury.
7. Potem dopiero optymalizować czas wykonania.

## 17. Jak Pracować Z Codexem/AI Nad Tym Projektem

Dobre polecenie startowe dla nowego Codexa:

```text
Przeczytaj OPEN_SPEC.md, sprawdź git log i aktualny branch. Nie usuwaj reguł rodzinnych 06/07 i 10/11/12 bez testów. Nie dodawaj reguł po koordynatach. Nie commituj backend/analysis_debug. Najpierw reprodukuj na PW-E-01 Rev2 i PW-E-02 Rev2, potem rób małe zmiany i sprawdzaj TypeScript + compileall.
```

Przed zmianami:

```powershell
git status --short
git log --oneline -5
py -3 -m compileall backend\core backend\main.py
cd frontend
npx tsc -p tsconfig.app.json --noEmit
```

Po zmianach:

```powershell
py -3 -m compileall backend\core backend\main.py
cd frontend
npx tsc -p tsconfig.app.json --noEmit
npx vite build
```

Jeżeli UI pokazuje coś innego niż lokalny test:

- sprawdzić, czy backend na `8000` jest aktualny,
- ubić stary proces po PID,
- uruchomić backend ponownie,
- zrobić nową analizę i patrzeć na nowe `analysis_id`.

## 18. Aktualne Commity Referencyjne

```text
6fb831a Niepewne bledy HITL debug
3186d5d Progres tekstowy
7d45d22 Mega Dobra optymalizacja-OBECNA
901c5b9 Bardzo dobra optymalizacja 2 bledy trzeba testow
b9b06cd ogranicz parent search po klastrowaniu
97fc492 dodaj profil wydajnosci analizy
```

Jeżeli trzeba wrócić do stabilniejszego stanu:

- `3186d5d` jest dobrym punktem tekstowym.
- `7d45d22` jest dobrym punktem optymalizacyjnym.
- `6fb831a` dodaje warstwę niepewności/HITL, ale nie rozwiązuje jeszcze wszystkich braków.

## 19. Słownik

- `final box`: zielony zaakceptowany box.
- `debugCandidate`: czerwony/pomarańczowy kandydat do sprawdzenia.
- `HITL`: Human In The Loop, człowiek poprawia wynik.
- `template`: wycięty wzorzec z legendy.
- `content_mask`: maska samej treści napisu po odjęciu ramek/linii.
- `parent`: pełniejszy symbol.
- `child/core`: mniejszy rdzeń symbolu.
- `ghost`: częściowy fałszywy duplikat wewnątrz większego symbolu.

## 20. Krótki Werdykt Techniczny

Projekt jest funkcjonalny i ma już sensowny szybki silnik. Największe ryzyko to dalsze dopisywanie wyjątków pod pojedyncze przypadki, bo to szybko popsuje uniwersalność. Najlepszy kierunek to:

- utrzymać image-based matching,
- rozwijać text-label matching,
- rozwijać HITL,
- logować niepewne przypadki,
- dopiero po zebraniu większej liczby PDF-ów wzmacniać ogólną logikę parent/core.

Ten projekt nie jest “gotowym magicznym OCR-em”. To hybrydowy detektor symboli technicznych oparty o template matching, maski kolorów, walidację geometryczną i ręczną korektę granicznych przypadków.

## 21. Pakiet Startowy Dla Kolegi

Jeżeli ktoś przejmuje projekt od zera, najpierw powinien zrobić dokładnie to:

1. Przeczytać cały ten plik.
2. Uruchomić `git status --short` i upewnić się, że nie ma przypadkowych zmian w silniku.
3. Sprawdzić `git log --oneline -8`, żeby wiedzieć, czy siedzi na branchu `codex-test-niewiadoma-optymalizacja`.
4. Odpalić backend i frontend.
5. Wrzucić `PW-E-02 Rev2.pdf` i sprawdzić znany punkt `2293,1548`, który ma być `12`.
6. Wrzucić `PW-E-01 Rev2 (1).pdf` i sprawdzić label/text przypadki `TM/MSW/TSM/INT/TV`.
7. Dopiero potem zmieniać kod.

Minimalny prompt dla jego Codexa/AI:

```text
Pracujesz nad ElektroScan. Najpierw przeczytaj OPEN_SPEC.md. To jest detektor symboli elektrycznych z PDF/obrazu, oparty głównie o OpenCV/template matching i HITL. Nie wolno hardcodować koordynat, nie wolno opierać produkcyjnej logiki na PDF text layer, nie commituj backend/analysis_debug. Aktualnie ważne są image-based text labels oraz debugCandidates/HITL. Przed zmianą sprawdź branch, status i ostatnie commity.
```

Jeżeli agent ma mały kontekst, wkleić mu tylko:

```text
Najważniejsze: nie psuj 12 przy bbox 2293,1548 na PW-E-02; nie usuwaj jeszcze reguł 06/07 i 10/11/12; text labels mają być image-based bez mapy MSW=05; debug/HITL ma pokazywać braki i niepewne boxy, ale nie zamieniać ich automatycznie w final bez bezpiecznych progów.
```

## 22. Złote Przypadki Regresyjne

Te przypadki są ważniejsze niż “średnie wrażenie z UI”. Jeżeli któryś z nich się zmienia, trzeba rozumieć dlaczego.

| PDF | Miejsce / bbox | Oczekiwane zachowanie | Dlaczego ważne |
| --- | --- | --- | --- |
| `PW-E-02 Rev2.pdf` | `2293,1548,48,31` | Ma być `12`, nie `11` | Historycznie największy problem, obecnie naprawiony. |
| `PW-E-02 Rev2.pdf` | `2742,975,31,31` | Znany problem `09` zamiast `06`, przynajmniej HITL/uncertain | Nie robić reguły po koordynacie. |
| `PW-E-02 Rev2.pdf` | MSW/GSW okolice `2293,1856` i odwrócone MSW | Label ma być rozstrzygany po treści, nie ramce | Testuje `content_mask`. |
| `PW-E-01 Rev2 (1).pdf` | `TM/TSM` blisko siebie | TM i TSM mają być łapane obrazowo | TSM ma czasem przesuniętą kreskę/prostokąt. |
| `PW-E-01 Rev2 (1).pdf` | `INT/TV` odwrócone/lustrzane | Ma działać przez obraz, nie PDF text | Testuje mirror/rotation labeli. |
| `PW-E-01 Rev2 (1).pdf` | `08 E 400V` przy dobrym przykładzie `1187,1767,46,44` | Ma zostać finalnie wykryte | Pilnuje, żeby `08` nie znikło. |
| `PW-E-01 Rev2 (1).pdf` | brakujący/trudny `08` obok `06 @1363,737` | Co najmniej debug `Brak?` / niepewny komponent | Obecnie trudny przypadek do HITL/rescue. |
| `PW-E-01 Rev2 (1).pdf` | zielone klastry `11/12/13` | Braki mają być widoczne w debug/HITL | Testuje partial ghosts i unexplained components. |

Docelowo te przypadki warto przenieść do plików w stylu:

```text
tests/golden/pw_e_02_rev2.json
tests/golden/pw_e_01_rev2.json
```

Na razie najważniejsze dane są w rozmowie i w debug payloadach kopiowanych z UI.

## 23. Rytuał Debugowania Nowego Błędu

Przy nowym błędzie nie zaczynać od pisania reguły. Najpierw zebrać dane:

1. Kliknąć problematyczny box w UI.
2. Skopiować debug payload.
3. Sprawdzić `analysis_id`, `analysis_session`, `source_pdf`, `hidden_layers_used`.
4. Sprawdzić `frontend_nearby_boxes`.
5. Jeżeli debug jest włączony, sprawdzić `frontend_nearby_debug_candidates`.
6. Porównać z podobnym poprawnym przypadkiem z tego samego PDF-a.
7. Zadać pytanie: czy różnica wynika z treści, ramki, koloru, overlapu, rotacji, skali, czy walidacji?
8. Dopiero potem zmieniać próg albo logikę.

Klasy problemów:

- `template widzi podobne, ale wybiera złą klasę`: konflikt klas, trzeba poprawić ranking/verification.
- `template w ogóle nie widzi`: brak wariantu, zbyt niski raw match, zbyt mały ROI albo obraz jest zniekształcony.
- `jest finalny box, ale powinien być inny`: `accepted_uncertain` i panel korekty.
- `jest kolorowy fragment bez boxa`: `unexplained_component`.
- `w środku większego symbolu pojawia się mały fałszywy box`: `partial_ghost` / overlap tłumiony przez parent.

Najlepszy format zgłoszenia błędu:

```text
PDF:
czy warstwy ukryte:
oczekiwany symbol:
aktualny symbol:
bbox:
debug payload:
screen/crop:
poprawny podobny przykład:
```

## 24. Inwarianty Projektu

Te zasady są prawie jak testy jednostkowe architektury:

- Każda analiza ma być świeża dla aktualnego wejścia.
- Cache globalny schematu jest zakazany na tym etapie.
- `analysis_debug` jest lokalną diagnostyką, nie częścią produktu.
- Warstwa PDF może pomagać przy renderze/warstwach, ale nie może być jedynym źródłem prawdy o symbolach.
- Template matching ma być obrazowy i odporny na PDF jako skan/zdjęcie.
- HITL poprawia wynik analizy, ale nie tworzy ukrytego globalnego uczenia pod jeden rzut.
- Reguły rodzinne są tymczasowym zabezpieczeniem, nie docelowym modelem wiedzy.
- Docelowy zamiennik reguł rodzinnych ma wynikać z geometrii masek i relacji `core -> parent`.
- Text labels są uniwersalną ścieżką “czytania z obrazu”, nie słownikiem nazw.
- Lepiej pokazać `Brak?` w debug niż cicho zgubić symbol.
- Lepiej zostawić jeden przypadek do HITL niż zepsuć pięć innych przez agresywny próg.

## 25. Mapa Ryzyk

Największe ryzyka techniczne:

- Overfit pod dwa przykładowe PDF-y.
- Zbyt duże zaufanie do ramek przy symbolach tekstowych.
- Zbyt duże zaufanie do `matchTemplate` bez walidacji kontekstu.
- Usunięcie reguł rodzinnych bez gotowego mechanizmu ogólnego.
- Debug kandydaci zalewający UI szumem.
- Stary backend na porcie `8000`, przez który testowana jest nie ta wersja kodu.
- Commitowanie snapshotów `backend/analysis_debug/`.

Największe ryzyko produktowe:

- Użytkownik będzie ufał zielonym boxom jak prawdzie absolutnej. Dlatego HITL i oznaczanie niepewności są ważniejsze niż udawanie 100%.

## 26. Co Jest Już Naprawdę Dobre

Projekt nie jest już zabawką ani prostym demem:

- Szybkość po optymalizacjach zeszła z prawie minuty do kilkunastu sekund na i7-13700KF.
- Problem `11 -> 12` na głównym PDF-ie został rozwiązany w aktualnym branchu.
- Label/text pipeline naprawił klasę problemów, gdzie ramka była podobna, a treść inna.
- UI potrafi kopiować bardzo bogaty debug payload.
- HITL/debug pokazuje nie tylko finalne wyniki, ale też miejsca niepewne.
- Kod jest już częściowo rozdzielony z jednego wielkiego `detector.py` na moduły core.

Najważniejsze: dalszy rozwój powinien być spokojny i systemowy. Ten projekt najbardziej cierpi wtedy, gdy ktoś robi “szybką poprawkę” pod jeden screen.
