# Decyzje Architektoniczne, Inwarianty i Mapa Ryzyk

## Czego Nie Robić

- **Nie hardcodować koordynat.** Reguły `jeżeli x=1299,y=722 to symbol 08` psują uniwersalność.
- **Nie dopisywać wyjątków per-koordynat.** Jeżeli PDF ma problem w punkcie, rozwiązuj go ogólnie.
- **Nie usuwać reguł rodzinnych 06/07 i 10/11/12** bez gotowego mechanizmu ogólnego i testów regresji.
- **Nie bazować produkcyjnie na PDF text layer.** Wejście może być skanem/zdjęciem.
- **Nie commitować `backend/analysis_debug/`.** To lokalna diagnostyka.
- **Nie commitować `backend/data/` ani projektowych snapshotów debug.** To dane
  robocze użytkownika/kontenera.
- **Nie robić agresywnego cache schematu/PDF/wyników** między analizami.
- **Nie obniżać DPI poniżej 300** bez twardego testu jakości.
- **Nie uznawać jednego PDF-a za całą specyfikację świata.**
- **Nie robić "szybkiej poprawki" pod jeden screen** — projekt najbardziej cierpi właśnie na tym.

## Co Można Robić

- Debug-only kandydaci i HITL.
- Narzędzia korekty w UI.
- Projektową persystencję PDF, wzorców i historii analiz.
- OCR opisów legendy jako fallback, jeżeli PDF text layer nie wystarcza.
- Uniwersalne relacje obrazowe `core -> fuller parent`.
- Image-content matching dla labeli.
- Profilowanie etapów (`summarize_analysis_performance.py`).
- Porównania snapshotów (`compare_analysis_snapshot.py`).
- Lokalne, per-request struktury przyspieszające.
- Refaktoryzację plików, jeśli zachowana jest regresja.

## Inwarianty Projektu

Prawie jak testy jednostkowe architektury:

- Każda analiza jest świeża dla aktualnego wejścia.
- Cache globalny schematu jest zakazany na tym etapie.
- `analysis_debug` jest lokalną diagnostyką, nie częścią produktu.
- Dane projektów są izolowane po `project_id`; projektowe wzorce jednego
  projektu nie mogą mieszać się z innym.
- Powrót do projektu ma przywracać preview PDF, warstwy, legendę, wzorce i
  ostatnią analizę, jeśli istnieje.
- Warstwa PDF może pomagać przy renderze/warstwach, ale nie może być jedynym źródłem prawdy o symbolach.
- Template matching musi być obrazowy i odporny na PDF jako skan/zdjęcie.
- HITL poprawia wynik analizy, ale nie tworzy ukrytego globalnego uczenia pod jeden rzut.
- Reguły rodzinne są tymczasowym zabezpieczeniem, nie docelowym modelem wiedzy.
- Docelowy zamiennik reguł rodzinnych wynika z geometrii masek i relacji `core -> parent`.
- Text labels są uniwersalną ścieżką "czytania z obrazu", nie słownikiem nazw.
- Lepiej pokazać `Brak?` w debug niż cicho zgubić symbol.
- Lepiej zostawić jeden przypadek do HITL niż zepsuć pięć innych przez agresywny próg.

## Decyzja 2026-05-11: Projekty I Legenda Są Częścią Produktu, Nie Debugiem

Auth, dashboard projektów, sesje, historia analiz i review legendy są teraz
podstawowym flow produktu. Nowe funkcje po zalogowaniu powinny używać endpointów
`/api/projects/{project_id}/...`, a legacy endpointy bez `project_id` traktować
jako fallback developerski.

Legenda jest bramką jakości analizy. Analiza planu ma być zablokowana, dopóki
wzorce nie są sprawdzone. Nazwy symboli mają być możliwie czytelne już po
ekstrakcji, ale użytkownik musi móc je poprawić ręcznie.

Nie rozwiązujemy problemów legend przez mapy typu "czwarty element zawsze jest
B" ani przez współrzędne konkretnego planu. Poprawki mają wynikać z geometrii
tabel, grupowania komponentów, OCR/PDF text i relacji wierszy.

## Decyzja 2026-04-30: Inspektor ROI Jest Lokalna Prawda Dla Gray

Jesli Inspektor ROI pokazuje mocny `PASS`, a finalna analiza nie pokazuje
symbolu, nie zaczynac od krecenia globalnym progiem. Najpierw przesledzic
kandydata przez fazy:

- `scan_raw`
- `gray_raw_budget`
- `raw_prefilter`
- `validation`
- `clustering`
- `format_results`

Ta zasada ujawnila trzy realne bledy: slepe odejmowanie legendy w finalnym
formatowaniu, globalny starvation peakow dla wydluzonych symboli i zbyt twardy
prog `purity` dla prawdziwej strong geometry. To sa lepsze poprawki niz
hardcode po koordynatach albo dopisywanie zasad typu "ten symbol zawsze tu".

Gray-only zmiany moga uzywac dark ink zones i progow kalibrowanych z legendy.
Kolorowy profil ma pozostac osobna, szybka sciezka.

## Mapa Ryzyk Technicznych

| Ryzyko | Prawdopodobieństwo | Wpływ |
| --- | --- | --- |
| Overfit pod dwa przykładowe PDF-y | Wysokie | Wysoki |
| Zbyt duże zaufanie do ramek przy symbolach tekstowych | Średnie | Wysoki |
| Zbyt duże zaufanie do `matchTemplate` bez walidacji kontekstu | Niskie | Wysoki |
| Usunięcie reguł rodzinnych bez gotowego mechanizmu ogólnego | Średnie | Wysoki |
| Debug kandydaci zalewający UI szumem | Niskie | Średni |
| Stary backend na porcie 8000 podczas testów | Wysokie | Średni |
| Commitowanie snapshotów `backend/analysis_debug/` | Niskie | Niski |

## Ryzyko Produktowe

Użytkownik będzie ufał zielonym boxom jak prawdzie absolutnej. Dlatego HITL i oznaczanie niepewności są ważniejsze niż udawanie 100%.

## Najbliższy Sensowny Plan Prac

Sensowna kolejność:

1. Ustabilizować HITL/debug — prawdziwe braki widoczne, fałszywe debug boxy nie przeszkadzają.
2. Panel/tryb listy "niepewne miejsca" z sortowaniem po powodzie i koordynatach.
3. Debug-only `template_probe` dla słabych, ale lokalnie sensownych prób template'u.
4. Uniwersalny mechanizm `core -> fuller parent`, równolegle logowany z obecnymi promocjami.
5. Dopiero po braku regresji wymieniać ręczne reguły rodzinne.
6. Zebrać więcej PDF-ów przed dużą zmianą architektury.
7. Optymalizować czas wykonania jako ostatnie.

## Werdykt Techniczny

Projekt jest funkcjonalny i ma sensowny szybki silnik. Najlepszy kierunek:

- Utrzymać image-based matching.
- Rozwijać text-label matching.
- Rozwijać HITL.
- Logować niepewne przypadki.
- Dopiero po zebraniu większej liczby PDF-ów wzmacniać ogólną logikę parent/core.

Ten projekt to hybrydowy detektor symboli technicznych oparty o template matching, maski kolorów, walidację geometryczną i ręczną korektę granicznych przypadków — nie "gotowy magiczny OCR".

## Co Jest Już Naprawdę Dobre

- Szybkość zeszła z ~minuty do kilkunastu sekund na i7-13700KF.
- Problem `11 -> 12` na `PW-E-02` naprawiony.
- Label/text pipeline naprawił klasę problemów ramka-treść.
- UI kopiuje bogaty debug payload z sąsiedztwem.
- HITL/debug pokazuje miejsca niepewne, nie tylko finalne wyniki.
- Kod rozdzielony na moduły core.
- Narzędzia diagnostyczne (`compare_analysis_snapshot`, `summarize_analysis_performance`).
- CostPanel i zarządzanie wzorcami przez UI bez operacji na plikach ręcznie.
