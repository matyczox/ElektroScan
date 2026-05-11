# Detekcja — Metryki, Labels, Promocje, HITL

> Aktualizacja: stary panel `Pokaz niepewne/brakujace` i `debugCandidates`
> zostaly usuniete z glownego UI. Do diagnozy brakow uzywamy teraz Inspektora
> ROI (`backend/core/roi_inspector.py`). Jesli ponizsze sekcje mowia o
> debugCandidates/HITL jako aktywnym panelu, traktuj je jako historyczny opis.

## Metryki Walidacji Kandydata

Każdy kandydat jest oceniany przez zestaw metryk. Wszystkie pola są dostępne w debug payload.

| Metryka | Znaczenie |
| --- | --- |
| `match` | Surowy wynik `cv2.matchTemplate` |
| `verification` | Złożona ocena po pełnej walidacji |
| `coverage` | Ile pikseli template'u jest wyjaśnione przez obraz planu |
| `purity` | Ile pikseli ROI pasuje do template'u |
| `context_purity` | Jak dobrze symbol wyjaśnia lokalny kontekst |
| `color_similarity` | Zgodność koloru z template'em |
| `rotation` | Rotacja wariantu (0 / 90 / 180 / 270) |
| `scale` | Skala wariantu (0.90 / 1.00 / 1.10) |
| `mirrored` | Czy wariant jest lustrzany |
| `source` | Skąd pochodzi: `template`, `template_promoted_*`, `template_content`, `unexplained_component` |
| `reason` | Finalne lub debugowe uzasadnienie decyzji |

### Interpretacja

- Wysoki `match` nie wystarcza — jeżeli `coverage/purity/context_purity` są złe, kandydat odpada.
- Niskie `context_purity` często oznacza zatłoczone miejsce albo `partial_ghost`.
- `verification=0` przy debug kandydacie: odpadł w walidacji, ale pokazujemy w HITL.

### Aktualne Progi (z detector_config.py)

```python
THRESHOLD_PRECISE = 0.55
THRESHOLD_DILATED = 0.45
TEXT_CONTENT_THRESHOLD = 0.58
MIN_COVERAGE_RATIO = 0.24
MIN_PURITY_RATIO = 0.08
MIN_CONTEXT_PURITY = 0.72
MIN_VERIFICATION_SCORE = 0.40
MAX_CENTROID_OFFSET_RATIO = 0.18
LOW_MATCH_STRICT_THRESHOLD = 0.58
```

## Text Labels (Symbole Tekstowe)

Dodane w commicie `3186d5d Progres tekstowy`. Uniwersalna ścieżka image-based dla symboli z napisem.

Przykłady symboli label: `TM`, `TAB`, `TSM`, `MSW`, `GSW`, `INT`, `TV`

### Zasada Działania

- Brak mapy `MSW=05`, `GSW=04`, `TSM=03` — silnik rozpoznaje obrazowo.
- Dla labeli tworzona jest `content_mask`: maska samej treści (litery/znaki) po odjęciu ramek i linii.
- Kandydat jest oceniany po pełnym wzorcu i po samej treści.
- Zwycięstwo zależy mocniej od `content_score` niż od ramki.

### Dlaczego To Ważne

- `MSW` i `GSW` mają podobne ramki — samo matching po ramce myli `04/05`.
- `TSM` może mieć prostokąt/kreskę przesuniętą, napis nadal czytelny.
- `INT/TV` mogą być odwrócone/lustrzane — trzeba patrzeć na obraz, nie na PDF text layer.

### Czego Nie Robić

- Nie wprowadzać słownika `MSW=05` — to rozbija uniwersalność.
- Nie bazować na PDF text layer — produkcyjnie wejście może być skanem/zdjęciem.
- Nie agresywnie zmieniać `content_mask` bez sprawdzenia regresji MSW/GSW.

## Nazewnictwo Wzorców Legendy

Nazwy wzorców po ekstrakcji legendy są częścią jakości detekcji, bo przechodzą
do wyników analizy i panelu korekty.

Aktualna kolejność źródeł nazwy:

1. Opis tekstowy z tego samego wiersza legendy.
2. OCR opisu, jeżeli tekst PDF nie jest dostępny albo legenda jest rastrowa.
3. Krótki indeks symbolu (`A`, `B`, `D1`, `GSW`, `MSW`) jako indeks/fallback.
4. Czytelny fallback UI/backend zamiast surowego `nieznany_symbol`.

Ważne: nie tworzyć słownika pod jeden PDF. Przypadki typu `A + kółko` oraz
`B + kwadrat` mają być rozwiązane przez grupowanie komponentów i wierszy, nie
przez kolejność elementów na konkretnym screenie.

## Rodzinne Promocje

Mechanizm w `backend/core/detector_promotions.py`. Mniejszy rdzeń (`child/core`) może zostać podniesiony do pełniejszego symbolu (`parent`), jeśli template'y są w relacji zawierania.

### Aktualnie Ręczne Reguły

- Rodzina `06/07`
- Rodzina `10/11/12`

**Nie usuwać gwałtownie.** Historycznie usunięcie tych reguł powracało z błędami.

### Docelowy Kierunek

Zastąpić relacjami wynikającymi z geometrii masek template'ów (zawieranie pikseli, dodatkowe piksele). Nie z ręcznego wpisania symbolu. Najpierw logować nowy mechanizm równolegle w debug, dopiero potem przełączać finalną decyzję.

## HITL i Debug Candidates

Warstwa dodana w commicie `6fb831a Niepewne bledy HITL debug`.

### Typy Kandydatów Debug

| Typ | Znaczenie |
| --- | --- |
| `accepted_uncertain` | Finalny zielony box istnieje, ale metryki są graniczne — człowiek powinien sprawdzić |
| `rejected_candidate` | Template coś widział, ale walidacja odrzuciła |
| `rejected_low_content` | Label/text kandydat odpadł po treści |
| `unexplained_component` | Kolorowy komponent planu niewyjaśniony przez finalne boxy |
| `overlap_conflict` | Kilka klas walczy o podobny obszar |
| `partial_ghost` | Częściowy duplikat/ghost w środku większego symbolu |

### W UI

- Zielone boxy: finalne detekcje.
- Czerwone/pomarańczowe boxy: debug, nie są finalną detekcją.
- `Sprawdź`: finalny box niepewny (`accepted_uncertain`).
- `Może`: kandydat odrzucony, potencjalnie przydatny.
- `Brak?`: niewyjaśniony komponent (`unexplained_component`).

### Ważne Zasady

- `Brak?` może być szeroki i obejmować kilka połączonych kresek — nie zna klasy symbolu.
- `accepted_uncertain` nie powinien mieć przycisku "Dodaj" — to już finalny box. Użytkownik zmienia klasę lub usuwa.
- `rejected_candidate` i `unexplained_component` mogą być ręcznie dodane przez użytkownika.
- Debug payload klikniętego boxa zawiera `frontend_nearby_boxes` i `frontend_nearby_debug_candidates` — kluczowe do diagnozy sąsiedztwa.
