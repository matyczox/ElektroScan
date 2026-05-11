# Znane Problemy i Złote Przypadki Regresyjne

Znane problemy nie są powodem do panicznego overfitu. Podejście: zebrać dane, zrozumieć, dopiero potem zmieniać.

## Aktualne Ryzyka Po Ostatnich Poprawkach

- OCR opisów legendy jest best-effort. Jeżeli PDF ma bardzo słabą jakość albo
  nietypową typografię, użytkownik nadal musi móc poprawić nazwę w
  `LegendReviewPanel`.
- Stare wzorce zapisane przed poprawkami mogą dalej mieć złą nazwę. Nie
  migrujemy ich magicznie; zalecane jest ponowne wyciągnięcie legendy albo
  ręczna zmiana nazwy.
- Nie dodawać wyjątków po koordynatach. Ostatnie poprawki dla `C1/D1`,
  `GSW/MSW` oraz `A/B` zostały zrobione przez ogólne grupowanie wierszy,
  komponentów i tekstu.
- Powrót do projektu musi odtwarzać canvas, wzorce i analizę. Czarny ekran po
  powrocie albo zablokowana analiza przy sprawdzonej legendzie to regresja.

## Legendy — Przypadki Do Pilnowania

| PDF / typ | Oczekiwane zachowanie | Ryzyko regresji |
| --- | --- | --- |
| Viking gray / raster | Wzorce dostają czytelne nazwy z OCR/opisu, nie `nieznany_symbol` | Brak Tesseract albo za agresywne czyszczenie OCR |
| Viking tabela C1/D1 | Kwadrat i indeks `C1`/`D1` są w tym samym cropie i mają właściwą nazwę | Crop brany z granic komórki zamiast komponentów |
| PW-E-02 kolor | `GSW` i `MSW` mają własne opisy, bez przejęcia sąsiedniego wiersza | Tekst przypisany po najbliższym x zamiast po wierszu |
| PW-E-02 kolor | `A + kółko` oraz `B + kwadrat` są osobnymi wzorcami | Litera `B` wpada do cropu `A` albo odwrotnie |
| Kolorowe klasyczne legendy | Symbol tekstowy i grafika z jednego wiersza są grupowane razem | Zbyt szerokie merge komponentów |

## Złote Przypadki Regresyjne

Te przypadki są ważniejsze niż "średnie wrażenie z UI". Jeżeli któryś się zmienia, trzeba rozumieć dlaczego.

Committed golden snapshoty:

- `backend/tests/golden/viking_bronisze_e8_gray_first_pdf_100pct.json` -
  pierwszy szary PDF zaakceptowany jako 100% aktualnego celu.

| PDF | Miejsce / bbox | Oczekiwane zachowanie | Dlaczego ważne |
| --- | --- | --- | --- |
| `PW-E-02 Rev2.pdf` | `2293,1548,48,31` | Ma być `12`, nie `11` | Historycznie największy problem, obecnie naprawiony |
| `PW-E-02 Rev2.pdf` | `2742,975,31,31` | `09` zamiast `06` — znany błąd, przynajmniej HITL/uncertain | Nie robić reguły po koordynacie |
| `PW-E-02 Rev2.pdf` | MSW/GSW okolice `2293,1856` i odwrócone MSW | Label rozstrzygany po treści, nie ramce | Testuje `content_mask` |
| `PW-E-01 Rev2 (1).pdf` | `TM/TSM` blisko siebie | Łapane obrazowo | TSM ma czasem przesuniętą kreskę/prostokąt |
| `PW-E-01 Rev2 (1).pdf` | `INT/TV` odwrócone/lustrzane | Działa przez obraz, nie PDF text | Testuje mirror/rotation labeli |
| `PW-E-01 Rev2 (1).pdf` | `08 E 400V` przy `1187,1767,46,44` | Finalnie wykryte | Pilnuje, żeby `08` nie znikło |
| `PW-E-01 Rev2 (1).pdf` | brakujący `08` obok `06 @1363,737` | Co najmniej debug `Brak?` / niepewny komponent | Trudny przypadek, HITL/rescue |
| `PW-E-01 Rev2 (1).pdf` | zielone klastry `11/12/13` | Braki widoczne w debug/HITL | Testuje partial ghosts i unexplained components |

Docelowo: przenieść do `tests/golden/pw_e_02_rev2.json` i `tests/golden/pw_e_01_rev2.json`. Na razie dane są w debug payloadach kopiowanych z UI. Do porównywania służy `backend/tools/compare_analysis_snapshot.py`.

## Znane Problemy

### 1. Brakujące / Trudne `08 E 400V`

Na `PW-E-01 Rev2 (1).pdf` część przypadków `08_E_400V` nie jest wykrywana.

**Dobry przykład (działa):**
```
symbol=08_E_400V_wypust_400V_zasilanie_kuchenki_zakonczony_puszka_nt
bbox=1187,1767,46,44
match=0.644, verification=0.638, coverage=0.720, purity=0.765
rotation=270, scale=1.000, mirrored=false
```

**Problemowy rejon:**
```
nearby: 06 @1363,737,32,31 | 09 @1299,722,34,34
```

**Diagnoza:**
- Dobry `08`: max match `~0.644`.
- Brakujący rejon: max match `08` wynosi tylko `~0.421`.
- Problem nie jest brakiem rotacji ani mirror.
- Kształt jest zlepiony/zniekształcony.

**Obecne rozwiązanie:** `Brak?` i `accepted_uncertain` w okolicy. Nie zmieniać finalnej klasy na siłę.

**Potencjalny kierunek:**
- Debug-only `template_probe`: słabe próby template'u przy niepewnym miejscu (`08_probe match=0.42`).
- Rescue dla parentów: mniejszy symbol + dodatkowy komponent razem pasują do pełniejszego template'u → HITL.

---

### 2. Zielone Układy `11/12/13`

Grupy `07`, `11`, `12`, `13` potrafią generować: poprawne `12`, dodatkowe `11`, brakujące `13`, duży `Brak?` obejmujący połączone symbole.

Często kilka symboli jest fizycznie zlepionych na rysunku — to nie zawsze błąd detektora.

**Obecne rozwiązanie:** `accepted_uncertain` dla zatłoczonych boksów, `unexplained_component` dla większych niepokrytych obszarów. HITL decyduje.

---

### 3. `06/09` Pomyłki

Na `PW-E-02 Rev2.pdf`:
```
powinno być: 06
wykrywa jako: 09
bbox: 2742,975,31,31
```

**Nie dopisywać reguły po tej koordynacie.**

Lepszy kierunek: porównać pełniejszy shape-score, logować lokalne konflikty, opracować ogólną regułę parent/core jeśli powtarza się rodzinnie.

---

### 4. `MSW/GSW` Ramki

`MSW` bywało czytane jako `04` zamiast `05` (ramka podobna do `GSW`). Obecnie text-label pipeline poprawił sytuację.

Uważać przy zmianach `content_mask`. Nie wprowadzać mapy `MSW=05`.

## Baseline dla Znanych PDF-ów

### `PW-E-02 Rev2.pdf`
- Oczekiwana liczba boxów: `134-139` (zależnie od warstw i etapu).
- `bbox=2293,1548` → ma być `12` (historycznie największy problem, naprawiony).
- `bbox=2742,975` → znany błąd `09` zamiast `06` (HITL).

### `PW-E-01 Rev2 / PW-E-01 Rev2 (1).pdf`
Sprawdzać:
- `TM`, `TSM`, `MSW`, `GSW`, `INT`, `TV` po image-content labels.
- `E 400V` przypadki `08`.
- Grupy zielonych `11/12/13`.
- Fałszywe środki `21/23` w zlepionych symbolach.

### `VIKING-BRONISZE-ELE-Rzuty-E8.pdf`

- Profil: `gray`.
- Status: pierwszy szary PDF zaakceptowany przez uzytkownika jako 100%.
- Golden: `backend/tests/golden/viking_bronisze_e8_gray_first_pdf_100pct.json`.
- Oczekiwany rozklad goldena: `01:7, 02:8, 03:11, 04:12, 05:13, 06:14, 07:16`.
- Ten baseline nie oznacza, ze inne szare PDF sa gotowe.

### `PW-E-02 Rev2.pdf` — kolorowa legenda klasyczna

- Ekstrakcja legendy powinna zwrócić około `22` wzorce dla aktualnego znanego
  przykładu.
- `GSW` i `MSW` muszą dostać właściwe opisy z własnych wierszy.
- Element `A + kółko` nie może zawierać fragmentu `B`.
- Element `B + kwadrat` musi zawierać literę `B` i zielony kwadrat razem.
- Jeśli nazwy wyglądają jak OCR-owy bełkot, najpierw sprawdzić przypisanie
  tekstu do wierszy, nie dopisywać mapy po kolejności.
