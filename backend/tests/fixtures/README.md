# Test Fixtures

Ten katalog trzyma stabilne wejscia do przyszlych testow regresji.

Najwazniejsza zasada: `backend/templates/` jest stanem roboczym aplikacji po
ostatniej ekstrakcji legendy. Testy nie powinny na nim polegac, bo kolejny PDF
albo klikniecie w UI moze go nadpisac.

Docelowy przeplyw testu gray:

1. Wez PDF testowy i fixture legendy z tego katalogu.
2. Runner wysyla PDF do `/api/preview` i sprawdza klasyfikacje
   `recommendedProfile` (`gray` albo `color`).
3. Runner czysci `/api/templates` i uploaduje zapisane PNG z `templates/`.
4. Runner odpala `/api/analyze` z `detector_profile=auto`.
5. Runner porownuje wynik z odpowiednim goldenem z `backend/tests/golden/`.
6. Jesli sa roznice, najpierw sprawdz Inspektorem ROI czy to realna poprawa,
   regresja, czy tylko przesuniecie w gesto upakowanym fragmencie.

Komenda:

```powershell
py -3 backend/tools/run_golden_regression.py --api-url http://localhost:8000
```

Mozesz puscic jeden przypadek:

```powershell
py -3 backend/tools/run_golden_regression.py --fixture viking_bronisze_e10_gray
```

## Dodawanie kolejnego PDF

Dla kazdego stabilnego przypadku potrzebujemy czterech rzeczy:

1. PDF w `test_pdfs/`.
2. PNG wzorcow legendy w `backend/tests/fixtures/<case>/templates/`.
3. `manifest.json` z `expectedProfile` ustawionym na `gray` albo `color`.
4. Golden JSON w `backend/tests/golden/`.

`expectedProfile` testuje klasyfikacje PDF-a. Runner najpierw pyta backend przez
`/api/preview`, czy PDF jest `gray` czy `color`, a dopiero potem odpala analize.
Jesli klasyfikacja sie pomyli, test pada przed detekcja.

Dla kolorowych PDF-ow robimy taki sam fixture, tylko:

- `expectedProfile`: `color`
- `analyzeProfile`: `auto`
- osobny golden kolorowy
- osobne PNG legendy kolorowej

Nie mieszaj template PNG z roznych PDF-ow. Jesli legenda zostala kliknieta albo
poprawiona recznie w UI, po zaakceptowaniu wyniku od razu zapisz stan
`/api/templates` do fixture. Inaczej golden bedzie mial boxy z jednej legendy, a
test bedzie puszczal inna legende i zacznie klamac.

## Status

- `viking_bronisze_e8_gray` jest aktywny i przechodzi.
- `viking_bronisze_e9_gray` jest aktywny i przechodzi z zapisanymi strefami
  zakazanymi na tekst/ramke.
- `viking_bronisze_e10_gray` jest aktywny i przechodzi jako baseline 90-95%.

Nie aktualizuj goldenow tylko dlatego, ze lokalny run dal inne liczby. Golden ma
bronic przed regresja, a nie zmuszac silnik do 100% na jednym PDF kosztem innych.
