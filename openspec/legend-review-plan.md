# Plan: Review i Akceptacja Wzorcow Legendy

## Kontekst

Ekstrakcja wzorcow z legendy jest krytycznym etapem calego procesu. Jesli wzorzec zostanie przyciety niepoprawnie, pozniejsza detekcja moze liczyc zle elementy mimo poprawnego algorytmu wyszukiwania.

Dlatego po automatycznym wyznaczeniu znakow legendy uzytkownik powinien przejsc przez kazdy wzorzec, zaakceptowac go albo poprawic recznie zaznaczenie na legendzie.

## Cel

- Po ekstrakcji legendy aplikacja pokazuje etap "Sprawdz wzorce legendy".
- Kazdy znaleziony znak ma status review: `pending`, `accepted`, `fixed`, `rejected`.
- Analiza planu jest zablokowana, dopoki wszystkie wymagane wzorce nie zostana zaakceptowane, poprawione albo odrzucone.
- Uzytkownik moze recznie zaznaczyc poprawny obszar wzorca bezposrednio na obrazie legendy.

## Proponowany UX

1. Uzytkownik wgrywa PDF i zaznacza obszar legendy.
2. Backend wycina automatyczne kandydaty wzorcow.
3. Frontend otwiera panel/modal "Sprawdz wzorce legendy".
4. Dla kazdego wzorca uzytkownik widzi:
   - nazwe/indeks znaku,
   - podglad wycinka,
   - status,
   - akcje: `Akceptuj`, `Popraw zaznaczenie`, `Zmien nazwe`, `Odrzuc`.
5. Jesli wycinek jest zly, uzytkownik wybiera "Popraw zaznaczenie" i rysuje prostokat na legendzie.
6. Nowy reczny crop zastapi automatyczny wzorzec dla tej sesji.
7. Po przejsciu wszystkich wzorcow odblokowuje sie analiza.

## Zachowanie Recznej Korekty

- Reczne zaznaczenie powinno operowac na wspolrzednych oryginalnego podgladu planu/legendy, a nie na przeskalowanych wspolrzednych DOM.
- Zaznaczenie uzytkownika jest zrodlem prawdy dla cropa.
- System moze nadal normalizowac obraz wzorca do wspolnego formatu, ale nie powinien automatycznie przycinac go ponownie w sposob, ktory moglby usunac fragment etykiety lub symbolu.
- Nalezy wspierac przypadki, gdzie wzorzec zawiera etykiete plus symbol, np. `C1` nad kwadratem albo `D1` nad wypelnieniem.
- Nazwa wzorca zostaje zachowana, chyba ze uzytkownik ja zmieni.

## Zmiany Backend

Proponowany model danych review item:

```ts
type LegendTemplateReviewItem = {
  id: string;
  name: string;
  imageBase64: string;
  status: "pending" | "accepted" | "fixed" | "rejected";
  extractionMethod: "classic" | "table" | "manual";
  sourceBBoxPx?: [number, number, number, number];
  correctedBBoxPx?: [number, number, number, number];
  warnings?: string[];
};
```

Proponowane API:

- `GET /api/templates/review?session_id=...` - lista wzorcow do sprawdzenia.
- `POST /api/templates/{template_id}/accept` - akceptacja wzorca.
- `POST /api/templates/{template_id}/crop` - zapis recznego cropa.
- `PATCH /api/templates/{template_id}` - zmiana nazwy lub statusu.
- `DELETE /api/templates/{template_id}` - odrzucenie/usuniecie wzorca.

MVP moze zaczac prosciej: stan review trzymany w frontendzie, a backend dostaje tylko finalne poprawione wzorce. Docelowo lepsze jest jednak utrwalenie statusow po stronie backendu, zeby reload strony nie zerowal pracy uzytkownika.

## Zmiany Frontend

- Dodac panel/modal `LegendReviewPanel`.
- Dodac stan `legendReviewItems` w glownej logice aplikacji.
- Dodac tryb canvasu do recznej korekty pojedynczego wzorca, np. `legend-template-fix`.
- Pokazywac progres, np. `12/18 zaakceptowane`.
- Zablokowac przycisk analizy, dopoki review nie jest kompletne.
- Umiescic akcje poprawy w naturalnym miejscu przy podgladzie wzorca, bez wymagania ponownego przechodzenia calego flow uploadu.

## Reguly Akceptacji

- Kazdy automatycznie znaleziony wzorzec musi zostac obsluzony.
- Uzytkownik moze odrzucic falszywy wzorzec.
- Uzytkownik moze dodac brakujacy wzorzec recznie z obszaru legendy.
- Po korekcie recznej system uzywa poprawionego wzorca w analizie.
- Analiza nie startuje, jesli istnieje wzorzec ze statusem `pending`.

## Testy

- Unit backend: zapis recznego cropa tworzy poprawny obraz i metadane.
- Unit backend: zaakceptowany/poprawiony/odrzucony wzorzec zmienia status zgodnie z oczekiwaniem.
- Frontend: przycisk analizy jest zablokowany przy statusie `pending`.
- Frontend: akceptacja, odrzucenie i reczna korekta aktualizuja progres.
- E2E: uzytkownik poprawia crop dla wzorca tabelarycznego typu `C1`/`D1`, a analiza korzysta z poprawionej wersji.

## Non-goals

- Brak globalnego uczenia modelu na podstawie korekt uzytkownika.
- Brak zmiany glownego algorytmu detekcji w tym etapie.
- Brak automatycznego OCR jako warunku MVP.

## Kolejnosc Implementacji

1. Dodac review UI z akceptacja, odrzuceniem i blokada analizy.
2. Dodac reczny crop wybranego wzorca z obszaru legendy.
3. Dodac mozliwosc recznego dodania brakujacego wzorca.
4. Utrwalic review metadata po stronie backendu.
5. Dodac testy jednostkowe i E2E dla pelnego flow.

## Status Implementacji

MVP zostalo wdrozone:

- Frontend: `LegendReviewPanel` pokazuje wzorce, statusy, progres oraz akcje akceptacji, odrzucenia, zmiany nazwy, poprawy cropa i dodania brakujacego wzorca.
- Frontend: analiza jest blokowana, dopoki istnieje wzorzec ze statusem `pending`.
- Canvas: tryb korekty wzorca pozwala narysowac nowy crop na podgladzie legendy.
- Backend: `POST /api/templates/{template_name}/crop` zapisuje reczny crop jako wzorzec.
- Backend: `PATCH /api/templates/{template_name}` zmienia nazwe wzorca.
- Backend/frontend: nowy PDF startuje z pusta baza wzorcow; `/api/preview`
  czysci `backend/templates/`, a frontend nie laduje starych wzorcow przy starcie.
- UI: poprawiono polskie znaki i layout listy "Baza Wzorcow".
- Testy: dodano testy komponentu review dla progresu, akceptacji i blokady pustego wzorca.

Do dopracowania pozniej:

- Trwale review metadata per sesja po stronie backendu.
- E2E na realnym PDF z reczna korekta `C1`/`D1`.
- Lepszy UX dodawania brakujacego wzorca bez `prompt`.
