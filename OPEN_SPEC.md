# ElektroScan AI OpenSpec

Aktualizacja: 2026-05-15

To jest główny punkt wejścia do dokumentacji projektu. Szczegółowe pliki źródłowe są w katalogu `openspec/`; ten plik trzyma aktualny skrót, zasady pracy i mapę dokumentów.

## Aktualny Stan

- Repozytorium robocze: `/Users/jakublewosz/Code/matiprojekt`
- Gałąź produkcyjna/demo: `main`
- Frontend: React/Vite, zwykle `http://127.0.0.1:5173`
- Backend: FastAPI, zwykle `http://127.0.0.1:8000`
- Uruchomienie całości: `docker compose up -d --build`
- Lokalny backend do testów: `PYTHONPATH=backend backend/venv/bin/python -m pytest backend/tests/unit`
- Lokalny frontend do testów: `cd frontend && npm run test -- --run`, build: `npm run build`

## Cel Produktu

ElektroScan AI analizuje plany elektryczne z PDF, pozwala użytkownikowi zaznaczyć legendę, wyciągnąć wzorce symboli, sprawdzić i nazwać je, a następnie wykryć wystąpienia tych symboli na planie. Wyniki służą do korekty projektu oraz eksportu zestawienia ilości elementów do Excela.

## Aktualny Przepływ Użytkownika

1. Użytkownik loguje się lub zakłada konto. Nie ma weryfikacji e-mail.
2. Tworzy projekt albo wraca do istniejącego.
3. W projekcie wgrywa lub przywraca PDF.
4. Zaznacza strefę legendy.
5. Uruchamia wyciąganie legendy z zaznaczenia.
6. Sprawdza wzorce w panelu `Sprawdź wzorce legendy`.
7. Może zaakceptować, odrzucić, przyciąć, dodać brakujący wzorzec albo poprawić nazwę.
8. Po sprawdzeniu wzorców może uruchomić analizę planu.
9. Wyniki wracają w prawym panelu, gdzie można rozwijać grupy, zmieniać nazwy/klasy i korygować detekcje.
10. W zakładce `Eksport` użytkownik pobiera plik `.xlsx` z aktualnym zestawieniem elementów i ilości.

Powrót do projektu ma przywracać podgląd PDF, warstwy, legendę, wzorce i ostatnią analizę. Wyjście z projektu nie powinno anulować trwającej analizy ani kasować wyniku.

## Najważniejsze Moduły

- `backend/main.py` - API FastAPI, auth, projekty, upload/preview, legendy, analiza, eksport XLSX, template management.
- `backend/core/legend_extractor.py` - ekstrakcja legendy z zaznaczenia, warianty tabelaryczne i klasyczne, OCR opisów, grupowanie symboli, normalizacja nazw.
- `backend/core/detector.py` - silnik template matching dla planu.
- `backend/core/detector_pdf.py` - pomocnicze warstwy PDF, tekst i strefy wykluczeń.
- `frontend/src/App.tsx` - główny stan aplikacji, auth, projekty, sesje, projektowy flow analizy.
- `frontend/src/components/CanvasView.tsx` - podgląd PDF, zaznaczanie stref, zoom, overlay wyników.
- `frontend/src/components/LegendReviewPanel.tsx` - weryfikacja i korekta wzorców legendy.
- `frontend/src/components/ResultsPanel.tsx` - panel wyników, korekty po analizie i eksportu XLSX.
- `frontend/src/symbolLabels.ts` - heurystyki przyjaznego nazewnictwa symboli po stronie UI.

## Dane I Przechowywanie

W Dockerze backend zapisuje dane w `/app/data`. Lokalnie odpowiednikiem jest `backend/data`.

Projektowe dane są rozdzielone po `project_id`:

- PDF i metadane podglądu,
- zaznaczone strefy,
- wyciągnięte wzorce legendy,
- snapshot ostatniej analizy,
- historia analiz,
- stan sesji/projektu.

Legacy/globalne endpointy nadal istnieją dla kompatybilności, ale nowe funkcje powinny używać endpointów projektowych.

## Eksport Wyników Do Excela

Eksport jest projektowym endpointem backendu i operuje na aktualnym stanie UI,
nie tylko na surowym snapshocie analizy. Frontend wysyła bieżące `results`,
`boxes`, `analysisContext` i `symbolLabels`, dzięki czemu plik uwzględnia:

- odrzucone fałszywe detekcje,
- ręczne zmiany klasy symbolu,
- przyjazne nazwy wzorców z legendy,
- agregację kilku wzorców pod tę samą nazwę elementu.

Backend generuje prawdziwy plik `.xlsx` bez dodatkowych zależności i zwraca go z
`Content-Disposition`. To zastępuje dawny kosztorys ilość × cena; obecny produkt
ma eksportować liczby elementów, nie liczyć ceny.

## Legenda I Nazewnictwo

Ekstrakcja legendy obsługuje kilka typów źródeł:

- legendy tabelaryczne z kolumną symboli i opisami,
- klasyczne legendy bez siatki tabeli,
- kolorowe legendy z symbolami tekstowymi i graficznymi,
- szare/czarne legendy rastrowe z OCR opisów.

Aktualne założenia:

- Nie używamy twardego szukania po koordynatach konkretnych PDF.
- Symbol i jego oznaczenie tekstowe mają być grupowane razem, jeżeli należą do jednej pozycji legendy.
- Opis po prawej stronie ma być preferowaną nazwą, a krótki indeks typu `A`, `B`, `D1`, `GSW`, `MSW` ma być używany tylko gdy jest faktyczną nazwą/indeksem.
- Stare nazwy typu `nieznany_symbol` mają być unikane; fallback powinien być czytelny i możliwy do poprawy w UI.
- Istniejące wcześniej zapisane błędne wzorce mogą wymagać ponownego wyciągnięcia legendy albo ręcznej zmiany nazwy.

Szczególne przypadki pokryte po ostatnich poprawkach:

- Viking gray legends: nazwy z OCR zamiast `nieznany_symbol`.
- Tabelaryczne C1/D1: kwadrat ma być razem z właściwym indeksem.
- Kolorowe PW-E-02: `GSW`/`MSW` nie mogą przejmować opisów z sąsiednich wierszy; elementy `A + kółko` i `B + kwadrat` nie mogą się mieszać.

## Granice Silnika

Silnik detekcji na planie pozostaje osobnym komponentem. Zmiany w UI, projektach, sesjach i panelach korekty nie powinny przypadkowo przepisywać `detector.py`. Zmiany w ekstrakcji legendy są dozwolone, ale muszą zachować szybkie działanie na kolorowych PDF i brak hardcodowanych współrzędnych.

## Weryfikacja

Po zmianach funkcjonalnych minimalny zestaw kontroli:

- `PYTHONPATH=backend backend/venv/bin/python -m pytest backend/tests/unit`
- `cd frontend && npm run test -- --run`
- `cd frontend && npm run build`
- eksport XLSX: `backend/tests/unit/test_analysis_export.py` oraz
  `frontend/src/tests/ResultsPanelExport.test.tsx`
- API health: `curl -s http://127.0.0.1:8000/api/health`
- Frontend: `curl -I http://127.0.0.1:5173/`

Po zmianach w legendzie warto ręcznie sprawdzić:

- projekt z legendą tabelaryczną,
- projekt Viking gray,
- projekt kolorowy PW-E-02,
- powrót do projektu po wyjściu na listę projektów,
- analizę po przywróceniu projektu z już zaznaczoną legendą.

## Mapa Dokumentów

- `openspec/current-context.md` - najważniejszy aktualny kontekst i zasady dla kolejnych sesji.
- `openspec/architecture.md` - architektura backend/frontend i przepływy danych.
- `openspec/workflow.md` - uruchamianie, testowanie i typowy workflow.
- `openspec/api.md` - endpointy i kontrakty API.
- `openspec/detection.md` - detekcja i heurystyki dopasowania.
- `openspec/known-issues.md` - ryzyka, ograniczenia i regresje do pilnowania.
- `openspec/changelog.md` - historia istotnych zmian.
- `openspec/decisions.md` - decyzje projektowe.
- `openspec/devops.md` - Docker, środowiska i operacje.
- `openspec/performance.md` - notatki wydajnościowe.
- Pliki planów `*-plan.md` - historyczne plany wdrożeń, zostawione jako kontekst.
