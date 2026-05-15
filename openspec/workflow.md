# Workflow — Uruchamianie, Praca z AI, Debugowanie

## Jak Uruchomić

Zalecany tryb dla pełnego projektu:

```bash
docker compose up -d --build
```

- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`
- Health backendu: `curl -s http://127.0.0.1:8000/api/health`
- Frontend smoke: `curl -I http://127.0.0.1:5173/`

Backend w Dockerze zapisuje dane w `/app/data`. Lokalnie odpowiednikiem jest
`backend/data`.

## Lokalny Tryb Dev

Backend:

```bash
cd /Users/jakublewosz/Code/matiprojekt
backend/venv/bin/python backend/main.py
```

Frontend:

```bash
cd /Users/jakublewosz/Code/matiprojekt/frontend
npm run dev -- --host 127.0.0.1
```

Jeżeli frontend albo backend pokazuje stary stan, sprawdź procesy i kontenery:

```bash
docker compose ps
docker compose logs backend --tail=80
docker compose logs frontend --tail=80
```

## Komendy Kontrolne

Przed zmianami:

```bash
git status --short
git log --oneline -5
```

Po zmianach backendowych:

```bash
PYTHONPATH=backend backend/venv/bin/python -m pytest backend/tests/unit
```

Po zmianach frontendowych:

```bash
cd frontend
npm run test -- --run
npm run build
```

Po zmianach w eksporcie:

```bash
PYTHONPATH=backend backend/venv/bin/python -m pytest backend/tests/unit/test_analysis_export.py -q
cd frontend && npm run test -- --run src/tests/ResultsPanelExport.test.tsx
```

Po zmianach w Dockerfile albo zależnościach:

```bash
docker compose build
docker compose up -d
```

## Aktualny Flow Produktowy

1. Użytkownik loguje się albo zakłada konto. Nie ma weryfikacji e-mail.
2. Tworzy projekt albo wybiera istniejący.
3. Wgrywa PDF do projektu lub wraca do ostatniej sesji PDF.
4. Wybiera warstwy PDF, jeżeli są potrzebne.
5. Zaznacza strefę legendy trybem `Legenda`.
6. Klika `Wyciągnij legendę z zaznaczenia`.
7. Przechodzi przez `Sprawdź wzorce legendy`.
8. Akceptuje, odrzuca, poprawia crop, dodaje brakujący wzorzec albo zmienia
   nazwę.
9. Uruchamia `Analizuj Plan`.
10. Koryguje wynik w prawym panelu: rozwija grupy, zmienia nazwę/klasę lub usuwa
    fałszywe detekcje.
11. Przechodzi do zakładki `Eksport` i pobiera `.xlsx` z aktualnym zestawieniem
    elementów oraz ilości.

Wyjście z projektu do listy projektów ma zachować stan. Powrót ma odtworzyć
preview PDF, warstwy, zaznaczoną legendę, sprawdzone wzorce i ostatnią analizę.

Eksport XLSX zastępuje dawny kosztorys. Plik ma być liczony z aktualnego stanu
UI po korektach, a nie tylko z pierwszej odpowiedzi backendu. Przed eksportem
użytkownik może odrzucić fałszywe boxy albo zmienić klasę detekcji; plik powinien
pokazać dokładnie te ilości, które widać w panelu wyników.

## Aktualny Flow Legendy

- Nowy PDF w projekcie czyści bazę wzorców tylko tego projektu.
- Zaznaczona strefa legendy jest źródłem prawdy; nie używać szukania po
  współrzędnych konkretnego pliku.
- Ekstraktor obsługuje tabele, klasyczne legendy, kolorowe symbole oraz szare
  legendy z OCR.
- Nazwa wzorca powinna pochodzić z opisu w tym samym wierszu, a krótki indeks
  typu `A`, `B`, `D1`, `GSW`, `MSW` jest pomocniczy.
- Analiza planu odblokowuje się dopiero wtedy, gdy nie ma wzorców `pending`.
- Stare wzorce zapisane przed poprawkami nazewnictwa mogą wymagać ponownego
  wyciągnięcia legendy albo ręcznej zmiany nazwy.

Przypadki, które warto ręcznie sprawdzić po zmianach w legendzie:

- tabelaryczne `C1`/`D1` z kwadratami,
- Viking gray z nazwami z OCR zamiast `nieznany_symbol`,
- kolorowe `GSW`/`MSW`,
- kolorowe `A + kółko` oraz `B + kwadrat`.

## Debugowanie Nowego Błędu

Nie zaczynać od reguły po koordynatach. Najpierw zebrać dane:

1. Zanotuj projekt, PDF, warstwy i profil detekcji.
2. Sprawdź, czy problem dotyczy ekstrakcji legendy czy analizy planu.
3. Jeżeli to legenda, zrób ponowny crop i sprawdź opis w `LegendReviewPanel`.
4. Jeżeli to detekcja, użyj Inspektora ROI na problematycznym miejscu.
5. Porównaj podobny poprawny przypadek z tego samego PDF.
6. Dopiero potem zmieniaj progi, ranking albo heurystyki.

## Format Zgłoszenia Błędu

```text
Projekt:
PDF:
Warstwy ukryte:
Profil:
Oczekiwany symbol/nazwa:
Aktualny symbol/nazwa:
bbox albo crop:
Czy problem jest w legendzie czy w analizie:
screen/crop:
Poprawny podobny przykład:
```

## Praca Z AI

Dobre polecenie startowe:

```text
Przeczytaj OPEN_SPEC.md oraz openspec/current-context.md. Sprawdź git status.
Nie hardcoduj koordynat ani nazw pod jeden PDF. Endpointy projektowe są
preferowane po zalogowaniu. Legendę poprawiaj przez ogólne reguły tabel,
wierszy, komponentów i OCR. Nie przywracaj starego panelu debugCandidates jako
domyślnego UI; do diagnostyki używaj Inspektora ROI.
```
