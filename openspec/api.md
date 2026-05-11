# API — Endpointy i Dane

Backend: FastAPI, port `8000`. Wszystkie odpowiedzi mają nagłówek `Cache-Control: no-store`.

## Endpointy

### Logowanie

```
POST /api/auth/register
POST /api/auth/login
GET /api/auth/me
PATCH /api/auth/me
POST /api/auth/password-reset/request
POST /api/auth/password-reset/confirm
GET /api/auth/sessions
DELETE /api/auth/sessions/{session_id}
POST /api/auth/logout-all
POST /api/auth/logout
```

Auth używa losowej sesji zapisanej w `HttpOnly` cookie
`elektroscan_session`. Hasła są hashowane po stronie backendu. Endpointy
projektowe wymagają zalogowanego użytkownika.

Rejestracja tworzy konto i sesję. Nie ma wymogu weryfikacji e-maila.
W lokalnym/dev trybie backend może zwrócić `resetToken` w odpowiedzi resetu
hasła, żeby dało się testować flow bez integracji mailowej. W produkcji należy
ustawić `ELEKTROSCAN_AUTH_DEV_TOKENS=false` i wysyłać token resetu kanałem
e-mail.

Reset hasła:

- `POST /api/auth/password-reset/request` zawsze zwraca neutralny komunikat,
  żeby nie ujawniać, czy konto istnieje.
- `POST /api/auth/password-reset/confirm` zużywa token jednorazowo, ustawia nowe
  hasło i usuwa aktywne sesje użytkownika.

Sesje:

- `GET /api/auth/sessions` zwraca aktywne sesje z flagą `isCurrent`.
- `DELETE /api/auth/sessions/{session_id}` usuwa jedną sesję.
- `POST /api/auth/logout-all` usuwa wszystkie sesje użytkownika.

### Projekty

```
GET /api/projects
POST /api/projects
GET /api/projects/{project_id}
PATCH /api/projects/{project_id}
DELETE /api/projects/{project_id}
GET /api/projects/{project_id}/analysis-runs
GET /api/projects/{project_id}/analysis-runs/{analysis_id}
```

Projekt jest właścicielskim workspace użytkownika. Dane robocze projektu żyją
w osobnych katalogach:

```text
backend/data/projects/{project_id}/uploads
backend/data/projects/{project_id}/templates
backend/data/projects/{project_id}/analysis_debug
```

Nowy upload PDF w projekcie czyści tylko `templates` tego projektu. Nie czyści
wzorców innych projektów ani globalnych legacy katalogów.

Dashboard projektu używa `latestSessionId`, `latestSourcePdf`,
`latestUploadAtUtc`, `latestAnalysisAtUtc` oraz `analysisCount` z listy
projektów. Po ponownym wejściu do projektu frontend używa `latestSessionId`, aby
odtworzyć ostatni podgląd PDF przez projektowe `render-preview` i `layers`.
Historia analiz jest zapisywana po udanym
`POST /api/projects/{project_id}/analyze` i dostępna przez `/analysis-runs`.
Frontend odtwarza ostatnią zakończoną analizę dla aktualnego `latestSessionId`
przez `GET /api/projects/{project_id}/analysis-runs/{analysis_id}`.

### Upload i Sesja

```
POST /api/preview
```
Upload pliku PDF. Renderuje podglad 300 DPI, tworzy `sessionId` i zwraca
diagnostyke PDF. Ten endpoint czysci `backend/templates/`, wiec nowy plan
zawsze startuje z pusta baza wzorcow.

```
GET /api/layers?session_id=...
```
Zwraca dostępne warstwy PDF dla sesji.

```
POST /api/render-preview?session_id=...
```
Renderuje podgląd PDF jako obraz (300 DPI).

```
POST /api/clear
```
Czyści stan sesji (uploads, wyniki).

### Analiza

```
POST /api/extract-legend?session_id=...
```
Ekstrahuje wzorce z recznie zaznaczonej strefy legendy PDF. Wymaga
`legend_zone`; brak strefy zwraca blad. Opcjonalnie przyjmuje
`hidden_layers`, `excluded_zones` i `detector_profile`.

**Response:** lista `patterns` ma `id`, `name`, `imgBase64` oraz poczatkowy
status `pending`. Frontend otwiera po tym review wzorcow.

```
POST /api/analyze?session_id=...
```
Uruchamia pełną detekcję symboli.

**Request body:**
```json
{
  "excluded_zones": [],
  "hidden_layers": [],
  "include_debug": true,
  "include_image": true
}
```

**Response:**
```json
{
  "results": [...],
  "boxes": [...],
  "resultImage": "base64...",
  "analysisContext": {
    "analysisId": "...",
    "sourcePdf": "...",
    "hiddenLayersUsed": [...],
    "hiddenLayersUnmatched": [...],
    "hiddenLayersRepr": "...",
    "performance": { ... }
  },
  "debugCandidates": [...]
}
```

Przy `include_debug=true` zapisywany jest snapshot JSON do `backend/analysis_debug/`.

### Zarządzanie Wzorcami (Templates)

```
GET /api/templates
```
Zwraca listę załadowanych wzorców z miniaturą base64, nazwą i ID.

```
POST /api/templates/upload
```
Dodaje nowy wzorzec PNG do `backend/templates/`.

```
POST /api/templates/{template_name}/crop
```
Zastepuje lub tworzy wzorzec na podstawie prostokata narysowanego przez
uzytkownika na podgladzie aktualnego PDF-a. Body:

```json
{
  "session_id": "...",
  "x": 100,
  "y": 200,
  "width": 80,
  "height": 60,
  "name": "C1",
  "hidden_layers": []
}
```

```
PATCH /api/templates/{template_name}
```
Zmienia nazwe wzorca. Zwraca zaktualizowany payload `pattern`.

```
DELETE /api/templates
```
Usuwa wszystkie wzorce z `backend/templates/`.

```
DELETE /api/templates/{template_name}
```
Usuwa jeden wzorzec po nazwie pliku.

**Uwaga:** aktualny review legendy uzywa `PATCH` do zmiany nazwy oraz
`POST /crop` do recznej korekty wzorca. Starszy `PatternModal` jest tylko
pomocniczym widokiem bazy.

Projektowe odpowiedniki obecnego workflow mają prefix:

```text
/api/projects/{project_id}/preview
/api/projects/{project_id}/layers
/api/projects/{project_id}/render-preview
/api/projects/{project_id}/extract-legend
/api/projects/{project_id}/analyze
/api/projects/{project_id}/inspect-roi
/api/projects/{project_id}/gray-debug-zones
/api/projects/{project_id}/templates
```

Frontend po zalogowaniu powinien używać wyłącznie endpointów projektowych.
Endpointy bez `project_id` zostają jako legacy/dev fallback.

## Debug Payload Boxa

Kliknięcie boxa w CanvasView kopiuje payload do schowka. Zawiera:

```json
{
  "symbol": "...",
  "bbox": "x,y,width,height",
  "match": 0.644,
  "verification": 0.638,
  "coverage": 0.720,
  "purity": 0.765,
  "context_purity": 0.82,
  "color_similarity": 0.91,
  "rotation": 270,
  "scale": 1.0,
  "mirrored": false,
  "source": "template",
  "reason": "...",
  "analysis_id": "...",
  "analysis_session": "...",
  "source_pdf": "...",
  "hidden_layers_used": [...],
  "hidden_layers_repr": "...",
  "frontend_nearby_boxes": [...],
  "frontend_debug_candidates_count": 3,
  "frontend_nearby_debug_candidates": [...]
}
```

## Frontend HITL — Co Może Użytkownik

- Po ekstrakcji legendy przejsc przez kazdy wzorzec w `LegendReviewPanel`.
- Zaakceptowac, odrzucic, zmienic nazwe albo poprawic crop wzorca.
- Dodac brakujacy wzorzec z obszaru legendy.
- Uruchomic analize dopiero po zakonczeniu review wszystkich wzorcow.
- Kliknąć box → skopiować debug payload.
- Usunąć fałszywy finalny box.
- Zmienić klasę finalnego boxa.
- Dodać debug-kandydata jako ręczny box.
- Dodać ręczny box z toolbaru (rysowanie na canvas).
- Ukryć debug-kandydata.

## Zarządzanie Wzorcami (UI)

- Sidebar wyświetla wzorce z miniaturą i nazwą.
- Nowy PDF startuje z pusta baza wzorcow; stare wzorce nie sa ladowane
  automatycznie przy starcie frontendu.
- Przycisk edycji otwiera `PatternModal` (zmiana nazwy lub usunięcie).
- Przycisk "Wyczyść całą bazę wiedzy" w Sidebar → `DELETE /api/templates`.

## Kosztorys Wykonawczy (CostPanel)

Panel po prawej stronie UI. Dla każdego symbolu z wyników detekcji:
- Ilość wykrytych instancji (readonly z wyników).
- Pole ceny netto PLN (edytowalne).
- Suma całkowita na dole.

Stan cen żyje tylko w React — nie jest persystowany ani wysyłany do backendu. Nie jest częścią silnika detekcji; to narzędzie pomocnicze do wstępnego kosztorysu.
