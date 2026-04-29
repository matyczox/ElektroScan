# API — Endpointy i Dane

Backend: FastAPI, port `8000`. Wszystkie odpowiedzi mają nagłówek `Cache-Control: no-store`.

## Endpointy

### Upload i Sesja

```
POST /api/upload
```
Upload pliku PDF. Zwraca `session_id` i listę warstw.

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
Ekstrahuje wzorce z legendy PDF. Opcjonalnie przyjmuje `hidden_layers`.

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
Dodaje nowy wzorzec (plik PNG/JPG) do `backend/templates/`.

```
DELETE /api/templates
```
Usuwa wszystkie wzorce z `backend/templates/`.

```
DELETE /api/templates/{template_name}
```
Usuwa jeden wzorzec po nazwie pliku.

**Uwaga:** edycja nazwy przez `PatternModal` w UI wykonuje sekwencję: `DELETE /api/templates/{id}` → `POST /api/templates/upload` z nową nazwą.

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

- Kliknąć box → skopiować debug payload.
- Usunąć fałszywy finalny box.
- Zmienić klasę finalnego boxa.
- Dodać debug-kandydata jako ręczny box.
- Dodać ręczny box z toolbaru (rysowanie na canvas).
- Ukryć debug-kandydata.

## Zarządzanie Wzorcami (UI)

- Sidebar wyświetla wzorce z miniaturą i nazwą.
- Przycisk edycji otwiera `PatternModal` (zmiana nazwy lub usunięcie).
- Przycisk "Wyczyść całą bazę wiedzy" w Sidebar → `DELETE /api/templates`.

## Kosztorys Wykonawczy (CostPanel)

Panel po prawej stronie UI. Dla każdego symbolu z wyników detekcji:
- Ilość wykrytych instancji (readonly z wyników).
- Pole ceny netto PLN (edytowalne).
- Suma całkowita na dole.

Stan cen żyje tylko w React — nie jest persystowany ani wysyłany do backendu. Nie jest częścią silnika detekcji; to narzędzie pomocnicze do wstępnego kosztorysu.
