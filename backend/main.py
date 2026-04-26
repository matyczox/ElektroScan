import os
import shutil
import uuid
import base64
import json
from datetime import datetime, timezone
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List

# Importujemy nasze core'owe moduły
from core.legend_extractor import pdf_to_png, extract_legend, get_pdf_layers, _normalize_layer_name
from core.detector import load_templates, detect_symbols, draw_results

app = FastAPI(title="ElektroScan AI API")

# Konfiguracja CORS - pozwala na komunikację z frontendem Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_response_caching(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Ścieżki robocze
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
ANALYSIS_DIR = BASE_DIR / "analysis_debug"
SESSION_META_SUFFIX = ".meta.json"

# Upewniamy się, że foldery istnieją
UPLOAD_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)


def _session_meta_path(session_id: str) -> Path:
    return UPLOAD_DIR / f"{session_id}{SESSION_META_SUFFIX}"


def _write_session_meta(session_id: str, *, source_pdf: str) -> None:
    _session_meta_path(session_id).write_text(
        json.dumps({"sourcePdf": source_pdf}, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_session_meta(session_id: str) -> dict:
    meta_path = _session_meta_path(session_id)
    if not meta_path.exists():
        return {}

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _analysis_snapshot_path(analysis_id: str) -> Path:
    return ANALYSIS_DIR / f"{analysis_id}.json"


def _write_analysis_snapshot(analysis_id: str, payload: dict) -> None:
    _analysis_snapshot_path(analysis_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_hidden_layer_debug(pdf_path: str, hidden_layers: list[str]) -> dict:
    available_layers = get_pdf_layers(pdf_path)
    available_names = [str(layer.get("name", "")) for layer in available_layers if layer.get("name")]
    normalized_available = {}
    for name in available_names:
        normalized_available.setdefault(_normalize_layer_name(name), []).append(name)

    requested = []
    matched = []
    unmatched = []

    for raw_name in hidden_layers:
        normalized = _normalize_layer_name(raw_name)
        matches = normalized_available.get(normalized, [])
        entry = {
            "value": raw_name,
            "repr": ascii(raw_name),
            "length": len(raw_name),
            "normalized": normalized,
            "matches": matches,
        }
        requested.append(entry)
        if matches:
            matched.append(raw_name)
        else:
            unmatched.append(raw_name)

    return {
        "requested": requested,
        "matched": matched,
        "unmatched": unmatched,
    }

@app.get("/")
async def root():
    return {"message": "ElektroScan AI API is running"}

class ExtractRequest(BaseModel):
    excluded_zones: Optional[List[dict]] = []
    hidden_layers: Optional[List[str]] = []

class RenderRequest(BaseModel):
    hidden_layers: Optional[List[str]] = []

@app.post("/api/preview")
async def api_preview(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Tylko pliki PDF są obsługiwane.")
    
    session_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{session_id}.pdf"
    
    # Zapis
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    _write_session_meta(session_id, source_pdf=file.filename or file_path.name)
        
    try:
        # Render podglądu (300 DPI — identycznie jak detekcja)
        plan_img = pdf_to_png(str(file_path), dpi=300)
        _, buffer_plan = cv2.imencode('.jpg', plan_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        plan_base64 = base64.b64encode(buffer_plan).decode('utf-8')
        
        return {
            "planPreview": f"data:image/jpeg;base64,{plan_base64}",
            "sessionId": session_id,
            "sourcePdf": file.filename or file_path.name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/layers")
async def api_layers(session_id: str):
    file_path = UPLOAD_DIR / f"{session_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")
    layers = get_pdf_layers(str(file_path))
    return {"layers": layers}

@app.post("/api/render-preview")
async def api_render_preview(session_id: str, body: RenderRequest = None):
    file_path = UPLOAD_DIR / f"{session_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")
    try:
        hidden_layers = body.hidden_layers if body else []
        plan_img = pdf_to_png(str(file_path), dpi=300, hidden_layers=hidden_layers)
        _, buffer_plan = cv2.imencode('.jpg', plan_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        plan_base64 = base64.b64encode(buffer_plan).decode('utf-8')
        return {
            "planPreview": f"data:image/jpeg;base64,{plan_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract-legend")
async def api_extract_legend(session_id: str, body: ExtractRequest = None):
    file_path = UPLOAD_DIR / f"{session_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")
        
    try:
        print(f"Renderowanie planu do ekstrakcji (300 DPI)")
        hidden_layers = body.hidden_layers if body else []
        plan_img = pdf_to_png(str(file_path), dpi=300, hidden_layers=hidden_layers)
        
        # Strefy wykluczone
        exclude_rects = []
        if body and body.excluded_zones:
            for zone in body.excluded_zones:
                try:
                    exclude_rects.append((
                        int(zone["x"]), int(zone["y"]),
                        int(zone["width"]), int(zone["height"])
                    ))
                except (KeyError, ValueError):
                    pass
        
        print("Ekstrakcja legendy...")
        if TEMPLATES_DIR.exists():
            shutil.rmtree(TEMPLATES_DIR)
        TEMPLATES_DIR.mkdir(exist_ok=True)
        
        symbols = extract_legend(
            str(file_path), 
            plan_img, 
            output_dir=str(TEMPLATES_DIR),
            exclude_rects=exclude_rects
        )
        
        # Generujemy podgląd jeszcze raz na wypadek gdyby UI go potrzebowało w pełnej rozdz.
        # Ale zwracamy wzorce.
        patterns_list = []
        for s in symbols:
            _, buffer_s = cv2.imencode('.png', s.image)
            img_b64 = base64.b64encode(buffer_s).decode('utf-8')
            patterns_list.append({
                "name": s.name,
                "imgBase64": f"data:image/png;base64,{img_b64}"
            })
            
        return {
            "patterns": patterns_list
        }
        
    except Exception as e:
        print(f"Błąd podczas ekstrakcji: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")

from pydantic import BaseModel
from typing import Optional, List

class AnalyzeRequest(BaseModel):
    excluded_zones: Optional[List[dict]] = []
    hidden_layers: Optional[List[str]] = []

@app.post("/api/analyze")
async def api_analyze(session_id: str, body: AnalyzeRequest = None):
    plan_path = UPLOAD_DIR / f"{session_id}.pdf"
    
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")
        
    try:
        # 1. Ładujemy plan
        hidden_layers = body.hidden_layers if body else []
        session_meta = _read_session_meta(session_id)
        plan_img = pdf_to_png(str(plan_path), dpi=300, hidden_layers=hidden_layers)
        
        # 2. Ładujemy wzorce
        templates = load_templates(str(TEMPLATES_DIR))
        
        # 3. Strefy wykluczone → lista krotek (x, y, w, h)
        exclude_rects = []
        if body and body.excluded_zones:
            for zone in body.excluded_zones:
                try:
                    exclude_rects.append((
                        int(zone["x"]), int(zone["y"]),
                        int(zone["width"]), int(zone["height"])
                    ))
                except (KeyError, ValueError):
                    pass
            print(f"Strefy wykluczone: {exclude_rects}")

        hidden_layer_debug = _build_hidden_layer_debug(str(plan_path), hidden_layers)
        
        # 4. Detekcja
        results = detect_symbols(
            plan_img,
            templates,
            exclude_rects=exclude_rects,
            pdf_path=str(plan_path),
            pdf_dpi=300,
            hidden_layers=hidden_layers,
        )
        
        # 5. Rysujemy ramki
        result_img = draw_results(plan_img, results)
        
        # 6. Konwersja wyniku do base64
        _, buffer_res = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result_base64 = base64.b64encode(buffer_res).decode('utf-8')
        
        # Przygotowujemy dane o ramkach dla frontendu (opcjonalnie)
        # Na razie wysyłamy gotowy obraz i listę wyników
        
        analysis_context = {
            "analysisId": str(uuid.uuid4()),
            "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
            "sessionId": session_id,
            "sourcePdf": session_meta.get("sourcePdf", plan_path.name),
            "hiddenLayersUsed": hidden_layers,
            "excludedZonesUsed": exclude_rects,
            "hiddenLayerDebug": hidden_layer_debug,
        }

        formatted_results = []
        all_boxes = []

        for r in results:
            # Podsumowanie per typ symbolu (do CostPanel)
            formatted_results.append({
                "name": r.symbol_name,
                "count": r.count,
                "color": r.color,
            })
            # Każda indywidualna detekcja (do Canvas)
            for det in r.detections:
                all_boxes.append({
                    "id": f"{r.symbol_name}_{det.x}_{det.y}",
                    "symbolName": r.symbol_name,
                    "x": det.x,
                    "y": det.y,
                    "width": det.width,
                    "height": det.height,
                    "confidence": det.confidence,
                    "verificationScore": det.verification_score,
                    "source": det.source,
                    "rotation": det.rotation,
                    "scale": det.scale,
                    "mirrored": det.mirrored,
                    "coverage": det.coverage,
                    "purity": det.purity,
                    "contextPurity": det.context_purity,
                    "colorSimilarity": det.color_similarity,
                    "analysisId": analysis_context["analysisId"],
                    "analysisGeneratedUtc": analysis_context["generatedAtUtc"],
                    "analysisSession": analysis_context["sessionId"],
                    "sourcePdf": analysis_context["sourcePdf"],
                    "hiddenLayersUsed": analysis_context["hiddenLayersUsed"],
                    "color": r.color,
                })

        response_payload = {
            "message": "Analiza zakończona",
            "analysisContext": analysis_context,
            "results": formatted_results,
            "boxes": all_boxes,
            "resultImage": f"data:image/jpeg;base64,{result_base64}"
        }

        try:
            _write_analysis_snapshot(
                analysis_context["analysisId"],
                {
                    "analysisContext": analysis_context,
                    "results": formatted_results,
                    "boxes": all_boxes,
                    "resultImageLength": len(response_payload["resultImage"]),
                },
            )
        except OSError as snapshot_error:
            print(f"Nie udało się zapisać snapshotu analizy: {snapshot_error}")

        return response_payload
        
    except Exception as e:
        print(f"Błąd podczas analizy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/templates")
async def api_get_templates():
    patterns_list = []
    if TEMPLATES_DIR.exists():
        for file_path in TEMPLATES_DIR.glob("*.png"):
            img = cv2.imread(str(file_path))
            if img is not None:
                _, buffer = cv2.imencode('.png', img)
                img_b64 = base64.b64encode(buffer).decode('utf-8')
                patterns_list.append({
                    "name": file_path.stem,
                    "imgBase64": f"data:image/png;base64,{img_b64}"
                })
    return {"patterns": patterns_list}

@app.post("/api/templates/upload")
async def api_upload_template(file: UploadFile = File(...)):
    """Ręczny upload wzorca PNG do bazy wiedzy."""
    if not file.filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Tylko pliki PNG są obsługiwane.")
    
    safe_name = Path(file.filename).stem
    dest_path = TEMPLATES_DIR / f"{safe_name}.png"
    
    try:
        with dest_path.open("wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        return {"message": f"Wzorzec '{safe_name}' dodany do bazy.", "name": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/templates")
async def api_delete_templates():
    if TEMPLATES_DIR.exists():
        shutil.rmtree(TEMPLATES_DIR)
        TEMPLATES_DIR.mkdir(exist_ok=True)
    return {"message": "Baza wiedzy wyczyszczona."}

@app.delete("/api/templates/{template_name}")
async def api_delete_template(template_name: str):
    target = TEMPLATES_DIR / f"{template_name}.png"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono wzorca.")

    target.unlink()
    return {"message": f"Wzorzec '{template_name}' usunięty."}

@app.post("/api/clear")
async def api_clear():
    # Czyścimy wszystko
    for folder in [UPLOAD_DIR, TEMPLATES_DIR]:
        if folder.exists():
            shutil.rmtree(folder)
            folder.mkdir(exist_ok=True)
    return {"message": "Wyczyszczono dane robocze."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
