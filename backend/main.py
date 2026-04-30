import base64
import json
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import fitz
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.detector import detect_symbols, draw_results, load_templates
from core.roi_inspector import inspect_roi

# Importujemy nasze core'owe moduły
from core.legend_extractor import _normalize_layer_name, extract_legend, get_pdf_layers, pdf_to_png

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

SNAPSHOT_EXECUTOR = ThreadPoolExecutor(max_workers=1)
ANALYSIS_PROGRESS: dict[str, dict] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


VERBOSE_LOGS = _env_flag("ELEKTROSCAN_VERBOSE_LOGS", default=True)
DEFAULT_ANALYSIS_DEBUG = _env_flag("ELEKTROSCAN_ANALYSIS_DEBUG", default=True)


def _log(message: str) -> None:
    if VERBOSE_LOGS:
        print(message)


def _set_analysis_progress(
    session_id: str,
    stage: str,
    percent: float,
    detail: str = "",
    *,
    analysis_id: str | None = None,
    done: bool = False,
    error: str | None = None,
) -> None:
    previous = ANALYSIS_PROGRESS.get(session_id, {})
    ANALYSIS_PROGRESS[session_id] = {
        "sessionId": session_id,
        "analysisId": analysis_id or previous.get("analysisId"),
        "stage": stage,
        "percent": round(max(0.0, min(100.0, float(percent))), 1),
        "detail": detail,
        "done": done,
        "error": error,
        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
    }


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


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _slowest_stages(timings_ms: dict[str, float], limit: int = 8) -> list[dict]:
    ignored = {"total", "totalBeforeSnapshot"}
    ranked = sorted(
        (
            (name, value)
            for name, value in timings_ms.items()
            if name not in ignored and isinstance(value, (int, float))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"name": name, "ms": round(float(value), 3)} for name, value in ranked[:limit]]


def _build_hidden_layer_debug(pdf_path: str, hidden_layers: list[str]) -> dict:
    available_layers = get_pdf_layers(pdf_path)
    available_names = [
        str(layer.get("name", "")) for layer in available_layers if layer.get("name")
    ]
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


def _normalize_detector_profile(profile: str | None) -> str:
    value = (profile or "auto").strip().lower()
    return value if value in {"auto", "color", "gray"} else "auto"


def _ink_profile_stats(plan_img: np.ndarray) -> dict:
    arr = plan_img.astype(np.int16)
    max_channel = arr.max(axis=2)
    min_channel = arr.min(axis=2)
    saturation = max_channel - min_channel
    ink = max_channel < 245
    ink_pixels = int(np.count_nonzero(ink))
    total_pixels = int(plan_img.shape[0] * plan_img.shape[1])

    if ink_pixels == 0 or total_pixels == 0:
        return {
            "inkPct": 0.0,
            "colorfulInkPct": 0.0,
            "grayInkPct": 0.0,
            "recommendedProfile": "color",
        }

    colorful = ink & (saturation > 35)
    colorful_pixels = int(np.count_nonzero(colorful))
    gray_pixels = ink_pixels - colorful_pixels
    colorful_ink_pct = (colorful_pixels / ink_pixels) * 100.0

    return {
        "inkPct": round((ink_pixels / total_pixels) * 100.0, 3),
        "colorfulInkPct": round(colorful_ink_pct, 3),
        "grayInkPct": round((gray_pixels / ink_pixels) * 100.0, 3),
        "recommendedProfile": "gray" if colorful_ink_pct < 1.0 else "color",
    }


def _build_pdf_diagnostics(pdf_path: str, plan_img: np.ndarray | None = None) -> dict:
    diagnostics = {
        "pages": 0,
        "layers": 0,
        "textCharsPage1": 0,
        "textBlocksPage1": 0,
        "drawingsPage1": 0,
        "imagesPage1": 0,
        "inkPct": 0.0,
        "colorfulInkPct": 0.0,
        "grayInkPct": 0.0,
        "recommendedProfile": "color",
    }

    try:
        diagnostics["layers"] = len(get_pdf_layers(pdf_path))
    except Exception:
        pass

    doc = fitz.open(pdf_path)
    try:
        diagnostics["pages"] = int(doc.page_count)
        if doc.page_count:
            page = doc.load_page(0)
            text_blocks = page.get_text("blocks")
            diagnostics["textBlocksPage1"] = int(
                sum(1 for block in text_blocks if len(block) > 6 and block[6] == 0)
            )
            diagnostics["textCharsPage1"] = int(len(page.get_text("text") or ""))
            try:
                diagnostics["drawingsPage1"] = int(len(page.get_drawings()))
            except Exception:
                diagnostics["drawingsPage1"] = 0
            try:
                diagnostics["imagesPage1"] = int(len(page.get_images(full=True)))
            except Exception:
                diagnostics["imagesPage1"] = 0
    finally:
        doc.close()

    if plan_img is None:
        try:
            plan_img = pdf_to_png(pdf_path, dpi=100)
        except Exception:
            plan_img = None

    if plan_img is not None:
        diagnostics.update(_ink_profile_stats(plan_img))

    return diagnostics


@app.get("/")
async def root():
    return {"message": "ElektroScan AI API is running"}


class LegendZone(BaseModel):
    page: Optional[int] = 0
    x: float
    y: float
    width: float
    height: float


def _zone_to_rect(zone: Optional[LegendZone]) -> tuple[int, int, int, int] | None:
    if zone is None:
        return None
    rect = (
        int(round(zone.x)),
        int(round(zone.y)),
        int(round(zone.width)),
        int(round(zone.height)),
    )
    if rect[2] <= 0 or rect[3] <= 0:
        return None
    return rect


def _clamp_rect_to_image(
    rect: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    image_h, image_w = image_shape[:2]
    x, y, w, h = rect
    x1 = max(0, min(image_w, x))
    y1 = max(0, min(image_h, y))
    x2 = max(0, min(image_w, x + w))
    y2 = max(0, min(image_h, y + h))
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _outside_plan_zone_rects(
    plan_zone: Optional[LegendZone],
    image_shape: tuple[int, ...],
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
    rect = _zone_to_rect(plan_zone)
    if rect is None:
        return None, []
    clamped = _clamp_rect_to_image(rect, image_shape)
    if clamped is None:
        return None, []

    image_h, image_w = image_shape[:2]
    x, y, w, h = clamped
    x2 = x + w
    y2 = y + h
    outside: list[tuple[int, int, int, int]] = []
    if y > 0:
        outside.append((0, 0, image_w, y))
    if y2 < image_h:
        outside.append((0, y2, image_w, image_h - y2))
    if x > 0:
        outside.append((0, y, x, h))
    if x2 < image_w:
        outside.append((x2, y, image_w - x2, h))
    return clamped, outside


class ExtractRequest(BaseModel):
    excluded_zones: Optional[List[dict]] = []
    hidden_layers: Optional[List[str]] = []
    legend_zone: Optional[LegendZone] = None
    detector_profile: Optional[str] = "auto"


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
        pdf_diagnostics = _build_pdf_diagnostics(str(file_path), plan_img)
        _, buffer_plan = cv2.imencode(".jpg", plan_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        plan_base64 = base64.b64encode(buffer_plan).decode("utf-8")

        return {
            "planPreview": f"data:image/jpeg;base64,{plan_base64}",
            "sessionId": session_id,
            "sourcePdf": file.filename or file_path.name,
            "pdfDiagnostics": pdf_diagnostics,
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


@app.get("/api/pdf-diagnostics")
async def api_pdf_diagnostics(session_id: str):
    file_path = UPLOAD_DIR / f"{session_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")
    try:
        return {"pdfDiagnostics": _build_pdf_diagnostics(str(file_path))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/render-preview")
async def api_render_preview(session_id: str, body: RenderRequest = None):
    file_path = UPLOAD_DIR / f"{session_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")
    try:
        hidden_layers = body.hidden_layers if body else []
        plan_img = pdf_to_png(str(file_path), dpi=300, hidden_layers=hidden_layers)
        pdf_diagnostics = _build_pdf_diagnostics(str(file_path), plan_img)
        _, buffer_plan = cv2.imencode(".jpg", plan_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        plan_base64 = base64.b64encode(buffer_plan).decode("utf-8")
        return {
            "planPreview": f"data:image/jpeg;base64,{plan_base64}",
            "pdfDiagnostics": pdf_diagnostics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-legend")
async def api_extract_legend(session_id: str, body: ExtractRequest = None):
    file_path = UPLOAD_DIR / f"{session_id}.pdf"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")

    try:
        _log("Renderowanie planu do ekstrakcji (300 DPI)")
        hidden_layers = body.hidden_layers if body else []
        requested_profile = _normalize_detector_profile(body.detector_profile if body else "auto")
        plan_img = pdf_to_png(str(file_path), dpi=300, hidden_layers=hidden_layers)
        pdf_diagnostics = _build_pdf_diagnostics(str(file_path), plan_img)
        resolved_profile = (
            pdf_diagnostics.get("recommendedProfile", "color")
            if requested_profile == "auto"
            else requested_profile
        )
        mask_mode = resolved_profile if resolved_profile in {"color", "gray"} else "auto"

        # Strefy wykluczone
        exclude_rects = []
        if body and body.excluded_zones:
            for zone in body.excluded_zones:
                try:
                    exclude_rects.append(
                        (int(zone["x"]), int(zone["y"]), int(zone["width"]), int(zone["height"]))
                    )
                except (KeyError, ValueError):
                    pass

        legend_rect_px = None
        if body and body.legend_zone:
            legend_rect_px = (
                int(round(body.legend_zone.x)),
                int(round(body.legend_zone.y)),
                int(round(body.legend_zone.width)),
                int(round(body.legend_zone.height)),
            )

        _log("Ekstrakcja legendy...")
        if TEMPLATES_DIR.exists():
            shutil.rmtree(TEMPLATES_DIR)
        TEMPLATES_DIR.mkdir(exist_ok=True)

        symbols = extract_legend(
            str(file_path),
            plan_img,
            output_dir=str(TEMPLATES_DIR),
            exclude_rects=exclude_rects,
            legend_rect_px=legend_rect_px,
            mask_mode=mask_mode,
        )

        # Generujemy podgląd jeszcze raz na wypadek gdyby UI go potrzebowało w pełnej rozdz.
        # Ale zwracamy wzorce.
        patterns_list = []
        for s in symbols:
            _, buffer_s = cv2.imencode(".png", s.image)
            img_b64 = base64.b64encode(buffer_s).decode("utf-8")
            patterns_list.append(
                {"id": s.name, "name": s.name, "imgBase64": f"data:image/png;base64,{img_b64}"}
            )

        return {
            "patterns": patterns_list,
            "legendZoneUsed": legend_rect_px,
            "legendMaskMode": mask_mode,
            "detectorProfileRequested": requested_profile,
            "detectorProfileUsed": resolved_profile,
            "pdfDiagnostics": pdf_diagnostics,
        }

    except Exception as e:
        print(f"Błąd podczas ekstrakcji: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")


class AnalyzeRequest(BaseModel):
    excluded_zones: Optional[List[dict]] = []
    hidden_layers: Optional[List[str]] = []
    include_debug: Optional[bool] = None
    include_debug_candidates: Optional[bool] = None
    include_image: Optional[bool] = None
    detector_profile: Optional[str] = "auto"
    legend_zone: Optional[LegendZone] = None
    plan_zone: Optional[LegendZone] = None


class RoiInspectRequest(BaseModel):
    hidden_layers: Optional[List[str]] = []
    detector_profile: Optional[str] = "auto"
    roi: LegendZone
    top_n: Optional[int] = 15


@app.get("/api/analysis-progress")
async def api_analysis_progress(session_id: str):
    return {
        "progress": ANALYSIS_PROGRESS.get(
            session_id,
            {
                "sessionId": session_id,
                "analysisId": None,
                "stage": "idle",
                "percent": 0.0,
                "detail": "",
                "done": False,
                "error": None,
                "updatedAtUtc": None,
            },
        )
    }


@app.post("/api/analyze")
def api_analyze(session_id: str, body: AnalyzeRequest = None):
    plan_path = UPLOAD_DIR / f"{session_id}.pdf"

    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")

    request_start = time.perf_counter()
    timings_ms: dict[str, float] = {}
    counters: dict[str, int] = {}
    analysis_id = str(uuid.uuid4())
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    _set_analysis_progress(session_id, "setup", 1, "Start analizy", analysis_id=analysis_id)

    try:
        phase_start = time.perf_counter()
        hidden_layers = body.hidden_layers if body else []
        include_debug = (
            body.include_debug
            if body and body.include_debug is not None
            else DEFAULT_ANALYSIS_DEBUG
        )
        include_debug_candidates = (
            bool(body.include_debug_candidates)
            if body and body.include_debug_candidates is not None
            else False
        )
        requested_profile = _normalize_detector_profile(body.detector_profile if body else "auto")
        include_image = body.include_image if body and body.include_image is not None else True
        session_meta = _read_session_meta(session_id)
        timings_ms["requestSetup"] = _elapsed_ms(phase_start)

        # 1. Ładujemy plan
        _set_analysis_progress(session_id, "render_pdf", 5, "Renderowanie PDF", analysis_id=analysis_id)
        phase_start = time.perf_counter()
        plan_img = pdf_to_png(str(plan_path), dpi=300, hidden_layers=hidden_layers)
        timings_ms["renderPdf"] = _elapsed_ms(phase_start)
        counters["planWidth"] = int(plan_img.shape[1])
        counters["planHeight"] = int(plan_img.shape[0])
        counters["hiddenLayers"] = len(hidden_layers)

        _set_analysis_progress(session_id, "diagnostics", 8, "Diagnostyka PDF", analysis_id=analysis_id)
        phase_start = time.perf_counter()
        pdf_diagnostics = _build_pdf_diagnostics(str(plan_path), plan_img)
        resolved_profile = (
            pdf_diagnostics.get("recommendedProfile", "color")
            if requested_profile == "auto"
            else requested_profile
        )
        if resolved_profile not in {"color", "gray"}:
            resolved_profile = "color"
        timings_ms["pdfDiagnostics"] = _elapsed_ms(phase_start)
        counters["pdfLayers"] = int(pdf_diagnostics.get("layers", 0))
        counters["pdfDrawingsPage1"] = int(pdf_diagnostics.get("drawingsPage1", 0))
        counters["pdfImagesPage1"] = int(pdf_diagnostics.get("imagesPage1", 0))

        # 2. Ładujemy wzorce
        _set_analysis_progress(session_id, "load_templates", 10, "Ladowanie wzorcow", analysis_id=analysis_id)
        phase_start = time.perf_counter()
        templates = load_templates(str(TEMPLATES_DIR))
        timings_ms["loadTemplates"] = _elapsed_ms(phase_start)
        counters["templatesLoaded"] = len(templates)

        # 3. Strefy wykluczone → lista krotek (x, y, w, h)
        phase_start = time.perf_counter()
        exclude_rects = []
        manual_exclude_rects = []
        legend_rect = None
        plan_zone_rect = None
        plan_zone_outside_rects = []
        if body and body.excluded_zones:
            for zone in body.excluded_zones:
                try:
                    rect = (int(zone["x"]), int(zone["y"]), int(zone["width"]), int(zone["height"]))
                    exclude_rects.append(rect)
                    manual_exclude_rects.append(rect)
                except (KeyError, ValueError):
                    pass
            _log(f"Strefy wykluczone: {exclude_rects}")
        if body and body.legend_zone:
            legend_rect = _zone_to_rect(body.legend_zone)
        if legend_rect is not None:
            exclude_rects.append(legend_rect)
            _log(f"Strefa legendy wykluczona z analizy: {legend_rect}")
        if body and body.plan_zone:
            plan_zone_rect, plan_zone_outside_rects = _outside_plan_zone_rects(
                body.plan_zone,
                plan_img.shape,
            )
            if plan_zone_rect is not None:
                exclude_rects.extend(plan_zone_outside_rects)
                _log(
                    "Strefa planu aktywna: "
                    f"{plan_zone_rect}; poza planem wykluczono {len(plan_zone_outside_rects)} prostokaty"
                )
        timings_ms["parseExcludedZones"] = _elapsed_ms(phase_start)
        counters["excludedZones"] = len(exclude_rects)
        counters["manualExcludedZones"] = len(manual_exclude_rects)
        counters["planZoneOutsideRects"] = len(plan_zone_outside_rects)

        phase_start = time.perf_counter()
        hidden_layer_debug = (
            _build_hidden_layer_debug(str(plan_path), hidden_layers) if include_debug else None
        )
        timings_ms["hiddenLayerDebug"] = _elapsed_ms(phase_start)

        # 4. Detekcja
        detector_profile: dict = {}

        def detector_progress(stage: str, percent: float, detail: str = "") -> None:
            _set_analysis_progress(
                session_id,
                stage,
                percent,
                detail,
                analysis_id=analysis_id,
            )

        phase_start = time.perf_counter()
        results = detect_symbols(
            plan_img,
            templates,
            exclude_rects=exclude_rects,
            pdf_path=str(plan_path),
            pdf_dpi=300,
            hidden_layers=hidden_layers,
            debug_profile=detector_profile if include_debug else None,
            detector_profile=resolved_profile,
            include_debug_candidates=include_debug_candidates,
            progress_callback=detector_progress,
        )
        timings_ms["detectSymbolsTotal"] = _elapsed_ms(phase_start)

        result_image_payload: str | None = None
        if include_image:
            _set_analysis_progress(session_id, "draw_results", 96, "Rysowanie wynikow", analysis_id=analysis_id)
            phase_start = time.perf_counter()
            result_img = draw_results(plan_img, results)
            timings_ms["drawResults"] = _elapsed_ms(phase_start)

            phase_start = time.perf_counter()
            _, buffer_res = cv2.imencode(".jpg", result_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            timings_ms["encodeResultJpeg"] = _elapsed_ms(phase_start)
            counters["resultJpegBytes"] = int(len(buffer_res))

            phase_start = time.perf_counter()
            result_base64 = base64.b64encode(buffer_res).decode("utf-8")
            result_image_payload = f"data:image/jpeg;base64,{result_base64}"
            timings_ms["base64Result"] = _elapsed_ms(phase_start)
        else:
            timings_ms["drawResults"] = 0.0
            timings_ms["encodeResultJpeg"] = 0.0
            timings_ms["base64Result"] = 0.0
            counters["resultJpegBytes"] = 0

        # Przygotowujemy dane o ramkach dla frontendu (opcjonalnie)
        # Na razie wysyłamy gotowy obraz i listę wyników

        analysis_context = {
            "analysisId": analysis_id,
            "generatedAtUtc": generated_at_utc,
            "sessionId": session_id,
            "sourcePdf": session_meta.get("sourcePdf", plan_path.name),
            "hiddenLayersUsed": hidden_layers,
            "excludedZonesUsed": exclude_rects,
            "manualExcludedZonesUsed": manual_exclude_rects,
            "legendZoneUsed": legend_rect,
            "planZoneUsed": plan_zone_rect,
            "planZoneOutsideExcluded": plan_zone_outside_rects,
            "detectorProfileRequested": requested_profile,
            "detectorProfileUsed": resolved_profile,
            "includeDebugCandidates": include_debug_candidates,
            "pdfDiagnostics": pdf_diagnostics,
        }
        if include_debug:
            analysis_context["hiddenLayerDebug"] = hidden_layer_debug

        _set_analysis_progress(session_id, "format_response", 98, "Przygotowanie odpowiedzi", analysis_id=analysis_id)
        phase_start = time.perf_counter()
        formatted_results = []
        all_boxes = []

        for r in results:
            # Podsumowanie per typ symbolu (do CostPanel)
            formatted_results.append(
                {
                    "name": r.symbol_name,
                    "count": r.count,
                    "color": r.color,
                }
            )
            # Każda indywidualna detekcja (do Canvas)
            for det in r.detections:
                box_payload = {
                    "id": f"{r.symbol_name}_{det.x}_{det.y}",
                    "symbolName": r.symbol_name,
                    "x": det.x,
                    "y": det.y,
                    "width": det.width,
                    "height": det.height,
                    "confidence": det.confidence,
                    "color": r.color,
                }
                if include_debug:
                    box_payload.update(
                        {
                            "verificationScore": det.verification_score,
                            "source": det.source,
                            "rotation": det.rotation,
                            "scale": det.scale,
                            "mirrored": det.mirrored,
                            "coverage": det.coverage,
                            "purity": det.purity,
                            "contextPurity": det.context_purity,
                            "colorSimilarity": det.color_similarity,
                            "isTextLabel": det.is_text_label,
                            "contentScore": det.content_score,
                            "contentBBox": det.content_bbox,
                            "contentSource": det.content_source,
                            "analysisId": analysis_context["analysisId"],
                            "analysisGeneratedUtc": analysis_context["generatedAtUtc"],
                            "analysisSession": analysis_context["sessionId"],
                            "sourcePdf": analysis_context["sourcePdf"],
                            "hiddenLayersUsed": analysis_context["hiddenLayersUsed"],
                        }
                    )
                all_boxes.append(box_payload)
        timings_ms["formatResultsAndBoxes"] = _elapsed_ms(phase_start)
        counters["resultGroups"] = len(formatted_results)
        counters["boxes"] = len(all_boxes)

        timings_ms["totalBeforeSnapshot"] = _elapsed_ms(request_start)
        timings_ms["total"] = timings_ms["totalBeforeSnapshot"]
        performance = {
            "backendTimingsMs": timings_ms,
            "backendCounters": counters,
            "detector": detector_profile,
            "slowestStages": _slowest_stages(
                {
                    **timings_ms,
                    **{
                        f"detector.{name}": value
                        for name, value in detector_profile.get("timingsMs", {}).items()
                    },
                }
            ),
        }
        if include_debug:
            analysis_context["performance"] = performance

        response_payload = {
            "message": "Analiza zakończona",
            "analysisContext": analysis_context,
            "results": formatted_results,
            "boxes": all_boxes,
            "resultImage": result_image_payload,
        }
        if include_debug:
            response_payload["performance"] = performance
            response_payload["debugCandidates"] = (
                detector_profile.get("debugCandidates", []) if include_debug_candidates else []
            )

        snapshot_queued = False
        try:
            phase_start = time.perf_counter()
            snapshot_payload = {
                "analysisContext": analysis_context,
                "results": formatted_results,
                "boxes": all_boxes,
                "debugCandidates": (
                    detector_profile.get("debugCandidates", [])
                    if include_debug and include_debug_candidates
                    else []
                ),
                "resultImageLength": len(response_payload["resultImage"] or ""),
                "performance": performance,
            }
            SNAPSHOT_EXECUTOR.submit(
                _write_analysis_snapshot,
                analysis_context["analysisId"],
                snapshot_payload,
            )
            performance["backendTimingsMs"]["snapshotQueue"] = _elapsed_ms(phase_start)
            snapshot_queued = True
        except OSError as snapshot_error:
            print(f"Nie udało się zapisać snapshotu analizy: {snapshot_error}")

        counters["snapshotQueued"] = 1 if snapshot_queued else 0
        performance["backendTimingsMs"]["total"] = _elapsed_ms(request_start)
        performance["slowestStages"] = _slowest_stages(
            {
                **performance["backendTimingsMs"],
                **{
                    f"detector.{name}": value
                    for name, value in detector_profile.get("timingsMs", {}).items()
                },
            }
        )
        print(
            "Analysis performance:"
            f" analysis_id={analysis_context['analysisId']},"
            f" total_ms={performance['backendTimingsMs']['total']:.0f},"
            f" slowest={performance['slowestStages']}"
        )

        _set_analysis_progress(
            session_id,
            "done",
            100,
            "Analiza zakonczona",
            analysis_id=analysis_id,
            done=True,
        )

        return response_payload

    except Exception as e:
        print(f"Błąd podczas analizy: {str(e)}")
        _set_analysis_progress(
            session_id,
            "error",
            100,
            str(e),
            analysis_id=analysis_id,
            done=True,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inspect-roi")
def api_inspect_roi(session_id: str, body: RoiInspectRequest):
    plan_path = UPLOAD_DIR / f"{session_id}.pdf"
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")

    try:
        hidden_layers = body.hidden_layers if body else []
        requested_profile = _normalize_detector_profile(body.detector_profile if body else "auto")
        plan_img = pdf_to_png(str(plan_path), dpi=300, hidden_layers=hidden_layers)
        pdf_diagnostics = _build_pdf_diagnostics(str(plan_path), plan_img)
        resolved_profile = (
            pdf_diagnostics.get("recommendedProfile", "color")
            if requested_profile == "auto"
            else requested_profile
        )
        if resolved_profile not in {"color", "gray"}:
            resolved_profile = "color"

        templates = load_templates(str(TEMPLATES_DIR))
        roi = (
            int(round(body.roi.x)),
            int(round(body.roi.y)),
            int(round(body.roi.width)),
            int(round(body.roi.height)),
        )
        return {
            "inspection": inspect_roi(
                plan_img,
                templates,
                roi,
                detector_profile=resolved_profile,
                top_n=body.top_n or 15,
            ),
            "detectorProfileRequested": requested_profile,
            "detectorProfileUsed": resolved_profile,
            "pdfDiagnostics": pdf_diagnostics,
        }
    except Exception as e:
        print(f"Blad inspektora ROI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/templates")
async def api_get_templates():
    patterns_list = []
    if TEMPLATES_DIR.exists():
        for file_path in TEMPLATES_DIR.glob("*.png"):
            img = cv2.imread(str(file_path))
            if img is not None:
                _, buffer = cv2.imencode(".png", img)
                img_b64 = base64.b64encode(buffer).decode("utf-8")
                patterns_list.append(
                    {
                        "id": file_path.stem,
                        "name": file_path.stem,
                        "imgBase64": f"data:image/png;base64,{img_b64}",
                    }
                )
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
