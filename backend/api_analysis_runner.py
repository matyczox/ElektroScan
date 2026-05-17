"""Analysis runner service used by API route wrappers."""

from __future__ import annotations

import base64
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import HTTPException

from api_analysis_utils import (
    DEFAULT_ANALYSIS_DEBUG,
    _build_hidden_layer_debug,
    _elapsed_ms,
    _log,
    _normalize_detector_profile,
    _slowest_stages,
)
from api_models import AnalyzeRequest
from api_rendering import ANALYSIS_DPI, _build_pdf_diagnostics, _render_pdf_for_session
from api_workspace import (
    _analysis_snapshot_path,
    _read_session_meta,
    _session_pdf_path,
    _set_analysis_progress,
    _write_analysis_snapshot,
)
from api_zones import _outside_plan_zone_rects, _zone_to_rect
from auth_store import record_analysis_run
from core.detector import detect_symbols, draw_results, load_templates


SNAPSHOT_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _analyze_session(
    session_id: str,
    body: AnalyzeRequest | None,
    *,
    upload_dir: Path,
    templates_dir: Path,
    analysis_dir: Path,
    project_id: str | None = None,
):
    plan_path = _session_pdf_path(session_id, upload_dir)

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
        requested_profile = _normalize_detector_profile(body.detector_profile if body else "auto")
        include_image = body.include_image if body and body.include_image is not None else True
        session_meta = _read_session_meta(session_id, upload_dir)
        timings_ms["requestSetup"] = _elapsed_ms(phase_start)

        # 1. Ładujemy plan
        _set_analysis_progress(
            session_id, "render_pdf", 5, "Renderowanie PDF", analysis_id=analysis_id
        )
        phase_start = time.perf_counter()
        plan_img, render_cache_hit = _render_pdf_for_session(
            session_id,
            str(plan_path),
            dpi=ANALYSIS_DPI,
            hidden_layers=hidden_layers,
        )
        timings_ms["renderPdf"] = _elapsed_ms(phase_start)
        counters["planWidth"] = int(plan_img.shape[1])
        counters["planHeight"] = int(plan_img.shape[0])
        counters["hiddenLayers"] = len(hidden_layers)
        counters["renderCacheHit"] = 1 if render_cache_hit else 0

        _set_analysis_progress(
            session_id, "diagnostics", 8, "Diagnostyka PDF", analysis_id=analysis_id
        )
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
        _set_analysis_progress(
            session_id, "load_templates", 10, "Ladowanie wzorcow", analysis_id=analysis_id
        )
        phase_start = time.perf_counter()
        templates = load_templates(str(templates_dir))
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
            pdf_dpi=ANALYSIS_DPI,
            hidden_layers=hidden_layers,
            debug_profile=detector_profile if include_debug else None,
            detector_profile=resolved_profile,
            progress_callback=detector_progress,
        )
        timings_ms["detectSymbolsTotal"] = _elapsed_ms(phase_start)

        result_image_payload: str | None = None
        if include_image:
            _set_analysis_progress(
                session_id, "draw_results", 96, "Rysowanie wynikow", analysis_id=analysis_id
            )
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
            "pdfDiagnostics": pdf_diagnostics,
        }
        if include_debug:
            analysis_context["hiddenLayerDebug"] = hidden_layer_debug

        _set_analysis_progress(
            session_id, "format_response", 98, "Przygotowanie odpowiedzi", analysis_id=analysis_id
        )
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
                    "visualBBox": det.visual_bbox,
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

        snapshot_queued = False
        try:
            phase_start = time.perf_counter()
            snapshot_payload = {
                "analysisContext": analysis_context,
                "results": formatted_results,
                "boxes": all_boxes,
                "resultImageLength": len(response_payload["resultImage"] or ""),
                "performance": performance,
            }
            SNAPSHOT_EXECUTOR.submit(
                _write_analysis_snapshot,
                analysis_context["analysisId"],
                snapshot_payload,
                analysis_dir,
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
        if project_id is not None:
            record_analysis_run(
                analysis_id=analysis_context["analysisId"],
                project_id=project_id,
                session_id=session_id,
                source_pdf=analysis_context["sourcePdf"],
                snapshot_path=str(_analysis_snapshot_path(analysis_context["analysisId"], analysis_dir)),
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

