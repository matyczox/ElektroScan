import base64
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api_models import (
    AuthRegisterRequest,
    AuthLoginRequest,
    AuthProfileUpdateRequest,
    PasswordResetRequest,
    PasswordResetConfirmRequest,
    ProjectCreateRequest,
    ProjectUpdateRequest,
    LegendZone,
    ExtractRequest,
    RenderRequest,
    AnalyzeRequest,
    AnalysisExportResult,
    AnalysisExportBox,
    AnalysisExportRequest,
    RoiInspectRequest,
    GrayDebugZonesRequest,
    TemplateCropRequest,
    TemplateUpdateRequest,
)
from analysis_export import (
    _build_analysis_export_rows,
    _build_analysis_export_xlsx,
    _export_filename,
)
from api_analysis_utils import (
    _log,
    _normalize_detector_profile,
    _normalize_legend_engine,
)
from api_analysis_runner import _analyze_session
from api_auth import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_DAYS,
    _clear_auth_cookie,
    _current_user_from_request,
    _dev_auth_token_payload,
    _set_auth_cookie,
    require_user,
)
from api_rendering import (
    ANALYSIS_DPI,
    PREVIEW_DPI,
    _build_pdf_diagnostics,
    _clear_render_cache,
    _pdf_page_size_at_dpi,
    _preview_response_meta,
    _render_pdf_for_session,
    _render_preview_response_for_session,
)
from api_template_service import (
    _crop_template_response,
    _delete_template_response,
    _templates_response,
    _update_template_response,
    _upload_template_response,
)
from api_workspace import (
    ANALYSIS_DIR,
    ANALYSIS_PROGRESS,
    TEMPLATES_DIR,
    UPLOAD_DIR,
    _analysis_snapshot_path,
    _clear_directory_contents,
    _ensure_project_workspace,
    _project_analysis_dir,
    _project_or_404,
    _project_templates_dir,
    _project_upload_dir,
    _read_session_meta,
    _require_project_session,
    _session_file_or_404,
    _session_pdf_path,
    _set_analysis_progress,
    _template_path_for_id,
    _write_analysis_snapshot,
    _write_session_meta,
)
from api_zones import (
    _clamp_rect_to_image,
    _extract_exclude_rects_from_request,
    _outside_plan_zone_rects,
    _zone_to_rect,
)
from template_store import (
    _append_extracted_templates,
    _clean_template_display_label,
    _display_template_name,
    _display_template_name_for_path,
    _legend_display_labels_from_drafts,
    _load_template_labels,
    _next_template_index,
    _renamed_template_stem,
    _template_labels_path,
    _template_payload_from_path,
    _write_template_labels,
)
from auth_store import (
    archive_project_for_user,
    authenticate_user,
    create_auth_session,
    create_password_reset_token_for_email,
    create_project,
    create_user,
    delete_auth_session,
    delete_auth_session_by_id,
    delete_auth_sessions_for_user,
    get_analysis_run_for_project,
    init_database,
    list_analysis_runs_for_project,
    list_auth_sessions_for_user,
    list_projects_for_user,
    record_project_upload_session,
    reset_password_with_token,
    update_user_profile,
    update_project_for_user,
)
from core import detector_gray as gray_strategy
from core.detector import load_templates
from core.detector_config import GRAY_SCALES
from core.detector_masks import _ink_mask
from core.detector_templates import _prepare_variants

# Importujemy nasze core'owe moduły
from core.legend_extractor import (
    _clean_ocr_label_text,
    _normalize_layer_name,
    extract_legend_detailed,
    get_pdf_layers,
    pdf_to_png,
)
from core.roi_inspector import inspect_roi


def _allowed_cors_origins() -> list[str]:
    raw = os.getenv("ELEKTROSCAN_CORS_ORIGINS")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


app = FastAPI(title="ElektroScan AI API")
init_database()

# Konfiguracja CORS - pozwala na komunikację z frontendem Vite
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


@app.middleware("http")
async def disable_response_caching(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/")
async def root():
    return {"message": "ElektroScan AI API is running"}






async def _build_preview_response(
    file: UploadFile,
    *,
    upload_dir: Path,
    templates_dir: Path,
) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Tylko pliki PDF są obsługiwane.")

    session_id = str(uuid.uuid4())
    file_path = _session_pdf_path(session_id, upload_dir)
    _clear_directory_contents(templates_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    _clear_render_cache()

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    _write_session_meta(session_id, source_pdf=filename or file_path.name, upload_dir=upload_dir)

    try:
        # Render podglądu (300 DPI — identycznie jak detekcja)
        plan_img, cache_hit = _render_pdf_for_session(
            session_id,
            str(file_path),
            dpi=PREVIEW_DPI,
        )
        analysis_size = _pdf_page_size_at_dpi(str(file_path), dpi=ANALYSIS_DPI)
        pdf_diagnostics = _build_pdf_diagnostics(str(file_path), plan_img)
        _, buffer_plan = cv2.imencode(".jpg", plan_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        plan_base64 = base64.b64encode(buffer_plan).decode("utf-8")

        return {
            "planPreview": f"data:image/jpeg;base64,{plan_base64}",
            "sessionId": session_id,
            "sourcePdf": filename or file_path.name,
            "pdfDiagnostics": pdf_diagnostics,
            **_preview_response_meta(
                plan_img,
                preview_dpi=PREVIEW_DPI,
                analysis_size=analysis_size,
                cache_hit=cache_hit,
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/register")
async def api_auth_register(body: AuthRegisterRequest, response: Response):
    try:
        user = create_user(email=body.email, password=body.password, name=body.name)
        token, session = create_auth_session(user["id"], ttl_days=SESSION_TTL_DAYS)
        _set_auth_cookie(response, token)
        return {"user": user, "session": session}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
async def api_auth_login(body: AuthLoginRequest, response: Response):
    user = authenticate_user(body.email, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Niepoprawny e-mail lub hasło.")
    token, session = create_auth_session(user["id"], ttl_days=SESSION_TTL_DAYS)
    _set_auth_cookie(response, token)
    return {"user": user, "session": session}


@app.get("/api/auth/me")
async def api_auth_me(request: Request):
    return {"user": _current_user_from_request(request)}


@app.patch("/api/auth/me")
async def api_auth_update_me(
    body: AuthProfileUpdateRequest,
    user: dict = Depends(require_user),
):
    try:
        updated = update_user_profile(user["id"], name=body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if updated is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono użytkownika.")
    return {"user": updated}


@app.post("/api/auth/password-reset/request")
async def api_auth_password_reset_request(body: PasswordResetRequest):
    reset_result = create_password_reset_token_for_email(body.email)
    payload = {
        "message": "Jeśli konto istnieje, wysłano instrukcję resetu hasła.",
        "passwordReset": None,
    }
    if reset_result is not None:
        reset_token, reset_info = reset_result
        payload["passwordReset"] = {
            **reset_info,
            **_dev_auth_token_payload("resetToken", reset_token),
        }
    return payload


@app.post("/api/auth/password-reset/confirm")
async def api_auth_password_reset_confirm(body: PasswordResetConfirmRequest, response: Response):
    try:
        user = reset_password_with_token(body.token, body.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if user is None:
        raise HTTPException(status_code=400, detail="Token resetu jest niepoprawny albo wygasł.")
    _clear_auth_cookie(response)
    return {"user": user, "message": "Hasło zostało zmienione. Zaloguj się ponownie."}


@app.get("/api/auth/sessions")
async def api_auth_sessions(request: Request, user: dict = Depends(require_user)):
    sessions = list_auth_sessions_for_user(
        user["id"],
        current_token=request.cookies.get(SESSION_COOKIE_NAME),
    )
    return {"sessions": sessions}


@app.delete("/api/auth/sessions/{session_id}")
async def api_auth_delete_session(
    session_id: str,
    request: Request,
    response: Response,
    user: dict = Depends(require_user),
):
    sessions = list_auth_sessions_for_user(
        user["id"],
        current_token=request.cookies.get(SESSION_COOKIE_NAME),
    )
    current_session = next((item for item in sessions if item["id"] == session_id), None)
    if current_session is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji.")
    delete_auth_session_by_id(user["id"], session_id)
    if current_session.get("isCurrent"):
        _clear_auth_cookie(response)
    return {"message": "Sesja została usunięta.", "deletedCurrentSession": current_session.get("isCurrent")}


@app.post("/api/auth/logout-all")
async def api_auth_logout_all(response: Response, user: dict = Depends(require_user)):
    deleted = delete_auth_sessions_for_user(user["id"])
    _clear_auth_cookie(response)
    return {"message": "Wylogowano ze wszystkich sesji.", "deletedSessions": deleted}


@app.post("/api/auth/logout")
async def api_auth_logout(request: Request, response: Response):
    delete_auth_session(request.cookies.get(SESSION_COOKIE_NAME))
    _clear_auth_cookie(response)
    return {"message": "Wylogowano."}


@app.get("/api/projects")
async def api_projects(user: dict = Depends(require_user)):
    return {"projects": list_projects_for_user(user["id"])}


@app.post("/api/projects")
async def api_create_project(body: ProjectCreateRequest, user: dict = Depends(require_user)):
    try:
        project = create_project(
            user["id"],
            name=body.name,
            description=body.description or "",
        )
        _ensure_project_workspace(project["id"])
        return {"project": project}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/projects/{project_id}")
async def api_project(project_id: str, user: dict = Depends(require_user)):
    return {"project": _project_or_404(project_id, user)}


@app.patch("/api/projects/{project_id}")
async def api_update_project(
    project_id: str,
    body: ProjectUpdateRequest,
    user: dict = Depends(require_user),
):
    try:
        project = update_project_for_user(
            project_id,
            user["id"],
            name=body.name,
            description=body.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if project is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono projektu.")
    return {"project": project}


@app.delete("/api/projects/{project_id}")
async def api_delete_project(project_id: str, user: dict = Depends(require_user)):
    if not archive_project_for_user(project_id, user["id"]):
        raise HTTPException(status_code=404, detail="Nie znaleziono projektu.")
    return {"message": "Projekt usunięty."}


@app.get("/api/projects/{project_id}/analysis-runs")
async def api_project_analysis_runs(project_id: str, user: dict = Depends(require_user)):
    _project_or_404(project_id, user)
    return {"analysisRuns": list_analysis_runs_for_project(project_id)}


@app.get("/api/projects/{project_id}/analysis-runs/{analysis_id}")
async def api_project_analysis_run(
    project_id: str,
    analysis_id: str,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    run = get_analysis_run_for_project(project_id, analysis_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono analizy.")
    snapshot_path = _analysis_snapshot_path(analysis_id, _project_analysis_dir(project_id))
    snapshot = None
    if snapshot_path.exists():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            snapshot = None
    return {"analysisRun": run, "snapshot": snapshot}


@app.post("/api/projects/{project_id}/preview")
async def api_project_preview(
    project_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    payload = await _build_preview_response(
        file,
        upload_dir=_project_upload_dir(project_id),
        templates_dir=_project_templates_dir(project_id),
    )
    record_project_upload_session(
        session_id=payload["sessionId"],
        project_id=project_id,
        source_pdf=payload["sourcePdf"],
    )
    return payload






@app.post("/api/preview")
async def api_preview(file: UploadFile = File(...)):
    return await _build_preview_response(
        file,
        upload_dir=UPLOAD_DIR,
        templates_dir=TEMPLATES_DIR,
    )


@app.get("/api/layers")
async def api_layers(session_id: str):
    file_path = _session_file_or_404(session_id, UPLOAD_DIR)
    layers = get_pdf_layers(str(file_path))
    return {"layers": layers}


@app.get("/api/projects/{project_id}/layers")
async def api_project_layers(
    project_id: str,
    session_id: str,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _require_project_session(project_id, session_id)
    file_path = _session_file_or_404(session_id, _project_upload_dir(project_id))
    layers = get_pdf_layers(str(file_path))
    return {"layers": layers}


@app.get("/api/pdf-diagnostics")
async def api_pdf_diagnostics(session_id: str):
    file_path = _session_file_or_404(session_id, UPLOAD_DIR)
    try:
        return {"pdfDiagnostics": _build_pdf_diagnostics(str(file_path))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/projects/{project_id}/pdf-diagnostics")
async def api_project_pdf_diagnostics(
    project_id: str,
    session_id: str,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _require_project_session(project_id, session_id)
    file_path = _session_file_or_404(session_id, _project_upload_dir(project_id))
    try:
        return {"pdfDiagnostics": _build_pdf_diagnostics(str(file_path))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/render-preview")
async def api_render_preview(session_id: str, body: RenderRequest = None):
    file_path = _session_file_or_404(session_id, UPLOAD_DIR)
    try:
        return _render_preview_response_for_session(session_id, file_path, body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/projects/{project_id}/render-preview")
async def api_project_render_preview(
    project_id: str,
    session_id: str,
    body: RenderRequest = None,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _require_project_session(project_id, session_id)
    file_path = _session_file_or_404(session_id, _project_upload_dir(project_id))
    try:
        return _render_preview_response_for_session(session_id, file_path, body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _extract_legend_response(
    session_id: str,
    body: ExtractRequest | None,
    *,
    upload_dir: Path,
    templates_dir: Path,
) -> dict:
    file_path = _session_file_or_404(session_id, upload_dir)

    try:
        _log("Renderowanie planu do ekstrakcji (300 DPI)")
        hidden_layers = body.hidden_layers if body else []
        requested_profile = _normalize_detector_profile(body.detector_profile if body else "auto")
        requested_legend_engine = _normalize_legend_engine(body.legend_engine if body else "auto")
        include_legend_debug = bool(body.include_legend_debug) if body else False
        plan_img, _cache_hit = _render_pdf_for_session(
            session_id,
            str(file_path),
            dpi=ANALYSIS_DPI,
            hidden_layers=hidden_layers,
            copy_image=True,
        )
        pdf_diagnostics = _build_pdf_diagnostics(str(file_path), plan_img)
        resolved_profile = (
            pdf_diagnostics.get("recommendedProfile", "color")
            if requested_profile == "auto"
            else requested_profile
        )
        mask_mode = resolved_profile if resolved_profile in {"color", "gray"} else "auto"

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

        if legend_rect_px is None:
            raise HTTPException(
                status_code=400,
                detail="Brak strefy legendy. Zaznacz obszar legendy na planie przed ekstrakcją.",
            )

        _log("Ekstrakcja legendy...")
        templates_dir.mkdir(parents=True, exist_ok=True)
        added_template_ids: set[str] = set()
        with tempfile.TemporaryDirectory(dir=templates_dir.parent) as extraction_dir:
            legend_bundle = extract_legend_detailed(
                str(file_path),
                plan_img,
                output_dir=extraction_dir,
                dpi=ANALYSIS_DPI,
                exclude_rects=exclude_rects,
                legend_rect_px=legend_rect_px,
                mask_mode=mask_mode,
                hidden_layers=hidden_layers,
                legend_engine=requested_legend_engine,
                include_debug_primitives=include_legend_debug,
            )
            symbols = legend_bundle.extracted_symbols
            legend_rect_px = legend_bundle.used_legend_rect_px_300

            if symbols:
                display_labels = _legend_display_labels_from_drafts(
                    legend_bundle.vector_drafts,
                    len(symbols),
                )
                added_template_ids = _append_extracted_templates(
                    Path(extraction_dir),
                    templates_dir,
                    display_labels=display_labels,
                )

        legend_metadata = {
            "legendEngineRequested": legend_bundle.engine_requested,
            "legendEngineUsed": legend_bundle.engine_used,
            "legendFallbackReason": legend_bundle.fallback_reason,
            "legendPageProfile": legend_bundle.page_profile,
            "sceneTransform": legend_bundle.scene_transform,
        }
        if include_legend_debug:
            legend_metadata["legendVectorDrafts"] = legend_bundle.vector_drafts or []
            legend_metadata["legendVectorPrimitives"] = legend_bundle.vector_primitives or []

        if not symbols:
            patterns_list = []
            labels = _load_template_labels(templates_dir)
            for template_path in sorted(templates_dir.glob("*.png")):
                payload = _template_payload_from_path(template_path, labels)
                if payload is None:
                    continue
                payload["status"] = "existing"
                patterns_list.append(payload)
            return {
                "patterns": patterns_list,
                "legendExtractedCount": 0,
                "legendAddedIds": [],
                "legendZoneUsed": legend_rect_px,
                "legendMaskMode": mask_mode,
                "detectorProfileRequested": requested_profile,
                "detectorProfileUsed": resolved_profile,
                "pdfDiagnostics": pdf_diagnostics,
                **legend_metadata,
            }

        patterns_list = []
        labels = _load_template_labels(templates_dir)
        for template_path in sorted(templates_dir.glob("*.png")):
            payload = _template_payload_from_path(template_path, labels)
            if payload is None:
                continue
            payload["status"] = "pending" if payload["id"] in added_template_ids else "existing"
            patterns_list.append(payload)

        return {
            "patterns": patterns_list,
            "legendExtractedCount": len(added_template_ids),
            "legendAddedIds": sorted(added_template_ids),
            "legendZoneUsed": legend_rect_px,
            "legendMaskMode": mask_mode,
            "detectorProfileRequested": requested_profile,
            "detectorProfileUsed": resolved_profile,
            "pdfDiagnostics": pdf_diagnostics,
            **legend_metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Błąd podczas ekstrakcji: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")


@app.post("/api/extract-legend")
async def api_extract_legend(session_id: str, body: ExtractRequest = None):
    return await _extract_legend_response(
        session_id,
        body,
        upload_dir=UPLOAD_DIR,
        templates_dir=TEMPLATES_DIR,
    )


@app.post("/api/projects/{project_id}/extract-legend")
async def api_project_extract_legend(
    project_id: str,
    session_id: str,
    body: ExtractRequest = None,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _require_project_session(project_id, session_id)
    return await _extract_legend_response(
        session_id,
        body,
        upload_dir=_project_upload_dir(project_id),
        templates_dir=_project_templates_dir(project_id),
    )


















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
    return _analyze_session(
        session_id,
        body,
        upload_dir=UPLOAD_DIR,
        templates_dir=TEMPLATES_DIR,
        analysis_dir=ANALYSIS_DIR,
    )


@app.post("/api/projects/{project_id}/analyze")
def api_project_analyze(
    project_id: str,
    session_id: str,
    body: AnalyzeRequest = None,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _require_project_session(project_id, session_id)
    return _analyze_session(
        session_id,
        body,
        upload_dir=_project_upload_dir(project_id),
        templates_dir=_project_templates_dir(project_id),
        analysis_dir=_project_analysis_dir(project_id),
        project_id=project_id,
    )


@app.post("/api/projects/{project_id}/analysis-export")
def api_project_analysis_export(
    project_id: str,
    body: AnalysisExportRequest,
    user: dict = Depends(require_user),
):
    project = _project_or_404(project_id, user)
    rows = _build_analysis_export_rows(body, _project_templates_dir(project_id))
    if not rows:
        raise HTTPException(status_code=400, detail="Brak wyników analizy do eksportu.")

    workbook = _build_analysis_export_xlsx(
        project=project,
        rows=rows,
        analysis_context=body.analysis_context,
    )
    filename = _export_filename(project, body.analysis_context)
    return Response(
        content=workbook,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
        },
    )


def _inspect_roi_response(
    session_id: str,
    body: RoiInspectRequest,
    *,
    upload_dir: Path,
    templates_dir: Path,
) -> dict:
    plan_path = _session_file_or_404(session_id, upload_dir)

    try:
        hidden_layers = body.hidden_layers if body else []
        requested_profile = _normalize_detector_profile(body.detector_profile if body else "auto")
        plan_img, _cache_hit = _render_pdf_for_session(
            session_id,
            str(plan_path),
            dpi=ANALYSIS_DPI,
            hidden_layers=hidden_layers,
        )
        pdf_diagnostics = _build_pdf_diagnostics(str(plan_path), plan_img)
        resolved_profile = (
            pdf_diagnostics.get("recommendedProfile", "color")
            if requested_profile == "auto"
            else requested_profile
        )
        if resolved_profile not in {"color", "gray"}:
            resolved_profile = "color"

        templates = load_templates(str(templates_dir))
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


@app.post("/api/inspect-roi")
def api_inspect_roi(session_id: str, body: RoiInspectRequest):
    return _inspect_roi_response(
        session_id,
        body,
        upload_dir=UPLOAD_DIR,
        templates_dir=TEMPLATES_DIR,
    )


@app.post("/api/projects/{project_id}/inspect-roi")
def api_project_inspect_roi(
    project_id: str,
    session_id: str,
    body: RoiInspectRequest,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _require_project_session(project_id, session_id)
    return _inspect_roi_response(
        session_id,
        body,
        upload_dir=_project_upload_dir(project_id),
        templates_dir=_project_templates_dir(project_id),
    )


def _gray_debug_zones_response(
    session_id: str,
    body: GrayDebugZonesRequest | None,
    *,
    upload_dir: Path,
    templates_dir: Path,
) -> dict:
    plan_path = _session_file_or_404(session_id, upload_dir)

    try:
        hidden_layers = body.hidden_layers if body else []
        plan_img, _cache_hit = _render_pdf_for_session(
            session_id,
            str(plan_path),
            dpi=ANALYSIS_DPI,
            hidden_layers=hidden_layers,
        )
        templates = load_templates(str(templates_dir))
        exclude_rects, plan_zone_rect, plan_zone_outside_rects = (
            _extract_exclude_rects_from_request(
                body,
                plan_img.shape,
            )
        )

        raw_dilated = _ink_mask(plan_img, dilate=True)
        plan_masks_by_template = {template_id: raw_dilated for template_id in range(len(templates))}
        gray_masks = gray_strategy.build_gray_scan_masks(
            plan_image=plan_img,
            templates=templates,
            plan_masks_by_template=plan_masks_by_template,
            exclude_rects=exclude_rects,
            raw_dilated=raw_dilated,
        )

        rois_by_rect: dict[tuple[int, int, int, int], int] = {}
        template_roi_counts: dict[str, int] = {}
        for template_id, template in enumerate(templates):
            variants = _prepare_variants(template_id, template, scales=GRAY_SCALES)
            if variants:
                max_width = max(variant.width for variant in variants)
                max_height = max(variant.height for variant in variants)
            else:
                max_height, max_width = template.mask.shape[:2]
            rois, _uses_full_scan, _roi_area, _foreground = gray_strategy.build_gray_search_rois(
                gray_masks.zone_mask,
                plan_img.shape,
                max_width,
                max_height,
            )
            template_roi_counts[template.name] = len(rois)
            for rect in rois:
                rois_by_rect[rect] = rois_by_rect.get(rect, 0) + 1

        overlay = np.zeros((plan_img.shape[0], plan_img.shape[1], 4), dtype=np.uint8)
        zone_pixels = gray_masks.zone_mask > 0
        evidence_pixels = gray_masks.evidence_mask > 0
        overlay[zone_pixels] = (94, 197, 34, 55)  # green, BGRA
        overlay[evidence_pixels] = (22, 115, 249, 125)  # orange, BGRA

        display_rois = gray_strategy.coalesce_gray_debug_rois(
            list(rois_by_rect.keys()),
            plan_img.shape,
        )
        for x, y, w, h in display_rois:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 59, 150), 2)

        ok, buffer_overlay = cv2.imencode(".png", overlay)
        if not ok:
            raise RuntimeError("Nie udalo sie wygenerowac overlay stref.")

        return {
            "overlayImage": f"data:image/png;base64,{base64.b64encode(buffer_overlay).decode('utf-8')}",
            "imageWidth": int(plan_img.shape[1]),
            "imageHeight": int(plan_img.shape[0]),
            "zoneThreshold": int(gray_masks.zone_threshold),
            "evidenceThreshold": int(gray_masks.evidence_threshold),
            "zonePixels": int(gray_masks.zone_ink_pixels),
            "evidencePixels": int(gray_masks.evidence_ink_pixels),
            "roiCount": int(len(display_rois)),
            "roiRefs": int(sum(rois_by_rect.values())),
            "templates": int(len(templates)),
            "planZoneUsed": plan_zone_rect,
            "planZoneOutsideExcluded": plan_zone_outside_rects,
            "topTemplateRoiCounts": [
                {"template": name, "count": count}
                for name, count in sorted(
                    template_roi_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            ],
        }
    except Exception as e:
        print(f"Blad podgladu gray debug zones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/gray-debug-zones")
def api_gray_debug_zones(session_id: str, body: GrayDebugZonesRequest = None):
    return _gray_debug_zones_response(
        session_id,
        body,
        upload_dir=UPLOAD_DIR,
        templates_dir=TEMPLATES_DIR,
    )


@app.post("/api/projects/{project_id}/gray-debug-zones")
def api_project_gray_debug_zones(
    project_id: str,
    session_id: str,
    body: GrayDebugZonesRequest = None,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _require_project_session(project_id, session_id)
    return _gray_debug_zones_response(
        session_id,
        body,
        upload_dir=_project_upload_dir(project_id),
        templates_dir=_project_templates_dir(project_id),
    )


@app.get("/api/templates")
async def api_get_templates():
    return _templates_response(TEMPLATES_DIR)




@app.get("/api/projects/{project_id}/templates")
async def api_project_get_templates(
    project_id: str,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    return _templates_response(_project_templates_dir(project_id))


@app.post("/api/templates/upload")
async def api_upload_template(file: UploadFile = File(...)):
    return await _upload_template_response(file, TEMPLATES_DIR)




@app.post("/api/projects/{project_id}/templates/upload")
async def api_project_upload_template(
    project_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    return await _upload_template_response(file, _project_templates_dir(project_id))


@app.post("/api/templates/{template_name}/crop")
async def api_crop_template(template_name: str, body: TemplateCropRequest):
    return await _crop_template_response(
        template_name,
        body,
        upload_dir=UPLOAD_DIR,
        templates_dir=TEMPLATES_DIR,
    )




@app.post("/api/projects/{project_id}/templates/{template_name}/crop")
async def api_project_crop_template(
    project_id: str,
    template_name: str,
    body: TemplateCropRequest,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _require_project_session(project_id, body.session_id)
    return await _crop_template_response(
        template_name,
        body,
        upload_dir=_project_upload_dir(project_id),
        templates_dir=_project_templates_dir(project_id),
    )


@app.patch("/api/templates/{template_name}")
async def api_update_template(template_name: str, body: TemplateUpdateRequest):
    return _update_template_response(template_name, body, TEMPLATES_DIR)




@app.patch("/api/projects/{project_id}/templates/{template_name}")
async def api_project_update_template(
    project_id: str,
    template_name: str,
    body: TemplateUpdateRequest,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    return _update_template_response(template_name, body, _project_templates_dir(project_id))


@app.delete("/api/templates")
async def api_delete_templates():
    _clear_directory_contents(TEMPLATES_DIR)
    return {"message": "Baza wiedzy wyczyszczona."}


@app.delete("/api/projects/{project_id}/templates")
async def api_project_delete_templates(
    project_id: str,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    _clear_directory_contents(_project_templates_dir(project_id))
    return {"message": "Baza wiedzy wyczyszczona."}


@app.delete("/api/templates/{template_name}")
async def api_delete_template(template_name: str):
    return _delete_template_response(template_name, TEMPLATES_DIR)




@app.delete("/api/projects/{project_id}/templates/{template_name}")
async def api_project_delete_template(
    project_id: str,
    template_name: str,
    user: dict = Depends(require_user),
):
    _project_or_404(project_id, user)
    return _delete_template_response(template_name, _project_templates_dir(project_id))


@app.post("/api/clear")
async def api_clear():
    # Czyścimy wszystko
    for folder in [UPLOAD_DIR, TEMPLATES_DIR]:
        _clear_directory_contents(folder)
    return {"message": "Wyczyszczono dane robocze."}


@app.post("/api/projects/{project_id}/clear")
async def api_project_clear(project_id: str, user: dict = Depends(require_user)):
    _project_or_404(project_id, user)
    for folder in [
        _project_upload_dir(project_id),
        _project_templates_dir(project_id),
        _project_analysis_dir(project_id),
    ]:
        _clear_directory_contents(folder)
    return {"message": "Wyczyszczono dane robocze projektu."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
