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
from api_debug_service import _gray_debug_zones_response, _inspect_roi_response
from api_legend_service import _extract_legend_response
from api_preview_service import _build_preview_response
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
