import base64
import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import HTTPException, UploadFile

from api_rendering import (
    ANALYSIS_DPI,
    PREVIEW_DPI,
    _build_pdf_diagnostics,
    _clear_render_cache,
    _pdf_page_size_at_dpi,
    _preview_response_meta,
    _render_pdf_for_session,
)
from api_workspace import (
    _clear_directory_contents,
    _session_pdf_path,
    _write_session_meta,
)


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
