"""Template CRUD helpers used by API route wrappers."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
from fastapi import HTTPException, UploadFile

from api_models import TemplateCropRequest, TemplateUpdateRequest
from api_rendering import ANALYSIS_DPI, _render_pdf_for_session
from api_workspace import _session_file_or_404, _template_path_for_id
from api_zones import _clamp_rect_to_image
from template_store import (
    _delete_template_display_label,
    _load_template_labels,
    _safe_template_stem,
    _set_template_display_label,
    _template_payload_from_path,
)


def _templates_response(templates_dir: Path) -> dict:
    patterns_list = []
    labels = _load_template_labels(templates_dir)
    if templates_dir.exists():
        for file_path in sorted(templates_dir.glob("*.png")):
            payload = _template_payload_from_path(file_path, labels)
            if payload is not None:
                patterns_list.append(payload)
    return {"patterns": patterns_list}


async def _upload_template_response(file: UploadFile, templates_dir: Path) -> dict:
    """Manual PNG template upload."""
    if not file.filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Tylko pliki PNG sÄ… obsĹ‚ugiwane.")

    safe_name = Path(file.filename).stem
    templates_dir.mkdir(parents=True, exist_ok=True)
    dest_path = templates_dir / f"{safe_name}.png"

    try:
        with dest_path.open("wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
        return {"message": f"Wzorzec '{safe_name}' dodany do bazy.", "name": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _crop_template_response(
    template_name: str,
    body: TemplateCropRequest,
    *,
    upload_dir: Path,
    templates_dir: Path,
) -> dict:
    """Replace or create a template from a user-selected crop on the rendered plan."""

    plan_path = _session_file_or_404(body.session_id, upload_dir)

    try:
        plan_img, _cache_hit = _render_pdf_for_session(
            body.session_id,
            str(plan_path),
            dpi=ANALYSIS_DPI,
            hidden_layers=body.hidden_layers or [],
        )
        rect = (
            int(round(body.x)),
            int(round(body.y)),
            int(round(body.width)),
            int(round(body.height)),
        )
        clamped = _clamp_rect_to_image(rect, plan_img.shape)
        if clamped is None:
            raise HTTPException(status_code=400, detail="Zaznaczenie wzorca jest puste.")

        x, y, w, h = clamped
        crop = plan_img[y : y + h, x : x + w]
        if crop.size == 0:
            raise HTTPException(status_code=400, detail="Zaznaczenie wzorca jest puste.")

        old_path = _template_path_for_id(template_name, templates_dir)
        if old_path is not None:
            target_stem = old_path.stem
        else:
            target_stem = _safe_template_stem(template_name)

        templates_dir.mkdir(parents=True, exist_ok=True)
        target_path = templates_dir / f"{target_stem}.png"
        if old_path is not None and old_path != target_path and target_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"Wzorzec '{target_stem}' juĹĽ istnieje.",
            )

        ok, buffer = cv2.imencode(".png", crop)
        if not ok:
            raise RuntimeError("Nie udaĹ‚o siÄ™ zakodowaÄ‡ wzorca PNG.")
        target_path.write_bytes(buffer.tobytes())

        if old_path is not None and old_path != target_path:
            old_path.unlink(missing_ok=True)
            _delete_template_display_label(templates_dir, old_path.stem)

        if body.name:
            _set_template_display_label(templates_dir, target_stem, body.name)

        payload = _template_payload_from_path(target_path)
        if payload is None:
            raise RuntimeError("Nie udaĹ‚o siÄ™ odczytaÄ‡ zapisanego wzorca.")
        payload["status"] = "fixed"
        payload["correctedBBoxPx"] = [x, y, w, h]
        return {"message": f"Wzorzec '{payload['name']}' poprawiony.", "pattern": payload}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _update_template_response(
    template_name: str,
    body: TemplateUpdateRequest,
    templates_dir: Path,
) -> dict:
    target = _template_path_for_id(template_name, templates_dir)
    if target is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono wzorca.")

    if not body.name:
        payload = _template_payload_from_path(target)
        return {"message": "Brak zmian.", "pattern": payload}

    _set_template_display_label(templates_dir, target.stem, body.name)
    payload = _template_payload_from_path(target)
    if payload is None:
        raise HTTPException(status_code=500, detail="Nie udaĹ‚o siÄ™ odczytaÄ‡ wzorca po zmianie.")
    return {"message": f"Wzorzec zmieniony na '{payload['name']}'.", "pattern": payload}


def _delete_template_response(template_name: str, templates_dir: Path) -> dict:
    target = _template_path_for_id(template_name, templates_dir)
    if target is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono wzorca.")

    target.unlink()
    _delete_template_display_label(templates_dir, target.stem)
    return {"message": f"Wzorzec '{template_name}' usuniÄ™ty."}
