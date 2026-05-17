"""Workspace, session, and analysis state helpers for API routes."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from auth_store import get_project_for_user, project_session_exists
from template_store import _safe_template_stem


BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMPLATES_DIR = BASE_DIR / "templates"
ANALYSIS_DIR = BASE_DIR / "analysis_debug"
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
SESSION_META_SUFFIX = ".meta.json"
ANALYSIS_PROGRESS: dict[str, dict] = {}


for directory in [UPLOAD_DIR, TEMPLATES_DIR, ANALYSIS_DIR, DATA_DIR, PROJECTS_DIR]:
    directory.mkdir(exist_ok=True)


def _project_or_404(project_id: str, user: dict) -> dict:
    project = get_project_for_user(project_id, user["id"])
    if project is None:
        raise HTTPException(status_code=404, detail="Nie znaleziono projektu.")
    _ensure_project_workspace(project_id)
    return project


def _project_root(project_id: str) -> Path:
    return PROJECTS_DIR / project_id


def _project_upload_dir(project_id: str) -> Path:
    return _project_root(project_id) / "uploads"


def _project_templates_dir(project_id: str) -> Path:
    return _project_root(project_id) / "templates"


def _project_analysis_dir(project_id: str) -> Path:
    return _project_root(project_id) / "analysis_debug"


def _ensure_project_workspace(project_id: str) -> None:
    for directory in [
        _project_upload_dir(project_id),
        _project_templates_dir(project_id),
        _project_analysis_dir(project_id),
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def _require_project_session(project_id: str, session_id: str) -> None:
    if not project_session_exists(project_id, session_id):
        raise HTTPException(status_code=404, detail="Nie znaleziono sesji w tym projekcie.")


def _session_pdf_path(session_id: str, upload_dir: Path = UPLOAD_DIR) -> Path:
    return upload_dir / f"{session_id}.pdf"


def _session_file_or_404(session_id: str, upload_dir: Path) -> Path:
    file_path = _session_pdf_path(session_id, upload_dir)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku sesji.")
    return file_path


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


def _session_meta_path(session_id: str, upload_dir: Path = UPLOAD_DIR) -> Path:
    return upload_dir / f"{session_id}{SESSION_META_SUFFIX}"


def _write_session_meta(
    session_id: str,
    *,
    source_pdf: str,
    upload_dir: Path = UPLOAD_DIR,
) -> None:
    upload_dir.mkdir(parents=True, exist_ok=True)
    _session_meta_path(session_id, upload_dir).write_text(
        json.dumps({"sourcePdf": source_pdf}, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_session_meta(session_id: str, upload_dir: Path = UPLOAD_DIR) -> dict:
    meta_path = _session_meta_path(session_id, upload_dir)
    if not meta_path.exists():
        return {}

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _analysis_snapshot_path(analysis_id: str, analysis_dir: Path = ANALYSIS_DIR) -> Path:
    return analysis_dir / f"{analysis_id}.json"


def _write_analysis_snapshot(
    analysis_id: str,
    payload: dict,
    analysis_dir: Path = ANALYSIS_DIR,
) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _analysis_snapshot_path(analysis_id, analysis_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for entry in directory.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _template_path_for_id(template_id: str, templates_dir: Path = TEMPLATES_DIR) -> Path | None:
    """Find a template by exact stem, with a suffix fallback for legacy responses."""

    safe_stem = _safe_template_stem(template_id)
    exact = templates_dir / f"{safe_stem}.png"
    if exact.exists():
        return exact

    suffix_matches = sorted(templates_dir.glob(f"*_{safe_stem}.png"))
    return suffix_matches[0] if suffix_matches else None
