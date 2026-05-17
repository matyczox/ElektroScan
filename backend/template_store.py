"""Template file store and display-label helpers."""

from __future__ import annotations

import base64
import json
import re
import shutil
from pathlib import Path

import cv2
from fastapi import HTTPException

from core.legend_extractor import _clean_ocr_label_text

TEMPLATE_LABELS_FILENAME = ".template_labels.json"

def _next_template_index(templates_dir: Path) -> int:
    max_index = 0
    if templates_dir.exists():
        for existing in templates_dir.glob("*.png"):
            match = re.match(r"^(\d+)_", existing.stem)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1

def _template_labels_path(templates_dir: Path) -> Path:
    return templates_dir / TEMPLATE_LABELS_FILENAME

def _load_template_labels(templates_dir: Path) -> dict[str, str]:
    labels_path = _template_labels_path(templates_dir)
    if not labels_path.exists():
        return {}
    try:
        payload = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): str(value).strip()
        for key, value in payload.items()
        if str(key).strip() and str(value).strip()
    }

def _write_template_labels(templates_dir: Path, labels: dict[str, str]) -> None:
    templates_dir.mkdir(parents=True, exist_ok=True)
    clean_labels = {
        str(key): str(value).strip()
        for key, value in labels.items()
        if str(key).strip() and str(value).strip()
    }
    labels_path = _template_labels_path(templates_dir)
    if clean_labels:
        labels_path.write_text(
            json.dumps(clean_labels, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        labels_path.unlink(missing_ok=True)

def _clean_template_display_label(raw_label: str | None) -> str | None:
    label = " ".join(str(raw_label or "").replace("_", " ").split())
    if not label:
        return None
    if sum(1 for char in label if char.isalnum()) < 2:
        return None
    return label[:180].strip() or None

def _is_template_display_heading(label: str | None) -> bool:
    compact = re.sub(r"[^0-9A-Za-z]+", "", str(label or "")).casefold()
    return compact in {
        "legend",
        "legenda",
        "oznaczenia",
        "symbol",
        "opis",
        "nazwa",
        "indeks",
        "producent",
    }

def _set_template_display_label(templates_dir: Path, template_id: str, label: str | None) -> None:
    labels = _load_template_labels(templates_dir)
    clean_label = _clean_template_display_label(label)
    if clean_label:
        labels[template_id] = clean_label
    else:
        labels.pop(template_id, None)
    _write_template_labels(templates_dir, labels)

def _delete_template_display_label(templates_dir: Path, template_id: str) -> None:
    labels = _load_template_labels(templates_dir)
    if template_id in labels:
        labels.pop(template_id, None)
        _write_template_labels(templates_dir, labels)

def _append_extracted_templates(
    extraction_dir: Path,
    templates_dir: Path,
    display_labels: list[str] | None = None,
) -> set[str]:
    """Move freshly extracted templates into the project without deleting existing ones."""

    templates_dir.mkdir(parents=True, exist_ok=True)
    next_index = _next_template_index(templates_dir)
    added_ids: set[str] = set()
    labels = _load_template_labels(templates_dir)

    for label_index, template_path in enumerate(sorted(extraction_dir.glob("*.png"))):
        raw_stem = _safe_template_stem(template_path.stem)
        label = re.sub(r"^\d+_+", "", raw_stem).strip("_") or raw_stem

        while True:
            dest_stem = f"{next_index:02d}_{label}"
            dest_path = templates_dir / f"{dest_stem}.png"
            next_index += 1
            if not dest_path.exists():
                break

        shutil.move(str(template_path), dest_path)
        added_ids.add(dest_stem)
        display_label = None
        if display_labels and label_index < len(display_labels):
            display_label = _clean_template_display_label(display_labels[label_index])
        labels[dest_stem] = display_label or _display_template_name(raw_stem)

    _write_template_labels(templates_dir, labels)

    return added_ids

def _legend_display_labels_from_drafts(
    vector_drafts: list[dict] | None,
    expected_count: int,
) -> list[str] | None:
    if not vector_drafts or expected_count <= 0:
        return None

    def sort_key(draft: dict) -> tuple[float, float]:
        row_bbox = draft.get("row_bbox_pt") or draft.get("bbox_pt") or [0, 0, 0, 0]
        bbox = draft.get("bbox_pt") or row_bbox
        try:
            return (float(row_bbox[1]), float(bbox[0]))
        except Exception:
            return (0.0, 0.0)

    labels: list[str] = []
    for draft in sorted(vector_drafts, key=sort_key):
        if not isinstance(draft, dict):
            continue
        label = _clean_template_display_label(draft.get("name_draft"))
        if label and not _is_template_display_heading(label):
            labels.append(label)

    if len(labels) == expected_count:
        return labels
    if expected_count < len(labels) <= expected_count + max(2, expected_count // 4):
        return labels[:expected_count]
    return None

def _safe_template_stem(raw_name: str) -> str:
    """Return a safe template file stem while keeping human-readable labels."""

    value = Path(str(raw_name or "").replace("\\", "/")).stem.strip()
    value = re.sub(r"[^\w\s.-]", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", "_", value, flags=re.UNICODE).strip("._")
    if not value:
        raise HTTPException(status_code=400, detail="Nazwa wzorca jest pusta.")
    return value[:120]

def _renamed_template_stem(current_stem: str, raw_name: str) -> str:
    """Build a renamed template stem, preserving extractor numeric prefixes."""

    new_stem = _safe_template_stem(raw_name)
    current_match = re.match(r"^(\d+)_", _safe_template_stem(current_stem))
    if current_match and not re.match(r"^\d+_", new_stem):
        return f"{current_match.group(1)}_{new_stem}"
    return new_stem

def _display_template_name(stem: str) -> str:
    """Strip extraction ordering prefixes in UI labels."""

    label = re.sub(r"^\d+_+", "", stem)
    label = re.sub(r"_+", " ", label).strip()
    cleaned = _clean_ocr_label_text(label)
    return cleaned or label

def _display_template_name_for_path(file_path: Path, labels: dict[str, str] | None = None) -> str:
    label_map = labels if labels is not None else _load_template_labels(file_path.parent)
    clean_label = _clean_template_display_label(label_map.get(file_path.stem))
    return clean_label or _display_template_name(file_path.stem)

def _template_payload_from_path(file_path: Path, labels: dict[str, str] | None = None) -> dict | None:
    img = cv2.imread(str(file_path))
    if img is None:
        return None
    _, buffer = cv2.imencode(".png", img)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    return {
        "id": file_path.stem,
        "name": _display_template_name_for_path(file_path, labels),
        "imgBase64": f"data:image/png;base64,{img_b64}",
    }
