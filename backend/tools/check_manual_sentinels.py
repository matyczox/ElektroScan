from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _symbol_name(box: dict[str, Any]) -> str:
    return str(box.get("symbolName") or box.get("symbol_name") or "")


def _source(box: dict[str, Any]) -> str:
    return str(box.get("source") or "")


def _box_rect(box: dict[str, Any]) -> tuple[float, float, float, float]:
    x = float(box.get("x", 0))
    y = float(box.get("y", 0))
    return (x, y, x + float(box.get("width", 0)), y + float(box.get("height", 0)))


def _box_label(box: dict[str, Any]) -> str:
    return (
        f"{_symbol_name(box)}@{int(float(box.get('x', 0)))},"
        f"{int(float(box.get('y', 0)))} {int(float(box.get('width', 0)))}x"
        f"{int(float(box.get('height', 0)))} source={_source(box)}"
    )


def _sentinel_checks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    checks = payload.get("manualSentinels")
    if checks is None:
        checks = payload.get("manualSentinelChecks")
    if checks is None:
        checks = payload.get("metadata", {}).get("manualSentinels")
    if checks is None:
        checks = payload.get("metadata", {}).get("manualSentinelChecks")
    if checks is None:
        return []
    if not isinstance(checks, list) or not all(isinstance(item, dict) for item in checks):
        raise ValueError("manualSentinels/manualSentinelChecks must be a list of objects")
    return checks


def _boxes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    boxes = payload.get("boxes", [])
    if not isinstance(boxes, list):
        raise ValueError("candidate JSON must contain a list at key 'boxes'")
    return [box for box in boxes if isinstance(box, dict)]


def _expanded_roi(check: dict[str, Any]) -> tuple[float, float, float, float] | None:
    roi = check.get("roi")
    if roi is None:
        roi = check.get("bbox")
    if isinstance(roi, dict):
        x = float(roi["x"])
        y = float(roi["y"])
        w = float(roi.get("width", roi.get("w", 0)))
        h = float(roi.get("height", roi.get("h", 0)))
    elif isinstance(roi, (list, tuple)) and len(roi) >= 4:
        x, y, w, h = (float(value) for value in roi[:4])
    else:
        return None
    tolerance = float(check.get("tolerance", check.get("bboxTolerance", 30)))
    return (x - tolerance, y - tolerance, x + w + tolerance, y + h + tolerance)


def _near_point(check: dict[str, Any]) -> tuple[float, float, float] | None:
    near = check.get("near") or {}
    if not isinstance(near, dict) and not {"x", "y"} <= set(check):
        return None
    x = float(near.get("x", check.get("x", 0)))
    y = float(near.get("y", check.get("y", 0)))
    radius = float(near.get("radius", check.get("radius", 35)))
    return x, y, radius


def _matches_location(check: dict[str, Any], box: dict[str, Any]) -> bool:
    roi = _expanded_roi(check)
    if roi is not None:
        x0, y0, x1, y1 = _box_rect(box)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx0, ry0, rx1, ry1 = roi
        return rx0 <= cx <= rx1 and ry0 <= cy <= ry1

    near = _near_point(check)
    if near is None:
        return True
    x, y, radius = near
    return math.hypot(float(box.get("x", 0)) - x, float(box.get("y", 0)) - y) <= radius


def _expected_symbols(check: dict[str, Any]) -> set[str]:
    symbols = {str(name) for name in check.get("symbolNames", [])}
    symbols.update(str(name) for name in check.get("allowedSymbolNames", []))
    if "symbolName" in check:
        symbols.add(str(check["symbolName"]))
    return symbols


def _matches_sources(check: dict[str, Any], box: dict[str, Any]) -> bool:
    source = _source(box)
    source_in = {str(name) for name in check.get("sourceIn", [])}
    if "source" in check:
        source_in.add(str(check["source"]))
    if source_in and source not in source_in:
        return False

    source_not_in = {str(name) for name in check.get("sourceNotIn", [])}
    if "sourceNot" in check:
        source_not_in.add(str(check["sourceNot"]))
    return source not in source_not_in


def _check_matches(check: dict[str, Any], boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    symbols = _expected_symbols(check)
    matches: list[dict[str, Any]] = []
    for box in boxes:
        if not _matches_location(check, box):
            continue
        if symbols and _symbol_name(box) not in symbols:
            continue
        if not _matches_sources(check, box):
            continue
        matches.append(box)
    return matches


def _count_range(check: dict[str, Any], mode: str) -> tuple[int, int]:
    if mode in {"forbid", "forbid_near"}:
        return int(check.get("minCount", 0)), int(check.get("maxCount", 0))
    return int(check.get("minCount", 1)), int(check.get("maxCount", 999999))


def evaluate_sentinels(
    golden_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> tuple[bool, str]:
    checks = _sentinel_checks(golden_payload)
    boxes = _boxes(candidate_payload)
    if not checks:
        return True, "Manual sentinels: none"

    lines = [f"Manual sentinels: {len(checks)}"]
    ok = True
    for check in checks:
        check_id = str(check.get("id") or "sentinel")
        mode = str(check.get("mode") or "require_near").lower()
        if mode == "allow_any_near":
            mode = "require_near"
        matches = _check_matches(check, boxes)
        min_count, max_count = _count_range(check, mode)

        passed = min_count <= len(matches) <= max_count
        ok = ok and passed
        status = "fixed" if passed else "still wrong"
        lines.append(
            f"  - {check_id}: {status} "
            f"(matches={len(matches)}, expected={min_count}..{max_count})"
        )
        for match in matches[:5]:
            lines.append(f"      {_box_label(match)}")
    return ok, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check manual sentinel cases in an analysis JSON.")
    parser.add_argument("golden", type=Path, help="Golden/sentinel JSON with manualSentinels.")
    parser.add_argument("candidate", type=Path, help="Candidate analysis JSON with boxes.")
    args = parser.parse_args()

    ok, report = evaluate_sentinels(_load_json(args.golden), _load_json(args.candidate))
    print(report)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
