from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.detector_templates import load_templates  # noqa: E402
from core.legend_extractor import pdf_to_png  # noqa: E402
from core.roi_inspector import inspect_roi  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _repo_path(value: str | None, *, base: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if base is not None and (base / path).exists():
        return (base / path).resolve()
    return (REPO_ROOT / path).resolve()


def _boxes(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    boxes = payload.get("boxes", [])
    return [box for box in boxes if isinstance(box, dict)] if isinstance(boxes, list) else []


def _symbol(box: dict[str, Any]) -> str:
    return str(box.get("symbolName") or box.get("symbol_name") or "")


def _source(box: dict[str, Any]) -> str:
    return str(box.get("source") or "")


def _roi(case: dict[str, Any]) -> tuple[int, int, int, int]:
    values = case.get("roi") or case.get("bbox")
    if not isinstance(values, list | tuple) or len(values) < 4:
        raise ValueError(f"Case {case.get('id')} must define roi [x, y, width, height]")
    return tuple(int(round(float(value))) for value in values[:4])  # type: ignore[return-value]


def _intersects_or_center_near(
    box: dict[str, Any],
    roi: tuple[int, int, int, int],
    *,
    tolerance: int,
    min_overlap: float = 0.20,
) -> bool:
    rx, ry, rw, rh = roi
    rx0 = rx - tolerance
    ry0 = ry - tolerance
    rx1 = rx + rw + tolerance
    ry1 = ry + rh + tolerance
    bx0 = float(box.get("x", 0))
    by0 = float(box.get("y", 0))
    bx1 = bx0 + float(box.get("width", 0))
    by1 = by0 + float(box.get("height", 0))
    cx = (bx0 + bx1) / 2.0
    cy = (by0 + by1) / 2.0
    if rx0 <= cx <= rx1 and ry0 <= cy <= ry1:
        return True
    ix0 = max(rx0, bx0)
    iy0 = max(ry0, by0)
    ix1 = min(rx1, bx1)
    iy1 = min(ry1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return False
    box_area = max(1.0, (bx1 - bx0) * (by1 - by0))
    roi_area = max(1.0, (rx1 - rx0) * (ry1 - ry0))
    return inter / min(box_area, roi_area) >= min_overlap


def _nearby_boxes(
    boxes: list[dict[str, Any]],
    roi: tuple[int, int, int, int],
    *,
    tolerance: int,
) -> list[dict[str, Any]]:
    return [
        box
        for box in boxes
        if _intersects_or_center_near(box, roi, tolerance=tolerance)
    ]


def _evaluate_case(case: dict[str, Any], nearby: list[dict[str, Any]]) -> str:
    expected = {str(value) for value in case.get("expectedSymbols", [])}
    allowed = {str(value) for value in case.get("allowedSymbols", [])}
    forbidden = {str(value) for value in case.get("forbiddenSymbols", [])}
    source_not = str(case.get("sourceNot") or "")
    if str(case.get("reviewStatus")) == "manual_check":
        return "manual_check"
    if expected:
        return (
            "pass"
            if any(_symbol(box) in expected and (not source_not or _source(box) != source_not) for box in nearby)
            else "fail"
        )
    if allowed:
        return "pass" if any(_symbol(box) in allowed for box in nearby) else "manual_check"
    if forbidden:
        return "fail" if any(_symbol(box) in forbidden for box in nearby) else "pass"
    return "manual_check"


def _clamped_crop(
    image,
    roi: tuple[int, int, int, int],
    *,
    padding: int,
):
    x, y, w, h = roi
    height, width = image.shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(width, x + w + padding)
    y1 = min(height, y + h + padding)
    crop = image[y0:y1, x0:x1].copy()
    cv2.rectangle(
        crop,
        (x - x0, y - y0),
        (x + w - x0, y + h - y0),
        (0, 165, 255),
        3,
    )
    return crop


def _candidate_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            {
                "symbolName": candidate.get("symbolName"),
                "accepted": candidate.get("accepted"),
                "reason": candidate.get("reason"),
                "match": candidate.get("match"),
                "verification": candidate.get("verification"),
                "coverage": candidate.get("coverage"),
                "purity": candidate.get("purity"),
                "contextPurity": candidate.get("contextPurity"),
                "bbox": candidate.get("bbox"),
            }
        )
    return rows


def build_report(
    case_pack_path: Path,
    output_dir: Path,
    *,
    analysis_path: Path | None,
    templates_dir: Path | None,
    padding: int,
    roi_tolerance: int,
    top_n: int,
) -> dict[str, Any]:
    case_pack = _load_json(case_pack_path)
    pdf_path = _repo_path(str(case_pack.get("pdfPath") or ""), base=case_pack_path.parent)
    if pdf_path is None or not pdf_path.exists():
        raise RuntimeError(f"Missing pdfPath: {pdf_path}")

    analysis = _load_json(analysis_path) if analysis_path else None
    boxes = _boxes(analysis)
    templates = load_templates(str(templates_dir)) if templates_dir and templates_dir.exists() else []
    image = pdf_to_png(str(pdf_path), dpi=int(case_pack.get("dpi", 300)))

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    report_cases: list[dict[str, Any]] = []
    for index, case in enumerate(case_pack.get("cases", []), start=1):
        if not isinstance(case, dict):
            continue
        roi = _roi(case)
        crop_path = crops_dir / f"{index:02d}_{case.get('id', 'case')}.png"
        cv2.imwrite(str(crop_path), _clamped_crop(image, roi, padding=padding))
        nearby = _nearby_boxes(boxes, roi, tolerance=roi_tolerance)
        inspection = None
        if templates:
            inspection_payload = inspect_roi(
                image,
                templates,
                roi,
                detector_profile=str(case_pack.get("detectorProfile") or "color"),
                top_n=top_n,
            )
            inspection = {
                "roiInkPixels": inspection_payload.get("roiInkPixels"),
                "roiScanPixels": inspection_payload.get("roiScanPixels"),
                "roiColorScanPixels": inspection_payload.get("roiColorScanPixels"),
                "roiColorScanTemplate": inspection_payload.get("roiColorScanTemplate"),
                "candidates": _candidate_summary(inspection_payload.get("candidates", [])),
            }
        report_cases.append(
            {
                "id": case.get("id"),
                "reviewStatus": case.get("reviewStatus"),
                "roi": roi,
                "crop": str(crop_path.relative_to(output_dir)).replace("\\", "/"),
                "expectedSymbols": case.get("expectedSymbols", []),
                "allowedSymbols": case.get("allowedSymbols", []),
                "currentSymbols": case.get("currentSymbols", []),
                "notes": case.get("notes", ""),
                "status": _evaluate_case(case, nearby) if boxes else "not_evaluated",
                "nearbyBoxes": nearby,
                "roiInspection": inspection,
            }
        )

    report = {
        "casePack": case_pack_path.as_posix(),
        "pdfPath": pdf_path.as_posix(),
        "analysisPath": analysis_path.as_posix() if analysis_path else None,
        "templatesDir": templates_dir.as_posix() if templates_dir else None,
        "boxCount": len(boxes),
        "templateCount": len(templates),
        "cases": report_cases,
    }
    (output_dir / "case_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown_report(output_dir / "case_report.md", report)
    return report


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# PDF Case Report: {Path(str(report['pdfPath'])).name}",
        "",
        f"- Analysis boxes: `{report['boxCount']}`",
        f"- Templates: `{report['templateCount']}`",
        f"- Analysis JSON: `{report['analysisPath'] or '(none)'}`",
        "",
        "| Case | Status | Expected / Allowed | Nearby | Crop |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in report["cases"]:
        expected = ", ".join(case.get("expectedSymbols") or case.get("allowedSymbols") or [])
        nearby = ", ".join(
            f"{_symbol(box)}@{int(float(box.get('x', 0)))},{int(float(box.get('y', 0)))}"
            for box in case.get("nearbyBoxes", [])[:5]
        )
        lines.append(
            f"| `{case['id']}` | `{case['status']}` | `{expected}` | "
            f"{nearby or '(none)'} | [{case['crop']}]({case['crop']}) |"
        )
    lines.append("")
    lines.append("## ROI Inspector Top Candidates")
    for case in report["cases"]:
        lines.append("")
        lines.append(f"### `{case['id']}`")
        if not case.get("roiInspection"):
            lines.append("ROI inspector skipped: pass `--templates-dir` with reviewed templates.")
            continue
        for candidate in case["roiInspection"].get("candidates", [])[:8]:
            lines.append(
                "- "
                f"`{candidate.get('symbolName')}` "
                f"{'PASS' if candidate.get('accepted') else candidate.get('reason')} "
                f"match={candidate.get('match')} ver={candidate.get('verification')} "
                f"cov={candidate.get('coverage')} pur={candidate.get('purity')}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a diagnostic crop/report pack for a PDF case.")
    parser.add_argument("case_pack", type=Path, help="Case pack JSON.")
    parser.add_argument(
        "--analysis",
        type=Path,
        default=None,
        help="Optional candidate analysis JSON with boxes.",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=None,
        help="Optional reviewed templates directory for ROI inspector candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "backend" / "tests" / "output" / "pdf_case_report",
    )
    parser.add_argument("--padding", type=int, default=80)
    parser.add_argument("--roi-tolerance", type=int, default=35)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    report = build_report(
        args.case_pack,
        args.output_dir,
        analysis_path=args.analysis,
        templates_dir=args.templates_dir,
        padding=args.padding,
        roi_tolerance=args.roi_tolerance,
        top_n=args.top_n,
    )
    print(f"Wrote {args.output_dir / 'case_report.md'}")
    print(f"Cases: {len(report['cases'])}, boxes: {report['boxCount']}, templates: {report['templateCount']}")


if __name__ == "__main__":
    main()
