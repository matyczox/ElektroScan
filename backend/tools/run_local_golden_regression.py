from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_analysis_snapshot import compare_snapshots  # noqa: E402
from core.detector import detect_symbols  # noqa: E402
from core.detector_templates import load_templates  # noqa: E402
from core.legend_extractor import pdf_to_png  # noqa: E402


DEFAULT_FIXTURES_DIR = REPO_ROOT / "backend" / "tests" / "fixtures"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "backend" / "analysis_debug" / "local_regression"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _repo_path(value: str | None, *, base: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if base is not None and (base / path).exists():
        return (base / path).resolve()
    return (REPO_ROOT / path).resolve()


def _load_manifests(
    fixtures_dir: Path,
    names: set[str] | None,
) -> list[tuple[Path, dict[str, Any]]]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(fixtures_dir.glob("*/manifest.json")):
        manifest = _read_json(path)
        name = str(manifest.get("name") or path.parent.name)
        if names and name not in names and path.parent.name not in names:
            continue
        manifests.append((path, manifest))
    return manifests


def _detections_to_boxes(results: list[Any]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for group in results:
        for detection in group.detections:
            boxes.append(
                {
                    "symbolName": group.symbol_name,
                    "x": detection.x,
                    "y": detection.y,
                    "width": detection.width,
                    "height": detection.height,
                    "match": detection.confidence,
                    "verification": detection.verification_score,
                    "coverage": detection.coverage,
                    "purity": detection.purity,
                    "context_purity": detection.context_purity,
                    "rotation": detection.rotation,
                    "scale": detection.scale,
                    "mirrored": detection.mirrored,
                    "source": detection.source,
                }
            )
    return boxes


def _run_fixture(
    manifest_path: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    trace_symbols: list[str],
    trace_points: list[str],
    trace_radius: float,
    ablation: str | None,
    collect_performance_profile: bool,
) -> tuple[bool, dict[str, Any]]:
    name = str(manifest.get("name") or manifest_path.parent.name)
    if not manifest.get("enabled", True):
        print(f"SKIP {name}: disabled")
        return True, {"name": name, "skipped": True}

    pdf_path = _repo_path(manifest.get("pdfPath"), base=manifest_path.parent)
    templates_dir = _repo_path(manifest.get("templatesDir"), base=manifest_path.parent)
    golden_path = _repo_path(manifest.get("golden"), base=manifest_path.parent)
    if pdf_path is None or not pdf_path.exists():
        raise RuntimeError(f"{name}: missing pdfPath {pdf_path}")
    if templates_dir is None or not templates_dir.exists():
        raise RuntimeError(f"{name}: missing templatesDir {templates_dir}")
    if golden_path is None or not golden_path.exists():
        raise RuntimeError(f"{name}: missing golden {golden_path}")

    print(f"\n=== {name} ===")
    started = time.perf_counter()
    image = pdf_to_png(str(pdf_path), dpi=300)
    render_elapsed = time.perf_counter() - started

    templates_started = time.perf_counter()
    templates = load_templates(str(templates_dir))
    templates_elapsed = time.perf_counter() - templates_started
    expected_templates = manifest.get("templateCount")
    if expected_templates is not None and int(expected_templates) != len(templates):
        raise RuntimeError(
            f"{name}: expected {expected_templates} templates, loaded {len(templates)}"
        )

    excluded_zones = [
        (int(zone["x"]), int(zone["y"]), int(zone["width"]), int(zone["height"]))
        for zone in manifest.get("excludedZones", [])
    ]
    debug_profile: dict[str, Any] = {
        "performanceProfile": collect_performance_profile,
    }
    if trace_symbols or trace_points:
        debug_profile["trace"] = {
            "symbols": trace_symbols,
            "points": trace_points,
            "radius": trace_radius,
        }
    if ablation:
        debug_profile["ablation"] = ablation
    detect_started = time.perf_counter()
    results = detect_symbols(
        image,
        templates,
        exclude_rects=excluded_zones,
        pdf_path=str(pdf_path),
        pdf_dpi=300,
        detector_profile=str(manifest.get("detectorProfile") or manifest.get("expectedProfile") or "gray"),
        debug_profile=debug_profile,
        progress_callback=lambda *_args: None,
    )
    detect_elapsed = time.perf_counter() - detect_started

    boxes = _detections_to_boxes(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_suffix = f"_{ablation}" if ablation else ""
    output_path = output_dir / f"{name}_local_candidate{output_suffix}.json"
    candidate_write_started = time.perf_counter()
    output_path.write_text(
        json.dumps(
            {
                "analysisContext": {
                    "sourcePdf": pdf_path.name,
                    "detectorProfileUsed": manifest.get("detectorProfile", "gray"),
                    "ablation": ablation or "",
                },
                "boxes": boxes,
                "performance": {"detector": debug_profile},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    candidate_write_elapsed = time.perf_counter() - candidate_write_started

    detector_timings = debug_profile.get("timingsMs", {})
    counters = debug_profile.get("counters", {})
    print(
        "Timing: "
        f"render={render_elapsed:.1f}s templates={templates_elapsed:.3f}s "
        f"detect={detect_elapsed:.1f}s "
        f"scan={float(detector_timings.get('scan', 0.0)) / 1000.0:.1f}s "
        f"prepare={float(detector_timings.get('prepare', 0.0)) / 1000.0:.1f}s "
        f"write={candidate_write_elapsed:.3f}s"
    )
    print(
        "Counters: "
        f"boxes={len(boxes)} variants={counters.get('prepared_variants')} "
        f"rois={counters.get('search_rois')} raw={counters.get('raw_peaks')} "
        f"validated={counters.get('validated_template_hits')}"
    )
    scan_profile = debug_profile.get("scanProfile", {})
    scan_total = scan_profile.get("total", {}) if isinstance(scan_profile, dict) else {}
    if scan_total:
        print(
            "Scan profile: "
            f"calls={scan_total.get('calls')} pixels={scan_total.get('pixels')} "
            f"outputPixels={scan_total.get('outputPixels')} rawPeaks={scan_total.get('rawPeaks')}"
        )
    trace = debug_profile.get("candidateTrace", {})
    if trace:
        print("Trace:")
        for stage, stage_trace in trace.items():
            print(
                f"  {stage}: matched={stage_trace.get('matchedCandidates')} "
                f"total={stage_trace.get('totalCandidates')}"
            )

    focus = tuple(part.strip() for part in str(manifest.get("focus", "")).split(",") if part.strip())
    compare_started = time.perf_counter()
    report = compare_snapshots(
        golden_path,
        output_path,
        focus_prefixes=focus,
        center_tolerance=float(manifest.get("centerTolerance", 18)),
        size_tolerance=float(manifest.get("sizeTolerance", 0.35)),
    )
    compare_elapsed = time.perf_counter() - compare_started
    print(report)
    print(f"Runner compare: {compare_elapsed:.3f}s")

    failed = (
        "Missing focus boxes: 0" not in report
        or "Extra focus boxes: 0" not in report
        or "Class conflicts near golden focus boxes: 0" not in report
    )
    perf_record = {
        "name": name,
        "passed": not failed,
        "ablation": ablation or "",
        "renderSeconds": round(render_elapsed, 3),
        "templateLoadSeconds": round(templates_elapsed, 3),
        "detectSeconds": round(detect_elapsed, 3),
        "candidateWriteSeconds": round(candidate_write_elapsed, 3),
        "compareSeconds": round(compare_elapsed, 3),
        "boxes": len(boxes),
        "candidatePath": str(output_path),
        "goldenPath": str(golden_path),
        "timingsMs": detector_timings,
        "counters": counters,
        "threading": debug_profile.get("threading", {}),
        "scanProfile": debug_profile.get("scanProfile", {}),
        "hitFlowProfile": debug_profile.get("hitFlowProfile", {}),
        "grayVariantStrategy": debug_profile.get("grayVariantStrategy", {}),
        "slowestPhase": debug_profile.get("slowestPhase"),
    }
    return not failed, perf_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden regression directly in-process.")
    parser.add_argument("--fixture", action="append", help="Fixture name/folder to run.")
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trace-symbol", action="append", default=[], help="Symbol/prefix to trace.")
    parser.add_argument("--trace-point", action="append", default=[], help="Point to trace as x,y.")
    parser.add_argument("--trace-radius", type=float, default=80.0)
    parser.add_argument("--perf-json", type=Path, help="Write per-fixture performance JSON.")
    parser.add_argument(
        "--ablation",
        choices=["no-text-mirror", "text-mirror", "with-text-mirror"],
        help="Run a diagnostic variant. Expected to fail if it changes golden output.",
    )
    args = parser.parse_args()

    names = set(args.fixture or []) or None
    manifests = _load_manifests(args.fixtures_dir, names)
    if not manifests:
        raise SystemExit("No fixtures matched.")

    all_ok = True
    perf_records: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for manifest_path, manifest in manifests:
        fixture_ok, perf_record = _run_fixture(
            manifest_path,
            manifest,
            args.output_dir,
            trace_symbols=args.trace_symbol,
            trace_points=args.trace_point,
            trace_radius=args.trace_radius,
            ablation=args.ablation,
            collect_performance_profile=bool(args.perf_json),
        )
        perf_records.append(perf_record)
        all_ok = fixture_ok and all_ok

    total_elapsed = time.perf_counter() - total_started
    print(f"\nTotal local regression time: {total_elapsed:.1f}s")
    if args.perf_json:
        perf_path = args.perf_json if args.perf_json.is_absolute() else (REPO_ROOT / args.perf_json)
        perf_path.parent.mkdir(parents=True, exist_ok=True)
        perf_write_started = time.perf_counter()
        perf_path.write_text(
            json.dumps(
                {
                    "totalSeconds": round(total_elapsed, 3),
                    "ablation": args.ablation or "",
                    "fixtures": perf_records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        perf_write_elapsed = time.perf_counter() - perf_write_started
        print(f"Performance JSON: {perf_path} (write={perf_write_elapsed:.3f}s)")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
