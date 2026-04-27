from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _performance_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return (
        snapshot.get("performance")
        or snapshot.get("analysisContext", {}).get("performance")
        or {}
    )


def _top_timings(timings: dict[str, Any], limit: int) -> list[tuple[str, float]]:
    ignored = {"total", "totalBeforeSnapshot"}
    rows: list[tuple[str, float]] = []
    for name, value in timings.items():
        if name in ignored:
            continue
        if isinstance(value, (int, float)):
            rows.append((name, float(value)))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:limit]


def _format_ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.0f} ms"
    return "n/a"


def _snapshot_paths(inputs: list[str], latest: int) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:latest])
        elif path.is_file():
            paths.append(path)
    return paths


def summarize(path: Path, limit: int) -> str:
    snapshot = _load_json(path)
    context = snapshot.get("analysisContext", {})
    performance = _performance_from_snapshot(snapshot)
    backend_timings = performance.get("backendTimingsMs", {})
    backend_counters = performance.get("backendCounters", {})
    detector = performance.get("detector", {})
    detector_timings = detector.get("timingsMs", {})
    detector_counters = detector.get("counters", {})

    lines = [
        f"Snapshot: {path}",
        f"analysisId={context.get('analysisId', 'n/a')}",
        f"sourcePdf={context.get('sourcePdf', 'n/a')}",
        f"total={_format_ms(backend_timings.get('total') or backend_timings.get('totalBeforeSnapshot'))}",
    ]

    if not performance:
        lines.append("performance=(missing; run analysis again with current backend)")
        return "\n".join(lines)

    lines.append("")
    lines.append("Backend top stages:")
    for name, value in _top_timings(backend_timings, limit):
        lines.append(f"  {name}: {_format_ms(value)}")

    lines.append("")
    lines.append("Detector top stages:")
    for name, value in _top_timings(detector_timings, limit):
        lines.append(f"  {name}: {_format_ms(value)}")

    lines.append("")
    lines.append("Counters:")
    for key in (
        "templatesLoaded",
        "boxes",
        "resultJpegBytes",
        "raw_peaks",
        "validated_template_hits",
        "prefilter_hits",
        "parent_search_input_hits",
        "parent_search_candidates",
        "final_hits",
    ):
        value = backend_counters.get(key, detector_counters.get(key))
        if value is not None:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ElektroScan analysis performance snapshots.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["backend/analysis_debug"],
        help="Snapshot JSON files or directories. Defaults to backend/analysis_debug.",
    )
    parser.add_argument("--latest", type=int, default=3, help="How many newest snapshots to read per directory.")
    parser.add_argument("--top", type=int, default=8, help="How many timing stages to print.")
    args = parser.parse_args()

    paths = _snapshot_paths(args.paths, args.latest)
    if not paths:
        print("No snapshot JSON files found.")
        return

    for index, path in enumerate(paths):
        if index:
            print("\n" + "-" * 72 + "\n")
        print(summarize(path, args.top))


if __name__ == "__main__":
    main()
