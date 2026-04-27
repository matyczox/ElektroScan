from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_FOCUS_PREFIXES = ("06", "07", "10", "11", "12")


def _load_boxes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    boxes = payload.get("boxes", [])
    if not isinstance(boxes, list):
        raise ValueError(f"{path} does not contain a list at key 'boxes'")
    return [box for box in boxes if isinstance(box, dict)]


def _symbol_name(box: dict[str, Any]) -> str:
    return str(box.get("symbolName") or box.get("symbol_name") or "")


def _source(box: dict[str, Any]) -> str:
    return str(box.get("source") or "")


def _prefix(symbol_name: str) -> str:
    return symbol_name.split("_", 1)[0]


def _is_focus(symbol_name: str, focus_prefixes: tuple[str, ...]) -> bool:
    return _prefix(symbol_name) in focus_prefixes


def _center(box: dict[str, Any]) -> tuple[float, float]:
    return (
        float(box.get("x", 0)) + float(box.get("width", 0)) / 2.0,
        float(box.get("y", 0)) + float(box.get("height", 0)) / 2.0,
    )


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx, ly = _center(left)
    rx, ry = _center(right)
    return math.hypot(lx - rx, ly - ry)


def _size_close(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> bool:
    for key in ("width", "height"):
        left_value = max(1.0, float(left.get(key, 0)))
        right_value = max(1.0, float(right.get(key, 0)))
        if abs(left_value - right_value) / max(left_value, right_value) > tolerance:
            return False
    return True


def _counts(boxes: list[dict[str, Any]]) -> Counter[str]:
    return Counter(_symbol_name(box) for box in boxes)


def _match_same_symbol(
    golden_boxes: list[dict[str, Any]],
    candidate_boxes: list[dict[str, Any]],
    center_tolerance: float,
    size_tolerance: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    pairs: list[tuple[float, int, int]] = []

    for golden_index, golden_box in enumerate(golden_boxes):
        golden_symbol = _symbol_name(golden_box)
        for candidate_index, candidate_box in enumerate(candidate_boxes):
            if _symbol_name(candidate_box) != golden_symbol:
                continue
            if not _size_close(golden_box, candidate_box, size_tolerance):
                continue
            distance = _distance(golden_box, candidate_box)
            if distance <= center_tolerance:
                pairs.append((distance, golden_index, candidate_index))

    matched_golden: set[int] = set()
    matched_candidate: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for distance, golden_index, candidate_index in sorted(pairs):
        if golden_index in matched_golden or candidate_index in matched_candidate:
            continue
        matched_golden.add(golden_index)
        matched_candidate.add(candidate_index)
        matches.append((golden_index, candidate_index, distance))

    return matches, matched_golden, matched_candidate


def _nearest_class_conflicts(
    golden_boxes: list[dict[str, Any]],
    candidate_boxes: list[dict[str, Any]],
    unmatched_golden: set[int],
    unmatched_candidate: set[int],
    center_tolerance: float,
    focus_prefixes: tuple[str, ...],
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    conflicts: list[tuple[dict[str, Any], dict[str, Any], float]] = []

    for golden_index in sorted(unmatched_golden):
        golden_box = golden_boxes[golden_index]
        if not _is_focus(_symbol_name(golden_box), focus_prefixes):
            continue

        best: tuple[float, int] | None = None
        for candidate_index in unmatched_candidate:
            candidate_box = candidate_boxes[candidate_index]
            if _symbol_name(candidate_box) == _symbol_name(golden_box):
                continue
            distance = _distance(golden_box, candidate_box)
            if distance <= center_tolerance and (best is None or distance < best[0]):
                best = (distance, candidate_index)

        if best is not None:
            conflicts.append((golden_box, candidate_boxes[best[1]], best[0]))

    return conflicts


def _box_label(box: dict[str, Any]) -> str:
    symbol = _symbol_name(box)
    return (
        f"{symbol}@{int(float(box.get('x', 0)))},"
        f"{int(float(box.get('y', 0)))}"
        f" {int(float(box.get('width', 0)))}x{int(float(box.get('height', 0)))}"
        f" source={_source(box)}"
    )


def compare_snapshots(
    golden_path: Path,
    candidate_path: Path,
    focus_prefixes: tuple[str, ...],
    center_tolerance: float,
    size_tolerance: float,
) -> str:
    golden_boxes = _load_boxes(golden_path)
    candidate_boxes = _load_boxes(candidate_path)
    matches, matched_golden, matched_candidate = _match_same_symbol(
        golden_boxes,
        candidate_boxes,
        center_tolerance=center_tolerance,
        size_tolerance=size_tolerance,
    )

    unmatched_golden = set(range(len(golden_boxes))) - matched_golden
    unmatched_candidate = set(range(len(candidate_boxes))) - matched_candidate
    source_shifts = [
        (golden_boxes[golden_index], candidate_boxes[candidate_index], distance)
        for golden_index, candidate_index, distance in matches
        if _source(golden_boxes[golden_index]) != _source(candidate_boxes[candidate_index])
    ]
    class_conflicts = _nearest_class_conflicts(
        golden_boxes,
        candidate_boxes,
        unmatched_golden,
        unmatched_candidate,
        center_tolerance=center_tolerance,
        focus_prefixes=focus_prefixes,
    )

    golden_counts = _counts(golden_boxes)
    candidate_counts = _counts(candidate_boxes)
    focus_symbols = sorted(
        {
            symbol
            for symbol in set(golden_counts) | set(candidate_counts)
            if _is_focus(symbol, focus_prefixes)
        }
    )

    lines = [
        f"Golden:   {golden_path} ({len(golden_boxes)} boxes)",
        f"Candidate:{candidate_path} ({len(candidate_boxes)} boxes)",
        f"Matched same-symbol boxes: {len(matches)}",
        "",
        "Focus counts:",
    ]
    for symbol in focus_symbols:
        golden_count = golden_counts.get(symbol, 0)
        candidate_count = candidate_counts.get(symbol, 0)
        delta = candidate_count - golden_count
        lines.append(f"  {symbol}: golden={golden_count} candidate={candidate_count} delta={delta:+d}")

    focus_missing = [
        golden_boxes[index]
        for index in sorted(unmatched_golden)
        if _is_focus(_symbol_name(golden_boxes[index]), focus_prefixes)
    ]
    focus_extra = [
        candidate_boxes[index]
        for index in sorted(unmatched_candidate)
        if _is_focus(_symbol_name(candidate_boxes[index]), focus_prefixes)
    ]

    lines.extend(["", f"Missing focus boxes: {len(focus_missing)}"])
    lines.extend(f"  - {_box_label(box)}" for box in focus_missing[:20])
    if len(focus_missing) > 20:
        lines.append(f"  ... {len(focus_missing) - 20} more")

    lines.extend(["", f"Extra focus boxes: {len(focus_extra)}"])
    lines.extend(f"  - {_box_label(box)}" for box in focus_extra[:20])
    if len(focus_extra) > 20:
        lines.append(f"  ... {len(focus_extra) - 20} more")

    lines.extend(["", f"Class conflicts near golden focus boxes: {len(class_conflicts)}"])
    for golden_box, candidate_box, distance in class_conflicts[:20]:
        lines.append(f"  - golden {_box_label(golden_box)} -> candidate {_box_label(candidate_box)} dist={distance:.1f}")
    if len(class_conflicts) > 20:
        lines.append(f"  ... {len(class_conflicts) - 20} more")

    focus_source_shifts = [
        (golden_box, candidate_box, distance)
        for golden_box, candidate_box, distance in source_shifts
        if _is_focus(_symbol_name(golden_box), focus_prefixes)
    ]
    lines.extend(["", f"Focus source shifts: {len(focus_source_shifts)}"])
    for golden_box, candidate_box, distance in focus_source_shifts[:20]:
        lines.append(
            f"  - {_symbol_name(golden_box)}@{int(float(golden_box.get('x', 0)))},"
            f"{int(float(golden_box.get('y', 0)))} "
            f"{_source(golden_box)} -> {_source(candidate_box)} dist={distance:.1f}"
        )
    if len(focus_source_shifts) > 20:
        lines.append(f"  ... {len(focus_source_shifts) - 20} more")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ElektroScan analysis snapshots.")
    parser.add_argument("golden", type=Path, help="Golden JSON snapshot, e.g. DOBRYLOGPDF.JSON.")
    parser.add_argument("candidate", type=Path, help="Candidate analysis_debug JSON snapshot.")
    parser.add_argument(
        "--focus",
        default=",".join(DEFAULT_FOCUS_PREFIXES),
        help="Comma-separated numeric prefixes to emphasize.",
    )
    parser.add_argument("--center-tolerance", type=float, default=18.0)
    parser.add_argument("--size-tolerance", type=float, default=0.35)
    args = parser.parse_args()

    focus_prefixes = tuple(part.strip() for part in args.focus.split(",") if part.strip())
    print(
        compare_snapshots(
            args.golden,
            args.candidate,
            focus_prefixes=focus_prefixes,
            center_tolerance=args.center_tolerance,
            size_tolerance=args.size_tolerance,
        )
    )


if __name__ == "__main__":
    main()
