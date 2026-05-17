"""Runtime/debug option parsing for the detector pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass

from core.detector_context import trace_points_from_value, trace_values_from_value


@dataclass(frozen=True)
class DetectorRuntimeOptions:
    initial_debug_profile: dict
    collect_performance_profile: bool
    ablation_no_text_mirror: bool
    gray_force_text_mirror: bool
    disable_text_mirror: bool
    trace_symbols: set[str]
    trace_points: list[tuple[float, float]]
    trace_radius: float
    trace_max_items: int


def build_runtime_options(
    *,
    debug_profile: dict | None,
    detector_profile: str,
) -> DetectorRuntimeOptions:
    initial_debug_profile = dict(debug_profile or {})
    collect_performance_profile = bool(initial_debug_profile.get("performanceProfile"))
    ablation_value = str(
        initial_debug_profile.get("ablation") or os.getenv("ELEKTROSCAN_ABLATION", "")
    ).strip().lower()
    ablation_no_text_mirror = ablation_value in {
        "no-text-mirror",
        "no_text_mirror",
        "notextmirror",
    } or os.getenv("ELEKTROSCAN_ABLATION_NO_TEXT_MIRROR", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    gray_text_mirror_override = os.getenv("ELEKTROSCAN_GRAY_TEXT_MIRROR", "").strip().lower()
    gray_force_text_mirror = ablation_value in {
        "text-mirror",
        "text_mirror",
        "with-text-mirror",
        "with_text_mirror",
    } or gray_text_mirror_override in {"1", "true", "yes", "on"}
    disable_text_mirror = ablation_no_text_mirror or (
        detector_profile == "gray" and not gray_force_text_mirror
    )

    trace_input = initial_debug_profile.get("candidateTrace") or initial_debug_profile.get("trace") or {}
    if not isinstance(trace_input, dict):
        trace_input = {}

    trace_symbols = set(trace_values_from_value(trace_input.get("symbols") or trace_input.get("symbol")))
    trace_symbols.update(trace_values_from_value(os.getenv("ELEKTROSCAN_TRACE_SYMBOLS")))
    trace_points = trace_points_from_value(trace_input.get("points") or trace_input.get("point"))
    trace_points.extend(trace_points_from_value(os.getenv("ELEKTROSCAN_TRACE_POINTS")))
    trace_radius = float(trace_input.get("radius") or os.getenv("ELEKTROSCAN_TRACE_RADIUS", 80))
    trace_max_items = int(trace_input.get("maxItems") or os.getenv("ELEKTROSCAN_TRACE_MAX_ITEMS", 40))

    return DetectorRuntimeOptions(
        initial_debug_profile=initial_debug_profile,
        collect_performance_profile=collect_performance_profile,
        ablation_no_text_mirror=ablation_no_text_mirror,
        gray_force_text_mirror=gray_force_text_mirror,
        disable_text_mirror=disable_text_mirror,
        trace_symbols=trace_symbols,
        trace_points=trace_points,
        trace_radius=trace_radius,
        trace_max_items=trace_max_items,
    )
