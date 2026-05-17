"""Small analysis utility helpers shared by API routes and services."""

from __future__ import annotations

import os
import time

from core.legend_extractor import _normalize_layer_name, get_pdf_layers


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


VERBOSE_LOGS = _env_flag("ELEKTROSCAN_VERBOSE_LOGS", default=True)
DEFAULT_ANALYSIS_DEBUG = _env_flag("ELEKTROSCAN_ANALYSIS_DEBUG", default=True)


def _log(message: str) -> None:
    if VERBOSE_LOGS:
        print(message)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _slowest_stages(timings_ms: dict[str, float], limit: int = 8) -> list[dict]:
    ignored = {"total", "totalBeforeSnapshot"}
    ranked = sorted(
        (
            (name, value)
            for name, value in timings_ms.items()
            if name not in ignored and isinstance(value, (int, float))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"name": name, "ms": round(float(value), 3)} for name, value in ranked[:limit]]


def _build_hidden_layer_debug(pdf_path: str, hidden_layers: list[str]) -> dict:
    available_layers = get_pdf_layers(pdf_path)
    available_names = [
        str(layer.get("name", "")) for layer in available_layers if layer.get("name")
    ]
    normalized_available = {}
    for name in available_names:
        normalized_available.setdefault(_normalize_layer_name(name), []).append(name)

    requested = []
    matched = []
    unmatched = []

    for raw_name in hidden_layers:
        normalized = _normalize_layer_name(raw_name)
        matches = normalized_available.get(normalized, [])
        entry = {
            "value": raw_name,
            "repr": ascii(raw_name),
            "length": len(raw_name),
            "normalized": normalized,
            "matches": matches,
        }
        requested.append(entry)
        if matches:
            matched.append(raw_name)
        else:
            unmatched.append(raw_name)

    return {
        "requested": requested,
        "matched": matched,
        "unmatched": unmatched,
    }


def _normalize_detector_profile(profile: str | None) -> str:
    value = (profile or "auto").strip().lower()
    return value if value in {"auto", "color", "gray"} else "auto"


def _normalize_legend_engine(engine: str | None) -> str:
    value = (engine or "auto").strip().lower()
    return value if value in {"auto", "raster", "vector_first"} else "auto"
