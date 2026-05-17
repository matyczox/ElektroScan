"""Shared data models for legend extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExtractedSymbol:
    """One extracted legend symbol template."""

    name: str
    image: np.ndarray
    index: int
    pixel_count: int = 0


@dataclass
class LegendExtractionBundle:
    """Detailed extraction result used by vector-first and raster fallback flows."""

    extracted_symbols: list[ExtractedSymbol]
    used_legend_rect_px_300: tuple[int, int, int, int]
    engine_requested: str
    engine_used: str
    fallback_reason: str | None = None
    page_profile: dict | None = None
    scene_transform: dict | None = None
    vector_drafts: list[dict] | None = None
    vector_primitives: list[dict] | None = None
