"""Gray-PDF detector entrypoint.

This wrapper intentionally locks the shared pipeline to ``detector_profile=gray``.
Color PDFs should never route through this path.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.detector_config import DEFAULT_PDF_DPI
from core.detector_models import DetectionResult, TemplateInfo
from core.detector_pipeline import _detect_symbols_pipeline


def detect_symbols_gray(
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    *,
    subtract_legend: bool = True,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
    pdf_path: str | None = None,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
    debug_profile: dict | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> list[DetectionResult]:
    return _detect_symbols_pipeline(
        plan_image,
        templates,
        subtract_legend=subtract_legend,
        exclude_rects=exclude_rects,
        pdf_path=pdf_path,
        pdf_dpi=pdf_dpi,
        hidden_layers=hidden_layers,
        debug_profile=debug_profile,
        detector_profile="gray",
        progress_callback=progress_callback,
    )
