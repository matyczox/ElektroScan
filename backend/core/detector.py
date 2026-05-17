"""Public detector router.

Keep the color and gray entrypoints separated so tuning gray PDFs cannot
accidentally change the fast color-PDF path.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.detector_color_engine import detect_symbols_color
from core.detector_gray_engine import detect_symbols_gray
from core.detector_models import DetectionResult, TemplateInfo
from core.detector_templates import load_templates
from core.detector_visualization import draw_results


def detect_symbols(
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    subtract_legend: bool = True,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
    pdf_path: str | None = None,
    pdf_dpi: int = 300,
    hidden_layers: list[str] | None = None,
    debug_profile: dict | None = None,
    detector_profile: str = "color",
    progress_callback=None,
) -> list[DetectionResult]:
    """Route detection to the explicit color or gray engine."""

    if detector_profile == "gray":
        return detect_symbols_gray(
            plan_image,
            templates,
            subtract_legend=subtract_legend,
            exclude_rects=exclude_rects,
            pdf_path=pdf_path,
            pdf_dpi=pdf_dpi,
            hidden_layers=hidden_layers,
            debug_profile=debug_profile,
            progress_callback=progress_callback,
        )

    return detect_symbols_color(
        plan_image,
        templates,
        subtract_legend=subtract_legend,
        exclude_rects=exclude_rects,
        pdf_path=pdf_path,
        pdf_dpi=pdf_dpi,
        hidden_layers=hidden_layers,
        debug_profile=debug_profile,
        progress_callback=progress_callback,
    )


if __name__ == "__main__":
    import sys

    plan_path = sys.argv[1] if len(sys.argv) > 1 else "wygenerowany_plan_300dpi.png"
    templates_dir = sys.argv[2] if len(sys.argv) > 2 else "templates"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "wynik.png"
    profile = sys.argv[4] if len(sys.argv) > 4 else "color"

    print(f"Loading plan: {plan_path}")
    plan = cv2.imread(plan_path)
    if plan is None:
        print(f"Error: cannot read {plan_path}")
        sys.exit(1)

    print(f"Loading templates from: {templates_dir}")
    loaded_templates = load_templates(templates_dir)
    print(f"Loaded {len(loaded_templates)} templates.\n")

    results = detect_symbols(plan, loaded_templates, detector_profile=profile)
    total = sum(result.count for result in results)
    for result in results:
        print(f"{result.symbol_name[:43]:<45} | {result.count:>5}")
    print("-" * 68)
    print(f"{'TOTAL':<45} | {total:>5}")

    output_image = draw_results(plan, results)
    cv2.imwrite(output_path, output_image)
    print(f"\nSaved result: {output_path}")
