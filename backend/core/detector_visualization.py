"""Visualization helpers for detector results."""

from __future__ import annotations

import cv2
import numpy as np

from core.detector_models import DetectionResult


def draw_results(
    plan_image: np.ndarray,
    results: list[DetectionResult],
) -> np.ndarray:
    """Draw detection boxes on a copy of the plan image."""

    output = plan_image.copy()

    for result in results:
        color = np.random.randint(0, 255, size=3).tolist()
        for det in result.detections:
            cv2.rectangle(
                output,
                (det.x, det.y),
                (det.x + det.width, det.y + det.height),
                color,
                2,
            )

    return output
