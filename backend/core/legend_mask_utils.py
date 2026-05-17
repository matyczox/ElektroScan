"""Raster mask utilities shared by legend extraction paths."""

from __future__ import annotations

import cv2
import numpy as np

HSV_LOWER = np.array([0, 30, 50])
HSV_UPPER = np.array([180, 255, 255])


def _hsv_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Create a binary mask of colored pixels."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)


def _ink_mask(image_bgr: np.ndarray, threshold: int = 238) -> np.ndarray:
    """Create a binary mask for dark ink in gray/black PDFs."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ink_pixels = gray < threshold
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color_pixels = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER) > 0
    ink_pixels = np.logical_and(ink_pixels, np.logical_not(color_pixels))
    return np.where(ink_pixels, 255, 0).astype(np.uint8)


def _legend_symbol_mask(image_bgr: np.ndarray, mask_mode: str = "auto") -> tuple[np.ndarray, str]:
    """Pick HSV color masking or dark-ink masking for legend segmentation."""
    requested = (mask_mode or "auto").lower()
    if requested not in {"auto", "color", "gray"}:
        requested = "auto"

    color_mask = _hsv_mask(image_bgr)
    if requested == "color":
        return color_mask, "color"

    ink_mask = _ink_mask(image_bgr)
    if requested == "gray":
        return ink_mask, "gray"

    color_pixels = int(cv2.countNonZero(color_mask))
    ink_pixels = int(cv2.countNonZero(ink_mask))
    if ink_pixels == 0:
        return color_mask, "color"

    color_ratio = color_pixels / max(ink_pixels, 1)
    if color_pixels < 100 or color_ratio < 0.08:
        return ink_mask, "gray"
    return color_mask, "color"


def _visible_ink_mask(image_bgr: np.ndarray, gray_threshold: int = 235) -> np.ndarray:
    """Mask visible dark or colored drawing pixels."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark_pixels = gray < gray_threshold
    color_pixels = _hsv_mask(image_bgr) > 0
    return np.where(np.logical_or(dark_pixels, color_pixels), 255, 0).astype(np.uint8)
