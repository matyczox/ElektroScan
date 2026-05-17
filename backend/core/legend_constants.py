"""Shared constants for raster legend extraction."""

from __future__ import annotations

import numpy as np

HSV_LOWER = np.array([0, 30, 50])
HSV_UPPER = np.array([180, 255, 255])
GLUE_KERNEL = np.ones((4, 40), np.uint8)
MIN_SYMBOL_SIZE = 15
SYMBOL_PADDING = 2
CELL_BORDER_TRIM = 2
MIN_PIXEL_DENSITY = 0.05
TEXT_TOLERANCE_Y = 15
TEXT_MAX_DISTANCE_X = 250
TEXT_MIN_OVERLAP_X = -15
MAX_FILENAME_LENGTH = 80
