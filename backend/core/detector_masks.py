"""Compatibility facade for detector mask helpers.

New code should import focused helpers from:
- detector_mask_builders for raw/color/ink masks and search ROIs,
- detector_shape_metrics for binary-mask shape scoring,
- detector_hit_validation for hit validation policy.
"""

from __future__ import annotations

from core.detector_hit_validation import ValidationMaskCache, _validate_template_hit
from core.detector_mask_builders import (
    _build_search_rois,
    _cached_dilated_mask,
    _color_mask_for_template,
    _dominant_hsv_color,
    _find_local_maxima,
    _hsv_mask,
    _hue_distance,
    _ink_mask,
    _roi_color_similarity,
    _suppress_long_strokes,
)
from core.detector_shape_metrics import (
    _context_purity,
    _extract_label_content_mask,
    _foreground_path_variance_ratio,
    _foreground_span_ratios,
    _label_content_score,
    _mask_bbox,
    _mask_centroid,
    _roi_mask,
    _thickness_normalized_mask,
    _tight_mask_crop,
)

__all__ = [
    "ValidationMaskCache",
    "_build_search_rois",
    "_cached_dilated_mask",
    "_color_mask_for_template",
    "_context_purity",
    "_dominant_hsv_color",
    "_extract_label_content_mask",
    "_find_local_maxima",
    "_foreground_path_variance_ratio",
    "_foreground_span_ratios",
    "_hsv_mask",
    "_hue_distance",
    "_ink_mask",
    "_label_content_score",
    "_mask_bbox",
    "_mask_centroid",
    "_roi_color_similarity",
    "_roi_mask",
    "_suppress_long_strokes",
    "_thickness_normalized_mask",
    "_tight_mask_crop",
    "_validate_template_hit",
]
