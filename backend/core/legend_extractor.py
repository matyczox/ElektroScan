"""
Compatibility wrapper for legend extraction public APIs and legacy helper imports.

Implementation lives in smaller legend_* modules so detector/legend debugging can stay
inside manageable files. Keep re-exports here stable for existing tests, tools and API
callers that import private helpers from core.legend_extractor.
"""

from __future__ import annotations

from .legend_common import _next_template_index
from .legend_constants import (
    CELL_BORDER_TRIM,
    GLUE_KERNEL,
    HSV_LOWER,
    HSV_UPPER,
    MAX_FILENAME_LENGTH,
    MIN_PIXEL_DENSITY,
    MIN_SYMBOL_SIZE,
    SYMBOL_PADDING,
    TEXT_MAX_DISTANCE_X,
    TEXT_MIN_OVERLAP_X,
    TEXT_TOLERANCE_Y,
)
from .legend_label_extractor import (
    _get_classic_row_label_text,
    _get_row_index_text,
    _get_row_label_text,
    _get_row_symbol_code_text,
    _get_symbol_text_inside_region,
    _get_table_description_label_ocr,
    _get_table_description_label_text,
    _get_visual_row_index_text,
    _is_row_label_prefix,
    _ocr_text_from_image,
)
from .legend_mask_utils import _hsv_mask, _ink_mask, _legend_symbol_mask, _visible_ink_mask
from .legend_models import ExtractedSymbol, LegendExtractionBundle
from .legend_orchestration import (
    _build_color_text_row_vector_drafts,
    _build_scene_transform_metadata,
    _extract_raster_bundle,
    _normalize_legend_engine,
    _symbol_pixel_count,
    _vector_drafts_are_usable,
    _write_vector_drafts_as_symbols,
    extract_legend,
    extract_legend_detailed,
)
from .legend_pdf_render import (
    _normalize_layer_name,
    _prepare_doc_with_hidden_layers,
    get_pdf_layers,
    pdf_to_png,
)
from .legend_raster_extractor import _extract_legend_raster_current, _is_spurious_color_legend_fragment
from .legend_row_extractors import (
    _color_classic_row_symbol_bboxes,
    _detect_gray_description_cut,
    _filter_gray_legend_symbol_contours,
    _get_classic_row_label_ocr,
    _gray_row_symbol_bboxes,
    _group_gray_row_spans,
    _group_visual_gray_row_spans,
    _rect_to_contour,
    _strip_gray_legend_descriptions,
    _visual_gray_description_row_spans,
)
from .legend_table_drafts import (
    _build_color_table_display_drafts,
    _build_gray_table_display_drafts,
    _build_table_grid_description_drafts,
    _is_descriptive_table_label,
    _table_region_color_score,
)
from .legend_table_geometry import (
    _cell_has_content,
    _detect_legend_format,
    _expand_legend_rect_to_table,
    _find_left_symbol_gutter_x,
    _first_table_symbol_column_right,
    _merge_close_indices,
    _remove_bottom_neighbor_label_components,
    _row_has_table_separator,
    _table_grid_region_candidates,
    _table_symbol_images_have_color,
    _table_symbol_images_look_valid,
    _table_symbol_quality,
    _table_symbols_need_expansion,
    _tighten_gray_legend_symbol_crop,
    _trim_selection_to_table_grid,
)
from .legend_table_raw import (
    _extract_gray_table_legend_raw,
    _extract_left_gutter_table_legend_raw,
    _extract_table_legend_raw,
)
from .legend_text import (
    _clean_ocr_label_text,
    _clean_table_description_ocr_text,
    _sanitize_filename,
    _symbol_text_token,
)
from .legend_visual_code import _read_visual_symbol_code
from .legend_vector_drafts import VectorLegendDraft, build_vector_legend_drafts
from .legend_vector_profile import PageProfile, profile_legend_region
