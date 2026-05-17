"""Vector-first legend extraction orchestration with raster fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

try:
    from .legend_common import _next_template_index
    from .legend_mask_utils import _legend_symbol_mask
    from .legend_models import ExtractedSymbol, LegendExtractionBundle
    from .legend_pdf_render import _prepare_doc_with_hidden_layers
    from .legend_raster_extractor import _extract_legend_raster_current
    from .legend_row_extractors import _color_classic_row_symbol_bboxes
    from .legend_scene_transform import build_scene_transform, hash_pdf_file, rect_px300_to_pt
    from .legend_table_drafts import _build_color_table_display_drafts, _build_gray_table_display_drafts
    from .legend_text import _sanitize_filename
    from .legend_vector_drafts import VectorLegendDraft, build_vector_legend_drafts
    from .legend_vector_profile import PageProfile, profile_legend_region
except ImportError:  # pragma: no cover
    from legend_common import _next_template_index
    from legend_mask_utils import _legend_symbol_mask
    from legend_models import ExtractedSymbol, LegendExtractionBundle
    from legend_pdf_render import _prepare_doc_with_hidden_layers
    from legend_raster_extractor import _extract_legend_raster_current
    from legend_row_extractors import _color_classic_row_symbol_bboxes
    from legend_scene_transform import build_scene_transform, hash_pdf_file, rect_px300_to_pt
    from legend_table_drafts import _build_color_table_display_drafts, _build_gray_table_display_drafts
    from legend_text import _sanitize_filename
    from legend_vector_drafts import VectorLegendDraft, build_vector_legend_drafts
    from legend_vector_profile import PageProfile, profile_legend_region

def _normalize_legend_engine(legend_engine: str | None) -> str:
    engine = str(legend_engine or "auto").strip().lower()
    if engine not in {"auto", "raster", "vector_first"}:
        return "auto"
    return engine


def _vector_drafts_are_usable(drafts: list[VectorLegendDraft]) -> tuple[bool, str | None]:
    if len(drafts) < 2:
        return False, "insufficient_vector_drafts"
    mean_confidence = sum(draft.confidence for draft in drafts) / len(drafts)
    if mean_confidence < 0.58:
        return False, "low_vector_draft_confidence"
    return True, None


def _symbol_pixel_count(symbol_image: np.ndarray) -> int:
    if symbol_image.size == 0:
        return 0
    gray = cv2.cvtColor(symbol_image, cv2.COLOR_BGR2GRAY)
    return int(cv2.countNonZero((gray > 12).astype(np.uint8)))


def _write_vector_drafts_as_symbols(
    drafts: list[VectorLegendDraft],
    output_path: Path,
) -> list[ExtractedSymbol]:
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[ExtractedSymbol] = []
    counter = _next_template_index(output_path)
    for draft in sorted(drafts, key=lambda item: (item.row_bbox_pt[1], item.bbox_pt[0])):
        if draft.image_bgr is None or draft.image_bgr.size == 0:
            continue
        safe_name = _sanitize_filename(draft.name_draft) or f"symbol_{counter:02d}"
        filename = f"{counter:02d}_{safe_name}.png"
        file_path = output_path / filename
        ok, buf = cv2.imencode(".png", draft.image_bgr)
        if not ok:
            continue
        file_path.write_bytes(buf.tobytes())
        results.append(
            ExtractedSymbol(
                name=safe_name,
                image=draft.image_bgr,
                index=counter,
                pixel_count=_symbol_pixel_count(draft.image_bgr),
            )
        )
        counter += 1
    return results


def _build_color_text_row_vector_drafts(
    page: fitz.Page,
    plan_image: np.ndarray,
    legend_rect_px: tuple[int, int, int, int],
    transform,
    profile: PageProfile,
) -> list[VectorLegendDraft]:
    """Use PDF text rows plus color mask components for safer color legend drafts."""

    x_start, y_start, width, height = legend_rect_px
    if width <= 0 or height <= 0:
        return []

    image_h, image_w = plan_image.shape[:2]
    x0 = max(0, int(x_start))
    y0 = max(0, int(y_start))
    x1 = min(image_w, int(x_start + width))
    y1 = min(image_h, int(y_start + height))
    if x1 <= x0 or y1 <= y0:
        return []

    legend_area = plan_image[y0:y1, x0:x1]
    raw_symbol_mask, mask_used = _legend_symbol_mask(legend_area, "color")
    if mask_used != "color":
        return []

    try:
        text_words = page.get_text("words")
    except Exception:
        return []

    _symbol_mask, bboxes, labels = _color_classic_row_symbol_bboxes(
        raw_symbol_mask,
        text_words,
        x_start=x0,
        y_start=y0,
        scale=transform.dpi / 72.0,
    )
    drafts: list[VectorLegendDraft] = []
    for idx, local_bbox in enumerate(bboxes, start=1):
        lx, ly, lw, lh = local_bbox
        if lw <= 0 or lh <= 0:
            continue

        local_mask = raw_symbol_mask[ly : ly + lh, lx : lx + lw]
        color_roi = legend_area[ly : ly + lh, lx : lx + lw]
        if local_mask.size == 0 or color_roi.size == 0:
            continue
        symbol_image = np.zeros_like(color_roi)
        symbol_image[local_mask > 0] = color_roi[local_mask > 0]
        if _symbol_pixel_count(symbol_image) < 8:
            continue

        abs_bbox_px = (x0 + lx, y0 + ly, lw, lh)
        bbox_pt = rect_px300_to_pt(abs_bbox_px, transform)
        label = labels.get(local_bbox) or f"symbol_{idx:02d}"
        drafts.append(
            VectorLegendDraft(
                draft_id=f"vlegend:color-row:{idx}",
                bbox_pt=bbox_pt,
                bbox_px_300=abs_bbox_px,
                row_bbox_pt=bbox_pt,
                name_draft=label,
                symbol_code=None,
                confidence=0.84 if profile.legend_kind_hint in {"rows", "table"} else 0.74,
                primitive_refs=[],
                review_required=True,
                label_source="right_text",
                structure_source="row_anchor",
                fallback_eligible=True,
                image_bgr=symbol_image,
            )
        )
    return drafts


def _extract_raster_bundle(
    *,
    pdf_path: str,
    plan_image: np.ndarray,
    output_dir: str,
    dpi: int,
    exclude_rects: list[tuple[int, int, int, int]] | None,
    legend_rect_px: tuple[int, int, int, int],
    mask_mode: str,
    hidden_layers: list[str] | None,
    engine_requested: str,
    fallback_reason: str | None,
    page_profile: PageProfile | None = None,
    scene_transform: dict | None = None,
    vector_drafts: list[VectorLegendDraft] | None = None,
    vector_primitives: list | None = None,
) -> LegendExtractionBundle:
    raster_result = _extract_legend_raster_current(
        pdf_path,
        plan_image,
        output_dir=output_dir,
        dpi=dpi,
        exclude_rects=exclude_rects,
        legend_rect_px=legend_rect_px,
        mask_mode=mask_mode,
        return_used_rect=True,
        hidden_layers=hidden_layers,
    )
    symbols, used_rect = raster_result
    return LegendExtractionBundle(
        extracted_symbols=symbols,
        used_legend_rect_px_300=used_rect,
        engine_requested=engine_requested,
        engine_used="raster",
        fallback_reason=fallback_reason,
        page_profile=page_profile.to_json() if page_profile else None,
        scene_transform=scene_transform,
        vector_drafts=[draft.to_json() for draft in (vector_drafts or [])],
        vector_primitives=vector_primitives or [],
    )


def _build_scene_transform_metadata(
    pdf_path: str,
    *,
    dpi: int,
    hidden_layers: list[str] | None,
) -> dict | None:
    doc: fitz.Document | None = None
    try:
        try:
            pdf_hash = hash_pdf_file(pdf_path)
        except Exception:
            pdf_hash = ""
        doc = _prepare_doc_with_hidden_layers(pdf_path, hidden_layers=hidden_layers)
        page = doc.load_page(0)
        return build_scene_transform(
            page,
            dpi=dpi,
            hidden_layers=hidden_layers,
            source_pdf_sha256=pdf_hash,
        ).to_json()
    except Exception:
        return None
    finally:
        if doc is not None:
            doc.close()


def extract_legend_detailed(
    pdf_path: str,
    plan_image: np.ndarray,
    output_dir: str = "templates",
    dpi: int = 300,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
    legend_rect_px: tuple[int, int, int, int] | None = None,
    mask_mode: str = "auto",
    hidden_layers: list[str] | None = None,
    legend_engine: str = "auto",
    include_debug_primitives: bool = False,
) -> LegendExtractionBundle:
    """Vector-first legend extraction orchestrator with raster fallback."""

    if legend_rect_px is None:
        raise ValueError(
            "legend_rect_px is required before legend extraction."
        )

    engine_requested = _normalize_legend_engine(legend_engine)
    output_path = Path(output_dir)

    if engine_requested == "raster":
        scene_transform_json = _build_scene_transform_metadata(
            pdf_path,
            dpi=dpi,
            hidden_layers=hidden_layers,
        )
        return _extract_raster_bundle(
            pdf_path=pdf_path,
            plan_image=plan_image,
            output_dir=output_dir,
            dpi=dpi,
            exclude_rects=exclude_rects,
            legend_rect_px=legend_rect_px,
            mask_mode=mask_mode,
            hidden_layers=hidden_layers,
            engine_requested=engine_requested,
            fallback_reason=None,
            scene_transform=scene_transform_json,
        )

    if engine_requested == "auto" and str(mask_mode).lower() == "gray":
        scene_transform_json = _build_scene_transform_metadata(
            pdf_path,
            dpi=dpi,
            hidden_layers=hidden_layers,
        )
        bundle = _extract_raster_bundle(
            pdf_path=pdf_path,
            plan_image=plan_image,
            output_dir=output_dir,
            dpi=dpi,
            exclude_rects=exclude_rects,
            legend_rect_px=legend_rect_px,
            mask_mode=mask_mode,
            hidden_layers=hidden_layers,
            engine_requested=engine_requested,
            fallback_reason="gray_mask_mode",
            scene_transform=scene_transform_json,
        )
        doc: fitz.Document | None = None
        try:
            try:
                pdf_hash = hash_pdf_file(pdf_path)
            except Exception:
                pdf_hash = ""
            doc = _prepare_doc_with_hidden_layers(pdf_path, hidden_layers=hidden_layers)
            page = doc.load_page(0)
            transform = build_scene_transform(
                page,
                dpi=dpi,
                hidden_layers=hidden_layers,
                source_pdf_sha256=pdf_hash,
            )
            label_drafts = _build_gray_table_display_drafts(
                page,
                plan_image.copy(),
                legend_rect_px,
                transform,
                mask_mode=mask_mode,
                exclude_rects=exclude_rects,
            )
            bundle.vector_drafts = [draft.to_json() for draft in label_drafts]
        except Exception:
            bundle.vector_drafts = []
        finally:
            if doc is not None:
                doc.close()
        return bundle

    doc: fitz.Document | None = None
    profile: PageProfile | None = None
    scene_transform_json: dict | None = None
    fallback_reason: str | None = None

    try:
        pdf_hash = hash_pdf_file(pdf_path)
    except Exception:
        pdf_hash = ""

    try:
        doc = _prepare_doc_with_hidden_layers(pdf_path, hidden_layers=hidden_layers)
        page = doc.load_page(0)
        transform = build_scene_transform(
            page,
            dpi=dpi,
            hidden_layers=hidden_layers,
            source_pdf_sha256=pdf_hash,
        )
        scene_transform_json = transform.to_json()
        profile = profile_legend_region(page, legend_rect_px, transform)
        should_attempt = engine_requested == "vector_first" or profile.attempt_vector

        if engine_requested == "auto" and str(mask_mode).lower() == "color" and should_attempt:
            fallback_reason = "color_vector_auto_guard"
            bundle = _extract_raster_bundle(
                pdf_path=pdf_path,
                plan_image=plan_image,
                output_dir=output_dir,
                dpi=dpi,
                exclude_rects=exclude_rects,
                legend_rect_px=legend_rect_px,
                mask_mode=mask_mode,
                hidden_layers=hidden_layers,
                engine_requested=engine_requested,
                fallback_reason=fallback_reason,
                page_profile=profile,
                scene_transform=scene_transform_json,
            )

            label_drafts: list[VectorLegendDraft] = []
            primitive_refs = []
            try:
                vector_plan_image = plan_image.copy()
                if exclude_rects:
                    for ex, ey, ew, eh in exclude_rects:
                        cv2.rectangle(
                            vector_plan_image,
                            (int(ex), int(ey)),
                            (int(ex + ew), int(ey + eh)),
                            (255, 255, 255),
                            -1,
                        )
                label_drafts = _build_color_table_display_drafts(
                    page,
                    vector_plan_image.copy(),
                    legend_rect_px,
                    transform,
                    expected_count=len(bundle.extracted_symbols),
                    exclude_rects=None,
                )
                if not label_drafts:
                    label_drafts = _build_color_text_row_vector_drafts(
                        page,
                        vector_plan_image,
                        legend_rect_px,
                        transform,
                        profile,
                    )
                if not label_drafts:
                    label_drafts, primitive_refs = build_vector_legend_drafts(
                        page,
                        vector_plan_image,
                        legend_rect_px,
                        transform,
                        profile,
                    )
            except Exception:
                label_drafts = []
                primitive_refs = []

            bundle.vector_drafts = [draft.to_json() for draft in label_drafts]
            bundle.vector_primitives = (
                [ref.to_json() for ref in primitive_refs] if include_debug_primitives else []
            )
            return bundle

        if not should_attempt:
            fallback_reason = profile.fallback_reason or "profile_not_vector_ready"
        else:
            vector_plan_image = plan_image.copy()
            if exclude_rects:
                for ex, ey, ew, eh in exclude_rects:
                    cv2.rectangle(
                        vector_plan_image,
                        (int(ex), int(ey)),
                        (int(ex + ew), int(ey + eh)),
                        (255, 255, 255),
                        -1,
                    )
            drafts = []
            primitive_refs = []
            if str(mask_mode).lower() == "color":
                drafts = _build_color_text_row_vector_drafts(
                    page,
                    vector_plan_image,
                    legend_rect_px,
                    transform,
                    profile,
                )
                if drafts and include_debug_primitives:
                    try:
                        _debug_drafts, primitive_refs = build_vector_legend_drafts(
                            page,
                            vector_plan_image,
                            legend_rect_px,
                            transform,
                            profile,
                        )
                    except Exception:
                        primitive_refs = []
            if not drafts:
                drafts, primitive_refs = build_vector_legend_drafts(
                    page,
                    vector_plan_image,
                    legend_rect_px,
                    transform,
                    profile,
                )
            usable, no_go_reason = _vector_drafts_are_usable(drafts)
            if usable:
                symbols = _write_vector_drafts_as_symbols(drafts, output_path)
                if symbols:
                    return LegendExtractionBundle(
                        extracted_symbols=symbols,
                        used_legend_rect_px_300=legend_rect_px,
                        engine_requested=engine_requested,
                        engine_used="vector_first",
                        fallback_reason=None,
                        page_profile=profile.to_json(),
                        scene_transform=scene_transform_json,
                        vector_drafts=[draft.to_json() for draft in drafts],
                        vector_primitives=(
                            [ref.to_json() for ref in primitive_refs]
                            if include_debug_primitives
                            else []
                        ),
                    )
                fallback_reason = "vector_draft_write_failed"
            else:
                fallback_reason = no_go_reason
    except Exception as exc:
        fallback_reason = f"vector_exception:{exc.__class__.__name__}"
    finally:
        if doc is not None:
            doc.close()

    return _extract_raster_bundle(
        pdf_path=pdf_path,
        plan_image=plan_image,
        output_dir=output_dir,
        dpi=dpi,
        exclude_rects=exclude_rects,
        legend_rect_px=legend_rect_px,
        mask_mode=mask_mode,
        hidden_layers=hidden_layers,
        engine_requested=engine_requested,
        fallback_reason=fallback_reason,
        page_profile=profile,
        scene_transform=scene_transform_json,
    )


def extract_legend(
    pdf_path: str,
    plan_image: np.ndarray,
    output_dir: str = "templates",
    dpi: int = 300,
    exclude_rects: list[tuple[int, int, int, int]] = None,
    legend_rect_px: tuple[int, int, int, int] | None = None,
    mask_mode: str = "auto",
    return_used_rect: bool = False,
    hidden_layers: list[str] | None = None,
    legend_engine: str = "auto",
    include_legend_debug: bool = False,
) -> list[ExtractedSymbol] | tuple[list[ExtractedSymbol], tuple[int, int, int, int]]:
    bundle = extract_legend_detailed(
        pdf_path,
        plan_image,
        output_dir=output_dir,
        dpi=dpi,
        exclude_rects=exclude_rects,
        legend_rect_px=legend_rect_px,
        mask_mode=mask_mode,
        hidden_layers=hidden_layers,
        legend_engine=legend_engine,
        include_debug_primitives=include_legend_debug,
    )
    if return_used_rect:
        return bundle.extracted_symbols, bundle.used_legend_rect_px_300
    return bundle.extracted_symbols
