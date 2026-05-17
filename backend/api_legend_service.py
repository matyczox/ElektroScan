import tempfile
from pathlib import Path

from fastapi import HTTPException

from api_analysis_utils import (
    _log,
    _normalize_detector_profile,
    _normalize_legend_engine,
)
from api_models import ExtractRequest
from api_rendering import ANALYSIS_DPI, _build_pdf_diagnostics, _render_pdf_for_session
from api_workspace import _session_file_or_404
from core.legend_extractor import extract_legend_detailed
from template_store import (
    _append_extracted_templates,
    _legend_display_labels_from_drafts,
    _load_template_labels,
    _template_payload_from_path,
)


async def _extract_legend_response(
    session_id: str,
    body: ExtractRequest | None,
    *,
    upload_dir: Path,
    templates_dir: Path,
) -> dict:
    file_path = _session_file_or_404(session_id, upload_dir)

    try:
        _log("Renderowanie planu do ekstrakcji (300 DPI)")
        hidden_layers = body.hidden_layers if body else []
        requested_profile = _normalize_detector_profile(body.detector_profile if body else "auto")
        requested_legend_engine = _normalize_legend_engine(body.legend_engine if body else "auto")
        include_legend_debug = bool(body.include_legend_debug) if body else False
        plan_img, _cache_hit = _render_pdf_for_session(
            session_id,
            str(file_path),
            dpi=ANALYSIS_DPI,
            hidden_layers=hidden_layers,
            copy_image=True,
        )
        pdf_diagnostics = _build_pdf_diagnostics(str(file_path), plan_img)
        resolved_profile = (
            pdf_diagnostics.get("recommendedProfile", "color")
            if requested_profile == "auto"
            else requested_profile
        )
        mask_mode = resolved_profile if resolved_profile in {"color", "gray"} else "auto"

        exclude_rects = []
        if body and body.excluded_zones:
            for zone in body.excluded_zones:
                try:
                    exclude_rects.append(
                        (int(zone["x"]), int(zone["y"]), int(zone["width"]), int(zone["height"]))
                    )
                except (KeyError, ValueError):
                    pass

        legend_rect_px = None
        if body and body.legend_zone:
            legend_rect_px = (
                int(round(body.legend_zone.x)),
                int(round(body.legend_zone.y)),
                int(round(body.legend_zone.width)),
                int(round(body.legend_zone.height)),
            )

        if legend_rect_px is None:
            raise HTTPException(
                status_code=400,
                detail="Brak strefy legendy. Zaznacz obszar legendy na planie przed ekstrakcją.",
            )

        _log("Ekstrakcja legendy...")
        templates_dir.mkdir(parents=True, exist_ok=True)
        added_template_ids: set[str] = set()
        with tempfile.TemporaryDirectory(dir=templates_dir.parent) as extraction_dir:
            legend_bundle = extract_legend_detailed(
                str(file_path),
                plan_img,
                output_dir=extraction_dir,
                dpi=ANALYSIS_DPI,
                exclude_rects=exclude_rects,
                legend_rect_px=legend_rect_px,
                mask_mode=mask_mode,
                hidden_layers=hidden_layers,
                legend_engine=requested_legend_engine,
                include_debug_primitives=include_legend_debug,
            )
            symbols = legend_bundle.extracted_symbols
            legend_rect_px = legend_bundle.used_legend_rect_px_300

            if symbols:
                display_labels = _legend_display_labels_from_drafts(
                    legend_bundle.vector_drafts,
                    len(symbols),
                )
                added_template_ids = _append_extracted_templates(
                    Path(extraction_dir),
                    templates_dir,
                    display_labels=display_labels,
                )

        legend_metadata = {
            "legendEngineRequested": legend_bundle.engine_requested,
            "legendEngineUsed": legend_bundle.engine_used,
            "legendFallbackReason": legend_bundle.fallback_reason,
            "legendPageProfile": legend_bundle.page_profile,
            "sceneTransform": legend_bundle.scene_transform,
        }
        if include_legend_debug:
            legend_metadata["legendVectorDrafts"] = legend_bundle.vector_drafts or []
            legend_metadata["legendVectorPrimitives"] = legend_bundle.vector_primitives or []

        if not symbols:
            patterns_list = []
            labels = _load_template_labels(templates_dir)
            for template_path in sorted(templates_dir.glob("*.png")):
                payload = _template_payload_from_path(template_path, labels)
                if payload is None:
                    continue
                payload["status"] = "existing"
                patterns_list.append(payload)
            return {
                "patterns": patterns_list,
                "legendExtractedCount": 0,
                "legendAddedIds": [],
                "legendZoneUsed": legend_rect_px,
                "legendMaskMode": mask_mode,
                "detectorProfileRequested": requested_profile,
                "detectorProfileUsed": resolved_profile,
                "pdfDiagnostics": pdf_diagnostics,
                **legend_metadata,
            }

        patterns_list = []
        labels = _load_template_labels(templates_dir)
        for template_path in sorted(templates_dir.glob("*.png")):
            payload = _template_payload_from_path(template_path, labels)
            if payload is None:
                continue
            payload["status"] = "pending" if payload["id"] in added_template_ids else "existing"
            patterns_list.append(payload)

        return {
            "patterns": patterns_list,
            "legendExtractedCount": len(added_template_ids),
            "legendAddedIds": sorted(added_template_ids),
            "legendZoneUsed": legend_rect_px,
            "legendMaskMode": mask_mode,
            "detectorProfileRequested": requested_profile,
            "detectorProfileUsed": resolved_profile,
            "pdfDiagnostics": pdf_diagnostics,
            **legend_metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Błąd podczas ekstrakcji: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Błąd serwera: {str(e)}")
