import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

from api_analysis_utils import _normalize_detector_profile
from api_models import GrayDebugZonesRequest, RoiInspectRequest
from api_rendering import ANALYSIS_DPI, _build_pdf_diagnostics, _render_pdf_for_session
from api_workspace import _session_file_or_404
from api_zones import _extract_exclude_rects_from_request
from core import detector_gray as gray_strategy
from core.detector import load_templates
from core.detector_config import GRAY_SCALES
from core.detector_masks import _ink_mask
from core.detector_templates import _prepare_variants
from core.roi_inspector import inspect_roi


def _inspect_roi_response(
    session_id: str,
    body: RoiInspectRequest,
    *,
    upload_dir: Path,
    templates_dir: Path,
) -> dict:
    plan_path = _session_file_or_404(session_id, upload_dir)

    try:
        hidden_layers = body.hidden_layers if body else []
        requested_profile = _normalize_detector_profile(body.detector_profile if body else "auto")
        plan_img, _cache_hit = _render_pdf_for_session(
            session_id,
            str(plan_path),
            dpi=ANALYSIS_DPI,
            hidden_layers=hidden_layers,
        )
        pdf_diagnostics = _build_pdf_diagnostics(str(plan_path), plan_img)
        resolved_profile = (
            pdf_diagnostics.get("recommendedProfile", "color")
            if requested_profile == "auto"
            else requested_profile
        )
        if resolved_profile not in {"color", "gray"}:
            resolved_profile = "color"

        templates = load_templates(str(templates_dir))
        roi = (
            int(round(body.roi.x)),
            int(round(body.roi.y)),
            int(round(body.roi.width)),
            int(round(body.roi.height)),
        )
        return {
            "inspection": inspect_roi(
                plan_img,
                templates,
                roi,
                detector_profile=resolved_profile,
                top_n=body.top_n or 15,
            ),
            "detectorProfileRequested": requested_profile,
            "detectorProfileUsed": resolved_profile,
            "pdfDiagnostics": pdf_diagnostics,
        }
    except Exception as e:
        print(f"Blad inspektora ROI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _gray_debug_zones_response(
    session_id: str,
    body: GrayDebugZonesRequest | None,
    *,
    upload_dir: Path,
    templates_dir: Path,
) -> dict:
    plan_path = _session_file_or_404(session_id, upload_dir)

    try:
        hidden_layers = body.hidden_layers if body else []
        plan_img, _cache_hit = _render_pdf_for_session(
            session_id,
            str(plan_path),
            dpi=ANALYSIS_DPI,
            hidden_layers=hidden_layers,
        )
        templates = load_templates(str(templates_dir))
        exclude_rects, plan_zone_rect, plan_zone_outside_rects = (
            _extract_exclude_rects_from_request(
                body,
                plan_img.shape,
            )
        )

        raw_dilated = _ink_mask(plan_img, dilate=True)
        plan_masks_by_template = {template_id: raw_dilated for template_id in range(len(templates))}
        gray_masks = gray_strategy.build_gray_scan_masks(
            plan_image=plan_img,
            templates=templates,
            plan_masks_by_template=plan_masks_by_template,
            exclude_rects=exclude_rects,
            raw_dilated=raw_dilated,
        )

        rois_by_rect: dict[tuple[int, int, int, int], int] = {}
        template_roi_counts: dict[str, int] = {}
        for template_id, template in enumerate(templates):
            variants = _prepare_variants(template_id, template, scales=GRAY_SCALES)
            if variants:
                max_width = max(variant.width for variant in variants)
                max_height = max(variant.height for variant in variants)
            else:
                max_height, max_width = template.mask.shape[:2]
            rois, _uses_full_scan, _roi_area, _foreground = gray_strategy.build_gray_search_rois(
                gray_masks.zone_mask,
                plan_img.shape,
                max_width,
                max_height,
            )
            template_roi_counts[template.name] = len(rois)
            for rect in rois:
                rois_by_rect[rect] = rois_by_rect.get(rect, 0) + 1

        overlay = np.zeros((plan_img.shape[0], plan_img.shape[1], 4), dtype=np.uint8)
        zone_pixels = gray_masks.zone_mask > 0
        evidence_pixels = gray_masks.evidence_mask > 0
        overlay[zone_pixels] = (94, 197, 34, 55)  # green, BGRA
        overlay[evidence_pixels] = (22, 115, 249, 125)  # orange, BGRA

        display_rois = gray_strategy.coalesce_gray_debug_rois(
            list(rois_by_rect.keys()),
            plan_img.shape,
        )
        for x, y, w, h in display_rois:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 59, 150), 2)

        ok, buffer_overlay = cv2.imencode(".png", overlay)
        if not ok:
            raise RuntimeError("Nie udalo sie wygenerowac overlay stref.")

        return {
            "overlayImage": f"data:image/png;base64,{base64.b64encode(buffer_overlay).decode('utf-8')}",
            "imageWidth": int(plan_img.shape[1]),
            "imageHeight": int(plan_img.shape[0]),
            "zoneThreshold": int(gray_masks.zone_threshold),
            "evidenceThreshold": int(gray_masks.evidence_threshold),
            "zonePixels": int(gray_masks.zone_ink_pixels),
            "evidencePixels": int(gray_masks.evidence_ink_pixels),
            "roiCount": int(len(display_rois)),
            "roiRefs": int(sum(rois_by_rect.values())),
            "templates": int(len(templates)),
            "planZoneUsed": plan_zone_rect,
            "planZoneOutsideExcluded": plan_zone_outside_rects,
            "topTemplateRoiCounts": [
                {"template": name, "count": count}
                for name, count in sorted(
                    template_roi_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:10]
            ],
        }
    except Exception as e:
        print(f"Blad podgladu gray debug zones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
