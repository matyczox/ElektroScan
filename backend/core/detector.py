"""
detector.py - CPU-friendly symbol detection for electrical plans.

This module now orchestrates the pipeline. The heavy helpers live in sibling
modules so detector behavior stays easier to audit and tune.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from core.detector_clustering import _bbox_metrics, _cluster_candidates, _prefilter_candidates, _prefilter_raw_template_hits
from core.detector_config import (
    DEFAULT_PDF_DPI,
    DEBUG_ACCEPTED_UNCERTAIN_CANDIDATES_LIMIT,
    DEBUG_ACCEPTED_UNCERTAIN_MAX_CONTEXT_PURITY,
    DEBUG_ACCEPTED_UNCERTAIN_MAX_SCORE,
    DEBUG_ACCEPTED_UNCERTAIN_NEARBY_PX,
    DEBUG_CANDIDATES_LIMIT,
    DEBUG_CONFLICT_CANDIDATES_LIMIT,
    DEBUG_CONFLICT_MAX_SCORE_DROP,
    DEBUG_CONFLICT_MIN_IOM,
    DEBUG_NEAR_UNCERTAIN_COMPONENT_PX,
    DEBUG_REJECTED_CANDIDATES_LIMIT,
    DEBUG_REJECTED_MIN_MATCH,
    DEBUG_UNEXPLAINED_CANDIDATES_LIMIT,
    DEBUG_UNEXPLAINED_MAX_AREA,
    DEBUG_UNEXPLAINED_MIN_AREA,
    DEBUG_UNEXPLAINED_PER_UNCERTAIN_BOX,
    DETECTOR_POSTPROCESS_MAX_WORKERS,
    DETECTOR_SCAN_MAX_WORKERS,
    LABEL_CONTENT_SCAN_MIN_PIXELS,
    MAX_PEAKS_PER_VARIANT,
    MAX_TEXT_CONTENT_PEAKS_PER_VARIANT,
    MIN_TEMPLATE_PIXELS,
    OPENCV_NUM_THREADS,
    PRECISE_KEYWORDS,
    TEXT_CONTENT_THRESHOLD,
    THRESHOLD_DILATED,
    THRESHOLD_PRECISE,
    _safe_cpu_count,
)
from core.detector_masks import (
    _build_search_rois,
    _color_mask_for_template,
    _find_local_maxima,
    _hsv_mask,
    _tight_mask_crop,
    _validate_template_hit,
)
from core.detector_models import CandidateHit, Detection, DetectionResult, TemplateInfo, TemplateVariant
from core.detector_pdf import _collect_pdf_text_hits, _estimate_legend_exclude_rect
from core.detector_promotions import _maybe_promote_socket_06_to_07, _maybe_promote_switch_parent_search
from core.detector_templates import _build_socket_07_promotions, _prepare_variants, load_templates


def _is_content_scan_eligible(variant: TemplateVariant) -> bool:
    """Keep OCR-like scans limited to useful label glyph masks."""

    if variant.content_mask is None or variant.content_bbox is None:
        return False
    if variant.content_pixel_count < LABEL_CONTENT_SCAN_MIN_PIXELS:
        return False

    content_crop = _tight_mask_crop(variant.content_mask)
    if content_crop is None:
        return False
    if content_crop.shape == variant.transformed_mask.shape:
        return False

    return True


def _hit_score(hit: CandidateHit) -> float:
    """Return the best available confidence-like score for debug ranking."""

    return float(hit.verification_score or hit.content_score or hit.match_score)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = [int(value) for value in bbox]
    return (x + w / 2.0, y + h / 2.0)


def _debug_candidate_from_hit(
    hit: CandidateHit,
    templates: list[TemplateInfo],
    reason: str,
    *,
    related_hit: CandidateHit | None = None,
) -> dict:
    """Serialize a non-final candidate for HITL/debug overlays."""

    x, y, w, h = [int(value) for value in hit.bbox]
    payload = {
        "id": f"debug_{reason}_{hit.template_id}_{x}_{y}_{w}_{h}_{hit.rotation}_{int(hit.scale * 100)}_{int(hit.mirrored)}",
        "reason": reason,
        "symbolName": templates[hit.template_id].name,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "confidence": round(float(hit.match_score), 3),
        "verificationScore": round(float(hit.verification_score), 3),
        "coverage": round(float(hit.coverage), 3),
        "purity": round(float(hit.purity), 3),
        "contextPurity": round(float(hit.context_purity), 3),
        "colorSimilarity": round(float(hit.color_similarity), 3),
        "rotation": int(hit.rotation),
        "scale": round(float(hit.scale), 3),
        "mirrored": bool(hit.mirrored),
        "source": hit.source,
        "isTextLabel": bool(hit.is_text_label),
        "contentScore": round(float(hit.content_score), 3),
    }
    if related_hit is not None:
        rx, ry, rw, rh = [int(value) for value in related_hit.bbox]
        payload["relatedFinal"] = {
            "symbolName": templates[related_hit.template_id].name,
            "bbox": [rx, ry, rw, rh],
            "verificationScore": round(float(related_hit.verification_score), 3),
            "source": related_hit.source,
        }
    return payload


def _debug_candidate_from_component(
    component_bbox: tuple[int, int, int, int],
    area: int,
) -> dict:
    """Serialize an unexplained foreground component for HITL/debug overlays."""

    x, y, w, h = component_bbox
    return {
        "id": f"debug_unexplained_component_{x}_{y}_{w}_{h}",
        "reason": "unexplained_component",
        "symbolName": "possible_missed",
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "confidence": 0.0,
        "verificationScore": 0.0,
        "coverage": 0.0,
        "purity": 0.0,
        "contextPurity": 0.0,
        "colorSimilarity": 0.0,
        "rotation": 0,
        "scale": 1.0,
        "mirrored": False,
        "source": "unexplained_component",
        "area": int(area),
    }


def _bbox_center_inside(
    inner_bbox: tuple[int, int, int, int],
    outer_bbox: tuple[int, int, int, int],
) -> bool:
    """Return True when the inner bbox center falls inside outer bbox."""

    x, y, w, h = [int(value) for value in inner_bbox]
    ox, oy, ow, oh = [int(value) for value in outer_bbox]
    cx = x + w / 2.0
    cy = y + h / 2.0
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def _is_symbol_like_unexplained_component(
    bbox: tuple[int, int, int, int],
    area: int,
) -> bool:
    """Filter debug-only components down to compact, symbol-like shapes."""

    _, _, w, h = [int(value) for value in bbox]
    if w <= 3 or h <= 3:
        return False

    bbox_area = max(1, w * h)
    density = float(area) / float(bbox_area)
    aspect = max(w / max(1, h), h / max(1, w))

    # Long, flat connected components are usually circuit text labels or notes.
    if aspect >= 3.2 and min(w, h) < 24:
        return False

    # Very sparse huge strokes are usually architectural/background remnants.
    if bbox_area > 4500 and density < 0.16:
        return False

    return True


def _unexplained_component_score(
    bbox: tuple[int, int, int, int],
    area: int,
) -> float:
    """Rank compact debug components ahead of long annotation fragments."""

    _, _, w, h = [int(value) for value in bbox]
    aspect = max(w / max(1, h), h / max(1, w))
    bbox_area = max(1, w * h)
    density = float(area) / float(bbox_area)
    return float(area) * density / max(1.0, aspect)


def _accepted_uncertainty_score(hit: CandidateHit, final_hits: list[CandidateHit]) -> float:
    """Rank accepted detections that are worth human review in crowded areas."""

    score = _hit_score(hit)
    cx, cy = _bbox_center(hit.bbox)
    nearby = 0
    for other in final_hits:
        if other is hit:
            continue
        ox, oy = _bbox_center(other.bbox)
        if abs(cx - ox) <= DEBUG_ACCEPTED_UNCERTAIN_NEARBY_PX and abs(cy - oy) <= DEBUG_ACCEPTED_UNCERTAIN_NEARBY_PX:
            nearby += 1

    if nearby == 0:
        return 0.0

    low_score = max(0.0, DEBUG_ACCEPTED_UNCERTAIN_MAX_SCORE - score)
    low_context = max(0.0, DEBUG_ACCEPTED_UNCERTAIN_MAX_CONTEXT_PURITY - float(hit.context_purity)) * 2.0
    low_coverage = max(0.0, 0.70 - float(hit.coverage)) * 0.50
    quality_signal = low_score + low_context + low_coverage
    if quality_signal <= 0.0:
        return 0.0

    crowding = min(nearby, 4) * 0.04
    return quality_signal + crowding


def _collect_debug_candidates(
    *,
    rejected_hits: list[CandidateHit],
    validated_candidates: list[CandidateHit],
    final_hits: list[CandidateHit],
    templates: list[TemplateInfo],
    plan_image: np.ndarray,
    exclude_rects: list[tuple[int, int, int, int]],
) -> list[dict]:
    """Build non-final hints for HITL without changing accepted detections."""

    debug_candidates: list[dict] = []
    seen: set[tuple[str, int, int, int, int, str]] = set()

    def add(payload: dict) -> None:
        key = (
            str(payload.get("symbolName", "")),
            int(payload.get("x", 0)),
            int(payload.get("y", 0)),
            int(payload.get("width", 0)),
            int(payload.get("height", 0)),
            str(payload.get("reason", "")),
        )
        if key in seen:
            return
        seen.add(key)
        debug_candidates.append(payload)

    rejected_count = 0
    for candidate in sorted(rejected_hits, key=_hit_score, reverse=True):
        if len(debug_candidates) >= DEBUG_CANDIDATES_LIMIT or rejected_count >= DEBUG_REJECTED_CANDIDATES_LIMIT:
            break
        if candidate.match_score < DEBUG_REJECTED_MIN_MATCH:
            continue
        reason = "rejected_low_content" if candidate.is_text_label else "rejected_candidate"
        add(_debug_candidate_from_hit(candidate, templates, reason))
        rejected_count += 1

    accepted_uncertain_count = 0
    accepted_uncertain_hits = [
        (_accepted_uncertainty_score(hit, final_hits), hit)
        for hit in final_hits
        if not hit.is_text_label
    ]
    for uncertainty, hit in sorted(accepted_uncertain_hits, key=lambda item: item[0], reverse=True):
        if (
            len(debug_candidates) >= DEBUG_CANDIDATES_LIMIT
            or accepted_uncertain_count >= DEBUG_ACCEPTED_UNCERTAIN_CANDIDATES_LIMIT
        ):
            break
        if uncertainty <= 0.0:
            break
        add(_debug_candidate_from_hit(hit, templates, "accepted_uncertain", related_hit=hit))
        accepted_uncertain_count += 1

    if len(debug_candidates) < DEBUG_CANDIDATES_LIMIT:
        explained = np.zeros(plan_image.shape[:2], dtype=np.uint8)
        uncertain_halos = np.zeros(plan_image.shape[:2], dtype=np.uint8)
        uncertain_boxes: list[tuple[int, int, int, int]] = []
        for hit in final_hits:
            x, y, w, h = [int(value) for value in hit.bbox]
            uncertainty = _accepted_uncertainty_score(hit, final_hits)
            if uncertainty > 0.0:
                pad = DEBUG_NEAR_UNCERTAIN_COMPONENT_PX
                cv2.rectangle(
                    uncertain_halos,
                    (max(0, x - pad), max(0, y - pad)),
                    (min(plan_image.shape[1], x + w + pad), min(plan_image.shape[0], y + h + pad)),
                    255,
                    -1,
                )
                uncertain_boxes.append((x, y, w, h))
            else:
                cv2.rectangle(explained, (x, y), (x + w, y + h), 255, -1)
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(explained, (ex, ey), (ex + ew, ey + eh), 255, -1)

        plan_mask = _hsv_mask(plan_image, dilate=False)
        unexplained = cv2.bitwise_and(plan_mask, cv2.bitwise_not(explained))
        components, _, stats, _ = cv2.connectedComponentsWithStats(unexplained, connectivity=8)
        components_payloads: list[tuple[float, dict]] = []
        near_uncertain_payloads: dict[int, list[tuple[float, dict]]] = {
            index: [] for index in range(len(uncertain_boxes))
        }
        for component_id in range(1, components):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < DEBUG_UNEXPLAINED_MIN_AREA or area > DEBUG_UNEXPLAINED_MAX_AREA:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            component_bbox = (x, y, w, h)
            cx, cy = _bbox_center(component_bbox)
            near_uncertain_indexes = [
                index
                for index, uncertain_box in enumerate(uncertain_boxes)
                if abs(cx - _bbox_center(uncertain_box)[0]) <= DEBUG_NEAR_UNCERTAIN_COMPONENT_PX
                and abs(cy - _bbox_center(uncertain_box)[1]) <= DEBUG_NEAR_UNCERTAIN_COMPONENT_PX
            ]
            near_uncertain = bool(near_uncertain_indexes)
            if not near_uncertain and not _is_symbol_like_unexplained_component(component_bbox, area):
                continue
            if near_uncertain and (w <= 3 or h <= 3):
                continue
            score = _unexplained_component_score(component_bbox, area) + (1000.0 if near_uncertain else 0.0)
            payload = _debug_candidate_from_component(component_bbox, area)
            components_payloads.append((score, payload))
            for index in near_uncertain_indexes:
                near_uncertain_payloads[index].append((score, payload))

        unexplained_count = 0
        for payloads in near_uncertain_payloads.values():
            for _, payload in sorted(payloads, key=lambda item: item[0], reverse=True)[:DEBUG_UNEXPLAINED_PER_UNCERTAIN_BOX]:
                if (
                    len(debug_candidates) >= DEBUG_CANDIDATES_LIMIT
                    or unexplained_count >= DEBUG_UNEXPLAINED_CANDIDATES_LIMIT
                ):
                    break
                add(payload)
                unexplained_count += 1
            if (
                len(debug_candidates) >= DEBUG_CANDIDATES_LIMIT
                or unexplained_count >= DEBUG_UNEXPLAINED_CANDIDATES_LIMIT
            ):
                break

        for _, payload in sorted(components_payloads, key=lambda item: item[0], reverse=True):
            if (
                len(debug_candidates) >= DEBUG_CANDIDATES_LIMIT
                or unexplained_count >= DEBUG_UNEXPLAINED_CANDIDATES_LIMIT
            ):
                break
            add(payload)
            unexplained_count += 1

    conflict_count = 0
    for candidate in sorted(validated_candidates, key=_hit_score, reverse=True):
        if len(debug_candidates) >= DEBUG_CANDIDATES_LIMIT or conflict_count >= DEBUG_CONFLICT_CANDIDATES_LIMIT:
            break
        for final_hit in final_hits:
            if candidate is final_hit:
                continue
            if candidate.template_id == final_hit.template_id and candidate.bbox == final_hit.bbox:
                continue
            _, _, iom, _ = _bbox_metrics(candidate.bbox, final_hit.bbox)
            if iom < DEBUG_CONFLICT_MIN_IOM:
                continue
            candidate_score = _hit_score(candidate)
            final_score = _hit_score(final_hit)
            if candidate_score + DEBUG_CONFLICT_MAX_SCORE_DROP < final_score:
                continue
            if _bbox_center_inside(candidate.bbox, final_hit.bbox) and candidate_score <= final_score + 0.03:
                continue
            if candidate.template_id == final_hit.template_id:
                reason = "partial_ghost"
            else:
                reason = "overlap_conflict"
            add(_debug_candidate_from_hit(candidate, templates, reason, related_hit=final_hit))
            conflict_count += 1
            break

    return debug_candidates[:DEBUG_CANDIDATES_LIMIT]


def detect_symbols(
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    subtract_legend: bool = True,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
    pdf_path: str | None = None,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
    debug_profile: dict | None = None,
) -> list[DetectionResult]:
    """
    Detect symbols on a rendered plan using template matching plus PDF-text fallback.
    """

    exclude_rects = list(exclude_rects or [])

    if not templates:
        return []

    timings: dict[str, float] = {}

    legend_rect = _estimate_legend_exclude_rect(
        pdf_path=pdf_path or "",
        image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
    )
    if legend_rect is not None:
        exclude_rects.append(legend_rect)

    color_masks_cache: dict[str, np.ndarray] = {}

    def _get_plan_mask(template: TemplateInfo) -> np.ndarray:
        if template.dominant_hsv is not None:
            cache_key = f"{template.dominant_hsv}_{template.requires_precision}"
            if cache_key not in color_masks_cache:
                mask = _color_mask_for_template(
                    plan_image,
                    template.dominant_hsv,
                    dilate=not template.requires_precision,
                    hsv_image=plan_hsv,
                )
                for ex, ey, ew, eh in exclude_rects:
                    cv2.rectangle(mask, (ex, ey), (ex + ew, ey + eh), 0, -1)
                color_masks_cache[cache_key] = mask
            return color_masks_cache[cache_key]

        fallback = _hsv_mask(plan_image, dilate=False, hsv_image=plan_hsv)
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(fallback, (ex, ey), (ex + ew, ey + eh), 0, -1)
        return fallback

    phase_start = time.perf_counter()
    pdf_hits_by_template = _collect_pdf_text_hits(
        pdf_path=pdf_path or "",
        templates=templates,
        plan_image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
        exclude_rects=exclude_rects,
    )
    pdf_candidates = [hit for hits in pdf_hits_by_template.values() for hit in hits]
    timings["pdf_text"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    plan_hsv = cv2.cvtColor(plan_image, cv2.COLOR_BGR2HSV)

    variant_workers = max(1, min(len(templates), DETECTOR_POSTPROCESS_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=variant_workers) as pool:
        prepared_variant_items = list(
            pool.map(
                lambda item: (item[0], _prepare_variants(item[0], item[1])),
                enumerate(templates),
            )
        )
    variants_by_template = dict(prepared_variant_items)
    variants_lookup = {
        (variant.template_id, variant.scale, variant.rotation, variant.mirrored): variant
        for variants in variants_by_template.values()
        for variant in variants
    }
    socket_07_promotions = _build_socket_07_promotions(templates, variants_by_template)
    parent_ids_by_child: dict[int, set[int]] = {}
    for rules in socket_07_promotions.values():
        for rule in rules:
            parent_ids_by_child.setdefault(rule.child_template_id, set()).add(rule.parent_template_id)
    plan_masks_by_template: dict[int, np.ndarray] = {}
    unique_mask_keys = {
        f"{template.dominant_hsv}_{template.requires_precision}"
        for template in templates
        if template.dominant_hsv is not None
    }

    def _build_cached_color_mask(cache_key: str) -> tuple[str, np.ndarray]:
        hsv_text, precision_text = cache_key.rsplit("_", 1)
        dominant_hsv = tuple(int(part.strip()) for part in hsv_text.strip("()").split(","))
        requires_precision = precision_text == "True"
        mask = _color_mask_for_template(
            plan_image,
            dominant_hsv,  # type: ignore[arg-type]
            dilate=not requires_precision,
            hsv_image=plan_hsv,
        )
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(mask, (ex, ey), (ex + ew, ey + eh), 0, -1)
        return cache_key, mask

    if unique_mask_keys:
        mask_workers = max(1, min(len(unique_mask_keys), DETECTOR_POSTPROCESS_MAX_WORKERS))
        with ThreadPoolExecutor(max_workers=mask_workers) as pool:
            color_masks_cache.update(dict(pool.map(_build_cached_color_mask, unique_mask_keys)))

    for template_id, template in enumerate(templates):
        plan_masks_by_template[template_id] = _get_plan_mask(template)
    plan_mask_foregrounds = {
        template_id: int(cv2.countNonZero(plan_mask))
        for template_id, plan_mask in plan_masks_by_template.items()
    }
    max_variant_size_by_template = {
        template_id: (
            max((variant.width for variant in variants), default=templates[template_id].mask.shape[1]),
            max((variant.height for variant in variants), default=templates[template_id].mask.shape[0]),
        )
        for template_id, variants in variants_by_template.items()
    }
    search_rois_by_template: dict[int, list[tuple[int, int, int, int]]] = {}
    search_roi_stats_by_template: dict[int, tuple[bool, int, int]] = {}
    def _prepare_search_roi(item: tuple[int, np.ndarray]) -> tuple[int, list[tuple[int, int, int, int]], tuple[bool, int, int]]:
        template_id, plan_mask = item
        max_width, max_height = max_variant_size_by_template[template_id]
        rois, uses_full_scan, roi_area, foreground_pixels = _build_search_rois(
            plan_mask,
            plan_image.shape,
            max_width,
            max_height,
        )
        return template_id, rois, (uses_full_scan, roi_area, foreground_pixels)

    roi_workers = max(1, min(len(plan_masks_by_template), DETECTOR_POSTPROCESS_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=roi_workers) as pool:
        roi_items = list(pool.map(_prepare_search_roi, plan_masks_by_template.items()))
    for template_id, rois, stats in roi_items:
        search_rois_by_template[template_id] = rois
        search_roi_stats_by_template[template_id] = stats
    dilated_plan_masks_by_template: dict[int, np.ndarray] = {}
    timings["prepare"] = time.perf_counter() - phase_start

    diagnostics = {
        "raw_peaks": 0,
        "raw_prefilter_hits": 0,
        "raw_prefilter_removed": 0,
        "prepared_variants": sum(len(variants) for variants in variants_by_template.values()),
        "skipped_empty_color_masks": 0,
        "validated_template_hits": 0,
        "promoted_targeted_hits": 0,
        "parent_search_input_hits": 0,
        "parent_search_candidates": 0,
        "promoted_parent_search_hits": 0,
        "pdf_text_hits": len(pdf_candidates),
        "prefilter_hits": 0,
        "pre_parent_clusters": 0,
        "final_hits": 0,
        "search_rois": sum(len(rois) for rois in search_rois_by_template.values()),
        "full_scan_templates": sum(1 for uses_full, _, _ in search_roi_stats_by_template.values() if uses_full),
        "roi_area_pixels": sum(area for _, area, _ in search_roi_stats_by_template.values()),
        "roi_foreground_pixels": sum(pixels for _, _, pixels in search_roi_stats_by_template.values()),
    }

    def _scan_template(template_id: int) -> list[CandidateHit]:
        template = templates[template_id]
        threshold = THRESHOLD_PRECISE if template.requires_precision else THRESHOLD_DILATED
        plan_mask = plan_masks_by_template[template_id]
        search_rois = search_rois_by_template.get(template_id, [])
        if plan_mask_foregrounds.get(template_id, 0) < MIN_TEMPLATE_PIXELS or not search_rois:
            return []

        template_hits: list[CandidateHit] = []
        for variant in variants_by_template.get(template_id, []):
            if variant.height > plan_mask.shape[0] or variant.width > plan_mask.shape[1]:
                continue

            variant_peaks: list[tuple[int, int, float]] = []
            too_many_peaks = False
            for roi_x, roi_y, roi_w, roi_h in search_rois:
                if variant.height > roi_h or variant.width > roi_w:
                    continue

                roi_plan_mask = plan_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                match_result = cv2.matchTemplate(
                    roi_plan_mask,
                    variant.transformed_mask,
                    cv2.TM_CCOEFF_NORMED,
                )
                peaks = _find_local_maxima(
                    match_result,
                    threshold=threshold,
                    template_width=variant.width,
                    template_height=variant.height,
                )
                if peaks:
                    variant_peaks.extend((roi_x + px, roi_y + py, score) for px, py, score in peaks)
                if len(variant_peaks) > MAX_PEAKS_PER_VARIANT:
                    too_many_peaks = True
                    break

            if too_many_peaks:
                variant_peaks.sort(key=lambda item: item[2], reverse=True)
                variant_peaks = variant_peaks[:MAX_PEAKS_PER_VARIANT]

            for px, py, score in variant_peaks:
                template_hits.append(
                    CandidateHit(
                        template_id=template_id,
                        scale=variant.scale,
                        rotation=variant.rotation,
                        mirrored=variant.mirrored,
                        transformed_mask=variant.transformed_mask,
                        content_mask=variant.content_mask,
                        pixel_count=variant.pixel_count,
                        content_pixel_count=variant.content_pixel_count,
                        content_bbox=variant.content_bbox,
                        bbox=(px, py, variant.width, variant.height),
                        match_score=score,
                        dominant_hsv=template.dominant_hsv,
                        source="template",
                        is_text_label=template.is_text_label,
                    )
                )

            if not template.is_text_label or not _is_content_scan_eligible(variant):
                continue

            content_crop = _tight_mask_crop(variant.content_mask)
            if content_crop is None or variant.content_bbox is None:
                continue

            content_x, content_y, content_w, content_h = variant.content_bbox
            content_peaks: list[tuple[int, int, float]] = []
            too_many_content_peaks = False
            for roi_x, roi_y, roi_w, roi_h in search_rois:
                if content_h > roi_h or content_w > roi_w:
                    continue

                roi_plan_mask = plan_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                match_result = cv2.matchTemplate(
                    roi_plan_mask,
                    content_crop,
                    cv2.TM_CCOEFF_NORMED,
                )
                peaks = _find_local_maxima(
                    match_result,
                    threshold=TEXT_CONTENT_THRESHOLD,
                    template_width=content_w,
                    template_height=content_h,
                )
                if peaks:
                    content_peaks.extend((roi_x + px, roi_y + py, score) for px, py, score in peaks)
                if len(content_peaks) > MAX_TEXT_CONTENT_PEAKS_PER_VARIANT:
                    too_many_content_peaks = True
                    break

            if too_many_content_peaks:
                content_peaks.sort(key=lambda item: item[2], reverse=True)
                content_peaks = content_peaks[:MAX_TEXT_CONTENT_PEAKS_PER_VARIANT]

            for px, py, score in content_peaks:
                template_hits.append(
                    CandidateHit(
                        template_id=template_id,
                        scale=variant.scale,
                        rotation=variant.rotation,
                        mirrored=variant.mirrored,
                        transformed_mask=variant.content_mask,
                        content_mask=variant.content_mask,
                        pixel_count=variant.content_pixel_count,
                        content_pixel_count=variant.content_pixel_count,
                        content_bbox=variant.content_bbox,
                        bbox=(px - content_x, py - content_y, variant.width, variant.height),
                        match_score=score,
                        dominant_hsv=template.dominant_hsv,
                        source="template_content",
                        is_text_label=True,
                    )
                )

        return template_hits

    template_ids_to_scan = [
        template_id
        for template_id in variants_by_template
        if plan_mask_foregrounds.get(template_id, 0) >= MIN_TEMPLATE_PIXELS
    ]
    diagnostics["skipped_empty_color_masks"] = len(variants_by_template) - len(template_ids_to_scan)
    raw_template_hits: list[CandidateHit] = []
    phase_start = time.perf_counter()
    scan_workers = max(1, min(len(template_ids_to_scan), DETECTOR_SCAN_MAX_WORKERS))
    if template_ids_to_scan:
        with ThreadPoolExecutor(max_workers=scan_workers) as pool:
            for hits in pool.map(_scan_template, template_ids_to_scan):
                raw_template_hits.extend(hits)
    else:
        scan_workers = 0
    timings["scan"] = time.perf_counter() - phase_start

    diagnostics["raw_peaks"] = len(raw_template_hits)
    phase_start = time.perf_counter()
    raw_before_prefilter = len(raw_template_hits)
    raw_template_hits = _prefilter_raw_template_hits(raw_template_hits)
    diagnostics["raw_prefilter_hits"] = len(raw_template_hits)
    diagnostics["raw_prefilter_removed"] = raw_before_prefilter - len(raw_template_hits)
    timings["raw_prefilter"] = time.perf_counter() - phase_start

    validated_candidates: list[CandidateHit] = list(pdf_candidates)
    phase_start = time.perf_counter()
    postprocess_workers = max(1, DETECTOR_POSTPROCESS_MAX_WORKERS)

    def _validate_and_promote_hit(hit: CandidateHit) -> tuple[CandidateHit, CandidateHit, CandidateHit | None]:
        plan_mask = plan_masks_by_template[hit.template_id]
        if _validate_template_hit(hit, plan_mask, plan_image):
            promoted_hit = _maybe_promote_socket_06_to_07(
                hit,
                plan_image,
                templates,
                plan_masks_by_template,
                dilated_plan_masks_by_template,
                variants_lookup,
                socket_07_promotions,
            )
            return hit, promoted_hit, None
        return hit, hit, hit

    validated_hits: list[CandidateHit] = []
    rejected_hits: list[CandidateHit] = []
    validation_workers = max(1, min(len(raw_template_hits), postprocess_workers))
    if raw_template_hits:
        with ThreadPoolExecutor(max_workers=validation_workers) as pool:
            for validation_result in pool.map(_validate_and_promote_hit, raw_template_hits):
                if validation_result is None:
                    continue
                original_hit, promoted_hit, rejected_hit = validation_result
                if rejected_hit is not None:
                    rejected_hits.append(rejected_hit)
                    continue
                if promoted_hit.template_id != original_hit.template_id or promoted_hit.bbox != original_hit.bbox:
                    diagnostics["promoted_targeted_hits"] += 1
                validated_hits.append(promoted_hit)
    validated_candidates.extend(validated_hits)

    diagnostics["validated_template_hits"] = len(validated_candidates) - len(pdf_candidates)
    timings["validation_targeted"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    prefiltered_candidates = _prefilter_candidates(validated_candidates)
    diagnostics["prefilter_hits"] = len(prefiltered_candidates)
    timings["prefilter"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    pre_parent_candidates = _cluster_candidates(prefiltered_candidates, parent_ids_by_child)
    diagnostics["pre_parent_clusters"] = len(pre_parent_candidates)
    timings["pre_parent_clustering"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    def _search_parent_hit(hit: CandidateHit) -> tuple[CandidateHit, dict[str, int]]:
        local_stats: dict[str, int] = {}
        promoted_hit = _maybe_promote_switch_parent_search(
            hit,
            plan_image,
            templates,
            plan_masks_by_template,
            dilated_plan_masks_by_template,
            variants_lookup,
            socket_07_promotions,
            local_stats,
        )
        return promoted_hit, local_stats

    parent_search_candidates: list[CandidateHit] = []
    parent_search_workers = max(1, min(len(pre_parent_candidates), postprocess_workers))
    if pre_parent_candidates:
        with ThreadPoolExecutor(max_workers=parent_search_workers) as pool:
            for hit, (promoted_hit, local_stats) in zip(pre_parent_candidates, pool.map(_search_parent_hit, pre_parent_candidates)):
                diagnostics["parent_search_input_hits"] += local_stats.get("parent_search_input_hits", 0)
                diagnostics["parent_search_candidates"] += local_stats.get("parent_search_candidates", 0)
                if promoted_hit.template_id != hit.template_id or promoted_hit.bbox != hit.bbox:
                    diagnostics["promoted_parent_search_hits"] += 1
                parent_search_candidates.append(promoted_hit)
    timings["parent_search"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    final_hits = _cluster_candidates(parent_search_candidates, parent_ids_by_child)
    diagnostics["final_hits"] = len(final_hits)
    timings["clustering"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    debug_candidates = (
        _collect_debug_candidates(
            rejected_hits=rejected_hits,
            validated_candidates=prefiltered_candidates,
            final_hits=final_hits,
            templates=templates,
            plan_image=plan_image,
            exclude_rects=exclude_rects,
        )
        if debug_profile is not None
        else []
    )
    diagnostics["debug_candidates"] = len(debug_candidates)
    timings["debug_candidates"] = time.perf_counter() - phase_start

    timings_ms = {
        name: round(seconds * 1000.0, 3)
        for name, seconds in timings.items()
    }
    if debug_profile is not None:
        debug_profile.clear()
        debug_profile.update(
            {
                "timingsMs": timings_ms,
                "counters": {key: int(value) for key, value in diagnostics.items()},
                "threading": {
                    "scanWorkers": int(scan_workers),
                    "configuredScanWorkers": int(DETECTOR_SCAN_MAX_WORKERS),
                    "validationWorkers": int(validation_workers if raw_template_hits else 0),
                    "parentSearchWorkers": int(parent_search_workers if pre_parent_candidates else 0),
                    "configuredPostprocessWorkers": int(DETECTOR_POSTPROCESS_MAX_WORKERS),
                    "opencvThreads": int(OPENCV_NUM_THREADS),
                    "cpuCount": int(_safe_cpu_count()),
                },
                "searchRoi": {
                    "totalRois": int(diagnostics["search_rois"]),
                    "fullScanTemplates": int(diagnostics["full_scan_templates"]),
                    "roiAreaPixels": int(diagnostics["roi_area_pixels"]),
                    "foregroundPixels": int(diagnostics["roi_foreground_pixels"]),
                    "fullImageAreaPixels": int(plan_image.shape[0] * plan_image.shape[1]),
                },
                "debugCandidates": debug_candidates,
                "slowestPhase": max(timings_ms.items(), key=lambda item: item[1])[0]
                if timings_ms
                else None,
            }
        )

    print(
        "Detection diagnostics:"
        f" prepared_variants={diagnostics['prepared_variants']},"
        f" skipped_empty_color_masks={diagnostics['skipped_empty_color_masks']},"
        f" raw_peaks={diagnostics['raw_peaks']},"
        f" raw_after_prefilter={diagnostics['raw_prefilter_hits']}(-{diagnostics['raw_prefilter_removed']}),"
        f" validated_template_hits={diagnostics['validated_template_hits']},"
        f" promoted_targeted_hits={diagnostics['promoted_targeted_hits']},"
        f" parent_search_input_hits={diagnostics['parent_search_input_hits']},"
        f" parent_search_candidates={diagnostics['parent_search_candidates']},"
        f" promoted_parent_search_hits={diagnostics['promoted_parent_search_hits']},"
        f" pdf_text_hits={diagnostics['pdf_text_hits']},"
        f" after_prefilter={diagnostics['prefilter_hits']},"
        f" pre_parent_clusters={diagnostics['pre_parent_clusters']},"
        f" final_clusters={diagnostics['final_hits']},"
        f" rois={diagnostics['search_rois']} full_scan_templates={diagnostics['full_scan_templates']},"
        f" threads=scan:{scan_workers}/{DETECTOR_SCAN_MAX_WORKERS}|post:{postprocess_workers}/{DETECTOR_POSTPROCESS_MAX_WORKERS}|opencv:{OPENCV_NUM_THREADS},"
        f" timings_ms="
        f"pdf_text:{timings_ms['pdf_text']:.0f}|"
        f"prepare:{timings_ms['prepare']:.0f}|"
        f"scan:{timings_ms['scan']:.0f}|"
        f"raw_prefilter:{timings_ms['raw_prefilter']:.0f}|"
        f"validation_targeted:{timings_ms['validation_targeted']:.0f}|"
        f"prefilter:{timings_ms['prefilter']:.0f}|"
        f"pre_parent_clustering:{timings_ms['pre_parent_clustering']:.0f}|"
        f"parent_search:{timings_ms['parent_search']:.0f}|"
        f"clustering:{timings_ms['clustering']:.0f}"
    )

    per_template: dict[int, list[Detection]] = {}
    for hit in final_hits:
        x, y, w, h = [int(value) for value in hit.bbox]
        detection = Detection(
            symbol_name=templates[hit.template_id].name,
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=round(hit.match_score, 3),
            source=hit.source,
            rotation=hit.rotation,
            scale=hit.scale,
            mirrored=hit.mirrored,
            coverage=round(hit.coverage, 3),
            purity=round(hit.purity, 3),
            context_purity=round(hit.context_purity, 3),
            color_similarity=round(hit.color_similarity, 3),
            verification_score=round(hit.verification_score, 3),
            is_text_label=hit.is_text_label,
            content_score=round(hit.content_score, 3),
            content_bbox=(
                (
                    x + hit.content_bbox[0],
                    y + hit.content_bbox[1],
                    hit.content_bbox[2],
                    hit.content_bbox[3],
                )
                if hit.content_bbox is not None
                else None
            ),
            content_source=hit.source if hit.is_text_label else "",
        )
        per_template.setdefault(hit.template_id, []).append(detection)

    results: list[DetectionResult] = []
    for template_id, detections in per_template.items():
        detections.sort(key=lambda det: (det.verification_score, det.confidence), reverse=True)

        count = len(detections)
        if subtract_legend and legend_rect is None:
            count = max(0, count - 1)

        if count <= 0:
            continue

        results.append(
            DetectionResult(
                symbol_name=templates[template_id].name,
                count=count,
                color="#22c55e",
                detections=detections[:count] if subtract_legend and legend_rect is None else detections,
            )
        )

    results.sort(key=lambda result: result.symbol_name.lower())
    return results


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


if __name__ == "__main__":
    import sys

    plan_path = sys.argv[1] if len(sys.argv) > 1 else "wygenerowany_plan_300dpi.png"
    templates_dir = sys.argv[2] if len(sys.argv) > 2 else "templates"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "wynik.png"

    print(f"Loading plan: {plan_path}")
    plan = cv2.imread(plan_path)
    if plan is None:
        print(f"Error: cannot read {plan_path}")
        sys.exit(1)

    print(f"Loading templates from: {templates_dir}")
    templates = load_templates(templates_dir)
    print(f"Loaded {len(templates)} templates.\n")

    print(f"{'NAME':<45} | {'TYPE':<10} | {'COUNT':>5}")
    print("-" * 68)

    results = detect_symbols(plan, templates)

    total = 0
    for result in results:
        mode = "[PRECISE]" if any(word in result.symbol_name.lower() for word in PRECISE_KEYWORDS) else "[DILATE]"
        print(f"{result.symbol_name[:43]:<45} | {mode:<10} | {result.count:>5}")
        total += result.count

    print("-" * 68)
    print(f"{'TOTAL':<45} | {'':10} | {total:>5}")

    output_image = draw_results(plan, results)
    cv2.imwrite(output_path, output_image)
    print(f"\nSaved result: {output_path}")
