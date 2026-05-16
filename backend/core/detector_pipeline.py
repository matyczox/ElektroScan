"""
detector.py - CPU-friendly symbol detection for electrical plans.

This module now orchestrates the pipeline. The heavy helpers live in sibling
modules so detector behavior stays easier to audit and tune.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
import re

import cv2
import numpy as np

from core import detector_color as color_strategy
from core import detector_gray as gray_strategy
from core.detector_clustering import (
    _cluster_candidates,
    _dedupe_raw_template_hits_before_validation,
    _prefilter_candidates,
    _prefilter_raw_template_hits,
    _suppress_color_local_fragments,
)
from core.detector_config import (
    DEFAULT_PDF_DPI,
    DETECTOR_POSTPROCESS_MAX_WORKERS,
    DETECTOR_SCAN_MAX_WORKERS,
    GRAY_DARK_EVIDENCE_THRESHOLD,
    GRAY_DARK_INK_THRESHOLD,
    GRAY_DARK_ZONE_THRESHOLD,
    GRAY_SCALES,
    GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
    GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
    GRAY_TINY_FRAGMENT_MAX_DIMENSION,
    GRAY_TINY_FRAGMENT_MAX_SCALE,
    OPENCV_NUM_THREADS,
    SCALES,
    _safe_cpu_count,
)
from core.detector_context import (
    DetectionTemplateContext,
    bbox_iom,
    center_distance,
    center_inside,
    expanded_box,
    hue_close,
    token_family,
    trace_points_from_value,
    trace_values_from_value,
)
from core.detector_diagnostics import (
    build_candidate_stage_counts,
    build_hit_flow_profile,
    build_roi_strategy_profile,
)
from core.detector_masks import (
    _build_search_rois,
    _hsv_mask,
    _ink_mask,
)
from core.detector_models import (
    CandidateHit,
    Detection,
    DetectionResult,
    TemplateInfo,
)
from core.detector_pdf import (
    _collect_pdf_text_exclude_rects,
    _collect_pdf_text_hits,
    _estimate_legend_exclude_rect,
    _estimate_title_block_exclude_rects,
)
from core.detector_pdf_policy import (
    apply_color_pdf_text_class_hints,
    filter_pdf_text_fallbacks,
)
from core.detector_parent_search import search_parent_candidates
from core.detector_postprocess import dedupe_final_hits
from core.detector_scanning import scan_template_candidates
from core.detector_templates import (
    _build_socket_07_promotions,
    _prepare_variants,
    _template_numeric_prefix,
)
from core.detector_validation import validate_template_candidates


def _detect_symbols_pipeline(
    plan_image: np.ndarray,
    templates: list[TemplateInfo],
    subtract_legend: bool = True,
    exclude_rects: list[tuple[int, int, int, int]] | None = None,
    pdf_path: str | None = None,
    pdf_dpi: int = DEFAULT_PDF_DPI,
    hidden_layers: list[str] | None = None,
    debug_profile: dict | None = None,
    detector_profile: str = "color",
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> list[DetectionResult]:
    """
    Detect symbols on a rendered plan using template matching plus PDF-text fallback.
    """

    exclude_rects = list(exclude_rects or [])

    if not templates:
        return []

    timings: dict[str, float] = {}

    def _progress(stage: str, percent: float, detail: str = "") -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, max(0.0, min(100.0, float(percent))), detail)
        except Exception:
            pass

    detector_profile = detector_profile if detector_profile in {"color", "gray"} else "color"
    initial_debug_profile = dict(debug_profile or {})
    collect_performance_profile = bool(initial_debug_profile.get("performanceProfile"))
    ablation_value = str(
        initial_debug_profile.get("ablation") or os.getenv("ELEKTROSCAN_ABLATION", "")
    ).strip().lower()
    ablation_no_text_mirror = ablation_value in {
        "no-text-mirror",
        "no_text_mirror",
        "notextmirror",
    } or os.getenv("ELEKTROSCAN_ABLATION_NO_TEXT_MIRROR", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    gray_text_mirror_override = os.getenv("ELEKTROSCAN_GRAY_TEXT_MIRROR", "").strip().lower()
    gray_force_text_mirror = ablation_value in {
        "text-mirror",
        "text_mirror",
        "with-text-mirror",
        "with_text_mirror",
    } or gray_text_mirror_override in {"1", "true", "yes", "on"}
    disable_text_mirror = ablation_no_text_mirror or (
        detector_profile == "gray" and not gray_force_text_mirror
    )

    trace_input = initial_debug_profile.get("candidateTrace") or initial_debug_profile.get("trace") or {}
    if not isinstance(trace_input, dict):
        trace_input = {}

    trace_symbols = set(trace_values_from_value(trace_input.get("symbols") or trace_input.get("symbol")))
    trace_symbols.update(trace_values_from_value(os.getenv("ELEKTROSCAN_TRACE_SYMBOLS")))
    trace_points = trace_points_from_value(trace_input.get("points") or trace_input.get("point"))
    trace_points.extend(trace_points_from_value(os.getenv("ELEKTROSCAN_TRACE_POINTS")))
    trace_radius = float(trace_input.get("radius") or os.getenv("ELEKTROSCAN_TRACE_RADIUS", 80))
    trace_max_items = int(trace_input.get("maxItems") or os.getenv("ELEKTROSCAN_TRACE_MAX_ITEMS", 40))
    candidate_trace_enabled = bool(trace_symbols or trace_points)
    candidate_trace: dict[str, dict] = {}

    template_context = DetectionTemplateContext(templates)
    _template_tokens = template_context.template_tokens
    _template_primary_token = template_context.template_primary_token
    _token_family = token_family
    _template_token_family = template_context.template_token_family
    _template_name = template_context.template_name
    _is_visual_pdf_text_blocked = template_context.is_visual_pdf_text_blocked
    _l_label_group = template_context.l_label_group
    _is_magenta_family_template = template_context.is_magenta_family_template
    _magenta_template_code = template_context.magenta_template_code
    _is_tb11_wave_template = template_context.is_tb11_wave_template
    _center_distance = center_distance
    _center_inside = center_inside
    _bbox_iom = bbox_iom
    _hue_close = hue_close
    _expanded_box = expanded_box

    def _apply_color_label_disambiguation(
        accepted_hits: list[CandidateHit],
        rejected_hits: list[CandidateHit],
        rejection_reason_by_hit_id: dict[int, str],
        pdf_hits: list[CandidateHit],
    ) -> tuple[int, list[CandidateHit]]:
        """Use exact color labels as local family evidence, never as standalone detections."""

        if detector_profile != "color" or not pdf_hits:
            return 0, []
        # Exact PDF text is evidence for review/debug in the color path. It must not
        # relabel visual pictograms because plan-side labels are often circuit IDs,
        # not symbol classes.
        return 0, []

        allowed_reasons = {
            "low_match_strict",
            "content_score",
            "color_text_fragment",
            "color_text_geometry",
            "color_text_weak_match",
            "noisy_partial",
        }
        visual_pool = [
            hit
            for hit in accepted_hits
            if hit.source != "pdf_text" and hit.dominant_hsv is not None
        ]
        visual_pool.extend(
            hit
            for hit in rejected_hits
            if hit.source != "pdf_text"
            and hit.dominant_hsv is not None
            and rejection_reason_by_hit_id.get(id(hit)) in allowed_reasons
        )

        if not visual_pool:
            return 0, []

        changed = 0
        rescued: list[CandidateHit] = []
        seen_rescues: set[tuple[int, int, int, int, int]] = set()
        accepted_ids = {id(hit) for hit in accepted_hits}

        def candidate_allowed_for_label(candidate: CandidateHit, target_id: int, family: str) -> bool:
            if not _hue_close(candidate.dominant_hsv, templates[target_id].dominant_hsv):
                return False
            candidate_family = _template_token_family(candidate.template_id)
            if family == "L":
                target_group = _l_label_group(target_id)
                return bool(target_group) and _l_label_group(candidate.template_id) == target_group
            if family in {"AW", "EW"}:
                return candidate_family in {"AW", "EW"}
            return False

        def local_to_label(candidate: CandidateHit, pdf_hit: CandidateHit, target_id: int) -> bool:
            target_template = templates[target_id]
            th, tw = target_template.mask.shape[:2]
            search_box = _expanded_box(
                pdf_hit.bbox,
                pad_x=max(45.0, tw * 0.85),
                pad_y=max(45.0, th * 0.85),
            )
            if _center_inside(candidate.bbox, search_box):
                return True
            return _center_distance(candidate.bbox, pdf_hit.bbox) <= max(80.0, float(max(tw, th)))

        def color_evidence_for_label(
            template_id: int,
            bbox: tuple[int, int, int, int],
            *,
            pad: int,
        ) -> tuple[int, float]:
            plan_mask = plan_masks_by_template.get(template_id)
            if plan_mask is None:
                return 0, 0.0
            x, y, w, h = _expanded_box(bbox, pad_x=pad, pad_y=pad)
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(plan_mask.shape[1], x + w)
            y1 = min(plan_mask.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                return 0, 0.0
            roi = plan_mask[y0:y1, x0:x1]
            pixels = int(cv2.countNonZero(roi))
            return pixels, pixels / max(1, int(roi.size))

        def target_shape_confirms(template_id: int, candidate: CandidateHit) -> bool:
            if candidate.template_id == template_id:
                return True
            plan_mask = plan_masks_by_template.get(template_id)
            if plan_mask is None:
                return False
            x, y, w, h = candidate.bbox
            if w <= 0 or h <= 0:
                return False
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(plan_mask.shape[1], x + w)
            y1 = min(plan_mask.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                return False
            roi = plan_mask[y0:y1, x0:x1]
            if int(cv2.countNonZero(roi)) < 20:
                return False

            target_mask = templates[template_id].mask
            if candidate.scale and abs(float(candidate.scale) - 1.0) >= 0.001:
                target_mask = cv2.resize(
                    target_mask,
                    (
                        max(1, int(round(target_mask.shape[1] * float(candidate.scale)))),
                        max(1, int(round(target_mask.shape[0] * float(candidate.scale)))),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            if candidate.rotation % 360 == 90:
                target_mask = cv2.rotate(target_mask, cv2.ROTATE_90_CLOCKWISE)
            elif candidate.rotation % 360 == 180:
                target_mask = cv2.rotate(target_mask, cv2.ROTATE_180)
            elif candidate.rotation % 360 == 270:
                target_mask = cv2.rotate(target_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if candidate.mirrored:
                target_mask = cv2.flip(target_mask, 1)
            if target_mask.shape[:2] != roi.shape[:2]:
                target_mask = cv2.resize(
                    target_mask,
                    (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            target_pixels = max(1, int(cv2.countNonZero(target_mask)))
            roi_pixels = max(1, int(cv2.countNonZero(roi)))
            overlap = int(cv2.countNonZero(cv2.bitwise_and(roi, target_mask)))
            coverage = overlap / target_pixels
            purity = overlap / roi_pixels
            return coverage >= 0.46 and purity >= 0.48

        for pdf_hit in pdf_hits:
            target_id = pdf_hit.template_id
            token = _template_primary_token(target_id)
            family = _token_family(token)
            if family not in {"L", "EW"}:
                continue
            if family == "L" and not _l_label_group(target_id):
                continue

            local_candidates = [
                hit
                for hit in visual_pool
                if candidate_allowed_for_label(hit, target_id, family)
                and local_to_label(hit, pdf_hit, target_id)
                and target_shape_confirms(target_id, hit)
            ]
            if not local_candidates:
                continue

            def score(hit: CandidateHit) -> tuple[float, ...]:
                accepted = 1.0 if id(hit) in accepted_ids else 0.0
                return (
                    accepted,
                    float(hit.verification_score),
                    float(hit.match_score),
                    float(hit.coverage),
                    float(hit.purity),
                    -_center_distance(hit.bbox, pdf_hit.bbox),
                )

            winner = max(local_candidates, key=score)
            winner_was_accepted = id(winner) in accepted_ids
            if winner.template_id == target_id and winner.source != "pdf_text" and winner_was_accepted:
                continue

            boosted = replace(
                winner,
                template_id=target_id,
                source="template_label_disambiguation",
                promoted_from_template_id=winner.template_id,
                match_score=round(max(float(winner.match_score), 0.62), 4),
                verification_score=round(max(float(winner.verification_score), 0.72), 4),
                coverage=round(max(float(winner.coverage), 0.62), 4),
                purity=round(max(float(winner.purity), 0.62), 4),
                context_purity=round(max(float(winner.context_purity), 0.55), 4),
                color_similarity=round(max(float(winner.color_similarity), 0.95), 4),
                dominant_hsv=templates[target_id].dominant_hsv,
                is_text_label=templates[target_id].is_text_label,
                content_score=round(max(float(winner.content_score), 0.75), 4),
            )
            key = (boosted.template_id, *boosted.bbox)
            if key in seen_rescues:
                continue
            seen_rescues.add(key)

            if id(winner) in accepted_ids:
                accepted_hits.append(boosted)
                accepted_ids.add(id(boosted))
            else:
                rescued.append(boosted)
            changed += 1

        return changed, rescued

    def _reconcile_magenta_family_hits(
        final_hits: list[CandidateHit],
        candidates: list[CandidateHit],
    ) -> tuple[list[CandidateHit], int]:
        """Replace local magenta family losers with a stronger same-place sibling."""

        if detector_profile != "color":
            return final_hits, 0

        output = list(final_hits)
        replacements = 0
        magenta_candidates = [
            hit
            for hit in candidates
            if hit.source != "pdf_text"
            and _is_magenta_family_template(hit.template_id)
            and hit.coverage >= 0.50
            and hit.purity >= 0.55
            and hit.context_purity >= 0.20
        ]
        if not magenta_candidates:
            return final_hits, 0

        def same_local_symbol(left: CandidateHit, right: CandidateHit) -> bool:
            iom = _bbox_iom(left.bbox, right.bbox)
            center_distance = _center_distance(left.bbox, right.bbox)
            centers_nested = _center_inside(left.bbox, right.bbox) or _center_inside(
                right.bbox,
                left.bbox,
            )
            return (
                iom >= 0.55
                or (
                    center_distance
                    <= max(
                        24.0,
                        min(left.bbox[2], left.bbox[3], right.bbox[2], right.bbox[3]) * 0.48,
                    )
                    and iom >= 0.22
                )
                or (centers_nested and iom >= 0.35)
            )

        def score(hit: CandidateHit) -> float:
            return (
                float(hit.verification_score) * 3.0
                + float(hit.match_score) * 1.4
                + float(hit.coverage) * 0.9
                + float(hit.purity) * 0.5
                + float(hit.context_purity) * 0.5
            )

        for idx, final_hit in enumerate(list(output)):
            if not _is_magenta_family_template(final_hit.template_id):
                continue
            local = [hit for hit in magenta_candidates if same_local_symbol(hit, final_hit)]
            if not local:
                continue
            best = max(local, key=score)
            if best.template_id == final_hit.template_id and best.bbox == final_hit.bbox:
                continue
            if score(best) <= score(final_hit) + 0.12:
                continue
            output[idx] = replace(
                best,
                source="template_family_reclass",
                verification_score=round(max(float(best.verification_score), 0.58), 4),
                context_purity=round(max(float(best.context_purity), 0.24), 4),
            )
            replacements += 1

        return output, replacements

    def _rescue_magenta_family_hits(
        final_hits: list[CandidateHit],
        candidates: list[CandidateHit],
    ) -> tuple[list[CandidateHit], int]:
        """Use a local color-mask scan to restore weak magenta-family symbols."""

        if detector_profile != "color":
            return final_hits, 0

        magenta_ids = [
            template_id
            for template_id in range(len(templates))
            if _is_magenta_family_template(template_id)
        ]
        if not magenta_ids:
            return final_hits, 0

        seed_hits = [
            hit
            for hit in candidates
            if hit.source != "pdf_text"
            and _is_magenta_family_template(hit.template_id)
            and hit.match_score >= 0.24
            and hit.coverage >= 0.36
            and hit.purity >= 0.34
        ]
        if not seed_hits:
            return final_hits, 0

        def close_enough(left: CandidateHit, right: CandidateHit) -> bool:
            return _bbox_iom(left.bbox, right.bbox) >= 0.08 or _center_distance(left.bbox, right.bbox) <= 42.0

        groups: list[list[CandidateHit]] = []
        for hit in sorted(seed_hits, key=lambda item: (item.bbox[1], item.bbox[0])):
            for group in groups:
                if any(close_enough(hit, other) for other in group):
                    group.append(hit)
                    break
            else:
                groups.append([hit])

        def split_vertical_stacks(group: list[CandidateHit]) -> list[list[CandidateHit]]:
            if len(group) <= 1:
                return [group]
            heights = sorted(max(1, hit.bbox[3]) for hit in group)
            median_height = float(heights[len(heights) // 2])
            split_gap = max(30.0, median_height * 0.62)
            ordered = sorted(group, key=lambda hit: hit.bbox[1] + hit.bbox[3] / 2.0)
            split_groups: list[list[CandidateHit]] = [[ordered[0]]]
            previous_center = ordered[0].bbox[1] + ordered[0].bbox[3] / 2.0
            for hit in ordered[1:]:
                center_y = hit.bbox[1] + hit.bbox[3] / 2.0
                if center_y - previous_center > split_gap:
                    split_groups.append([hit])
                else:
                    split_groups[-1].append(hit)
                previous_center = center_y
            return split_groups

        groups = [
            split_group
            for group in groups
            for split_group in split_vertical_stacks(group)
        ]

        def rotated_mask(mask: np.ndarray, rotation: int, mirrored: bool) -> np.ndarray:
            out = mask
            if rotation % 360 == 90:
                out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
            elif rotation % 360 == 180:
                out = cv2.rotate(out, cv2.ROTATE_180)
            elif rotation % 360 == 270:
                out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if mirrored:
                out = cv2.flip(out, 1)
            return out

        def scaled_mask(mask: np.ndarray, scale: float) -> np.ndarray:
            if abs(scale - 1.0) < 0.001:
                return mask
            width = max(1, int(round(mask.shape[1] * scale)))
            height = max(1, int(round(mask.shape[0] * scale)))
            return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        def group_bbox(group: list[CandidateHit]) -> tuple[int, int, int, int]:
            x0 = min(hit.bbox[0] for hit in group)
            y0 = min(hit.bbox[1] for hit in group)
            x1 = max(hit.bbox[0] + hit.bbox[2] for hit in group)
            y1 = max(hit.bbox[1] + hit.bbox[3] for hit in group)
            return (x0, y0, x1 - x0, y1 - y0)

        def magenta_union_roi(
            bbox: tuple[int, int, int, int],
            *,
            pad: int = 8,
        ) -> np.ndarray | None:
            x, y, w, h = _expanded_box(bbox, pad_x=pad, pad_y=pad)
            first_mask = next(
                (plan_masks_by_template.get(template_id) for template_id in magenta_ids),
                None,
            )
            if first_mask is None:
                return None
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(first_mask.shape[1], x + w)
            y1 = min(first_mask.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                return None
            combined = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            for template_id in magenta_ids:
                plan_mask = plan_masks_by_template.get(template_id)
                if plan_mask is None:
                    continue
                combined = cv2.bitwise_or(combined, plan_mask[y0:y1, x0:x1])
            return combined

        def side_arrow_score(bbox: tuple[int, int, int, int]) -> float:
            roi = magenta_union_roi(bbox)
            if roi is None or int(cv2.countNonZero(roi)) < 35:
                return 0.0
            coords = cv2.findNonZero(roi)
            if coords is None:
                return 0.0
            x, y, w, h = cv2.boundingRect(coords)
            if w < 10 or h < 12:
                return 0.0
            trimmed = roi[y : y + h, x : x + w]
            right_band = trimmed[:, int(round(w * 0.62)) :]
            right_pixels = int(cv2.countNonZero(right_band))
            total_pixels = max(1, int(cv2.countNonZero(trimmed)))
            row_span = float(np.count_nonzero(np.any(right_band > 0, axis=1))) / max(1, h)
            column_weight = right_pixels / total_pixels
            return min(1.0, row_span * 1.25 + column_weight * 1.6)

        def local_best_for_template(
            template_id: int,
            search_bbox: tuple[int, int, int, int],
        ) -> tuple[float, tuple[int, int, int, int], int, float, float, float] | None:
            plan_mask = plan_masks_by_template.get(template_id)
            if plan_mask is None:
                return None
            pad_y = max(10.0, min(22.0, float(search_bbox[3]) * 0.22))
            sx, sy, sw, sh = _expanded_box(search_bbox, pad_x=28, pad_y=pad_y)
            x0 = max(0, sx)
            y0 = max(0, sy)
            x1 = min(plan_mask.shape[1], sx + sw)
            y1 = min(plan_mask.shape[0], sy + sh)
            if x1 <= x0 or y1 <= y0:
                return None
            search = plan_mask[y0:y1, x0:x1]
            if int(cv2.countNonZero(search)) < 90:
                return None

            base_mask = templates[template_id].mask
            best: tuple[float, tuple[int, int, int, int], int, float, float, float] | None = None
            for scale in (0.90, 1.00, 1.10):
                scaled = scaled_mask(base_mask, scale)
                for rotation in (0, 90, 180, 270):
                    for mirrored in (False,):
                        variant = rotated_mask(scaled, rotation, mirrored)
                        vh, vw = variant.shape[:2]
                        if vh < 8 or vw < 8 or vh > search.shape[0] or vw > search.shape[1]:
                            continue
                        response = cv2.matchTemplate(search, variant, cv2.TM_CCORR_NORMED)
                        _, max_value, _, max_loc = cv2.minMaxLoc(response)
                        bx = x0 + int(max_loc[0])
                        by = y0 + int(max_loc[1])
                        roi = plan_mask[by : by + vh, bx : bx + vw]
                        template_pixels = max(1, int(cv2.countNonZero(variant)))
                        roi_pixels = max(1, int(cv2.countNonZero(roi)))
                        overlap = int(cv2.countNonZero(cv2.bitwise_and(roi, variant)))
                        coverage = overlap / template_pixels
                        purity = overlap / roi_pixels
                        center_penalty = min(
                            0.45,
                            _center_distance((bx, by, vw, vh), search_bbox)
                            / max(1.0, float(max(search_bbox[2], search_bbox[3], vw, vh)))
                            * 0.55,
                        )
                        score = float(max_value) * 1.4 + coverage * 0.9 + purity * 0.6 - center_penalty
                        result = (score, (bx, by, vw, vh), rotation, float(max_value), coverage, purity)
                        if best is None or result[0] > best[0]:
                            best = result
            return best

        def local_final_near(bbox: tuple[int, int, int, int]) -> list[CandidateHit]:
            return [
                hit
                for hit in final_hits
                if _is_magenta_family_template(hit.template_id)
                and (
                    _bbox_iom(hit.bbox, bbox) >= 0.08
                    or _center_distance(hit.bbox, bbox) <= 42.0
                )
            ]

        output = list(final_hits)
        existing = {(hit.template_id, hit.bbox) for hit in output}
        rescued = 0

        for group in groups:
            bbox = group_bbox(group)
            existing_local = local_final_near(bbox)
            if any(
                hit.verification_score >= 0.72
                and hit.match_score >= 0.58
                and hit.context_purity >= 0.45
                and (
                    _bbox_iom(hit.bbox, bbox) >= 0.35
                    or _center_distance(hit.bbox, bbox) <= 24.0
                )
                for hit in existing_local
            ):
                continue

            best_id: int | None = None
            best_result: tuple[float, tuple[int, int, int, int], int, float, float, float] | None = None
            arrow_score = side_arrow_score(bbox)
            side_arrow_id: int | None = None
            side_arrow_result: tuple[float, tuple[int, int, int, int], int, float, float, float] | None = None
            for template_id in magenta_ids:
                result = local_best_for_template(template_id, bbox)
                if result is None:
                    continue
                code = _magenta_template_code(template_id)
                score, result_bbox, rotation, max_value, coverage, purity = result
                if code == 2:
                    score += 0.30 * arrow_score
                elif arrow_score >= 0.45:
                    score -= 0.12 * arrow_score
                if code in {6, 7, 8} and max(result_bbox[2], result_bbox[3]) < 48:
                    score -= 0.12
                adjusted = (score, result_bbox, rotation, max_value, coverage, purity)
                if code == 2:
                    side_arrow_id = template_id
                    side_arrow_result = adjusted
                if best_result is None or adjusted[0] > best_result[0]:
                    best_id = template_id
                    best_result = adjusted

            if (
                arrow_score >= 0.70
                and side_arrow_id is not None
                and side_arrow_result is not None
                and (
                    best_result is None
                    or side_arrow_result[0] >= best_result[0] - 0.18
                )
            ):
                best_id = side_arrow_id
                best_result = side_arrow_result

            if best_id is None or best_result is None:
                continue
            score, result_bbox, rotation, max_value, coverage, purity = best_result
            best_code = _magenta_template_code(best_id)
            min_score = 1.34 if best_code == 2 and arrow_score >= 0.70 else 1.42
            min_match = 0.24
            min_coverage = 0.38 if best_code == 2 and arrow_score >= 0.70 else 0.34
            min_purity = 0.34
            if score < min_score or max_value < min_match or coverage < min_coverage or purity < min_purity:
                continue

            target = templates[best_id]
            strongest_seed = max(
                group,
                key=lambda hit: (
                    float(hit.verification_score),
                    float(hit.match_score),
                    float(hit.coverage),
                    float(hit.purity),
                ),
            )
            rescued_hit = replace(
                strongest_seed,
                template_id=best_id,
                bbox=result_bbox,
                rotation=rotation,
                source="template_family_reclass",
                transformed_mask=target.mask,
                content_mask=target.content_mask,
                pixel_count=max(1, int(target.pixel_count)),
                content_pixel_count=max(0, int(target.content_pixel_count)),
                content_bbox=target.content_bbox,
                dominant_hsv=target.dominant_hsv,
                is_text_label=target.is_text_label,
                match_score=round(max(float(strongest_seed.match_score), float(max_value)), 4),
                verification_score=round(max(float(strongest_seed.verification_score), min(0.72, score / 2.4)), 4),
                coverage=round(max(float(strongest_seed.coverage), coverage), 4),
                purity=round(max(float(strongest_seed.purity), purity), 4),
                context_purity=round(max(float(strongest_seed.context_purity), 0.30), 4),
                color_similarity=round(max(float(strongest_seed.color_similarity), 0.95), 4),
            )

            replaced = False
            for idx, existing_hit in enumerate(output):
                if (
                    _is_magenta_family_template(existing_hit.template_id)
                    and (
                        _bbox_iom(existing_hit.bbox, rescued_hit.bbox) >= 0.25
                        or _center_distance(existing_hit.bbox, rescued_hit.bbox) <= 24.0
                    )
                ):
                    output[idx] = rescued_hit
                    replaced = True
                    rescued += 1
                    existing.add((rescued_hit.template_id, rescued_hit.bbox))
                    break
            if not replaced and (rescued_hit.template_id, rescued_hit.bbox) not in existing:
                output.append(rescued_hit)
                existing.add((rescued_hit.template_id, rescued_hit.bbox))
                rescued += 1

        return output, rescued

    def _rescue_color_family_hits(
        final_hits: list[CandidateHit],
        candidates: list[CandidateHit],
        rejection_reason_by_hit_id: dict[int, str] | None = None,
    ) -> tuple[list[CandidateHit], int]:
        """Restore strong validated color-family hits lost only by local arbitration."""

        if detector_profile != "color":
            return final_hits, 0

        rescued = 0
        output = list(final_hits)
        existing_keys = {(hit.template_id, hit.bbox) for hit in output}

        def near_same_template(candidate: CandidateHit) -> bool:
            for hit in output:
                if hit.template_id != candidate.template_id:
                    continue
                distance = _center_distance(hit.bbox, candidate.bbox)
                if is_tb11_wave(candidate):
                    if _bbox_iom(hit.bbox, candidate.bbox) >= 0.42 or distance <= 18.0:
                        return True
                    continue
                if distance <= max(24.0, min(candidate.bbox[2], candidate.bbox[3]) * 0.75):
                    return True
            return False

        def is_tb11_wave(candidate: CandidateHit) -> bool:
            return _is_tb11_wave_template(candidate.template_id)

        def is_long_l(candidate: CandidateHit) -> bool:
            return _l_label_group(candidate.template_id) == "long"

        def has_better_long_l_competitor(candidate: CandidateHit) -> bool:
            if not is_tb11_wave(candidate):
                return False
            return any(
                is_long_l(other)
                and other.source != "pdf_text"
                and (
                    _bbox_iom(candidate.bbox, other.bbox) >= 0.08
                    or _center_distance(candidate.bbox, other.bbox) <= 86.0
                )
                and other.match_score >= max(0.56, candidate.match_score + 0.04)
                and other.coverage >= 0.55
                and other.purity >= 0.62
                for other in candidates
            )

        def eligible(candidate: CandidateHit) -> bool:
            if candidate.source == "pdf_text" or candidate.dominant_hsv is None:
                return False
            if is_tb11_wave(candidate):
                return (
                    (
                        candidate.verification_score >= 0.55
                        or (
                            candidate.match_score >= 0.56
                            and rejection_reason_by_hit_id is not None
                            and rejection_reason_by_hit_id.get(id(candidate))
                            == "color_elongated_stroke_fragment"
                        )
                    )
                    and candidate.coverage >= 0.80
                    and candidate.purity >= 0.55
                    and candidate.context_purity >= 0.14
                    and not has_better_long_l_competitor(candidate)
                )
            return False

        for candidate in sorted(
            (hit for hit in candidates if eligible(hit)),
            key=lambda hit: (
                float(hit.verification_score),
                float(hit.match_score),
                float(hit.coverage),
                float(hit.purity),
            ),
            reverse=True,
        ):
            key = (candidate.template_id, candidate.bbox)
            if key in existing_keys or near_same_template(candidate):
                continue
            restored = replace(
                candidate,
                source=(
                    "template_long_l_rescue"
                    if is_long_l(candidate)
                    else "template_wave_rescue"
                ),
                verification_score=round(max(float(candidate.verification_score), 0.58), 4),
                context_purity=round(max(float(candidate.context_purity), 0.24), 4),
            )
            output.append(restored)
            existing_keys.add(key)
            rescued += 1

        return output, rescued

    def _promote_long_l_over_tb11_conflicts(
        final_hits: list[CandidateHit],
        candidates: list[CandidateHit],
    ) -> tuple[list[CandidateHit], int]:
        if detector_profile != "color":
            return final_hits, 0
        promoted = 0
        output: list[CandidateHit] = []
        long_l_pool = [
            hit
            for hit in candidates
            if _l_label_group(hit.template_id) == "long"
            and hit.source != "pdf_text"
            and hit.match_score >= 0.60
            and hit.coverage >= 0.68
            and hit.purity >= 0.70
            and hit.context_purity >= 0.35
            and max(hit.bbox[2], hit.bbox[3]) / max(1, min(hit.bbox[2], hit.bbox[3])) >= 2.45
            and min(hit.bbox[2], hit.bbox[3]) >= 32
        ]

        def tb11_conflict_candidate(hit: CandidateHit) -> bool:
            return (
                _is_tb11_wave_template(hit.template_id)
                and hit.source != "pdf_text"
                and hit.match_score >= 0.50
                and hit.coverage >= 0.70
                and hit.purity >= 0.50
                and hit.context_purity <= 0.35
            )

        def local_long_l(hit: CandidateHit) -> list[CandidateHit]:
            return [
                other
                for other in long_l_pool
                if (
                    _bbox_iom(hit.bbox, other.bbox) >= 0.40
                    or (
                        _center_distance(hit.bbox, other.bbox) <= 42.0
                        and _bbox_iom(hit.bbox, other.bbox) >= 0.20
                    )
                )
                and (
                    other.match_score >= hit.match_score + 0.06
                    or (
                        other.match_score >= 0.62
                        and other.coverage >= 0.73
                        and other.purity >= 0.72
                        and hit.context_purity <= 0.32
                    )
                )
                and max(other.bbox[2], other.bbox[3]) / max(1, min(other.bbox[2], other.bbox[3])) >= 2.45
            ]

        def score(candidate: CandidateHit) -> tuple[float, ...]:
            return (
                float(candidate.match_score),
                float(candidate.coverage),
                float(candidate.purity),
                float(candidate.context_purity),
                float(candidate.verification_score),
            )

        for hit in final_hits:
            if not _is_tb11_wave_template(hit.template_id):
                output.append(hit)
                continue
            local = local_long_l(hit)
            if not local:
                output.append(hit)
                continue
            best = max(local, key=score)
            output.append(
                replace(
                    best,
                    source="template_long_l_rescue",
                    verification_score=round(max(float(best.verification_score), 0.58), 4),
                    context_purity=round(max(float(best.context_purity), 0.24), 4),
                )
            )
            promoted += 1
        seen_keys = {(hit.template_id, hit.bbox) for hit in output}
        added_from_candidate_conflicts = 0
        for tb_candidate in sorted(
            (hit for hit in candidates if tb11_conflict_candidate(hit)),
            key=lambda hit: (float(hit.match_score), float(hit.coverage), float(hit.purity)),
            reverse=True,
        ):
            if added_from_candidate_conflicts >= 4:
                break
            if any(
                _l_label_group(hit.template_id) == "long"
                and (
                    _bbox_iom(hit.bbox, tb_candidate.bbox) >= 0.25
                    or _center_distance(hit.bbox, tb_candidate.bbox) <= 46.0
                )
                for hit in output
            ):
                continue
            local = local_long_l(tb_candidate)
            if not local:
                continue
            best = max(local, key=score)
            if best.match_score < 0.62 or best.coverage < 0.68 or best.purity < 0.72:
                continue
            key = (best.template_id, best.bbox)
            if key in seen_keys:
                continue
            output.append(
                replace(
                    best,
                    source="template_long_l_rescue",
                    verification_score=round(max(float(best.verification_score), 0.58), 4),
                    context_purity=round(max(float(best.context_purity), 0.24), 4),
                )
            )
            seen_keys.add(key)
            added_from_candidate_conflicts += 1
            promoted += 1
        return output, promoted

    def _suppress_tb11_long_l_conflicts(
        final_hits: list[CandidateHit],
        candidates: list[CandidateHit],
    ) -> tuple[list[CandidateHit], int]:
        if detector_profile != "color":
            return final_hits, 0
        suppressed = 0
        output: list[CandidateHit] = []
        long_l_pool = [
            hit
            for hit in (final_hits + candidates)
            if _l_label_group(hit.template_id) == "long"
            and hit.source != "pdf_text"
            and hit.match_score >= 0.56
            and hit.coverage >= 0.55
            and hit.purity >= 0.62
        ]
        for hit in final_hits:
            if _is_tb11_wave_template(hit.template_id):
                has_long_l = any(
                    (
                        _bbox_iom(hit.bbox, other.bbox) >= 0.08
                        or _center_distance(hit.bbox, other.bbox) <= 86.0
                    )
                    and other.match_score >= max(0.56, hit.match_score + 0.04)
                    for other in long_l_pool
                )
                if has_long_l:
                    suppressed += 1
                    continue
            output.append(hit)
        return output, suppressed

    def _suppress_long_l_rescue_over_tb11_waves(
        final_hits: list[CandidateHit],
        pdf_label_hits: list[CandidateHit] | None = None,
    ) -> tuple[list[CandidateHit], int]:
        if detector_profile != "color":
            return final_hits, 0

        pdf_label_hits = pdf_label_hits or []
        tb11_waves = [
            hit
            for hit in final_hits
            if _is_tb11_wave_template(hit.template_id)
            and hit.source != "pdf_text"
            and hit.match_score >= 0.50
            and hit.verification_score >= 0.50
            and hit.coverage >= 0.75
            and hit.purity >= 0.52
        ]
        if not tb11_waves:
            return final_hits, 0

        output: list[CandidateHit] = []
        suppressed = 0

        def color_visual_bbox(hit: CandidateHit) -> tuple[int, int, int, int] | None:
            if hit.source == "pdf_text" or hit.dominant_hsv is None:
                return None
            plan_mask = plan_masks_by_template.get(hit.template_id)
            if plan_mask is None:
                return None
            x, y, w, h = [int(value) for value in hit.bbox]
            image_h, image_w = plan_mask.shape[:2]
            x0 = max(0, min(image_w, x))
            y0 = max(0, min(image_h, y))
            x1 = max(0, min(image_w, x + w))
            y1 = max(0, min(image_h, y + h))
            if x1 <= x0 or y1 <= y0:
                return None
            roi = plan_mask[y0:y1, x0:x1]
            ys, xs = np.where(roi > 0)
            if len(xs) < 4:
                return None
            pad = 3
            vx0 = max(x0, x0 + int(xs.min()) - pad)
            vy0 = max(y0, y0 + int(ys.min()) - pad)
            vx1 = min(x1, x0 + int(xs.max()) + 1 + pad)
            vy1 = min(y1, y0 + int(ys.max()) + 1 + pad)
            visual_w = vx1 - vx0
            visual_h = vy1 - vy0
            if visual_w <= 0 or visual_h <= 0:
                return None
            if visual_w * visual_h < max(16, int(0.03 * w * h)):
                return None
            return (vx0, vy0, visual_w, visual_h)

        def has_local_l_label_evidence(hit: CandidateHit) -> bool:
            if hit.content_score > 0.05:
                return True
            if hit.dominant_hsv is None:
                return False
            search_box = _expanded_box(
                hit.bbox,
                pad_x=max(72.0, hit.bbox[2] * 0.45),
                pad_y=max(64.0, hit.bbox[3] * 1.60),
            )
            for label in pdf_label_hits:
                if label.source != "pdf_text" or _template_token_family(label.template_id) != "L":
                    continue
                if label.dominant_hsv is not None and not _hue_close(hit.dominant_hsv, label.dominant_hsv):
                    continue
                if (
                    _center_inside(label.bbox, search_box)
                    or _bbox_iom(hit.bbox, label.bbox) >= 0.04
                    or _center_distance(hit.bbox, label.bbox) <= max(96.0, hit.bbox[2] * 0.85)
                ):
                    return True
            return False

        def wave_relation(hit: CandidateHit, wave: CandidateHit) -> bool:
            hx, hy, hw, hh = hit.bbox
            wx, wy, ww, wh = wave.bbox
            hcx = hx + hw / 2.0
            hcy = hy + hh / 2.0
            wcx = wx + ww / 2.0
            wcy = wy + wh / 2.0
            x_overlap = max(0, min(hx + hw, wx + ww) - max(hx, wx))
            x_gap = max(0, max(hx, wx) - min(hx + hw, wx + ww))
            horizontally_related = (
                x_overlap >= min(max(8.0, ww * 0.25), max(8.0, hw * 0.18))
                or x_gap <= max(12.0, min(hw, ww) * 0.22)
                or abs(hcx - wcx) <= max(48.0, hw * 0.55)
            )
            vertically_related = abs(hcy - wcy) <= max(34.0, (hh + wh) * 0.75)
            return (
                _bbox_iom(hit.bbox, wave.bbox) >= 0.16
                or (
                    _center_distance(hit.bbox, wave.bbox) <= 92.0
                    and _bbox_iom(hit.bbox, wave.bbox) >= 0.06
                )
                or (horizontally_related and vertically_related)
            )

        for hit in final_hits:
            if hit.source == "template_long_l_rescue" and _l_label_group(hit.template_id) == "long":
                related_waves = [wave for wave in tb11_waves if wave_relation(hit, wave)]
                visual_bbox = color_visual_bbox(hit)
                visual_h = visual_bbox[3] if visual_bbox is not None else hit.bbox[3]
                visually_thin_fragment = (
                    hit.content_score <= 0.05
                    and hit.bbox[2] >= hit.bbox[3] * 2.45
                    and visual_h <= max(22, int(round(hit.bbox[3] * 0.55)))
                )
                if visually_thin_fragment and not has_local_l_label_evidence(hit):
                    suppressed += 1
                    continue
                thin_rescue = hit.bbox[3] <= max(24, int(round(hit.bbox[2] * 0.22)))
                weak_context = hit.context_purity <= 0.34 or hit.verification_score <= 0.62
                if (
                    len(related_waves) >= 2
                    or (
                        related_waves
                        and weak_context
                        and (
                            thin_rescue
                            or hit.source == "template_long_l_rescue"
                        )
                    )
                ):
                    suppressed += 1
                    continue
            output.append(hit)
        return output, suppressed

    def _apply_final_label_resolver(
        final_hits: list[CandidateHit],
        visual_pool: list[CandidateHit],
        pdf_label_hits: list[CandidateHit],
    ) -> tuple[list[CandidateHit], int]:
        """Use exact labels to relabel or rescue a nearby visual hit after clustering."""

        if detector_profile != "color" or not pdf_label_hits:
            return final_hits, 0

        output = list(final_hits)
        changed = 0
        seen_keys = {(hit.template_id, hit.bbox) for hit in output}

        def target_allowed(template_id: int, family: str) -> bool:
            if family == "L":
                return bool(_l_label_group(template_id))
            return family == "EW"

        def candidate_allowed(candidate: CandidateHit, target_id: int, family: str) -> bool:
            if candidate.source == "pdf_text" or candidate.dominant_hsv is None:
                return False
            if not _hue_close(candidate.dominant_hsv, templates[target_id].dominant_hsv):
                return False
            candidate_family = _template_token_family(candidate.template_id)
            if family == "L":
                return bool(_l_label_group(candidate.template_id)) and (
                    _l_label_group(candidate.template_id) == _l_label_group(target_id)
                )
            return candidate_family in {"AW", "EW"}

        def local_to_label(
            candidate: CandidateHit,
            pdf_hit: CandidateHit,
            target_id: int,
            family: str,
        ) -> bool:
            th, tw = templates[target_id].mask.shape[:2]
            if family == "L":
                pad_x = max(90.0, tw * 1.20)
                pad_y = max(190.0, th * 3.20)
                max_distance = 210.0
            else:
                pad_x = max(92.0, tw * 0.75)
                pad_y = max(92.0, th * 0.75)
                max_distance = 140.0
            search_box = _expanded_box(
                pdf_hit.bbox,
                pad_x=pad_x,
                pad_y=pad_y,
            )
            return _center_inside(candidate.bbox, search_box) or _center_distance(
                candidate.bbox,
                pdf_hit.bbox,
            ) <= max_distance

        def good_enough(candidate: CandidateHit, family: str) -> bool:
            if any(id(candidate) == id(hit) for hit in output):
                return True
            if family == "L":
                if _l_label_group(candidate.template_id) == "block":
                    return (
                        candidate.match_score >= 0.72
                        and candidate.coverage >= 0.74
                        and candidate.purity >= 0.86
                        and candidate.context_purity >= 0.40
                    )
                return (
                    candidate.match_score >= 0.50
                    and candidate.coverage >= 0.50
                    and candidate.purity >= 0.62
                    and candidate.context_purity >= 0.13
                )
            return (
                min(candidate.bbox[2], candidate.bbox[3]) >= 82
                and candidate.bbox[2] * candidate.bbox[3] >= 7800
                and candidate.match_score >= 0.40
                and candidate.coverage >= 0.50
                and candidate.purity >= 0.55
                and candidate.context_purity >= 0.15
            )

        def target_geometry_mask(template_id: int, family: str) -> np.ndarray:
            template = templates[template_id]
            if family == "EW" and template.content_mask is not None and template.content_bbox is not None:
                x, y, w, h = template.content_bbox
                cropped = template.content_mask[y : y + h, x : x + w]
                if int(cv2.countNonZero(cropped)) > 0:
                    return cropped
            return template.mask

        def target_shape_confirms(template_id: int, candidate: CandidateHit, family: str) -> bool:
            if candidate.template_id == template_id:
                return True
            plan_mask = plan_masks_by_template.get(template_id)
            if plan_mask is None:
                return False
            x, y, w, h = candidate.bbox
            if w <= 0 or h <= 0:
                return False
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(plan_mask.shape[1], x + w)
            y1 = min(plan_mask.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                return False
            roi = plan_mask[y0:y1, x0:x1]
            if int(cv2.countNonZero(roi)) < 20:
                return False

            target_mask = target_geometry_mask(template_id, family)
            if candidate.scale and abs(float(candidate.scale) - 1.0) >= 0.001:
                target_mask = cv2.resize(
                    target_mask,
                    (
                        max(1, int(round(target_mask.shape[1] * float(candidate.scale)))),
                        max(1, int(round(target_mask.shape[0] * float(candidate.scale)))),
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )
            if candidate.rotation % 360 == 90:
                target_mask = cv2.rotate(target_mask, cv2.ROTATE_90_CLOCKWISE)
            elif candidate.rotation % 360 == 180:
                target_mask = cv2.rotate(target_mask, cv2.ROTATE_180)
            elif candidate.rotation % 360 == 270:
                target_mask = cv2.rotate(target_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if candidate.mirrored:
                target_mask = cv2.flip(target_mask, 1)
            if target_mask.shape[:2] != roi.shape[:2]:
                target_mask = cv2.resize(
                    target_mask,
                    (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            target_pixels = max(1, int(cv2.countNonZero(target_mask)))
            roi_pixels = max(1, int(cv2.countNonZero(roi)))
            overlap = int(cv2.countNonZero(cv2.bitwise_and(roi, target_mask)))
            coverage = overlap / target_pixels
            purity = overlap / roi_pixels
            if family == "EW":
                return coverage >= 0.40 and purity >= 0.42
            return coverage >= 0.46 and purity >= 0.48

        def rotated_mask(mask: np.ndarray, rotation: int) -> np.ndarray:
            if rotation % 360 == 90:
                return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            if rotation % 360 == 180:
                return cv2.rotate(mask, cv2.ROTATE_180)
            if rotation % 360 == 270:
                return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return mask

        def scaled_mask(mask: np.ndarray, scale: float) -> np.ndarray:
            if abs(scale - 1.0) < 0.001:
                return mask
            width = max(1, int(round(mask.shape[1] * scale)))
            height = max(1, int(round(mask.shape[0] * scale)))
            return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        def scan_template_near_label(
            pdf_hit: CandidateHit,
            target_id: int,
            family: str,
        ) -> CandidateHit | None:
            plan_mask = plan_masks_by_template.get(target_id)
            if plan_mask is None:
                return None
            target_mask = target_geometry_mask(target_id, family)
            th, tw = target_mask.shape[:2]
            pad_x = max(85.0, tw * (1.25 if family == "L" else 0.90))
            pad_y = max(85.0, th * (2.25 if family == "L" else 0.95))
            sx, sy, sw, sh = _expanded_box(pdf_hit.bbox, pad_x=pad_x, pad_y=pad_y)
            x0 = max(0, sx)
            y0 = max(0, sy)
            x1 = min(plan_mask.shape[1], sx + sw)
            y1 = min(plan_mask.shape[0], sy + sh)
            if x1 <= x0 or y1 <= y0:
                return None
            search = plan_mask[y0:y1, x0:x1]
            if int(cv2.countNonZero(search)) < (100 if family == "L" else 150):
                return None

            best: tuple[float, tuple[int, int, int, int], int, float, float, float] | None = None
            for scale in (0.90, 1.00, 1.10):
                scaled = scaled_mask(target_mask, scale)
                for rotation in (0, 90, 180, 270):
                    variant = rotated_mask(scaled, rotation)
                    vh, vw = variant.shape[:2]
                    if vh < 8 or vw < 8 or vh > search.shape[0] or vw > search.shape[1]:
                        continue
                    response = cv2.matchTemplate(search, variant, cv2.TM_CCORR_NORMED)
                    _, max_value, _, max_loc = cv2.minMaxLoc(response)
                    bx = x0 + int(max_loc[0])
                    by = y0 + int(max_loc[1])
                    roi = plan_mask[by : by + vh, bx : bx + vw]
                    template_pixels = max(1, int(cv2.countNonZero(variant)))
                    roi_pixels = max(1, int(cv2.countNonZero(roi)))
                    overlap = int(cv2.countNonZero(cv2.bitwise_and(roi, variant)))
                    coverage = overlap / template_pixels
                    purity = overlap / roi_pixels
                    aspect = max(vw, vh) / max(1, min(vw, vh))
                    distance_penalty = min(
                        0.35,
                        _center_distance((bx, by, vw, vh), pdf_hit.bbox)
                        / max(1.0, float(max(tw, th, vw, vh)))
                        * 0.24,
                    )
                    score_value = float(max_value) * 1.3 + coverage * 0.9 + purity * 0.6 - distance_penalty
                    if family == "L" and aspect < 2.35:
                        score_value -= 0.35
                    result = (score_value, (bx, by, vw, vh), rotation, float(max_value), coverage, purity)
                    if best is None or result[0] > best[0]:
                        best = result

            if best is None:
                return None
            score_value, bbox, rotation, max_value, coverage, purity = best
            if family == "L":
                if score_value < 1.58 or max_value < 0.50 or coverage < 0.54 or purity < 0.62:
                    return None
            elif score_value < 1.30 or max_value < 0.34 or coverage < 0.42 or purity < 0.44:
                return None

            template = templates[target_id]
            return replace(
                pdf_hit,
                source="template_label_disambiguation",
                bbox=bbox,
                rotation=rotation,
                transformed_mask=template.mask,
                content_mask=template.content_mask,
                pixel_count=max(1, int(template.pixel_count)),
                content_pixel_count=max(0, int(template.content_pixel_count)),
                content_bbox=template.content_bbox,
                match_score=round(max_value, 4),
                verification_score=round(min(0.82, max(0.58, score_value / 2.35)), 4),
                coverage=round(coverage, 4),
                purity=round(purity, 4),
                context_purity=round(max(0.45, min(0.80, purity)), 4),
                color_similarity=1.0,
                dominant_hsv=template.dominant_hsv,
                is_text_label=template.is_text_label,
                content_score=round(max(0.65, min(0.90, score_value / 2.2)), 4),
            )

        def score(
            candidate: CandidateHit,
            pdf_hit: CandidateHit,
            target_id: int,
            family: str,
        ) -> tuple[float, ...]:
            in_final = 1.0 if any(id(candidate) == id(hit) for hit in output) else 0.0
            distance = _center_distance(candidate.bbox, pdf_hit.bbox)
            target_match = 1.0 if candidate.template_id == target_id else 0.0
            if family == "L":
                return (
                    target_match,
                    -distance,
                    float(candidate.match_score),
                    float(candidate.coverage),
                    float(candidate.purity),
                    float(candidate.context_purity),
                    float(candidate.verification_score),
                    in_final,
                )
            return (
                target_match,
                in_final,
                -distance,
                float(candidate.verification_score),
                float(candidate.match_score),
                float(candidate.coverage),
                float(candidate.purity),
                float(candidate.context_purity),
            )

        for pdf_hit in pdf_label_hits:
            target_id = pdf_hit.template_id
            token = _template_primary_token(target_id)
            family = _token_family(token)
            if not target_allowed(target_id, family):
                continue

            loose_candidates = [
                hit
                for hit in visual_pool
                if candidate_allowed(hit, target_id, family)
                and local_to_label(hit, pdf_hit, target_id, family)
                and good_enough(hit, family)
            ]
            local_candidates = [
                hit
                for hit in loose_candidates
                if target_shape_confirms(target_id, hit, family)
            ]
            if not local_candidates:
                scanned = (
                    scan_template_near_label(pdf_hit, target_id, family)
                    if family == "EW"
                    else None
                )
                if scanned is None:
                    continue
                key = (scanned.template_id, scanned.bbox)
                if key not in seen_keys:
                    output.append(scanned)
                    seen_keys.add(key)
                    changed += 1
                continue
            winner = max(
                local_candidates,
                key=lambda candidate: score(candidate, pdf_hit, target_id, family),
            )
            winner_already_final = any(id(winner) == id(existing) for existing in output)
            if winner.template_id == target_id and winner_already_final:
                continue
            promoted = replace(
                winner,
                template_id=target_id,
                source="template_label_disambiguation",
                promoted_from_template_id=winner.template_id,
                match_score=round(max(float(winner.match_score), 0.62), 4),
                verification_score=round(max(float(winner.verification_score), 0.58), 4),
                coverage=round(max(float(winner.coverage), 0.58), 4),
                purity=round(max(float(winner.purity), 0.58), 4),
                context_purity=round(max(float(winner.context_purity), 0.45), 4),
                dominant_hsv=templates[target_id].dominant_hsv,
                is_text_label=templates[target_id].is_text_label,
                content_score=round(max(float(winner.content_score), 0.65), 4),
            )

            replaced = False
            for idx, existing in enumerate(output):
                if (
                    candidate_allowed(existing, target_id, family)
                    and (
                        id(existing) == id(winner)
                        or _bbox_iom(existing.bbox, winner.bbox) >= 0.18
                        or _center_distance(existing.bbox, winner.bbox) <= 42.0
                    )
                ):
                    output[idx] = promoted
                    replaced = True
                    changed += 1
                    break

            key = (promoted.template_id, promoted.bbox)
            if not replaced and key not in seen_keys:
                output.append(promoted)
                seen_keys.add(key)
                changed += 1
        return output, changed

    def _suppress_weak_short_l_fragments(
        final_hits: list[CandidateHit],
    ) -> tuple[list[CandidateHit], int]:
        if detector_profile != "color":
            return final_hits, 0
        suppressed = 0
        output: list[CandidateHit] = []
        for hit in final_hits:
            name = _template_name(hit.template_id)
            weak_short_l = (
                name in {"08_L8", "09_L9"}
                and hit.source != "template_label_disambiguation"
                and hit.match_score < 0.70
                and hit.context_purity < 0.50
            )
            if weak_short_l:
                has_fuller_neighbor = any(
                    other is not hit
                    and _template_name(other.template_id) not in {"08_L8", "09_L9"}
                    and other.source != "pdf_text"
                    and _center_distance(hit.bbox, other.bbox) <= 85.0
                    and (
                        other.verification_score >= hit.verification_score + 0.08
                        or other.context_purity >= 0.65
                    )
                    for other in final_hits
                )
                if has_fuller_neighbor:
                    suppressed += 1
                    continue
            output.append(hit)
        return output, suppressed

    def _trace_symbol_matches(symbol_name: str) -> bool:
        if not trace_symbols:
            return True
        return any(
            symbol_name == requested
            or symbol_name.startswith(f"{requested}_")
            or symbol_name.startswith(requested)
            for requested in trace_symbols
        )

    def _trace_box_distance(bbox: tuple[int, int, int, int]) -> float:
        if not trace_points:
            return 0.0
        x, y, w, h = bbox
        best = float("inf")
        for px, py in trace_points:
            dx = max(float(x) - px, 0.0, px - float(x + w))
            dy = max(float(y) - py, 0.0, py - float(y + h))
            best = min(best, float(np.hypot(dx, dy)))
        return best

    def _record_candidate_trace(
        stage: str,
        hits: list[CandidateHit],
        reason_by_id: dict[int, str] | None = None,
    ) -> None:
        if not candidate_trace_enabled:
            return

        matched: list[tuple[float, CandidateHit]] = []
        for hit in hits:
            if not (0 <= hit.template_id < len(templates)):
                continue
            symbol_name = templates[hit.template_id].name
            if not _trace_symbol_matches(symbol_name):
                continue
            distance = _trace_box_distance(hit.bbox)
            if trace_points and distance > trace_radius:
                continue
            matched.append((distance, hit))

        matched.sort(
            key=lambda item: (
                item[0],
                -float(item[1].verification_score),
                -float(item[1].match_score),
            )
        )
        items = []
        reason_by_id = reason_by_id or {}
        for distance, hit in matched[:trace_max_items]:
            x, y, w, h = hit.bbox
            item = {
                "symbolName": templates[hit.template_id].name,
                "templateId": int(hit.template_id),
                "bbox": [int(x), int(y), int(w), int(h)],
                "match": round(float(hit.match_score), 3),
                "verification": round(float(hit.verification_score), 3),
                "coverage": round(float(hit.coverage), 3),
                "purity": round(float(hit.purity), 3),
                "context": round(float(hit.context_purity), 3),
                "contentScore": round(float(hit.content_score), 3),
                "pixelCount": int(hit.pixel_count),
                "scale": round(float(hit.scale), 3),
                "rotation": int(hit.rotation),
                "mirrored": bool(hit.mirrored),
                "source": str(hit.source),
                "isTextLabel": bool(hit.is_text_label),
                "roiStrategy": str(hit.roi_strategy),
                "distance": round(float(distance), 3),
            }
            reason = reason_by_id.get(id(hit))
            if reason:
                item["reason"] = reason
            items.append(item)
        candidate_trace[stage] = {
            "totalCandidates": int(len(hits)),
            "matchedCandidates": int(len(matched)),
            "items": items,
        }

    legend_rect = _estimate_legend_exclude_rect(
        pdf_path=pdf_path or "",
        image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
    )
    if legend_rect is not None:
        exclude_rects.append(legend_rect)

    gray_text_exclude_rects: list[tuple[int, int, int, int]] = []
    gray_title_exclude_rects: list[tuple[int, int, int, int]] = []
    if detector_profile == "gray":
        gray_text_exclude_rects = _collect_pdf_text_exclude_rects(
            pdf_path=pdf_path or "",
            image_shape=plan_image.shape,
            dpi=pdf_dpi,
            hidden_layers=hidden_layers,
        )
        gray_title_exclude_rects = _estimate_title_block_exclude_rects(
            pdf_path=pdf_path or "",
            image_shape=plan_image.shape,
            dpi=pdf_dpi,
            hidden_layers=hidden_layers,
        )
        exclude_rects.extend(gray_title_exclude_rects)

    color_masks_cache: dict[str, np.ndarray] = {}
    ink_mask_cache: np.ndarray | None = None
    dilated_ink_mask_cache: np.ndarray | None = None
    empty_plan_mask_cache: np.ndarray | None = None

    def _get_ink_plan_mask(*, dilate: bool) -> np.ndarray:
        nonlocal ink_mask_cache, dilated_ink_mask_cache
        if ink_mask_cache is None:
            ink_mask_cache = _ink_mask(plan_image, dilate=False)
            for ex, ey, ew, eh in exclude_rects:
                cv2.rectangle(ink_mask_cache, (ex, ey), (ex + ew, ey + eh), 0, -1)
        if not dilate:
            return ink_mask_cache
        if dilated_ink_mask_cache is None:
            dilated_ink_mask_cache = cv2.dilate(
                ink_mask_cache,
                np.ones((3, 3), np.uint8),
                iterations=1,
            )
        return dilated_ink_mask_cache

    def _get_empty_plan_mask() -> np.ndarray:
        nonlocal empty_plan_mask_cache
        if empty_plan_mask_cache is None:
            empty_plan_mask_cache = np.zeros(plan_image.shape[:2], dtype=np.uint8)
        return empty_plan_mask_cache

    def _get_plan_mask(template: TemplateInfo) -> np.ndarray:
        if detector_profile == "gray":
            return _get_ink_plan_mask(dilate=True)

        if template.dominant_hsv is None:
            return _get_empty_plan_mask()

        if template.dominant_hsv is not None:
            cache_key = color_strategy.color_mask_cache_key(template)
            if cache_key not in color_masks_cache:
                mask = color_strategy.build_color_plan_mask(
                    plan_image=plan_image,
                    plan_hsv=plan_hsv,
                    template=template,
                    exclude_rects=exclude_rects,
                )
                if cache_key is not None:
                    color_masks_cache[cache_key] = mask
                return mask
            return color_masks_cache[cache_key]

        fallback = _hsv_mask(plan_image, dilate=False, hsv_image=plan_hsv)
        for ex, ey, ew, eh in exclude_rects:
            cv2.rectangle(fallback, (ex, ey), (ex + ew, ey + eh), 0, -1)
        return fallback

    phase_start = time.perf_counter()
    _progress("pdf_text", 12, "Odczyt pomocniczych tekstow PDF")
    pdf_hits_by_template = _collect_pdf_text_hits(
        pdf_path=pdf_path or "",
        templates=templates,
        plan_image_shape=plan_image.shape,
        dpi=pdf_dpi,
        hidden_layers=hidden_layers,
        exclude_rects=exclude_rects,
    )
    pdf_candidates = [hit for hits in pdf_hits_by_template.values() for hit in hits]
    has_pdf_text_assist = len(pdf_candidates) > 0
    timings["pdf_text"] = time.perf_counter() - phase_start

    phase_start = time.perf_counter()
    _progress("prepare", 18, "Przygotowanie masek i wariantow")
    plan_hsv = cv2.cvtColor(plan_image, cv2.COLOR_BGR2HSV) if detector_profile == "color" else None

    variant_workers = max(1, min(len(templates), DETECTOR_POSTPROCESS_MAX_WORKERS))
    used_scales = list(GRAY_SCALES) if detector_profile == "gray" else list(SCALES)
    with ThreadPoolExecutor(max_workers=variant_workers) as pool:
        prepared_variant_items = list(
            pool.map(
                lambda item: (
                    item[0],
                    _prepare_variants(
                        item[0],
                        item[1],
                        scales=used_scales,
                        include_gray_diagonal_rotations=detector_profile == "gray",
                        disable_text_mirror=disable_text_mirror,
                    ),
                ),
                enumerate(templates),
            )
        )
    variants_by_template = dict(prepared_variant_items)
    if detector_profile == "gray":
        variants_by_template = {
            template_id: [
                variant
                for variant in variants
                if not (
                    not templates[template_id].is_text_label
                    and variant.scale <= GRAY_TINY_FRAGMENT_MAX_SCALE
                    and max(variant.width, variant.height) <= GRAY_TINY_FRAGMENT_MAX_DIMENSION
                )
            ]
            for template_id, variants in variants_by_template.items()
        }
    variants_lookup = {
        (variant.template_id, variant.scale, variant.rotation, variant.mirrored): variant
        for variants in variants_by_template.values()
        for variant in variants
    }
    socket_07_promotions = (
        _build_socket_07_promotions(templates, variants_by_template)
        if detector_profile in {"gray", "color"}
        else {}
    )
    validation_promotions = socket_07_promotions if detector_profile in {"gray", "color"} else {}
    parent_ids_by_child: dict[int, set[int]] = {}
    for rules in socket_07_promotions.values():
        for rule in rules:
            parent_ids_by_child.setdefault(rule.child_template_id, set()).add(
                rule.parent_template_id
            )
    if detector_profile == "color":
        template_ids_by_prefix: dict[str, list[int]] = {}
        for template_id, template in enumerate(templates):
            prefix = _template_numeric_prefix(template.name)
            if prefix is None:
                continue
            template_ids_by_prefix.setdefault(prefix, []).append(template_id)
        for child_prefix, parent_prefixes in {
            "06": ("07",),
            "09": ("07",),
            "11": ("10", "12"),
        }.items():
            for child_id in template_ids_by_prefix.get(child_prefix, []):
                for parent_prefix in parent_prefixes:
                    parent_ids_by_child.setdefault(child_id, set()).update(
                        template_ids_by_prefix.get(parent_prefix, [])
                    )
    plan_masks_by_template: dict[int, np.ndarray] = {}
    template_by_mask_key: dict[str, TemplateInfo] = {}
    if detector_profile == "color":
        for template in templates:
            cache_key = color_strategy.color_mask_cache_key(template)
            if cache_key is not None:
                template_by_mask_key.setdefault(cache_key, template)
    unique_mask_keys = set(template_by_mask_key)

    def _build_cached_color_mask(cache_key: str) -> tuple[str, np.ndarray]:
        mask = color_strategy.build_color_plan_mask(
            plan_image=plan_image,
            plan_hsv=plan_hsv,
            template=template_by_mask_key[cache_key],
            exclude_rects=exclude_rects,
        )
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
            max(
                (variant.width for variant in variants),
                default=templates[template_id].mask.shape[1],
            ),
            max(
                (variant.height for variant in variants),
                default=templates[template_id].mask.shape[0],
            ),
        )
        for template_id, variants in variants_by_template.items()
    }

    # Gray stroke suppression - build dedicated dark scan masks before ROI.
    # Gray ROIs should be seeded from the same dark ink mask used by scanning,
    # otherwise pale architectural lines still create too many search windows.
    gray_suppressed_pixels = 0
    gray_raw_ink_pixels = 0
    gray_dark_ink_pixels = 0
    gray_dark_zone_pixels = 0
    gray_dark_evidence_pixels = 0
    gray_dark_threshold = GRAY_DARK_INK_THRESHOLD
    gray_dark_zone_threshold = GRAY_DARK_ZONE_THRESHOLD
    gray_dark_evidence_threshold = GRAY_DARK_EVIDENCE_THRESHOLD
    gray_dark_suppressed_pixels = 0
    gray_dark_zone_suppressed_pixels = 0
    scan_masks_by_template: dict[int, np.ndarray]
    scan_mask_kinds_by_template: dict[int, str]
    gray_zone_mask: np.ndarray | None = None
    gray_evidence_mask: np.ndarray | None = None
    timings["gray_suppress"] = 0.0
    if detector_profile == "gray":
        _t_suppress = time.perf_counter()
        _raw_dilated = _get_ink_plan_mask(dilate=True)
        gray_scan_masks = gray_strategy.build_gray_scan_masks(
            plan_image=plan_image,
            templates=templates,
            plan_masks_by_template=plan_masks_by_template,
            exclude_rects=exclude_rects,
            raw_dilated=_raw_dilated,
        )
        gray_raw_ink_pixels = gray_scan_masks.raw_ink_pixels
        gray_suppressed_pixels = gray_scan_masks.suppressed_pixels
        gray_dark_ink_pixels = gray_scan_masks.dark_ink_pixels
        gray_dark_zone_pixels = gray_scan_masks.zone_ink_pixels
        gray_dark_evidence_pixels = gray_scan_masks.evidence_ink_pixels
        gray_dark_threshold = gray_scan_masks.dark_threshold
        gray_dark_zone_threshold = gray_scan_masks.zone_threshold
        gray_dark_evidence_threshold = gray_scan_masks.evidence_threshold
        gray_dark_suppressed_pixels = gray_scan_masks.dark_suppressed_pixels
        gray_dark_zone_suppressed_pixels = gray_scan_masks.zone_suppressed_pixels
        scan_masks_by_template = gray_scan_masks.scan_masks_by_template
        scan_mask_kinds_by_template = gray_scan_masks.scan_mask_kinds_by_template
        gray_zone_mask = gray_scan_masks.zone_mask
        gray_evidence_mask = gray_scan_masks.evidence_mask
        timings["gray_suppress"] = time.perf_counter() - _t_suppress
    else:
        scan_masks_by_template = plan_masks_by_template
        scan_mask_kinds_by_template = {template_id: "color" for template_id in plan_masks_by_template}

    search_rois_by_template: dict[int, list[tuple[int, int, int, int]]] = {}
    search_roi_stats_by_template: dict[int, tuple[bool, int, int]] = {}
    search_roi_strategy_by_template: dict[int, str] = {}
    gray_search_component_index = (
        gray_strategy.build_gray_search_component_index(gray_zone_mask)
        if detector_profile == "gray" and gray_zone_mask is not None
        else None
    )

    def _prepare_search_roi(
        item: tuple[int, np.ndarray]
    ) -> tuple[int, list[tuple[int, int, int, int]], tuple[bool, int, int]]:
        template_id, plan_mask = item
        max_width, max_height = max_variant_size_by_template[template_id]
        if detector_profile == "gray":
            roi_seed_mask = gray_zone_mask if gray_zone_mask is not None else plan_mask
            roi_strategy, tile_size, max_tile_rois = gray_strategy.gray_tile_roi_strategy(
                templates[template_id]
            )
            roi_strategy, tile_size, max_tile_rois = (
                gray_strategy.adapt_gray_tile_roi_strategy_for_plan(
                    roi_strategy,
                    tile_size,
                    max_tile_rois,
                    gray_search_component_index,
                )
            )
            rois, uses_full_scan, roi_area, foreground_pixels = gray_strategy.build_gray_search_rois(
                roi_seed_mask,
                plan_image.shape,
                max_width,
                max_height,
                is_large_text_template=gray_strategy.use_large_text_tile_rois(
                    templates[template_id]
                ),
                component_index=gray_search_component_index,
                tile_size_override=tile_size,
                max_tile_rois_override=max_tile_rois,
                component_supplement_rois=80 if templates[template_id].is_text_label else 0,
            )
        else:
            roi_strategy = "color"
            rois, uses_full_scan, roi_area, foreground_pixels = _build_search_rois(
                plan_mask,
                plan_image.shape,
                max_width,
                max_height,
            )
        return template_id, rois, (uses_full_scan, roi_area, foreground_pixels), roi_strategy

    roi_workers = max(1, min(len(plan_masks_by_template), DETECTOR_POSTPROCESS_MAX_WORKERS))
    with ThreadPoolExecutor(max_workers=roi_workers) as pool:
        roi_items = list(pool.map(_prepare_search_roi, plan_masks_by_template.items()))
    for template_id, rois, stats, roi_strategy in roi_items:
        search_rois_by_template[template_id] = rois
        search_roi_stats_by_template[template_id] = stats
        search_roi_strategy_by_template[template_id] = roi_strategy
    dilated_plan_masks_by_template: dict[int, np.ndarray] = {}
    timings["prepare"] = time.perf_counter() - phase_start

    diagnostics = {
        "detector_profile": 1 if detector_profile == "gray" else 0,
        "gray_text_exclude_rects": len(gray_text_exclude_rects),
        "gray_title_exclude_rects": len(gray_title_exclude_rects),
        "gray_suppressed_pixels": gray_suppressed_pixels,
        "gray_suppressed_ratio": round(gray_suppressed_pixels / max(1, gray_raw_ink_pixels), 3),
        "gray_dark_ink_pixels": gray_dark_ink_pixels,
        "gray_dark_zone_pixels": gray_dark_zone_pixels,
        "gray_dark_evidence_pixels": gray_dark_evidence_pixels,
        "gray_dark_threshold": gray_dark_threshold,
        "gray_dark_zone_threshold": gray_dark_zone_threshold,
        "gray_dark_evidence_threshold": gray_dark_evidence_threshold,
        "gray_dark_suppressed_pixels": gray_dark_suppressed_pixels,
        "gray_dark_suppressed_ratio": round(
            gray_dark_suppressed_pixels / max(1, gray_dark_ink_pixels), 3
        ),
        "gray_dark_zone_suppressed_pixels": gray_dark_zone_suppressed_pixels,
        "gray_dark_zone_suppressed_ratio": round(
            gray_dark_zone_suppressed_pixels / max(1, gray_dark_zone_pixels), 3
        ),
        "raw_peaks": 0,
        "raw_budget_hits": 0,
        "raw_budget_removed": 0,
        "raw_prefilter_hits": 0,
        "raw_prefilter_removed": 0,
        "gray_near_threshold_recovery_candidates": 0,
        "gray_near_threshold_recovery_accepted": 0,
        "gray_near_threshold_recovery_rejected": 0,
        "gray_interrupted_recovery_candidates": 0,
        "gray_interrupted_recovery_accepted": 0,
        "gray_interrupted_recovery_rejected": 0,
        "color_recovery_candidates": 0,
        "color_recovery_accepted": 0,
        "color_recovery_rejected": 0,
        "prepared_variants": sum(len(variants) for variants in variants_by_template.values()),
        "gray_text_mirror_enabled": int(detector_profile == "gray" and not disable_text_mirror),
        "color_no_hsv_templates": sum(
            1
            for template in templates
            if detector_profile == "color" and template.dominant_hsv is None
        ),
        "skipped_empty_color_masks": 0,
        "validated_template_hits": 0,
        "promoted_targeted_hits": 0,
        "parent_search_input_hits": 0,
        "parent_search_candidates": 0,
        "promoted_parent_search_hits": 0,
        "pdf_text_hits": len(pdf_candidates),
        "pdf_text_label_disambiguation": 0,
        "color_family_rescued": 0,
        "prefilter_hits": 0,
        "pre_parent_clusters": 0,
        "final_hits": 0,
        "search_rois": sum(len(rois) for rois in search_rois_by_template.values()),
        "gray_roi_fast_templates": sum(
            1
            for strategy in search_roi_strategy_by_template.values()
            if strategy in {"large_text_fast", "fast_compact", "fast_compact_connected"}
        ),
        "gray_roi_fast_rois": sum(
            len(search_rois_by_template[template_id])
            for template_id, strategy in search_roi_strategy_by_template.items()
            if strategy in {"large_text_fast", "fast_compact", "fast_compact_connected"}
        ),
        "gray_roi_safe_templates": sum(
            1 for strategy in search_roi_strategy_by_template.values() if strategy == "safe_elongated"
        ),
        "gray_roi_safe_rois": sum(
            len(search_rois_by_template[template_id])
            for template_id, strategy in search_roi_strategy_by_template.items()
            if strategy == "safe_elongated"
        ),
        "full_scan_templates": sum(
            1 for uses_full, _, _ in search_roi_stats_by_template.values() if uses_full
        ),
        "roi_area_pixels": sum(area for _, area, _ in search_roi_stats_by_template.values()),
        "roi_foreground_pixels": sum(
            pixels for _, _, pixels in search_roi_stats_by_template.values()
        ),
    }

    scan_result = scan_template_candidates(
        templates=templates,
        variants_by_template=variants_by_template,
        scan_masks_by_template=scan_masks_by_template,
        scan_mask_kinds_by_template=scan_mask_kinds_by_template,
        search_rois_by_template=search_rois_by_template,
        roi_strategies_by_template=search_roi_strategy_by_template,
        plan_mask_foregrounds=plan_mask_foregrounds,
        detector_profile=detector_profile,
        progress_callback=_progress,
        collect_profile=collect_performance_profile,
    )
    raw_template_hits = scan_result.raw_template_hits
    scan_workers = scan_result.scan_workers
    diagnostics["skipped_empty_color_masks"] = scan_result.skipped_empty_color_masks
    diagnostics["scan_tasks"] = scan_result.scan_tasks
    diagnostics["scan_task_rois"] = scan_result.scan_task_rois
    timings["scan"] = scan_result.timing_seconds
    diagnostics["raw_peaks"] = len(raw_template_hits)
    diagnostics["gray_near_threshold_recovery_candidates"] = sum(
        1 for hit in raw_template_hits if hit.source == "template_near_threshold"
    )
    diagnostics["gray_interrupted_recovery_candidates"] = sum(
        1 for hit in raw_template_hits if hit.source == "template_interrupted_recovery"
    )
    diagnostics["color_recovery_candidates"] = sum(
        1 for hit in raw_template_hits if hit.source == "template_color_recovery"
    )
    raw_scan_hits = raw_template_hits
    _record_candidate_trace("raw_scan", raw_template_hits)

    # Per-scale histograms for diagnostics: count raw_peaks at each scale
    # before budget/prefilter so we can see whether matchTemplate fires at all
    # at a given scale (especially important on gray plans).
    raw_peaks_by_scale: dict[float, int] = {}
    if debug_profile is not None:
        for _hit in raw_template_hits:
            raw_peaks_by_scale[_hit.scale] = raw_peaks_by_scale.get(_hit.scale, 0) + 1

    gray_budget_profile: dict = {}
    phase_start = time.perf_counter()
    if detector_profile == "gray":
        raw_template_hits, gray_budget_profile = gray_strategy.gray_raw_budget(
            raw_template_hits,
            templates,
            plan_masks_by_template,
        )
        diagnostics["raw_budget_hits"] = len(raw_template_hits)
        diagnostics["raw_budget_removed"] = int(gray_budget_profile.get("removed", 0))
    else:
        diagnostics["raw_budget_hits"] = len(raw_template_hits)
    timings["raw_budget"] = time.perf_counter() - phase_start
    raw_budget_hits = raw_template_hits
    _record_candidate_trace("raw_budget", raw_template_hits)

    phase_start = time.perf_counter()
    raw_before_prefilter = len(raw_template_hits)
    gray_frame_raw_rescue_hits = (
        [hit for hit in raw_template_hits if gray_strategy.is_gray_frame_raw_rescue_hit(hit, templates)]
        if detector_profile == "gray"
        else []
    )
    detail = f"Odsiew slabych kandydatow ({raw_before_prefilter})"
    if gray_budget_profile.get("removed"):
        detail += f", ucieto {gray_budget_profile['removed']}"
    _progress("raw_prefilter", 62, detail)
    raw_template_hits = _prefilter_raw_template_hits(raw_template_hits)
    if gray_frame_raw_rescue_hits:
        existing_ids = {id(hit) for hit in raw_template_hits}
        restored = [hit for hit in gray_frame_raw_rescue_hits if id(hit) not in existing_ids]
        if restored:
            raw_template_hits.extend(restored)
    else:
        restored = []
    raw_template_hits = _dedupe_raw_template_hits_before_validation(raw_template_hits)
    diagnostics["raw_prefilter_hits"] = len(raw_template_hits)
    diagnostics["raw_prefilter_removed"] = raw_before_prefilter - len(raw_template_hits)
    diagnostics["gray_frame_raw_rescue_hits"] = len(gray_frame_raw_rescue_hits)
    diagnostics["gray_frame_raw_rescue_restored"] = len(restored)
    raw_prefilter_hits = raw_template_hits
    timings["raw_prefilter"] = time.perf_counter() - phase_start
    _record_candidate_trace("raw_prefilter", raw_template_hits)

    candidates_by_scale: dict[float, int] = {}
    if debug_profile is not None:
        for _hit in raw_template_hits:
            candidates_by_scale[_hit.scale] = candidates_by_scale.get(_hit.scale, 0) + 1

    validated_candidates: list[CandidateHit] = []
    postprocess_workers = max(1, DETECTOR_POSTPROCESS_MAX_WORKERS)
    validation_result = validate_template_candidates(
        raw_template_hits=raw_template_hits,
        plan_image=plan_image,
        templates=templates,
        plan_masks_by_template=plan_masks_by_template,
        dilated_plan_masks_by_template=dilated_plan_masks_by_template,
        variants_lookup=variants_lookup,
        socket_07_promotions=validation_promotions,
        plan_hsv=plan_hsv,
        postprocess_workers=postprocess_workers,
        progress_callback=_progress,
        gray_evidence_mask=gray_evidence_mask if detector_profile == "gray" else None,
        gray_relaxed_evidence_mask=gray_zone_mask if detector_profile == "gray" else None,
    )
    validated_hits = validation_result.validated_hits
    rejection_reasons = validation_result.rejection_reasons
    validation_workers = validation_result.validation_workers
    color_pdf_text_hinted = apply_color_pdf_text_class_hints(
        validated_hits,
        pdf_candidates,
        detector_profile=detector_profile,
        template_tokens=_template_tokens,
    )
    label_disambiguation_count, label_rescued_hits = _apply_color_label_disambiguation(
        validated_hits,
        validation_result.rejected_hits,
        validation_result.rejection_reason_by_hit_id,
        pdf_candidates,
    )
    if label_rescued_hits:
        validated_hits.extend(label_rescued_hits)
        _record_candidate_trace("label_disambiguation_rescued", label_rescued_hits)
    pdf_candidates, removed_pdf_text_fallbacks = filter_pdf_text_fallbacks(
        pdf_candidates,
        validated_hits,
        detector_profile=detector_profile,
        is_visual_pdf_text_blocked=_is_visual_pdf_text_blocked,
    )
    validated_candidates.extend(pdf_candidates)
    diagnostics["promoted_targeted_hits"] += validation_result.promoted_targeted_hits
    diagnostics["gray_near_threshold_recovery_accepted"] = sum(
        1 for hit in validated_hits if hit.source == "template_near_threshold"
    )
    diagnostics["gray_near_threshold_recovery_rejected"] = sum(
        1 for hit in validation_result.rejected_hits if hit.source == "template_near_threshold"
    )
    diagnostics["gray_interrupted_recovery_accepted"] = sum(
        1 for hit in validated_hits if hit.source == "template_interrupted_recovery"
    )
    diagnostics["gray_interrupted_recovery_rejected"] = sum(
        1
        for hit in validation_result.rejected_hits
        if hit.source == "template_interrupted_recovery"
    )
    diagnostics["color_recovery_accepted"] = sum(
        1 for hit in validated_hits if hit.source == "template_color_recovery"
    )
    diagnostics["color_recovery_rejected"] = sum(
        1 for hit in validation_result.rejected_hits if hit.source == "template_color_recovery"
    )
    validated_candidates.extend(validated_hits)
    _record_candidate_trace("validation_accepted", validated_hits)
    _record_candidate_trace(
        "validation_rejected",
        validation_result.rejected_hits,
        validation_result.rejection_reason_by_hit_id,
    )

    accepted_by_scale: dict[float, int] = {}
    if debug_profile is not None:
        for _hit in validated_hits:
            accepted_by_scale[_hit.scale] = accepted_by_scale.get(_hit.scale, 0) + 1

    diagnostics["validated_template_hits"] = len(validated_candidates) - len(pdf_candidates)
    diagnostics["pdf_text_fallback_removed"] = len(removed_pdf_text_fallbacks)
    diagnostics["pdf_text_class_hints"] = color_pdf_text_hinted
    diagnostics["pdf_text_label_disambiguation"] = label_disambiguation_count
    timings["validation_targeted"] = validation_result.timing_seconds

    phase_start = time.perf_counter()
    _progress("prefilter", 84, "Klastrowanie kandydatow")
    prefiltered_candidates = _prefilter_candidates(validated_candidates)
    diagnostics["prefilter_hits"] = len(prefiltered_candidates)
    timings["prefilter"] = time.perf_counter() - phase_start
    _record_candidate_trace("prefilter", prefiltered_candidates)

    phase_start = time.perf_counter()
    pre_parent_candidates = _cluster_candidates(
        prefiltered_candidates,
        parent_ids_by_child,
        mode=detector_profile,
        prefer_direct_color_family_parent=True,
    )
    pre_parent_ids = {id(hit) for hit in pre_parent_candidates}
    pre_parent_suppressed = [hit for hit in prefiltered_candidates if id(hit) not in pre_parent_ids]
    diagnostics["pre_parent_clusters"] = len(pre_parent_candidates)
    timings["pre_parent_clustering"] = time.perf_counter() - phase_start
    _record_candidate_trace("pre_parent_clusters", pre_parent_candidates)
    _record_candidate_trace(
        "pre_parent_cluster_suppressed",
        pre_parent_suppressed,
        {id(hit): "cluster_suppressed" for hit in pre_parent_suppressed},
    )

    if detector_profile == "color":
        parent_search_input = [
            hit
            for hit in prefiltered_candidates
            if (hit.template_id, hit.scale, hit.rotation, hit.mirrored) in socket_07_promotions
            and hit.dominant_hsv is not None
            and hit.source == "template"
            and hit.match_score >= 0.48
            and hit.coverage >= 0.65
            and hit.purity >= 0.55
        ]
    else:
        parent_search_input = pre_parent_candidates
    parent_search_result = search_parent_candidates(
        pre_parent_candidates=parent_search_input,
        detector_profile=detector_profile,
        plan_image=plan_image,
        templates=templates,
        plan_masks_by_template=plan_masks_by_template,
        dilated_plan_masks_by_template=dilated_plan_masks_by_template,
        variants_lookup=variants_lookup,
        socket_07_promotions=socket_07_promotions,
        plan_hsv=plan_hsv,
        allow_color_switch_10=True,
        postprocess_workers=postprocess_workers,
        progress_callback=_progress,
    )
    if detector_profile == "color":
        promoted_parent_hits = [
            hit
            for hit in parent_search_result.candidates
            if hit.source.startswith("template_parent_search_")
            or hit.source.startswith("template_promoted_")
        ]
        parent_search_candidates = pre_parent_candidates + promoted_parent_hits
    else:
        parent_search_candidates = parent_search_result.candidates
    parent_search_workers = parent_search_result.workers
    diagnostics["parent_search_input_hits"] += parent_search_result.input_hits
    diagnostics["parent_search_candidates"] += parent_search_result.attempted_candidates
    diagnostics["promoted_parent_search_hits"] += parent_search_result.promoted_hits
    timings["parent_search"] = parent_search_result.timing_seconds
    _record_candidate_trace("parent_search", parent_search_candidates)

    phase_start = time.perf_counter()
    _progress("final_clustering", 94, "Finalne laczenie wynikow")
    final_hits = _cluster_candidates(
        parent_search_candidates,
        parent_ids_by_child,
        mode=detector_profile,
        prefer_direct_color_family_parent=True,
    )
    final_cluster_ids = {id(hit) for hit in final_hits}
    final_cluster_suppressed = [
        hit for hit in parent_search_candidates if id(hit) not in final_cluster_ids
    ]
    _record_candidate_trace(
        "final_cluster_suppressed",
        final_cluster_suppressed,
        {id(hit): "same_place_loser" for hit in final_cluster_suppressed},
    )
    if detector_profile == "gray":
        final_hits, rescued_gray_frames, gray_rescue_trace = gray_strategy.rescue_validated_gray_frame_hits(
            final_hits,
            validated_hits,
            templates,
        )
        for stage_name, stage_trace in gray_rescue_trace.items():
            _record_candidate_trace(
                stage_name,
                stage_trace["hits"],
                stage_trace.get("reasons"),
            )
        gray_unresolved_strong_hits = gray_strategy.trace_unresolved_strong_gray_hits(
            final_hits,
            validated_hits,
            templates,
        )
    else:
        (
            final_hits,
            color_fragment_suppressed,
            color_fragment_suppression_reasons,
        ) = _suppress_color_local_fragments(final_hits)
        _record_candidate_trace(
            "color_fragment_suppressed",
            color_fragment_suppressed,
            color_fragment_suppression_reasons,
        )
        final_hits, color_magenta_reclassed = _reconcile_magenta_family_hits(
            final_hits,
            prefiltered_candidates,
        )
        final_hits, color_magenta_local_rescued = _rescue_magenta_family_hits(
            final_hits,
            prefiltered_candidates + validation_result.rejected_hits,
        )
        final_hits, color_family_rescued = _rescue_color_family_hits(
            final_hits,
            prefiltered_candidates + validation_result.rejected_hits,
            validation_result.rejection_reason_by_hit_id,
        )
        final_hits, long_l_over_tb11_promoted = _promote_long_l_over_tb11_conflicts(
            final_hits,
            prefiltered_candidates + validation_result.rejected_hits,
        )
        final_hits, tb11_long_l_suppressed = _suppress_tb11_long_l_conflicts(
            final_hits,
            prefiltered_candidates + validation_result.rejected_hits,
        )
        final_hits, final_label_disambiguation = _apply_final_label_resolver(
            final_hits,
            final_hits + prefiltered_candidates + validation_result.rejected_hits,
            pdf_candidates + removed_pdf_text_fallbacks,
        )
        final_hits, long_l_over_true_tb11_suppressed = _suppress_long_l_rescue_over_tb11_waves(
            final_hits,
            pdf_candidates + removed_pdf_text_fallbacks,
        )
        final_hits, weak_short_l_suppressed = _suppress_weak_short_l_fragments(
            final_hits,
        )
        final_hits, duplicate_final_suppressed = dedupe_final_hits(final_hits)
        diagnostics["color_magenta_reclassed"] = color_magenta_reclassed
        diagnostics["color_magenta_local_rescued"] = color_magenta_local_rescued
        diagnostics["color_family_rescued"] = color_family_rescued
        diagnostics["long_l_over_tb11_promoted"] = long_l_over_tb11_promoted
        diagnostics["tb11_long_l_suppressed"] = tb11_long_l_suppressed
        diagnostics["final_label_disambiguation"] = final_label_disambiguation
        diagnostics["long_l_over_true_tb11_suppressed"] = long_l_over_true_tb11_suppressed
        diagnostics["weak_short_l_suppressed"] = weak_short_l_suppressed
        diagnostics["duplicate_final_suppressed"] = duplicate_final_suppressed
        rescued_gray_frames = 0
        gray_unresolved_strong_hits = {"strongValidated": 0, "unresolved": 0, "items": []}
    diagnostics["gray_frame_final_rescued"] = rescued_gray_frames
    diagnostics["final_hits"] = len(final_hits)
    timings["clustering"] = time.perf_counter() - phase_start
    _record_candidate_trace("final", final_hits)

    timings_ms = {name: round(seconds * 1000.0, 3) for name, seconds in timings.items()}
    if debug_profile is not None:
        candidate_stage_counts = build_candidate_stage_counts(
            pdf_candidates=pdf_candidates,
            raw_scan_hits=raw_scan_hits,
            raw_budget_hits=raw_budget_hits,
            raw_prefilter_hits=raw_prefilter_hits,
            validation_rejected_hits=validation_result.rejected_hits,
            validated_hits=validated_hits,
            prefiltered_hits=prefiltered_candidates,
            pre_parent_hits=pre_parent_candidates,
            parent_search_hits=parent_search_candidates,
            final_hits=final_hits,
            rescued_gray_frames=rescued_gray_frames,
        )
        hit_flow_profile = {}
        roi_strategy_profile = {}
        if collect_performance_profile:
            hit_flow_profile = build_hit_flow_profile(
                templates=templates,
                raw_scan_hits=raw_scan_hits,
                raw_budget_hits=raw_budget_hits,
                raw_prefilter_hits=raw_prefilter_hits,
                validated_hits=validated_hits,
                pre_cluster_hits=pre_parent_candidates,
                final_hits=final_hits,
            )
            roi_strategy_profile = build_roi_strategy_profile(
                templates=templates,
                variants_by_template=variants_by_template,
                search_rois_by_template=search_rois_by_template,
                search_roi_stats_by_template=search_roi_stats_by_template,
                search_roi_strategy_by_template=search_roi_strategy_by_template,
                raw_scan_hits=raw_scan_hits,
                raw_budget_hits=raw_budget_hits,
                raw_prefilter_hits=raw_prefilter_hits,
                validated_hits=validated_hits,
                final_hits=final_hits,
            )
        debug_profile.clear()
        debug_profile.update(
            {
                "profileFlags": {
                    "debugProfile": True,
                    "performanceProfile": bool(collect_performance_profile),
                    "candidateTrace": bool(candidate_trace_enabled),
                },
                "timingsMs": timings_ms,
                "counters": {key: int(value) for key, value in diagnostics.items()},
                "profile": detector_profile,
                "threading": {
                    "scanStrategy": scan_result.scan_strategy,
                    "scanWorkers": int(scan_workers),
                    "scanTasks": int(scan_result.scan_tasks),
                    "scanTaskRois": int(scan_result.scan_task_rois),
                    "configuredScanWorkers": int(scan_result.configured_scan_workers),
                    "validationWorkers": int(validation_workers if raw_template_hits else 0),
                    "parentSearchWorkers": int(
                        parent_search_workers if pre_parent_candidates else 0
                    ),
                    "configuredPostprocessWorkers": int(DETECTOR_POSTPROCESS_MAX_WORKERS),
                    "opencvThreads": int(scan_result.opencv_threads),
                    "configuredOpencvThreads": int(OPENCV_NUM_THREADS),
                    "cpuCount": int(_safe_cpu_count()),
                },
                "searchRoi": {
                    "totalRois": int(diagnostics["search_rois"]),
                    "fullScanTemplates": int(diagnostics["full_scan_templates"]),
                    "roiAreaPixels": int(diagnostics["roi_area_pixels"]),
                    "foregroundPixels": int(diagnostics["roi_foreground_pixels"]),
                    "fullImageAreaPixels": int(plan_image.shape[0] * plan_image.shape[1]),
                },
                "grayRawBudget": gray_budget_profile,
                "candidateTrace": candidate_trace if candidate_trace_enabled else {},
                "candidateStageCounts": candidate_stage_counts,
                "scanProfile": scan_result.scan_profile,
                "grayNearThresholdRecovery": {
                    "candidates": int(diagnostics["gray_near_threshold_recovery_candidates"]),
                    "accepted": int(diagnostics["gray_near_threshold_recovery_accepted"]),
                    "rejected": int(diagnostics["gray_near_threshold_recovery_rejected"]),
                    "rejectionReasons": {
                        str(key): int(value)
                        for key, value in validation_result.rejection_reasons_by_source.get(
                            "template_near_threshold",
                            {},
                        ).items()
                    },
                },
                "grayInterruptedRecovery": {
                    "candidates": int(diagnostics["gray_interrupted_recovery_candidates"]),
                    "accepted": int(diagnostics["gray_interrupted_recovery_accepted"]),
                    "rejected": int(diagnostics["gray_interrupted_recovery_rejected"]),
                    "rejectionReasons": {
                        str(key): int(value)
                        for key, value in validation_result.rejection_reasons_by_source.get(
                            "template_interrupted_recovery",
                            {},
                        ).items()
                    },
                },
                "hitFlowProfile": hit_flow_profile,
                "roiStrategyProfile": roi_strategy_profile,
                "ablation": {"noTextMirror": bool(ablation_no_text_mirror)},
                "grayVariantStrategy": {
                    "textMirrorEnabled": bool(
                        detector_profile == "gray" and not disable_text_mirror
                    ),
                    "textMirrorOverride": bool(gray_force_text_mirror),
                },
                "grayStrokeSuppression": {
                    "removedPixels": int(diagnostics["gray_suppressed_pixels"]),
                    "rawPixels": int(gray_raw_ink_pixels),
                    "ratio": float(diagnostics["gray_suppressed_ratio"]),
                    "darkThreshold": int(diagnostics["gray_dark_threshold"]),
                    "darkPixels": int(diagnostics["gray_dark_ink_pixels"]),
                    "zoneThreshold": int(diagnostics["gray_dark_zone_threshold"]),
                    "zonePixels": int(diagnostics["gray_dark_zone_pixels"]),
                    "evidenceThreshold": int(diagnostics["gray_dark_evidence_threshold"]),
                    "evidencePixels": int(diagnostics["gray_dark_evidence_pixels"]),
                    "zoneRemovedPixels": int(diagnostics["gray_dark_zone_suppressed_pixels"]),
                    "zoneRemovedRatio": float(diagnostics["gray_dark_zone_suppressed_ratio"]),
                    "darkRemovedPixels": int(diagnostics["gray_dark_suppressed_pixels"]),
                    "darkRemovedRatio": float(diagnostics["gray_dark_suppressed_ratio"]),
                    "timingMs": round(timings_ms.get("gray_suppress", 0.0), 3),
                    "horizontalKernelPx": GRAY_SUPPRESS_HORIZONTAL_KERNEL_PX,
                    "verticalKernelPx": GRAY_SUPPRESS_VERTICAL_KERNEL_PX,
                },
                "scaleHistogram": {
                    "usedScales": [round(float(s), 3) for s in used_scales],
                    "rawPeaks": {
                        round(float(s), 3): int(c) for s, c in sorted(raw_peaks_by_scale.items())
                    },
                    "candidates": {
                        round(float(s), 3): int(c) for s, c in sorted(candidates_by_scale.items())
                    },
                    "accepted": {
                        round(float(s), 3): int(c) for s, c in sorted(accepted_by_scale.items())
                    },
                },
                "scanMaskRawHits": {
                    str(k): int(v) for k, v in sorted(scan_result.raw_hits_by_mask_kind.items())
                },
                "grayUnresolvedStrongHits": gray_unresolved_strong_hits,
                "rejectionReasons": {k: int(v) for k, v in rejection_reasons.items()},
                "slowestPhase": (
                    max(timings_ms.items(), key=lambda item: item[1])[0] if timings_ms else None
                ),
            }
        )

    if debug_profile is not None:
        def _fmt_scale_hist(hist: dict[float, int]) -> str:
            if not hist:
                return "{}"
            return "{" + ",".join(f"{s:.2f}:{hist.get(s, 0)}" for s in used_scales) + "}"

        rejection_summary = (
            ",".join(f"{k}:{v}" for k, v in sorted(rejection_reasons.items(), key=lambda kv: -kv[1]))
            if rejection_reasons
            else "-"
        )
        scan_mask_summary = (
            ",".join(f"{k}:{v}" for k, v in sorted(scan_result.raw_hits_by_mask_kind.items()))
            if scan_result.raw_hits_by_mask_kind
            else "-"
        )

        print(
            "Detection diagnostics:"
            f" profile={detector_profile},"
            f" templates={len(templates)},"
            f" used_scales=[{','.join(f'{s:.2f}' for s in used_scales)}],"
            f" prepared_variants={diagnostics['prepared_variants']},"
            f" color_no_hsv_templates={diagnostics['color_no_hsv_templates']},"
            f" skipped_empty_color_masks={diagnostics['skipped_empty_color_masks']},"
            f" raw_peaks={diagnostics['raw_peaks']} per_scale={_fmt_scale_hist(raw_peaks_by_scale)},"
            f" gray_near_threshold={diagnostics['gray_near_threshold_recovery_candidates']}/"
            f"{diagnostics['gray_near_threshold_recovery_accepted']}/"
            f"{diagnostics['gray_near_threshold_recovery_rejected']},"
            f" gray_interrupted={diagnostics['gray_interrupted_recovery_candidates']}/"
            f"{diagnostics['gray_interrupted_recovery_accepted']}/"
            f"{diagnostics['gray_interrupted_recovery_rejected']},"
            f" raw_peaks_by_mask={scan_mask_summary},"
            f" raw_budget={diagnostics['raw_budget_hits']}(-{diagnostics['raw_budget_removed']}),"
            f" raw_budget_protected={int(gray_budget_profile.get('geometryProtectedKept', 0))}/{int(gray_budget_profile.get('geometryProtected', 0))},"
            f" raw_after_prefilter={diagnostics['raw_prefilter_hits']}(-{diagnostics['raw_prefilter_removed']})"  # noqa: E501
            f" candidates_per_scale={_fmt_scale_hist(candidates_by_scale)},"
            f" validated_template_hits={diagnostics['validated_template_hits']}"
            f" accepted_per_scale={_fmt_scale_hist(accepted_by_scale)},"
            f" rejects=nms:{diagnostics['raw_prefilter_removed']}|suppression(budget):{diagnostics['raw_budget_removed']}|validation:[{rejection_summary}],"  # noqa: E501
            f" promoted_targeted_hits={diagnostics['promoted_targeted_hits']},"
            f" parent_search_input_hits={diagnostics['parent_search_input_hits']},"
            f" parent_search_candidates={diagnostics['parent_search_candidates']},"
            f" promoted_parent_search_hits={diagnostics['promoted_parent_search_hits']},"
            f" pdf_text_hits={diagnostics['pdf_text_hits']},"
            f" after_prefilter={diagnostics['prefilter_hits']},"
            f" pre_parent_clusters={diagnostics['pre_parent_clusters']},"
            f" final_clusters={diagnostics['final_hits']},"
            f" rois={diagnostics['search_rois']} full_scan_templates={diagnostics['full_scan_templates']},"  # noqa: E501
            f" scan_tasks={diagnostics['scan_tasks']} task_rois={diagnostics['scan_task_rois']},"
            f" gray_suppress={diagnostics['gray_suppressed_pixels']}px({diagnostics['gray_suppressed_ratio']:.0%}),"
            f" gray_dark<{diagnostics['gray_dark_threshold']}={diagnostics['gray_dark_ink_pixels']}px,"
            f" gray_zone<{diagnostics['gray_dark_zone_threshold']}={diagnostics['gray_dark_zone_pixels']}px,"
            f" gray_evidence<{diagnostics['gray_dark_evidence_threshold']}={diagnostics['gray_dark_evidence_pixels']}px,"
            f" gray_zone_suppress={diagnostics['gray_dark_zone_suppressed_pixels']}px({diagnostics['gray_dark_zone_suppressed_ratio']:.0%}),"
            f" gray_dark_suppress={diagnostics['gray_dark_suppressed_pixels']}px({diagnostics['gray_dark_suppressed_ratio']:.0%}),"
            f" threads={scan_result.scan_strategy}:scan:{scan_workers}/{scan_result.configured_scan_workers}|post:{postprocess_workers}/{DETECTOR_POSTPROCESS_MAX_WORKERS}|opencv:{scan_result.opencv_threads},"  # noqa: E501
            f" timings_ms="
            f"pdf_text:{timings_ms['pdf_text']:.0f}|"
            f"prepare:{timings_ms['prepare']:.0f}|"
            f"gray_suppress:{timings_ms['gray_suppress']:.0f}|"
            f"scan:{timings_ms['scan']:.0f}|"
            f"raw_prefilter:{timings_ms['raw_prefilter']:.0f}|"
            f"validation_targeted:{timings_ms['validation_targeted']:.0f}|"
            f"prefilter:{timings_ms['prefilter']:.0f}|"
            f"pre_parent_clustering:{timings_ms['pre_parent_clustering']:.0f}|"
            f"parent_search:{timings_ms['parent_search']:.0f}|"
            f"clustering:{timings_ms['clustering']:.0f}"
        )

    def _color_visual_bbox(hit: CandidateHit) -> tuple[int, int, int, int] | None:
        if detector_profile != "color" or hit.source == "pdf_text" or hit.dominant_hsv is None:
            return None
        plan_mask = plan_masks_by_template.get(hit.template_id)
        if plan_mask is None:
            return None
        x, y, w, h = [int(value) for value in hit.bbox]
        image_h, image_w = plan_mask.shape[:2]
        x0 = max(0, min(image_w, x))
        y0 = max(0, min(image_h, y))
        x1 = max(0, min(image_w, x + w))
        y1 = max(0, min(image_h, y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        roi = plan_mask[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if len(xs) < 4:
            return None
        pad = 3
        vx0 = max(x0, x0 + int(xs.min()) - pad)
        vy0 = max(y0, y0 + int(ys.min()) - pad)
        vx1 = min(x1, x0 + int(xs.max()) + 1 + pad)
        vy1 = min(y1, y0 + int(ys.max()) + 1 + pad)
        visual_w = vx1 - vx0
        visual_h = vy1 - vy0
        if visual_w <= 0 or visual_h <= 0:
            return None
        # Avoid replacing a normal symbol box with a single antialiased speck.
        if visual_w * visual_h < max(16, int(0.03 * w * h)):
            return None
        return (vx0, vy0, visual_w, visual_h)

    per_template: dict[int, list[Detection]] = {}
    _progress("format_results", 99, "Formatowanie wynikow")
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
            roi_strategy=hit.roi_strategy,
            visual_bbox=_color_visual_bbox(hit),
        )
        per_template.setdefault(hit.template_id, []).append(detection)

    results: list[DetectionResult] = []
    for template_id, detections in per_template.items():
        detections.sort(key=lambda det: (det.verification_score, det.confidence), reverse=True)

        count = len(detections)
        # Gray PDFs use explicit legend/text/plan-zone exclusions.  Blindly
        # subtracting one hit per symbol here can hide valid plan detections
        # that already passed validation, especially when the legend is outside
        # the selected plan zone.
        blind_legend_subtract = (
            detector_profile != "gray" and subtract_legend and legend_rect is None
        )
        if blind_legend_subtract:
            count = max(0, count - 1)

        if count <= 0:
            continue

        results.append(
            DetectionResult(
                symbol_name=templates[template_id].name,
                count=count,
                color="#22c55e",
                detections=(
                    detections[:count] if blind_legend_subtract else detections
                ),
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


