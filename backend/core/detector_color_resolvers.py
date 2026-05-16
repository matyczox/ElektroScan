"""Color-path postprocess resolvers for detector candidates.

These helpers intentionally avoid project- or coordinate-specific rules. They
operate on generic template/candidate geometry after clustering.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from core.detector_context import (
    DetectionTemplateContext,
    bbox_iom,
    center_distance,
    center_inside,
    expanded_box,
    hue_close,
    token_family,
)
from core.detector_models import CandidateHit, TemplateInfo
from core.detector_postprocess import dedupe_final_hits


def apply_color_postprocess(
    *,
    detector_profile: str,
    final_hits: list[CandidateHit],
    prefiltered_candidates: list[CandidateHit],
    rejected_hits: list[CandidateHit],
    rejection_reason_by_hit_id: dict[int, str],
    pdf_candidates: list[CandidateHit],
    removed_pdf_text_fallbacks: list[CandidateHit],
    templates: list[TemplateInfo],
    template_context: DetectionTemplateContext,
    plan_masks_by_template: dict[int, np.ndarray],
) -> tuple[list[CandidateHit], dict[str, int]]:
    """Apply behavior-preserving color-family postprocess stages."""

    _template_primary_token = template_context.template_primary_token
    _token_family = token_family
    _template_token_family = template_context.template_token_family
    _template_name = template_context.template_name
    _l_label_group = template_context.l_label_group
    _is_magenta_family_template = template_context.is_magenta_family_template
    _magenta_template_code = template_context.magenta_template_code
    _is_tb11_wave_template = template_context.is_tb11_wave_template
    _center_distance = center_distance
    _center_inside = center_inside
    _bbox_iom = bbox_iom
    _hue_close = hue_close
    _expanded_box = expanded_box

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

    all_candidates = prefiltered_candidates + rejected_hits
    final_hits, color_magenta_reclassed = _reconcile_magenta_family_hits(
        final_hits,
        prefiltered_candidates,
    )
    final_hits, color_magenta_local_rescued = _rescue_magenta_family_hits(
        final_hits,
        all_candidates,
    )
    final_hits, color_family_rescued = _rescue_color_family_hits(
        final_hits,
        all_candidates,
        rejection_reason_by_hit_id,
    )
    final_hits, long_l_over_tb11_promoted = _promote_long_l_over_tb11_conflicts(
        final_hits,
        all_candidates,
    )
    final_hits, tb11_long_l_suppressed = _suppress_tb11_long_l_conflicts(
        final_hits,
        all_candidates,
    )
    final_hits, final_label_disambiguation = _apply_final_label_resolver(
        final_hits,
        final_hits + all_candidates,
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
    return final_hits, {
        "color_magenta_reclassed": color_magenta_reclassed,
        "color_magenta_local_rescued": color_magenta_local_rescued,
        "color_family_rescued": color_family_rescued,
        "long_l_over_tb11_promoted": long_l_over_tb11_promoted,
        "tb11_long_l_suppressed": tb11_long_l_suppressed,
        "final_label_disambiguation": final_label_disambiguation,
        "long_l_over_true_tb11_suppressed": long_l_over_true_tb11_suppressed,
        "weak_short_l_suppressed": weak_short_l_suppressed,
        "duplicate_final_suppressed": duplicate_final_suppressed,
    }
