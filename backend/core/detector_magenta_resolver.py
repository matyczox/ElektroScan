"""Magenta family postprocess resolvers for color detections."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from core.detector_models import CandidateHit, TemplateInfo

def reconcile_magenta_family_hits(
    final_hits: list[CandidateHit],
    candidates: list[CandidateHit],
    *,
    detector_profile: str,
    _is_magenta_family_template,
    _bbox_iom,
    _center_distance,
    _center_inside
):
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
    return _reconcile_magenta_family_hits(final_hits, candidates)


def rescue_magenta_family_hits(
    final_hits: list[CandidateHit],
    candidates: list[CandidateHit],
    *,
    detector_profile: str,
    templates: list[TemplateInfo],
    plan_masks_by_template: dict[int, np.ndarray],
    _is_magenta_family_template,
    _magenta_template_code,
    _bbox_iom,
    _center_distance,
    _expanded_box
):
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
    return _rescue_magenta_family_hits(final_hits, candidates)
