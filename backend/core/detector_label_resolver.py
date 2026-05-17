"""Exact-label disambiguation resolver for color detections."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from core.detector_models import CandidateHit, TemplateInfo

def apply_final_label_resolver(
    final_hits: list[CandidateHit],
    candidates: list[CandidateHit],
    label_hits: list[CandidateHit],
    *,
    detector_profile: str,
    templates: list[TemplateInfo],
    plan_masks_by_template: dict[int, np.ndarray],
    _template_primary_token,
    _token_family,
    _template_token_family,
    _l_label_group,
    _bbox_iom,
    _center_distance,
    _center_inside,
    _expanded_box,
    _hue_close
):
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
    return _apply_final_label_resolver(final_hits, candidates, label_hits)


def suppress_weak_short_l_fragments(
    final_hits: list[CandidateHit],
    *,
    detector_profile: str,
    _template_name,
    _center_distance
):
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
    return _suppress_weak_short_l_fragments(final_hits)
