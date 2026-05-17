"""Long-L and TB11 conflict resolvers for color detections."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from core.detector_models import CandidateHit, TemplateInfo

def rescue_color_family_hits(
    final_hits: list[CandidateHit],
    candidates: list[CandidateHit],
    rejection_reason_by_hit_id: dict[int, str],
    *,
    detector_profile: str,
    _l_label_group,
    _bbox_iom,
    _center_distance,
    _is_tb11_wave_template
):
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
    return _rescue_color_family_hits(final_hits, candidates, rejection_reason_by_hit_id)


def promote_long_l_over_tb11_conflicts(
    final_hits: list[CandidateHit],
    candidates: list[CandidateHit],
    *,
    detector_profile: str,
    _l_label_group,
    _bbox_iom,
    _center_distance,
    _is_tb11_wave_template
):
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
    return _promote_long_l_over_tb11_conflicts(final_hits, candidates)


def suppress_tb11_long_l_conflicts(
    final_hits: list[CandidateHit],
    candidates: list[CandidateHit],
    *,
    detector_profile: str,
    _l_label_group,
    _bbox_iom,
    _center_distance,
    _is_tb11_wave_template
):
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
    return _suppress_tb11_long_l_conflicts(final_hits, candidates)


def suppress_long_l_rescue_over_tb11_waves(
    final_hits: list[CandidateHit],
    label_hits: list[CandidateHit],
    *,
    detector_profile: str,
    plan_masks_by_template: dict[int, np.ndarray],
    _l_label_group,
    _template_token_family,
    _is_tb11_wave_template,
    _bbox_iom,
    _center_distance,
    _center_inside,
    _expanded_box,
    _hue_close
):
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
    return _suppress_long_l_rescue_over_tb11_waves(final_hits, label_hits)
