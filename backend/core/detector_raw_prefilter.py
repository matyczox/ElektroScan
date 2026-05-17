"""Raw candidate prefiltering before validation/clustering."""

from __future__ import annotations

import cv2
import numpy as np

from core.detector_config import (
    PREFILTER_NMS_IOU_THRESHOLD,
    PREFILTER_NMS_MIN_CANDIDATES,
    RAW_PREFILTER_CENTER_DISTANCE_RATIO,
    RAW_PREFILTER_IOM_THRESHOLD,
    RAW_PREFILTER_IOU_THRESHOLD,
    RAW_PREFILTER_LOCAL_CENTER_DISTANCE_RATIO,
    RAW_PREFILTER_LOCAL_IOM_THRESHOLD,
    RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES,
    RAW_PREFILTER_MIN_CANDIDATES,
)
from core.detector_geometry import _bbox_metrics, _box_center
from core.detector_models import CandidateHit


def _prefilter_candidates(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Use a conservative NMS only when the candidate set becomes very large."""

    if len(candidates) < PREFILTER_NMS_MIN_CANDIDATES:
        return candidates

    boxes = [list(hit.bbox) for hit in candidates]
    scores = [
        float(
            hit.content_score
            if hit.is_text_label and hit.content_score > 0.0
            else hit.verification_score or hit.match_score
        )
        for hit in candidates
    ]

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        score_threshold=0.0,
        nms_threshold=PREFILTER_NMS_IOU_THRESHOLD,
    )
    if len(indices) == 0:
        return candidates

    keep = set(indices.flatten().tolist())
    return [candidate for idx, candidate in enumerate(candidates) if idx in keep]


def _raw_candidates_overlap_strongly(left: CandidateHit, right: CandidateHit) -> bool:
    """Detect duplicate raw peaks for the same template before expensive validation."""

    if left.is_text_label or right.is_text_label:
        return False

    inter_area, iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    if inter_area <= 0:
        return False

    if iou >= RAW_PREFILTER_IOU_THRESHOLD:
        return True

    return (
        iom >= RAW_PREFILTER_IOM_THRESHOLD
        and center_distance <= RAW_PREFILTER_CENTER_DISTANCE_RATIO
    )


def _should_keep_gray_scale_alternative(
    candidate: CandidateHit,
    existing: CandidateHit,
    kept: list[CandidateHit],
) -> bool:
    """Keep a nearby gray scale variant until full geometry validation decides."""

    if candidate.dominant_hsv is not None or existing.dominant_hsv is not None:
        return False
    if candidate.is_text_label or existing.is_text_label:
        return False
    if candidate.template_id != existing.template_id:
        return False
    if abs(float(candidate.scale) - float(existing.scale)) < 0.09:
        return False

    cx, cy = _box_center(candidate.bbox)
    candidate_diag = max(1.0, float(np.hypot(candidate.bbox[2], candidate.bbox[3])))
    nearby_alternatives = 0
    for other in kept:
        if other.template_id != candidate.template_id:
            continue
        ox, oy = _box_center(other.bbox)
        if float(np.hypot(cx - ox, cy - oy)) <= candidate_diag * 0.25:
            nearby_alternatives += 1

    return nearby_alternatives < 3


def _raw_candidates_are_local_alternatives(left: CandidateHit, right: CandidateHit) -> bool:
    """Return true for same-template gray alternatives in one local place."""

    if left.dominant_hsv is not None or right.dominant_hsv is not None:
        return False
    if left.template_id != right.template_id:
        return False
    if left.source != right.source:
        return False
    if left.roi_strategy != right.roi_strategy:
        return False

    inter_area, _iou, iom, center_distance = _bbox_metrics(left.bbox, right.bbox)
    return (
        inter_area > 0
        and iom >= RAW_PREFILTER_LOCAL_IOM_THRESHOLD
        and center_distance <= RAW_PREFILTER_LOCAL_CENTER_DISTANCE_RATIO
    )


def _limit_same_template_raw_local_alternatives(
    candidates: list[CandidateHit],
) -> list[CandidateHit]:
    """Cap duplicate same-template raw alternatives before full validation.

    This is intentionally not a class competition.  It only limits many raw
    variants from the same template/source/ROI strategy in the same local
    place, while keeping several alternatives so validation can still choose
    the scale/rotation that carries real evidence.
    """

    if len(candidates) < RAW_PREFILTER_MIN_CANDIDATES:
        return candidates
    if RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES <= 0:
        return candidates

    grouped: dict[int, list[CandidateHit]] = {}
    passthrough: list[CandidateHit] = []
    for candidate in candidates:
        if candidate.dominant_hsv is not None:
            passthrough.append(candidate)
            continue
        grouped.setdefault(candidate.template_id, []).append(candidate)

    grid_cell_px = 32
    filtered: list[CandidateHit] = list(passthrough)
    for template_hits in grouped.values():
        kept: list[CandidateHit] = []
        kept_grid: dict[tuple[str, str, int, int], list[CandidateHit]] = {}
        for candidate in sorted(template_hits, key=lambda hit: hit.match_score, reverse=True):
            _x, _y, w, h = candidate.bbox
            cx, cy = _box_center(candidate.bbox)
            cell_x = int(cx // grid_cell_px)
            cell_y = int(cy // grid_cell_px)
            radius_px = (
                max(1.0, float(np.hypot(w, h)))
                * RAW_PREFILTER_LOCAL_CENTER_DISTANCE_RATIO
            )
            cell_radius = int(np.ceil(radius_px / grid_cell_px)) + 1
            key_prefix = (str(candidate.source), str(candidate.roi_strategy))

            local_alternatives = 0
            for dx in range(-cell_radius, cell_radius + 1):
                if local_alternatives >= RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES:
                    break
                for dy in range(-cell_radius, cell_radius + 1):
                    bucket = kept_grid.get(
                        (key_prefix[0], key_prefix[1], cell_x + dx, cell_y + dy)
                    )
                    if not bucket:
                        continue
                    for existing in bucket:
                        if _raw_candidates_are_local_alternatives(candidate, existing):
                            local_alternatives += 1
                            if local_alternatives >= RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES:
                                break
                    if local_alternatives >= RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES:
                        break
            if local_alternatives >= RAW_PREFILTER_LOCAL_MAX_ALTERNATIVES:
                continue
            kept.append(candidate)
            kept_grid.setdefault(
                (key_prefix[0], key_prefix[1], cell_x, cell_y),
                [],
            ).append(candidate)
        filtered.extend(kept)

    filtered.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.match_score))
    return filtered


def _prefilter_raw_template_hits(candidates: list[CandidateHit]) -> list[CandidateHit]:
    """Drop near-identical raw candidates only inside the same template family member."""

    if len(candidates) < RAW_PREFILTER_MIN_CANDIDATES:
        return candidates

    grouped: dict[int, list[CandidateHit]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.template_id, []).append(candidate)

    grid_cell_px = 48
    filtered: list[CandidateHit] = []
    for template_hits in grouped.values():
        kept: list[CandidateHit] = []
        kept_grid: dict[tuple[int, int], list[CandidateHit]] = {}
        for candidate in sorted(template_hits, key=lambda hit: hit.match_score, reverse=True):
            _x, _y, w, h = candidate.bbox
            cx, cy = _box_center(candidate.bbox)
            cell_x = int(cx // grid_cell_px)
            cell_y = int(cy // grid_cell_px)
            radius_px = max(1.0, float(np.hypot(w, h)))
            cell_radius = int(np.ceil(radius_px / grid_cell_px)) + 1
            overlapping_existing = None
            for dx in range(-cell_radius, cell_radius + 1):
                if overlapping_existing is not None:
                    break
                for dy in range(-cell_radius, cell_radius + 1):
                    bucket = kept_grid.get((cell_x + dx, cell_y + dy))
                    if not bucket:
                        continue
                    overlapping_existing = next(
                        (
                            existing
                            for existing in bucket
                            if _raw_candidates_overlap_strongly(candidate, existing)
                        ),
                        None,
                    )
                    if overlapping_existing is not None:
                        break
            if overlapping_existing is not None and not _should_keep_gray_scale_alternative(
                candidate,
                overlapping_existing,
                kept,
            ):
                continue
            kept.append(candidate)
            kept_grid.setdefault((cell_x, cell_y), []).append(candidate)
        filtered.extend(kept)

    filtered.sort(key=lambda hit: (hit.bbox[1], hit.bbox[0], -hit.match_score))
    return filtered


def _dedupe_raw_template_hits_before_validation(
    candidates: list[CandidateHit],
) -> list[CandidateHit]:
    """Run conservative same-template raw de-duplication before validation."""

    return _limit_same_template_raw_local_alternatives(candidates)
