"""Socket-family geometry/evidence resolver for color detections."""

from __future__ import annotations

from dataclasses import replace
import re

from core.detector_models import CandidateHit, TemplateInfo
from core.detector_pdf import PdfWordBox

_SOCKET_CODE_RE = re.compile(r"^(0[1-5])_sym_0[1-5]$", re.IGNORECASE)


def _socket_code(template: TemplateInfo) -> int | None:
    match = _SOCKET_CODE_RE.match(template.name)
    return int(match.group(1)) if match else None


def _center_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return (((lx + lw / 2) - (rx + rw / 2)) ** 2 + ((ly + lh / 2) - (ry + rh / 2)) ** 2) ** 0.5


def _expanded(
    bbox: tuple[int, int, int, int],
    *,
    pad_x: float,
    pad_y: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    px = int(round(pad_x))
    py = int(round(pad_y))
    return (x - px, y - py, w + 2 * px, h + 2 * py)


def _center_inside(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    cx = ix + iw / 2
    cy = iy + ih / 2
    return ox <= cx <= ox + ow and oy <= cy <= oy + oh


def _bbox_iom(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1 = max(lx, rx)
    iy1 = max(ly, ry)
    ix2 = min(lx + lw, rx + rw)
    iy2 = min(ly + lh, ry + rh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1, min(lw * lh, rw * rh))


def _quality(hit: CandidateHit) -> float:
    return (
        float(hit.verification_score) * 1.20
        + float(hit.match_score) * 0.30
        + float(hit.coverage) * 0.20
        + float(hit.purity) * 0.18
        + float(hit.context_purity) * 0.08
    )


def _local_quality(hit: CandidateHit, anchor: CandidateHit) -> float:
    max_dim = max(1.0, float(max(anchor.bbox[2], anchor.bbox[3])))
    distance_penalty = min(1.0, _center_distance(hit.bbox, anchor.bbox) / max_dim) * 0.45
    return _quality(hit) - distance_penalty


def _word_near(
    words: list[PdfWordBox],
    bbox: tuple[int, int, int, int],
    tokens: set[str],
    *,
    pad_x: float,
    pad_y: float,
) -> bool:
    search_box = _expanded(bbox, pad_x=pad_x, pad_y=pad_y)
    return any(token in tokens and _center_inside(word_bbox, search_box) for token, word_bbox in words)


def _pair_digit_near_socket(
    words: list[PdfWordBox],
    bbox: tuple[int, int, int, int],
) -> bool:
    x, y, w, h = bbox
    search_box = _expanded(bbox, pad_x=14.0, pad_y=8.0)
    for token, word_bbox in words:
        if token != "2" or not _center_inside(word_bbox, search_box):
            continue
        wx, wy, ww, wh = word_bbox
        cx = wx + ww / 2
        cy = wy + wh / 2
        # A pair marker belongs to the socket when it is inside or just on the
        # symbol edge. Nearby room/circuit labels can also contain standalone
        # "2"; those often sit visibly left/above the symbol and must not
        # force a double-socket class.
        if (x - 14) <= cx <= (x + w + 8) and (y - 8) <= cy <= (y + h + 8):
            return True
    return False


def _pair_digit_inside_socket(
    words: list[PdfWordBox],
    bbox: tuple[int, int, int, int],
) -> bool:
    x, y, w, h = bbox
    for token, word_bbox in words:
        wx, wy, ww, wh = word_bbox
        cx = wx + ww / 2
        cy = wy + wh / 2
        inside = (x - 2) <= cx <= (x + w + 2) and (y - 4) <= cy <= (y + h + 4)
        if not inside:
            continue
        if token == "2":
            return True
        if token.startswith("2") and len(token) <= 4:
            return True
    return False


def _pair_digit_in_local_stack(
    words: list[PdfWordBox],
    bbox: tuple[int, int, int, int],
) -> bool:
    search_box = _expanded(bbox, pad_x=40.0, pad_y=34.0)
    return any(token == "2" and _center_inside(word_bbox, search_box) for token, word_bbox in words)


def _socket_template_ids(templates: list[TemplateInfo]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for template_id, template in enumerate(templates):
        code = _socket_code(template)
        if code is not None:
            mapping[code] = template_id
    return mapping


def _local_socket_candidates(
    hit: CandidateHit,
    candidates: list[CandidateHit],
    templates: list[TemplateInfo],
) -> list[CandidateHit]:
    local: list[CandidateHit] = []
    max_dim = max(hit.bbox[2], hit.bbox[3])
    for candidate in candidates:
        if candidate.source == "pdf_text":
            continue
        if not (0 <= candidate.template_id < len(templates)):
            continue
        if _socket_code(templates[candidate.template_id]) is None:
            continue
        if (
            _bbox_iom(hit.bbox, candidate.bbox) >= 0.24
            or _center_distance(hit.bbox, candidate.bbox) <= max(28.0, max_dim * 0.40)
        ):
            local.append(candidate)
    return local


def _same_socket_place(left: CandidateHit, right: CandidateHit) -> bool:
    return (
        _bbox_iom(left.bbox, right.bbox) >= 0.42
        or _center_distance(left.bbox, right.bbox) <= max(18.0, min(left.bbox[2], left.bbox[3]) * 0.45)
    )


def _prefer_socket_in_same_place(
    group: list[CandidateHit],
    templates: list[TemplateInfo],
    pdf_word_boxes: list[PdfWordBox],
) -> CandidateHit:
    by_code: dict[int, list[CandidateHit]] = {}
    for hit in group:
        code = _socket_code(templates[hit.template_id])
        if code is not None:
            by_code.setdefault(code, []).append(hit)

    def best(code: int) -> CandidateHit | None:
        hits = by_code.get(code) or []
        return max(hits, key=_quality, default=None)

    data_hit = best(5)
    if data_hit is not None and _word_near(
        pdf_word_boxes,
        data_hit.bbox,
        {"DATA"},
        pad_x=max(24.0, data_hit.bbox[2] * 0.55),
        pad_y=max(24.0, data_hit.bbox[3] * 0.55),
    ):
        return data_hit

    for double_code, single_code in ((4, 3), (2, 1)):
        double_hit = best(double_code)
        if double_hit is not None and (
            _pair_digit_inside_socket(pdf_word_boxes, double_hit.bbox)
            or _pair_digit_near_socket(pdf_word_boxes, double_hit.bbox)
        ):
            return double_hit
        single_hit = best(single_code)
        if single_hit is not None and double_hit is not None:
            if _quality(double_hit) >= _quality(single_hit) + 0.08:
                return double_hit
            return single_hit

    return max(group, key=_quality)


def _dedupe_same_place_sockets(
    hits: list[CandidateHit],
    templates: list[TemplateInfo],
    pdf_word_boxes: list[PdfWordBox],
) -> tuple[list[CandidateHit], int]:
    output: list[CandidateHit] = []
    used: set[int] = set()
    removed = 0
    socket_indexes = [
        index
        for index, hit in enumerate(hits)
        if 0 <= hit.template_id < len(templates) and _socket_code(templates[hit.template_id]) is not None
    ]

    for index, hit in enumerate(hits):
        if index in used:
            continue
        if index not in socket_indexes:
            output.append(hit)
            used.add(index)
            continue

        group_indexes = [
            other_index
            for other_index in socket_indexes
            if other_index not in used and _same_socket_place(hit, hits[other_index])
        ]
        if len(group_indexes) == 1:
            output.append(hit)
            used.add(index)
            continue

        group = [hits[other_index] for other_index in group_indexes]
        winner = _prefer_socket_in_same_place(group, templates, pdf_word_boxes)
        output.append(winner)
        used.update(group_indexes)
        removed += len(group_indexes) - 1

    return output, removed


def _socket_rescue_quality(hit: CandidateHit, templates: list[TemplateInfo]) -> bool:
    """True for a visually solid socket candidate that may have lost clustering."""

    if hit.source == "pdf_text" or not (0 <= hit.template_id < len(templates)):
        return False
    code = _socket_code(templates[hit.template_id])
    if code not in {3, 4}:
        return False
    return (
        hit.match_score >= 0.62
        and hit.verification_score >= 0.62
        and hit.coverage >= 0.76
        and hit.purity >= 0.74
    )


def _rescue_missing_socket_hits(
    hits: list[CandidateHit],
    candidates: list[CandidateHit],
    templates: list[TemplateInfo],
    pdf_word_boxes: list[PdfWordBox],
) -> tuple[list[CandidateHit], int]:
    """Add strong visual 03/04 socket candidates that were swallowed by clusters."""

    rescue_candidates = [
        candidate
        for candidate in candidates
        if _socket_rescue_quality(candidate, templates)
    ]
    if not rescue_candidates:
        return hits, 0

    output = list(hits)
    used: set[int] = set()
    rescued = 0
    for index, candidate in enumerate(rescue_candidates):
        if index in used:
            continue
        group_indexes = [
            other_index
            for other_index, other in enumerate(rescue_candidates)
            if other_index not in used and _same_socket_place(candidate, other)
        ]
        group = [rescue_candidates[other_index] for other_index in group_indexes]
        winner = _prefer_socket_in_same_place(group, templates, pdf_word_boxes)
        target_code = _socket_code(templates[winner.template_id])
        if target_code not in {3, 4}:
            used.update(group_indexes)
            continue
        if target_code == 3 and not _pair_digit_in_local_stack(pdf_word_boxes, winner.bbox):
            used.update(group_indexes)
            continue

        if any(
            0 <= existing.template_id < len(templates)
            and _socket_code(templates[existing.template_id]) in {3, 4}
            and _same_socket_place(existing, winner)
            for existing in output
        ):
            used.update(group_indexes)
            continue

        output.append(winner)
        used.update(group_indexes)
        rescued += 1

    return output, rescued


def _suppress_weak_unlabeled_socket_hits(
    hits: list[CandidateHit],
    templates: list[TemplateInfo],
    pdf_word_boxes: list[PdfWordBox],
) -> tuple[list[CandidateHit], int]:
    """Drop weak hermetic-socket impostors that lack local pair evidence."""

    output: list[CandidateHit] = []
    suppressed = 0
    for hit in hits:
        if not (0 <= hit.template_id < len(templates)):
            output.append(hit)
            continue
        code = _socket_code(templates[hit.template_id])
        weak_hermetic = (
            code in {3, 4}
            and hit.source == "template"
            and hit.match_score < 0.56
            and hit.verification_score < 0.60
            and hit.coverage < 0.68
            and hit.purity < 0.76
        )
        if not weak_hermetic:
            output.append(hit)
            continue
        if code == 4 and (
            _pair_digit_inside_socket(pdf_word_boxes, hit.bbox)
            or _pair_digit_near_socket(pdf_word_boxes, hit.bbox)
        ):
            output.append(hit)
            continue
        suppressed += 1

    return output, suppressed


def resolve_socket_family_hits(
    final_hits: list[CandidateHit],
    candidates: list[CandidateHit],
    *,
    detector_profile: str,
    templates: list[TemplateInfo],
    pdf_word_boxes: list[PdfWordBox],
) -> tuple[list[CandidateHit], int]:
    """Resolve visually similar socket variants by shape candidates plus local labels.

    The resolver is deliberately narrow to templates named ``01_sym_01`` through
    ``05_sym_05``. It never creates a detection from text alone; exact PDF words
    only disambiguate nearby visual candidates.
    """

    if detector_profile != "color":
        return final_hits, 0

    template_ids = _socket_template_ids(templates)
    if len(template_ids) < 2:
        return final_hits, 0

    all_candidates = list(final_hits) + list(candidates)
    output: list[CandidateHit] = []
    changed = 0

    for hit in final_hits:
        if not (0 <= hit.template_id < len(templates)):
            output.append(hit)
            continue
        current_code = _socket_code(templates[hit.template_id])
        if current_code is None:
            output.append(hit)
            continue

        local_candidates = _local_socket_candidates(hit, all_candidates, templates)
        if not local_candidates:
            output.append(hit)
            continue

        best_by_code: dict[int, CandidateHit] = {}
        for candidate in local_candidates:
            code = _socket_code(templates[candidate.template_id])
            if code is None:
                continue
            previous = best_by_code.get(code)
            if previous is None or _local_quality(candidate, hit) > _local_quality(previous, hit):
                best_by_code[code] = candidate

        if not best_by_code:
            output.append(hit)
            continue

        evidence_bbox = max(local_candidates, key=_quality).bbox
        has_data_label = _word_near(
            pdf_word_boxes,
            evidence_bbox,
            {"DATA"},
            pad_x=max(24.0, evidence_bbox[2] * 0.55),
            pad_y=max(24.0, evidence_bbox[3] * 0.55),
        )

        best_plain = max(
            (best_by_code[code] for code in (1, 2, 5) if code in best_by_code),
            key=_quality,
            default=None,
        )
        best_hermetic = max(
            (best_by_code[code] for code in (3, 4) if code in best_by_code),
            key=_quality,
            default=None,
        )
        plain_score = _quality(best_plain) if best_plain is not None else -1.0
        hermetic_score = _quality(best_hermetic) if best_hermetic is not None else -1.0

        if current_code in {3, 4} or hermetic_score >= plain_score + 0.04:
            pair_bbox = best_by_code[4].bbox if 4 in best_by_code else evidence_bbox
            has_pair_digit = _pair_digit_inside_socket(pdf_word_boxes, pair_bbox)
            if not has_pair_digit and 4 not in best_by_code:
                has_pair_digit = _pair_digit_near_socket(pdf_word_boxes, evidence_bbox)
            target_code = 4 if has_pair_digit else 3
        elif has_data_label and 5 in best_by_code and _quality(best_by_code[5]) >= plain_score - 0.03:
            target_code = 5
        else:
            code_1 = best_by_code.get(1)
            code_2 = best_by_code.get(2)
            pair_bbox = code_2.bbox if code_2 is not None else evidence_bbox
            has_pair_digit = _pair_digit_inside_socket(pdf_word_boxes, pair_bbox)
            if not has_pair_digit and code_2 is None:
                has_pair_digit = _pair_digit_near_socket(pdf_word_boxes, evidence_bbox)
            strong_visual_pair = (
                code_1 is not None
                and code_2 is not None
                and _quality(code_2) >= _quality(code_1) - 0.05
                and (code_2.bbox[2] * code_2.bbox[3]) >= (code_1.bbox[2] * code_1.bbox[3]) * 1.18
                and code_2.purity >= 0.70
                and code_2.coverage >= 0.86
            )
            target_code = 2 if (has_pair_digit or strong_visual_pair) else 1

        target_id = template_ids.get(target_code)
        if target_id is None:
            output.append(hit)
            continue

        replacement_source = best_by_code.get(target_code)
        if replacement_source is None:
            replacement_source = best_by_code.get(1) or best_by_code.get(3) or hit

        if hit.template_id == target_id and replacement_source is hit:
            output.append(hit)
            continue

        replacement = replace(
            replacement_source,
            template_id=target_id,
            source=(
                replacement_source.source
                if replacement_source.template_id == target_id
                else "template_label_disambiguation"
            ),
            promoted_from_template_id=(
                replacement_source.template_id
                if replacement_source.template_id != target_id
                else replacement_source.promoted_from_template_id
            ),
            dominant_hsv=templates[target_id].dominant_hsv,
            is_text_label=templates[target_id].is_text_label,
        )
        output.append(replacement)
        changed += 1

    output, rescued = _rescue_missing_socket_hits(
        output,
        all_candidates,
        templates,
        pdf_word_boxes,
    )
    output, weak_suppressed = _suppress_weak_unlabeled_socket_hits(
        output,
        templates,
        pdf_word_boxes,
    )
    output, deduped = _dedupe_same_place_sockets(output, templates, pdf_word_boxes)
    return output, changed + rescued + weak_suppressed + deduped
