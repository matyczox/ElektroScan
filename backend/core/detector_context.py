"""Small shared helpers for detector orchestration.

This module intentionally contains only pure helpers: token/category lookup,
trace input parsing, and basic bbox math. Keeping these outside
``detector_pipeline`` makes the pipeline easier to scan without changing
detector behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from core.detector_models import TemplateInfo


def trace_values_from_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def trace_points_from_value(value: object) -> list[tuple[float, float]]:
    raw_values: list[object]
    if value is None:
        raw_values = []
    elif isinstance(value, str):
        raw_values = [part.strip() for part in value.split(";") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    points: list[tuple[float, float]] = []
    for item in raw_values:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
            continue
        parts = str(item).replace(":", ",").split(",")
        if len(parts) >= 2:
            points.append((float(parts[0].strip()), float(parts[1].strip())))
    return points


def token_family(token: str) -> str:
    match = re.fullmatch(r"([A-Z]+)\d*", str(token or "").upper())
    return match.group(1) if match else ""


def center_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return float(np.hypot((lx + lw / 2.0) - (rx + rw / 2.0), (ly + lh / 2.0) - (ry + rh / 2.0)))


def center_inside(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
    *,
    pad: float = 0.0,
) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    cx = ix + iw / 2.0
    cy = iy + ih / 2.0
    return (ox - pad) <= cx <= (ox + ow + pad) and (oy - pad) <= cy <= (oy + oh + pad)


def bbox_iom(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1 = max(lx, rx)
    iy1 = max(ly, ry)
    ix2 = min(lx + lw, rx + rw)
    iy2 = min(ly + lh, ry + rh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1, min(lw * lh, rw * rh))


def hue_close(left: tuple[int, int, int] | None, right: tuple[int, int, int] | None) -> bool:
    if left is None or right is None:
        return False
    delta = abs(int(left[0]) - int(right[0]))
    return min(delta, 180 - delta) <= 18


def expanded_box(
    bbox: tuple[int, int, int, int],
    *,
    pad_x: float,
    pad_y: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    px = int(round(pad_x))
    py = int(round(pad_y))
    return (x - px, y - py, w + 2 * px, h + 2 * py)


@dataclass(frozen=True)
class DetectionTemplateContext:
    templates: list[TemplateInfo]

    def template_tokens(self, template_id: int) -> tuple[str, ...]:
        if not (0 <= template_id < len(self.templates)):
            return ()
        return tuple(str(token).upper() for token in self.templates[template_id].text_tokens)

    def template_primary_token(self, template_id: int) -> str:
        tokens = self.template_tokens(template_id)
        return tokens[0] if tokens else ""

    def template_token_family(self, template_id: int) -> str:
        return token_family(self.template_primary_token(template_id))

    def template_name(self, template_id: int) -> str:
        if not (0 <= template_id < len(self.templates)):
            return ""
        return self.templates[template_id].name

    def is_visual_pdf_text_blocked(self, template_id: int) -> bool:
        if not (0 <= template_id < len(self.templates)):
            return False
        template = self.templates[template_id]
        if self.template_token_family(template_id) in {"L", "AW", "EW", "TB"}:
            return True
        return self._is_sparse_coded_color_symbol(template)

    def _is_sparse_coded_color_symbol(self, template: TemplateInfo) -> bool:
        """Return true when a legend code labels a sparse pictogram, not text.

        Some color legends put a code like G14/G520 next to a small symbol in
        the same cell. The code is useful evidence, but the final detection
        still needs the pictogram geometry; otherwise PDF text alone becomes a
        fake visual hit. Dense socket blocks such as PEL stay eligible for
        text fallback.
        """

        if template.dominant_hsv is None or template.is_text_label:
            return False
        primary_token = (template.text_tokens[0] if template.text_tokens else "").upper()
        if not re.fullmatch(r"[A-Z]+\d+[A-Z0-9]*", primary_token):
            return False
        height, width = template.mask.shape[:2]
        area = max(1, int(width) * int(height))
        density = float(template.pixel_count) / float(area)
        aspect = max(float(width) / max(1.0, float(height)), float(height) / max(1.0, float(width)))
        return area <= 8_000 and density <= 0.38 and aspect <= 1.85

    def l_label_group(self, template_id: int) -> str:
        name = self.template_name(template_id).upper()
        if re.fullmatch(r"0[1-6]_L[1-6]", name):
            return "block"
        if name in {"07_L7", "10_L10", "11_L11", "12_L12", "13_L13"}:
            return "long"
        return ""

    def is_magenta_family_template(self, template_id: int) -> bool:
        template = self.templates[template_id] if 0 <= template_id < len(self.templates) else None
        return (
            template is not None
            and template.dominant_hsv is not None
            and 135 <= int(template.dominant_hsv[0]) <= 165
            and re.match(r"^\d+_sym_\d+", template.name) is not None
        )

    def magenta_template_code(self, template_id: int) -> int | None:
        match = re.match(r"^\d+_sym_(\d+)", self.template_name(template_id))
        return int(match.group(1)) if match else None

    def is_tb11_wave_template(self, template_id: int) -> bool:
        return self.template_name(template_id).startswith("28_TB11")
