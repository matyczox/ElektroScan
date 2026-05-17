"""Zone and rectangle helpers for API requests."""

from __future__ import annotations

from typing import Optional

from api_models import LegendZone


def _zone_to_rect(zone: Optional[LegendZone]) -> tuple[int, int, int, int] | None:
    if zone is None:
        return None
    rect = (
        int(round(zone.x)),
        int(round(zone.y)),
        int(round(zone.width)),
        int(round(zone.height)),
    )
    if rect[2] <= 0 or rect[3] <= 0:
        return None
    return rect


def _clamp_rect_to_image(
    rect: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    image_h, image_w = image_shape[:2]
    x, y, w, h = rect
    x1 = max(0, min(image_w, x))
    y1 = max(0, min(image_h, y))
    x2 = max(0, min(image_w, x + w))
    y2 = max(0, min(image_h, y + h))
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _outside_plan_zone_rects(
    plan_zone: Optional[LegendZone],
    image_shape: tuple[int, ...],
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
    rect = _zone_to_rect(plan_zone)
    if rect is None:
        return None, []

    clamped = _clamp_rect_to_image(rect, image_shape)
    if clamped is None:
        return None, []

    image_h, image_w = image_shape[:2]
    x, y, w, h = clamped
    pieces = [
        (0, 0, image_w, y),
        (0, y + h, image_w, image_h - (y + h)),
        (0, y, x, h),
        (x + w, y, image_w - (x + w), h),
    ]
    return clamped, [piece for piece in pieces if piece[2] > 1 and piece[3] > 1]


def _extract_exclude_rects_from_request(
    body,
    image_shape: tuple[int, ...],
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int, int]],
    tuple[int, int, int, int] | None,
    tuple[int, int, int, int] | None,
    list[tuple[int, int, int, int]],
]:
    exclude_rects: list[tuple[int, int, int, int]] = []
    manual_exclude_rects: list[tuple[int, int, int, int]] = []
    legend_rect = None
    plan_zone_rect = None
    plan_zone_outside_rects: list[tuple[int, int, int, int]] = []

    if body and body.excluded_zones:
        for zone in body.excluded_zones:
            try:
                rect = (int(zone["x"]), int(zone["y"]), int(zone["width"]), int(zone["height"]))
                exclude_rects.append(rect)
                manual_exclude_rects.append(rect)
            except (KeyError, ValueError):
                pass
    if body and body.legend_zone:
        legend_rect = _zone_to_rect(body.legend_zone)
    if legend_rect is not None:
        exclude_rects.append(legend_rect)
    if body and body.plan_zone:
        plan_zone_rect, plan_zone_outside_rects = _outside_plan_zone_rects(
            body.plan_zone,
            image_shape,
        )
        if plan_zone_rect is not None:
            exclude_rects.extend(plan_zone_outside_rects)

    return (
        exclude_rects,
        manual_exclude_rects,
        legend_rect,
        plan_zone_rect,
        plan_zone_outside_rects,
    )
