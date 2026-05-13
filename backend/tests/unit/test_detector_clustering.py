"""Tests for clustering helpers — bbox metrics, center helpers, overlap detection."""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.detector_clustering import (
    _bbox_metrics,
    _box_center,
    _center_inside_box,
    _candidate_rank_key,
    _select_cluster_winner,
)
from core.detector_models import CandidateHit


# ---------------------------------------------------------------------------
# _bbox_metrics
# ---------------------------------------------------------------------------

class TestBboxMetrics:
    def test_full_overlap_identical_boxes(self):
        box = (10, 10, 50, 50)
        inter_area, iou, iom, norm_dist = _bbox_metrics(box, box)
        assert inter_area == 2500
        assert iou == pytest.approx(1.0)
        assert iom == pytest.approx(1.0)
        assert norm_dist == pytest.approx(0.0)

    def test_no_overlap(self):
        box_a = (0, 0, 10, 10)
        box_b = (20, 20, 10, 10)
        inter_area, iou, iom, norm_dist = _bbox_metrics(box_a, box_b)
        assert inter_area == 0
        assert iou == pytest.approx(0.0)
        assert iom == pytest.approx(0.0)
        assert norm_dist > 1.0

    def test_partial_overlap(self):
        box_a = (0, 0, 20, 20)
        box_b = (10, 10, 20, 20)
        inter_area, iou, iom, _ = _bbox_metrics(box_a, box_b)
        assert inter_area == 100
        assert iou == pytest.approx(100 / 700, abs=1e-6)
        assert iom == pytest.approx(100 / 400, abs=1e-6)

    def test_inner_box_fully_contained(self):
        outer = (0, 0, 40, 40)
        inner = (10, 10, 10, 10)
        inter_area, iou, iom, _ = _bbox_metrics(outer, inner)
        assert inter_area == 100
        assert iom == pytest.approx(1.0), "inner is fully inside outer → IoM should be 1"
        assert iou < 1.0

    def test_iou_symmetry(self):
        box_a = (0, 0, 30, 30)
        box_b = (15, 15, 30, 30)
        _, iou_ab, _, _ = _bbox_metrics(box_a, box_b)
        _, iou_ba, _, _ = _bbox_metrics(box_b, box_a)
        assert iou_ab == pytest.approx(iou_ba)

    def test_touching_edges_no_overlap(self):
        box_a = (0, 0, 10, 10)
        box_b = (10, 0, 10, 10)
        inter_area, iou, iom, _ = _bbox_metrics(box_a, box_b)
        assert inter_area == 0
        assert iou == pytest.approx(0.0)

    def test_iou_bounded_0_1(self):
        for _ in range(10):
            x1, y1, w1, h1 = np.random.randint(0, 100, 4).tolist()
            x2, y2, w2, h2 = np.random.randint(0, 100, 4).tolist()
            w1, h1, w2, h2 = max(1, w1), max(1, h1), max(1, w2), max(1, h2)
            _, iou, iom, _ = _bbox_metrics((x1, y1, w1, h1), (x2, y2, w2, h2))
            assert 0.0 <= iou <= 1.0, f"IoU out of range: {iou}"
            assert 0.0 <= iom <= 1.0, f"IoM out of range: {iom}"


# ---------------------------------------------------------------------------
# _box_center
# ---------------------------------------------------------------------------

class TestBoxCenter:
    def test_simple(self):
        cx, cy = _box_center((10, 20, 30, 40))
        assert cx == pytest.approx(25.0)
        assert cy == pytest.approx(40.0)

    def test_origin(self):
        cx, cy = _box_center((0, 0, 10, 10))
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)

    def test_unit_box(self):
        cx, cy = _box_center((5, 5, 1, 1))
        assert cx == pytest.approx(5.5)
        assert cy == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# _center_inside_box
# ---------------------------------------------------------------------------

class TestCenterInsideBox:
    def test_center_inside(self):
        assert _center_inside_box((50.0, 50.0), (0, 0, 100, 100))

    def test_center_outside(self):
        assert not _center_inside_box((200.0, 200.0), (0, 0, 100, 100))

    def test_exact_corner_with_margin(self):
        box = (10, 10, 80, 80)
        # point at (10, 10) = box origin; margin extends inward 4px → still inside
        assert _center_inside_box((10.0, 10.0), box, margin_ratio=0.05)

    def test_just_outside_no_margin(self):
        box = (10, 10, 80, 80)
        # point at (9, 10) — one pixel left of box left edge
        assert not _center_inside_box((9.0, 10.0), box, margin_ratio=0.0)

    def test_just_inside_no_margin(self):
        box = (10, 10, 80, 80)
        assert _center_inside_box((10.0, 10.0), box, margin_ratio=0.0)


# ---------------------------------------------------------------------------
# _candidate_rank_key
# ---------------------------------------------------------------------------

def _make_hit(**kwargs) -> CandidateHit:
    mask = np.zeros((10, 10), dtype=np.uint8)
    defaults = dict(
        template_id=0,
        scale=1.0,
        rotation=0,
        mirrored=False,
        transformed_mask=mask,
        content_mask=None,
        pixel_count=100,
        content_pixel_count=0,
        content_bbox=None,
        bbox=(0, 0, 30, 30),
        match_score=0.5,
        dominant_hsv=None,
    )
    defaults.update(kwargs)
    return CandidateHit(**defaults)


class TestCandidateRankKey:
    def test_higher_verification_wins(self):
        weak = _make_hit(verification_score=0.5)
        strong = _make_hit(verification_score=0.9)
        assert _candidate_rank_key(strong) > _candidate_rank_key(weak)

    def test_text_label_uses_content_score_first(self):
        hit = _make_hit(is_text_label=True, content_score=0.95, verification_score=0.6)
        key = _candidate_rank_key(hit)
        assert key[0] == pytest.approx(0.95), "first key element must be content_score for text labels"

    def test_non_text_label_uses_verification_first(self):
        hit = _make_hit(is_text_label=False, verification_score=0.8, content_score=0.99)
        key = _candidate_rank_key(hit)
        assert key[0] == pytest.approx(0.8), "first key element must be verification_score for non-labels"


class TestColorFamilyParentSelection:
    def test_prefers_switch_10_parent_over_local_11_core(self):
        child_11 = _make_hit(
            template_id=11,
            bbox=(1360, 1530, 32, 37),
            match_score=0.808,
            verification_score=0.758,
            coverage=0.921,
            purity=0.842,
            context_purity=0.420,
            dominant_hsv=(60, 255, 221),
        )
        parent_10 = _make_hit(
            template_id=10,
            bbox=(1354, 1518, 39, 49),
            match_score=0.751,
            verification_score=0.708,
            coverage=0.842,
            purity=0.819,
            context_purity=0.395,
            dominant_hsv=(60, 255, 221),
        )

        winner = _select_cluster_winner(
            [child_11, parent_10],
            parent_ids_by_child={11: {10}},
        )

        assert winner is parent_10

    def test_prefers_switch_12_parent_even_when_11_has_higher_raw_score(self):
        child_11 = _make_hit(
            template_id=11,
            bbox=(2292, 1546, 37, 32),
            match_score=0.864,
            verification_score=0.821,
            coverage=0.921,
            purity=0.907,
            context_purity=0.561,
            dominant_hsv=(60, 255, 221),
        )
        parent_12 = _make_hit(
            template_id=12,
            bbox=(2292, 1547, 49, 32),
            match_score=0.744,
            verification_score=0.722,
            coverage=0.861,
            purity=0.827,
            context_purity=0.456,
            dominant_hsv=(60, 255, 221),
        )

        winner = _select_cluster_winner(
            [child_11, parent_12],
            parent_ids_by_child={11: {12}},
        )

        assert winner is parent_12


class TestColorTextLabelSelection:
    def test_keeps_fuller_text_label_when_tighter_sibling_loses_content_context(self):
        full_int = _make_hit(
            template_id=18,
            bbox=(621, 1136, 66, 32),
            match_score=0.526,
            verification_score=0.695,
            coverage=0.659,
            purity=0.776,
            context_purity=0.562,
            content_score=0.750,
            dominant_hsv=(60, 255, 221),
            is_text_label=True,
            mirrored=True,
        )
        tighter_tv = _make_hit(
            template_id=19,
            bbox=(623, 1138, 54, 28),
            match_score=0.716,
            verification_score=0.605,
            coverage=0.786,
            purity=0.869,
            context_purity=0.374,
            content_score=0.535,
            dominant_hsv=(60, 255, 221),
            is_text_label=True,
            mirrored=True,
        )

        winner = _select_cluster_winner(
            [full_int, tighter_tv],
            parent_ids_by_child={},
        )

        assert winner is full_int
