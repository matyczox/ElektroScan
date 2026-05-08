"""Tests for shared local candidate selection."""

from __future__ import annotations

from itertools import permutations

import numpy as np

from core.detector_models import CandidateHit
from core.detector_gray import (
    _dedupe_gray_overlapping_alternatives,
    _weak_gray_compact_fragment_loses_to_larger,
)
from core.detector_selection import (
    local_dominates,
    same_physical_place,
    select_local_winners,
)


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
        bbox=(0, 0, 50, 50),
        match_score=0.6,
        dominant_hsv=None,
        verification_score=0.6,
        coverage=0.9,
        purity=0.6,
        context_purity=0.3,
    )
    defaults.update(kwargs)
    return CandidateHit(**defaults)


def _winner_ids(hits: list[CandidateHit]) -> set[int]:
    return {hit.template_id for hit in hits}


def test_gray_local_selection_is_order_independent() -> None:
    winner = _make_hit(
        template_id=8,
        bbox=(0, 0, 60, 60),
        verification_score=0.72,
        match_score=0.70,
        purity=0.66,
    )
    loser = _make_hit(
        template_id=7,
        bbox=(3, 3, 60, 60),
        verification_score=0.61,
        match_score=0.61,
        coverage=0.80,
    )
    separate = _make_hit(template_id=17, bbox=(120, 0, 25, 25), verification_score=0.62)

    expected = {8, 17}
    for ordered in permutations([winner, loser, separate]):
        assert _winner_ids(select_local_winners(list(ordered), mode="gray")) == expected


def test_gray_selection_prefers_non_mirrored_duplicate_tie() -> None:
    normal = _make_hit(template_id=7, bbox=(10, 10, 45, 45), mirrored=False)
    mirrored = _make_hit(template_id=7, bbox=(10, 10, 45, 45), mirrored=True)

    winners = select_local_winners([mirrored, normal], mode="gray")

    assert len(winners) == 1
    assert winners[0].mirrored is False


def test_same_template_shifted_tiny_duplicates_are_same_place() -> None:
    first = _make_hit(template_id=3, bbox=(906, 4751, 12, 28), rotation=270)
    shifted = _make_hit(template_id=3, bbox=(899, 4750, 12, 28), rotation=90)
    separate = _make_hit(template_id=3, bbox=(906, 4774, 12, 28), rotation=270)
    diagonal_neighbor = _make_hit(template_id=3, bbox=(899, 5128, 12, 28), rotation=90)
    upper_neighbor = _make_hit(template_id=3, bbox=(906, 5122, 12, 28), rotation=270)

    assert same_physical_place(first, shifted, mode="gray")
    assert not same_physical_place(first, separate, mode="gray")
    assert not same_physical_place(upper_neighbor, diagonal_neighbor, mode="gray")


def test_gray_selection_does_not_collapse_through_bridge_candidate() -> None:
    left = _make_hit(template_id=1, bbox=(0, 0, 60, 60), verification_score=0.70, purity=0.66)
    bridge = _make_hit(
        template_id=2,
        bbox=(20, 0, 60, 60),
        verification_score=0.58,
        match_score=0.50,
        coverage=0.78,
    )
    right = _make_hit(template_id=3, bbox=(40, 0, 60, 60), verification_score=0.69, purity=0.66)

    winners = select_local_winners([bridge, right, left], mode="gray")

    assert _winner_ids(winners) == {1, 3}


def test_gray_final_dedupe_does_not_collapse_through_shifted_bridge() -> None:
    upper = _make_hit(
        template_id=3,
        bbox=(906, 5122, 12, 28),
        match_score=0.954,
        verification_score=0.821,
        coverage=1.0,
        purity=0.929,
        context_purity=0.259,
        rotation=270,
        scale=0.5,
    )
    bridge = _make_hit(
        template_id=3,
        bbox=(899, 5122, 12, 28),
        match_score=0.934,
        verification_score=0.805,
        coverage=1.0,
        purity=0.929,
        context_purity=0.226,
        rotation=90,
        scale=0.5,
    )
    diagonal = _make_hit(
        template_id=3,
        bbox=(899, 5128, 12, 28),
        match_score=0.878,
        verification_score=0.776,
        coverage=1.0,
        purity=0.929,
        context_purity=0.209,
        rotation=90,
        scale=0.5,
    )

    winners = _dedupe_gray_overlapping_alternatives([bridge, diagonal, upper])

    assert {(hit.bbox[0], hit.bbox[1]) for hit in winners} == {(906, 5122), (899, 5128)}


def test_large_local_winner_dominates_rescue_edge_ghost() -> None:
    large = _make_hit(
        template_id=1,
        bbox=(7005, 1243, 49, 154),
        verification_score=0.887,
        match_score=0.975,
        coverage=0.98,
        purity=0.86,
        context_purity=0.55,
        pixel_count=1400,
    )
    ghost = _make_hit(
        template_id=7,
        bbox=(7034, 1352, 45, 45),
        verification_score=0.668,
        match_score=0.629,
        coverage=0.673,
        purity=0.901,
        context_purity=0.578,
        pixel_count=900,
    )

    assert same_physical_place(large, ghost, mode="gray")
    assert local_dominates(large, ghost, mode="gray")
    assert select_local_winners([ghost, large], mode="gray") == [large]


def test_similar_gray_symbols_need_purity_margin_to_dominate() -> None:
    true_07 = _make_hit(
        template_id=7,
        bbox=(6283, 2534, 60, 60),
        verification_score=0.592,
        match_score=0.573,
        coverage=0.898,
        purity=0.612,
        context_purity=0.315,
    )
    false_08 = _make_hit(
        template_id=8,
        bbox=(6282, 2536, 63, 54),
        verification_score=0.661,
        match_score=0.678,
        coverage=0.989,
        purity=0.625,
        context_purity=0.319,
    )
    true_08 = _make_hit(
        template_id=8,
        bbox=(2542, 5606, 63, 54),
        verification_score=0.686,
        match_score=0.683,
        coverage=0.996,
        purity=0.694,
        context_purity=0.376,
    )
    false_07 = _make_hit(
        template_id=7,
        bbox=(2543, 5604, 60, 60),
        verification_score=0.618,
        match_score=0.587,
        coverage=0.898,
        purity=0.659,
        context_purity=0.375,
    )

    assert not local_dominates(false_08, true_07, mode="gray")
    assert local_dominates(true_08, false_07, mode="gray")


def test_true_e9_07_touching_label_is_not_same_physical_place() -> None:
    true_07 = _make_hit(
        template_id=7,
        bbox=(6283, 1321, 60, 60),
        verification_score=0.619,
        match_score=0.579,
        coverage=0.909,
        purity=0.668,
        context_purity=0.383,
    )
    nearby_label = _make_hit(
        template_id=1,
        bbox=(6230, 1359, 169, 54),
        verification_score=0.887,
        match_score=0.975,
        coverage=0.999,
        purity=0.864,
        context_purity=0.237,
    )

    assert not same_physical_place(true_07, nearby_label, mode="gray")
    assert _winner_ids(select_local_winners([nearby_label, true_07], mode="gray")) == {1, 7}


def test_partial_center_overlap_competes_without_touching_adjacent_label() -> None:
    false_07 = _make_hit(
        template_id=7,
        bbox=(1250, 5175, 50, 50),
        verification_score=0.605,
        match_score=0.627,
        coverage=0.998,
        purity=0.536,
        context_purity=0.213,
    )
    true_12 = _make_hit(
        template_id=12,
        bbox=(1275, 5169, 54, 60),
        verification_score=0.700,
        match_score=0.731,
        coverage=0.998,
        purity=0.686,
        context_purity=0.342,
    )
    adjacent_label = _make_hit(
        template_id=1,
        bbox=(6230, 1359, 169, 54),
        verification_score=0.771,
        match_score=0.776,
        coverage=0.869,
        purity=0.844,
        context_purity=0.338,
        is_text_label=True,
    )
    true_07 = _make_hit(
        template_id=7,
        bbox=(6283, 1321, 60, 60),
        verification_score=0.619,
        match_score=0.579,
        coverage=0.909,
        purity=0.668,
        context_purity=0.383,
    )

    assert not same_physical_place(true_12, false_07, mode="gray")
    assert _weak_gray_compact_fragment_loses_to_larger(false_07, true_12)
    assert not local_dominates(true_12, false_07, mode="gray")
    assert not same_physical_place(adjacent_label, true_07, mode="gray")
