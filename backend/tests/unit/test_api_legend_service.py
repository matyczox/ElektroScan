from api_legend_service import _exclude_rects_for_legend_extraction


def test_legend_extraction_ignores_exclusions_overlapping_active_legend():
    legend_rect = (100, 100, 200, 100)
    exclude_rects = [
        (0, 0, 20, 20),
        (150, 120, 30, 30),
        (99, 199, 10, 10),
        (400, 400, 20, 20),
    ]

    assert _exclude_rects_for_legend_extraction(exclude_rects, legend_rect) == [
        (0, 0, 20, 20),
        (400, 400, 20, 20),
    ]
