import cv2
import numpy as np

from api_rendering import _ink_profile_stats


def test_ink_profile_counts_colored_symbols_as_color():
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 20), (110, 20), (40, 40, 40), 3)
    cv2.rectangle(image, (20, 50), (100, 80), (0, 0, 255), -1)

    stats = _ink_profile_stats(image)

    assert stats["recommendedProfile"] == "color"
    assert stats["colorfulInkPct"] > 1.0
    assert stats["grayInkPct"] < 100.0


def test_ink_profile_recommends_gray_for_black_only_plan():
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 20), (110, 20), (40, 40, 40), 3)
    cv2.rectangle(image, (20, 50), (100, 80), (120, 120, 120), 2)

    stats = _ink_profile_stats(image)

    assert stats["recommendedProfile"] == "gray"
    assert stats["colorfulInkPct"] == 0.0
