import cv2
import numpy as np

from core.detector_masks import _color_mask_for_template, _validate_template_hit
from core.detector_models import CandidateHit
from core.detector_templates import load_templates
from core.roi_inspector import inspect_roi


def _hit(**overrides) -> CandidateHit:
    mask = np.ones((20, 20), dtype=np.uint8) * 255
    defaults = {
        "template_id": 0,
        "scale": 1.0,
        "rotation": 0,
        "mirrored": False,
        "transformed_mask": mask,
        "content_mask": mask,
        "pixel_count": int(cv2.countNonZero(mask)),
        "content_pixel_count": int(cv2.countNonZero(mask)),
        "content_bbox": (0, 0, 20, 20),
        "bbox": (10, 10, 20, 20),
        "match_score": 0.75,
        "dominant_hsv": (0, 255, 255),
        "source": "template",
        "is_text_label": False,
    }
    defaults.update(overrides)
    return CandidateHit(**defaults)


def _red_plan(width: int = 80, height: int = 80) -> np.ndarray:
    return np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)


def test_color_hit_rejects_wrong_hue_even_with_good_geometry():
    plan_image = _red_plan()
    plan_mask = np.zeros((80, 80), dtype=np.uint8)
    plan_mask[10:30, 10:30] = 255
    plan_image[10:30, 10:30] = (255, 0, 255)

    reasons: dict[str, int] = {}
    assert not _validate_template_hit(_hit(), plan_mask, plan_image, reasons=reasons)
    assert reasons == {"color_similarity": 1}


def test_red_color_mask_excludes_black_and_purple_annotations():
    plan_image = _red_plan(width=120, height=60)
    plan_image[10:30, 10:35] = (0, 0, 255)      # red symbol ink
    plan_image[10:30, 40:65] = (0, 0, 0)        # black annotation
    plan_image[10:30, 70:95] = (255, 0, 255)    # purple/magenta annotation

    mask = _color_mask_for_template(plan_image, (0, 255, 255), dilate=False)

    assert cv2.countNonZero(mask[10:30, 10:35]) > 0
    assert cv2.countNonZero(mask[10:30, 40:65]) == 0
    assert cv2.countNonZero(mask[10:30, 70:95]) == 0


def test_saturated_template_mask_excludes_neighboring_label_hue():
    hsv = np.array(
        [
            [
                [150, 255, 255],  # reviewed magenta symbol color
                [135, 255, 255],  # nearby purple label color
            ]
        ],
        dtype=np.uint8,
    )
    plan_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mask = _color_mask_for_template(plan_image, (150, 255, 255), dilate=False, hsv_image=hsv)

    assert mask[0, 0] > 0
    assert mask[0, 1] == 0


def test_color_roi_inspector_reports_template_color_scan_mask(tmp_path):
    template_image = _red_plan(width=40, height=30)
    template_image[6:24, 6:30] = (0, 0, 255)
    cv2.imwrite(str(tmp_path / "01_L1.png"), template_image)

    plan_image = _red_plan(width=140, height=70)
    plan_image[20:38, 20:44] = (0, 0, 255)
    plan_image[20:38, 50:75] = (0, 0, 0)
    plan_image[20:38, 82:108] = (255, 0, 255)
    templates = load_templates(str(tmp_path))

    inspection = inspect_roi(plan_image, templates, (16, 16, 100, 30), detector_profile="color")

    assert inspection["roiInkPixels"] > inspection["roiColorScanPixels"] > 0
    assert inspection["roiScanPixels"] == inspection["roiColorScanPixels"]
    assert inspection["roiColorScanTemplate"]["symbolName"] == "01_L1"


def test_color_roi_inspector_reports_color_preview_when_template_is_too_large(tmp_path):
    template_image = _red_plan(width=80, height=80)
    template_image[10:70, 10:70] = (0, 255, 0)
    cv2.imwrite(str(tmp_path / "01_green.png"), template_image)

    plan_image = _red_plan(width=100, height=80)
    plan_image[20:50, 20:50] = (0, 255, 0)
    templates = load_templates(str(tmp_path))

    inspection = inspect_roi(plan_image, templates, (16, 16, 40, 40), detector_profile="color")

    assert inspection["candidates"] == []
    assert inspection["roiColorScanPixels"] > 0
    assert inspection["roiColorScanTemplate"]["symbolName"] == "01_green"
    assert inspection["roiColorScanTemplate"]["previewOnly"] is True


def test_color_template_content_requires_strong_shape_agreement():
    plan_image = _red_plan()
    plan_mask = np.zeros((80, 80), dtype=np.uint8)
    plan_mask[10:30, 10:20] = 255
    plan_image[10:30, 10:20] = (0, 0, 255)

    reasons: dict[str, int] = {}
    hit = _hit(
        source="template_content",
        is_text_label=True,
        match_score=0.70,
    )

    assert not _validate_template_hit(hit, plan_mask, plan_image, reasons=reasons)
    assert reasons == {"color_content_fragment": 1}


def test_color_text_template_rejects_extra_linework_inside_bbox():
    template_mask = np.zeros((20, 40), dtype=np.uint8)
    template_mask[:, :20] = 255
    plan_image = _red_plan(width=100)
    plan_mask = np.zeros((80, 100), dtype=np.uint8)
    plan_mask[10:30, 10:30] = 255
    plan_mask[10:30, 30:40] = 255
    plan_image[10:30, 10:40] = (0, 0, 255)

    reasons: dict[str, int] = {}
    hit = _hit(
        transformed_mask=template_mask,
        content_mask=template_mask,
        pixel_count=int(cv2.countNonZero(template_mask)),
        content_pixel_count=int(cv2.countNonZero(template_mask)),
        content_bbox=(0, 0, 20, 20),
        bbox=(10, 10, 40, 20),
        match_score=0.60,
        is_text_label=True,
    )

    assert not _validate_template_hit(hit, plan_mask, plan_image, reasons=reasons)
    assert reasons == {"color_text_geometry": 1}


def test_color_elongated_stroke_rejects_weak_line_fragment():
    template_mask = np.zeros((24, 56), dtype=np.uint8)
    template_mask[8:16, 4:52] = 255
    plan_image = _red_plan(width=100)
    plan_mask = np.zeros((80, 100), dtype=np.uint8)
    plan_mask[18:26, 14:62] = 255
    plan_image[18:26, 14:62] = (0, 0, 255)

    reasons: dict[str, int] = {}
    hit = _hit(
        transformed_mask=template_mask,
        content_mask=None,
        pixel_count=int(cv2.countNonZero(template_mask)),
        content_pixel_count=0,
        content_bbox=None,
        bbox=(10, 10, 56, 24),
        match_score=0.50,
        is_text_label=False,
    )

    assert not _validate_template_hit(hit, plan_mask, plan_image, reasons=reasons)
    assert reasons == {"color_elongated_stroke_fragment": 1}


def test_color_elongated_stroke_accepts_strong_clean_symbol():
    template_mask = np.zeros((24, 56), dtype=np.uint8)
    template_mask[8:16, 4:52] = 255
    plan_image = _red_plan(width=100)
    plan_mask = np.zeros((80, 100), dtype=np.uint8)
    plan_mask[18:26, 14:62] = 255
    plan_image[18:26, 14:62] = (0, 0, 255)

    reasons: dict[str, int] = {}
    hit = _hit(
        transformed_mask=template_mask,
        content_mask=None,
        pixel_count=int(cv2.countNonZero(template_mask)),
        content_pixel_count=0,
        content_bbox=None,
        bbox=(10, 10, 56, 24),
        match_score=0.82,
        is_text_label=False,
    )

    assert _validate_template_hit(hit, plan_mask, plan_image, reasons=reasons)
    assert reasons == {}


def test_color_elongated_stroke_rejects_straight_line_fragment_even_when_scores_are_ok():
    template_mask = np.zeros((24, 56), dtype=np.uint8)
    template_mask[10:14, 2:54] = 255
    plan_image = _red_plan(width=100)
    plan_mask = np.zeros((80, 100), dtype=np.uint8)
    plan_mask[20:24, 12:64] = 255
    plan_image[20:24, 12:64] = (0, 0, 255)

    reasons: dict[str, int] = {}
    hit = _hit(
        transformed_mask=template_mask,
        content_mask=None,
        pixel_count=int(cv2.countNonZero(template_mask)),
        content_pixel_count=0,
        content_bbox=None,
        bbox=(10, 10, 56, 24),
        match_score=0.60,
        is_text_label=False,
    )

    assert not _validate_template_hit(hit, plan_mask, plan_image, reasons=reasons)
    assert reasons == {"color_straight_stroke_fragment": 1}


def test_roi_inspector_rejects_flat_red_line_as_wavy_symbol():
    template_mask = np.zeros((24, 56), dtype=np.uint8)
    template_mask[4:16, :] = 255
    plan_image = _red_plan(width=100)
    plan_mask = np.zeros((80, 100), dtype=np.uint8)
    plan_mask[14:26, 10:66] = 255
    plan_image[plan_mask > 0] = (0, 0, 255)

    reasons: dict[str, int] = {}
    hit = _hit(
        transformed_mask=template_mask,
        content_mask=None,
        pixel_count=int(cv2.countNonZero(template_mask)),
        content_pixel_count=0,
        content_bbox=None,
        bbox=(10, 10, 56, 24),
        match_score=0.56,
        source="roi_inspector",
        is_text_label=False,
    )

    assert not _validate_template_hit(hit, plan_mask, plan_image, reasons=reasons)
    assert reasons == {"color_flat_elongated_fragment": 1}


def test_roi_inspector_wavy_symbol_can_pass_guarded_low_match_geometry():
    template_mask = np.zeros((24, 56), dtype=np.uint8)
    points = np.array(
        [[2, 12], [10, 5], [18, 18], [28, 6], [38, 18], [54, 10]],
        dtype=np.int32,
    )
    cv2.polylines(template_mask, [points], isClosed=False, color=255, thickness=3)
    plan_image = _red_plan(width=100)
    plan_mask = np.zeros((80, 100), dtype=np.uint8)
    plan_mask[10:34, 10:66] = template_mask
    plan_image[plan_mask > 0] = (0, 0, 255)

    reasons: dict[str, int] = {}
    hit = _hit(
        transformed_mask=template_mask,
        content_mask=None,
        pixel_count=int(cv2.countNonZero(template_mask)),
        content_pixel_count=0,
        content_bbox=None,
        bbox=(10, 10, 56, 24),
        match_score=0.48,
        source="roi_inspector",
        is_text_label=False,
    )

    assert _validate_template_hit(hit, plan_mask, plan_image, reasons=reasons)
    assert reasons == {}
