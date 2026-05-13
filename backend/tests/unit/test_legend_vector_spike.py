import json
from pathlib import Path

import cv2
import fitz
import numpy as np

from core.detector import detect_symbols
from core.detector_templates import load_templates
from core.legend_extractor import extract_legend_detailed, pdf_to_png
from core.legend_scene_transform import (
    build_scene_transform,
    rect_pt_to_px300,
    rect_px300_to_pt,
)
from core.legend_vector_profile import profile_legend_region


def _assert_color_auto_matches_golden(
    tmp_path: Path,
    *,
    fixture_name: str,
    pdf_name: str,
    golden_name: str,
    legend_rect: tuple[int, int, int, int],
    expected_count: int,
) -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / fixture_name
    pdf_path = fixture_dir / pdf_name
    golden_path = Path(__file__).resolve().parents[1] / "golden" / golden_name
    if not pdf_path.exists() or not golden_path.exists():
        return

    plan_image = pdf_to_png(str(pdf_path), dpi=300)
    bundle = extract_legend_detailed(
        str(pdf_path),
        plan_image.copy(),
        output_dir=str(tmp_path / "templates"),
        dpi=300,
        legend_rect_px=legend_rect,
        mask_mode="color",
        legend_engine="auto",
    )
    results = detect_symbols(
        plan_image.copy(),
        load_templates(str(tmp_path / "templates")),
        subtract_legend=True,
        exclude_rects=[legend_rect],
        pdf_path=str(pdf_path),
        pdf_dpi=300,
        detector_profile="color",
    )

    actual = {
        (group.symbol_name, detection.x, detection.y, detection.width, detection.height)
        for group in results
        for detection in group.detections
    }
    golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
    expected = {
        (box["symbolName"], box["x"], box["y"], box["width"], box["height"])
        for box in golden_data["boxes"]
    }

    assert bundle.engine_used == "raster"
    assert bundle.fallback_reason == "color_vector_auto_guard"
    assert len(actual) == expected_count
    assert actual == expected


def _save_vector_legend_pdf(path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=360, height=360)
    labels = ["A OUTLET", "B SWITCH", "C TV", "D DATA", "E LIGHT", "F PANEL", "G SENSOR", "H BELL"]
    for idx, label in enumerate(labels):
        y = 34 + idx * 38
        page.draw_rect(fitz.Rect(22, y - 8, 36, y + 6), color=(1, 0, 0), width=1.2)
        page.draw_line(fitz.Point(42, y), fitz.Point(60, y), color=(1, 0, 0), width=1.0)
        page.insert_text(fitz.Point(74, y + 4), label, fontsize=10, color=(0, 0, 0))
    doc.save(path)
    doc.close()
    return path


def _save_image_only_pdf(path: Path) -> Path:
    image = np.full((360, 520, 3), 255, dtype=np.uint8)
    cv2.putText(image, "RASTER LEGEND", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".png", image)
    assert ok
    doc = fitz.open()
    page = doc.new_page(width=360, height=220)
    page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(path)
    doc.close()
    return path


def test_scene_transform_roundtrip_with_cropbox_and_rotation_metadata():
    doc = fitz.open()
    page = doc.new_page(width=240, height=180)
    page.set_cropbox(fitz.Rect(10, 10, 230, 170))
    page.set_rotation(0)
    transform = build_scene_transform(page, dpi=300, hidden_layers=["ARCH"], source_pdf_sha256="x")

    rect_px = (18, 29, 83, 47)
    rect_pt = rect_px300_to_pt(rect_px, transform)
    roundtrip = rect_pt_to_px300(rect_pt, transform)

    assert transform.cropbox_pt == (10.0, 10.0, 230.0, 170.0)
    assert transform.rotation_deg == 0
    assert all(abs(a - b) <= 1 for a, b in zip(rect_px, roundtrip))
    doc.close()


def test_profile_classifies_vector_and_image_only_regions(tmp_path):
    vector_pdf = _save_vector_legend_pdf(tmp_path / "vector_legend.pdf")
    doc = fitz.open(vector_pdf)
    page = doc.load_page(0)
    transform = build_scene_transform(page, dpi=300)
    legend_rect_px = rect_pt_to_px300((0.0, 0.0, 340.0, 340.0), transform)
    vector_profile = profile_legend_region(page, legend_rect_px, transform)
    doc.close()

    assert vector_profile.page_kind == "vector_rich"
    assert vector_profile.attempt_vector is True
    assert vector_profile.vector_path_count >= 8

    image_pdf = _save_image_only_pdf(tmp_path / "image_only.pdf")
    doc = fitz.open(image_pdf)
    page = doc.load_page(0)
    transform = build_scene_transform(page, dpi=300)
    legend_rect_px = rect_pt_to_px300((0.0, 0.0, 360.0, 220.0), transform)
    image_profile = profile_legend_region(page, legend_rect_px, transform)
    doc.close()

    assert image_profile.page_kind == "image_only"
    assert image_profile.attempt_vector is False
    assert image_profile.fallback_reason == "legend_image_dominant"


def test_vector_first_drafts_keep_png_template_contract(tmp_path):
    pdf_path = _save_vector_legend_pdf(tmp_path / "vector_extract.pdf")
    plan_image = pdf_to_png(str(pdf_path), dpi=300)
    legend_rect_px = (0, 0, plan_image.shape[1], plan_image.shape[0])

    bundle = extract_legend_detailed(
        str(pdf_path),
        plan_image,
        output_dir=str(tmp_path / "templates"),
        dpi=300,
        legend_rect_px=legend_rect_px,
        mask_mode="color",
        legend_engine="vector_first",
        include_debug_primitives=True,
    )

    assert bundle.engine_used == "vector_first"
    assert bundle.fallback_reason is None
    assert len(bundle.extracted_symbols) >= 3
    assert bundle.vector_primitives
    assert all(symbol.image.size > 0 for symbol in bundle.extracted_symbols)
    assert list((tmp_path / "templates").glob("*.png"))
    assert any("OUTLET" in symbol.name for symbol in bundle.extracted_symbols)


def test_image_only_auto_falls_back_to_raster_with_reason(tmp_path):
    pdf_path = _save_image_only_pdf(tmp_path / "image_fallback.pdf")
    plan_image = pdf_to_png(str(pdf_path), dpi=300)
    legend_rect_px = (0, 0, plan_image.shape[1], plan_image.shape[0])

    bundle = extract_legend_detailed(
        str(pdf_path),
        plan_image,
        output_dir=str(tmp_path / "templates"),
        dpi=300,
        legend_rect_px=legend_rect_px,
        mask_mode="color",
        legend_engine="auto",
    )

    assert bundle.engine_used == "raster"
    assert bundle.fallback_reason == "legend_image_dominant"
    assert bundle.page_profile["page_kind"] == "image_only"


def test_color_fixture_vector_first_keeps_multiline_descriptions_together(tmp_path):
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / "pw_e_01_rev2_color"
    pdf_path = fixture_dir / "PW-E-01 Rev2 (1).pdf"
    if not pdf_path.exists():
        return

    plan_image = pdf_to_png(str(pdf_path), dpi=300)
    bundle = extract_legend_detailed(
        str(pdf_path),
        plan_image,
        output_dir=str(tmp_path / "templates"),
        dpi=300,
        legend_rect_px=(3480, 409, 1184, 1462),
        mask_mode="color",
        legend_engine="vector_first",
    )

    names = [symbol.name for symbol in bundle.extracted_symbols]
    assert bundle.engine_used == "vector_first"
    assert len(names) == 22
    assert any("LOTOS" in name and "oprawa_oswietleniowa" in name for name in names)
    assert any("PLAFOND" in name and "oprawa_oswietleniowa" in name for name in names)
    assert not any(name.startswith("LOTOS_") for name in names)
    assert not any(name.startswith("PLAFOND_") for name in names)


def test_color_fixture_auto_uses_raster_guard_for_golden_compatibility(tmp_path):
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / "pw_e_02_rev2_color"
    pdf_path = fixture_dir / "PW-E-02 Rev2.pdf"
    if not pdf_path.exists():
        return

    plan_image = pdf_to_png(str(pdf_path), dpi=300)
    bundle = extract_legend_detailed(
        str(pdf_path),
        plan_image,
        output_dir=str(tmp_path / "templates"),
        dpi=300,
        legend_rect_px=(3506, 428, 1153, 1430),
        mask_mode="color",
        legend_engine="auto",
    )

    names = [symbol.name for symbol in bundle.extracted_symbols]
    assert bundle.engine_used == "raster"
    assert bundle.fallback_reason == "color_vector_auto_guard"
    assert len(names) == 22
    assert "R_orurowanie_do_TSM_TV_gniazdo_TV_1xRG6" in names
    assert (tmp_path / "templates" / "20_R_orurowanie_do_TSM_TV_gniazdo_TV_1xRG6.png").exists()


def test_pw_e_01_auto_extraction_stays_aligned_with_demo_golden(tmp_path):
    _assert_color_auto_matches_golden(
        tmp_path,
        fixture_name="pw_e_01_rev2_color",
        pdf_name="PW-E-01 Rev2 (1).pdf",
        golden_name="pw_e_01_rev2_color_demo.json",
        legend_rect=(3480, 409, 1184, 1462),
        expected_count=151,
    )


def test_pw_e_02_auto_extraction_stays_aligned_with_caution_golden(tmp_path):
    _assert_color_auto_matches_golden(
        tmp_path,
        fixture_name="pw_e_02_rev2_color",
        pdf_name="PW-E-02 Rev2.pdf",
        golden_name="pw_e_02_rev2_color_caution.json",
        legend_rect=(3506, 428, 1153, 1430),
        expected_count=134,
    )
