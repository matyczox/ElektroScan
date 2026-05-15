import cv2
import numpy as np

from main import TemplateUpdateRequest, _append_extracted_templates, _update_template_response


def _write_png(path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_append_extracted_templates_keeps_existing_files(tmp_path):
    templates_dir = tmp_path / "templates"
    extraction_dir = tmp_path / "extracted"
    templates_dir.mkdir()
    extraction_dir.mkdir()

    _write_png(templates_dir / "01_existing.png")
    _write_png(extraction_dir / "01_new_symbol.png")
    _write_png(extraction_dir / "02_other_symbol.png")

    added_ids = _append_extracted_templates(extraction_dir, templates_dir)

    assert (templates_dir / "01_existing.png").exists()
    assert (templates_dir / "02_new_symbol.png").exists()
    assert (templates_dir / "03_other_symbol.png").exists()
    assert added_ids == {"02_new_symbol", "03_other_symbol"}
    assert not list(extraction_dir.glob("*.png"))


def test_append_extracted_templates_stores_display_labels_without_renaming(tmp_path):
    templates_dir = tmp_path / "templates"
    extraction_dir = tmp_path / "extracted"
    templates_dir.mkdir()
    extraction_dir.mkdir()

    _write_png(extraction_dir / "01_symbol_01.png")
    _write_png(extraction_dir / "02_symbol_02.png")

    added_ids = _append_extracted_templates(
        extraction_dir,
        templates_dir,
        display_labels=["Rozdzielnica główna", "Gniazdo 230V"],
    )

    assert (templates_dir / "01_symbol_01.png").exists()
    assert (templates_dir / "02_symbol_02.png").exists()
    assert added_ids == {"01_symbol_01", "02_symbol_02"}

    labels_json = templates_dir / ".template_labels.json"
    assert labels_json.exists()
    assert "Rozdzielnica główna" in labels_json.read_text(encoding="utf-8")


def test_update_template_changes_display_label_not_detector_id(tmp_path):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    _write_png(templates_dir / "01_symbol_01.png")

    response = _update_template_response(
        "01_symbol_01",
        TemplateUpdateRequest(name="Rozdzielnica główna"),
        templates_dir,
    )

    assert (templates_dir / "01_symbol_01.png").exists()
    assert not (templates_dir / "01_Rozdzielnica_główna.png").exists()
    assert response["pattern"]["id"] == "01_symbol_01"
    assert response["pattern"]["name"] == "Rozdzielnica główna"
