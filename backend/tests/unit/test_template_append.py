import cv2
import numpy as np

from main import _append_extracted_templates


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
