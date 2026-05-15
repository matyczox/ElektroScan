import io
import zipfile

from main import (
    AnalysisExportRequest,
    _build_analysis_export_rows,
    _build_analysis_export_xlsx,
    _write_template_labels,
)


def test_analysis_export_rows_use_current_boxes_and_display_labels(tmp_path):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    _write_template_labels(
        templates_dir,
        {
            "01_TM": "rozdzielnica glowna mieszkaniowa",
            "02_TAB": "rozdzielnica administracyjna budynku",
        },
    )

    body = AnalysisExportRequest(
        results=[
            {"name": "01_TM", "count": 99, "color": "#ef4444"},
            {"name": "02_TAB", "count": 99, "color": "#f97316"},
        ],
        boxes=[
            {"symbolName": "01_TM", "color": "#ef4444"},
            {"symbolName": "01_TM", "color": "#ef4444"},
            {"symbolName": "02_TAB", "color": "#f97316"},
        ],
        symbolLabels={"02_TAB": "rozdzielnica administracyjna budynku"},
    )

    rows = _build_analysis_export_rows(body, templates_dir)

    assert rows == [
        {
            "element": "rozdzielnica glowna mieszkaniowa",
            "count": 2,
            "templateIds": ["01_TM"],
            "color": "#ef4444",
        },
        {
            "element": "rozdzielnica administracyjna budynku",
            "count": 1,
            "templateIds": ["02_TAB"],
            "color": "#f97316",
        },
    ]


def test_analysis_export_rows_aggregate_duplicate_display_names(tmp_path):
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    _write_template_labels(
        templates_dir,
        {
            "01_AW1": "oprawa awaryjna",
            "02_AW2": "oprawa awaryjna",
        },
    )

    body = AnalysisExportRequest(
        results=[
            {"name": "01_AW1", "count": 3},
            {"name": "02_AW2", "count": 4},
        ],
    )

    rows = _build_analysis_export_rows(body, templates_dir)

    assert rows == [
        {
            "element": "oprawa awaryjna",
            "count": 7,
            "templateIds": ["01_AW1", "02_AW2"],
            "color": None,
        }
    ]


def test_analysis_export_xlsx_is_valid_zip_with_sheet_rows():
    workbook = _build_analysis_export_xlsx(
        project={"name": "Projekt testowy"},
        rows=[
            {
                "element": "gniazdo 230V",
                "count": 12,
                "templateIds": ["01_gniazdo"],
                "color": "#22c55e",
            }
        ],
        analysis_context={"sourcePdf": "plan.pdf", "analysisId": "analysis-1"},
    )

    assert workbook.startswith(b"PK")
    with zipfile.ZipFile(io.BytesIO(workbook)) as archive:
        names = set(archive.namelist())
        assert "xl/workbook.xml" in names
        assert "xl/worksheets/sheet1.xml" in names
        sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")

    assert "gniazdo 230V" in sheet
    assert "<v>12</v>" in sheet
