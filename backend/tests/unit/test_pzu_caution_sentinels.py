import json
from pathlib import Path

from tools.check_manual_sentinels import evaluate_sentinels


def test_manual_sentinel_checker_reports_fixed_and_wrong_cases():
    golden = {
        "manualSentinelChecks": [
            {
                "id": "forbid-pdf-text",
                "mode": "forbid_near",
                "roi": [90, 190, 30, 30],
                "symbolName": "03_L3",
                "source": "pdf_text",
            },
            {
                "id": "require-visual-ew",
                "mode": "require_near",
                "roi": [300, 400, 30, 30],
                "symbolName": "18_EW1",
                "sourceNot": "pdf_text",
            },
            {
                "id": "allow-any-l-family",
                "mode": "allow_any_near",
                "roi": [500, 600, 80, 40],
                "allowedSymbolNames": ["07_L7", "10_L10", "11_L11", "12_L12", "13_L13"],
            },
        ]
    }
    candidate = {
        "boxes": [
            {
                "symbolName": "18_EW1",
                "x": 308,
                "y": 405,
                "width": 40,
                "height": 40,
                "source": "template_label_disambiguation",
            },
            {"symbolName": "12_L12", "x": 520, "y": 606, "width": 120, "height": 40, "source": "template"},
        ]
    }

    ok, report = evaluate_sentinels(golden, candidate)

    assert ok
    assert "forbid-pdf-text: fixed" in report
    assert "require-visual-ew: fixed" in report
    assert "allow-any-l-family: fixed" in report

    candidate["boxes"].append(
        {"symbolName": "03_L3", "x": 99, "y": 199, "width": 20, "height": 20, "source": "pdf_text"}
    )
    ok, report = evaluate_sentinels(golden, candidate)

    assert not ok
    assert "forbid-pdf-text: still wrong" in report


def test_pzu_caution_fixture_uses_full_caution_baseline_and_reviewed_templates():
    repo_root = Path(__file__).resolve().parents[3]
    fixture_dir = repo_root / "backend" / "tests" / "fixtures" / "pzu_bydgoszcz_el02_color"
    manifest = json.loads((fixture_dir / "manifest.json").read_text(encoding="utf-8"))
    golden = json.loads(
        (repo_root / "backend" / "tests" / "golden" / "pzu_bydgoszcz_el02_color_caution.json")
        .read_text(encoding="utf-8")
    )

    assert manifest["caution"] is True
    assert manifest["sentinelOnly"] is False
    assert manifest["snapshotCompareGate"] is False
    assert manifest["extractTemplatesFresh"] is False
    assert manifest["focus"] == "*"
    assert len(list((fixture_dir / "templates").glob("*.png"))) == 29
    assert golden["metadata"]["releaseGate"] is False
    assert golden["metadata"]["sentinelOnly"] is False
    assert len(golden["boxes"]) > 300
    assert len(golden["manualSentinels"]) >= 8
