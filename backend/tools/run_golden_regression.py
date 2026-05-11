from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from compare_analysis_snapshot import compare_snapshots


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_FIXTURES_DIR = REPO_ROOT / "backend" / "tests" / "fixtures"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "backend" / "analysis_debug" / "regression"


class RegressionError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _repo_path(value: str | None, *, base: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if base is not None and (base / path).exists():
        return (base / path).resolve()
    return (REPO_ROOT / path).resolve()


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RegressionError(f"{method} {url} failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RegressionError(f"{method} {url} failed: {exc}") from exc
    return json.loads(raw.decode("utf-8"))


def _multipart_request(
    url: str,
    *,
    files: dict[str, Path],
    fields: dict[str, str] | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    boundary = f"----ElektroScanRegression{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in (fields or {}).items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    for name, path in files.items():
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{path.name}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode("utf-8")
        )
        body.extend(path.read_bytes())
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RegressionError(f"POST {url} failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RegressionError(f"POST {url} failed: {exc}") from exc
    return json.loads(raw.decode("utf-8"))


def _load_manifests(fixtures_dir: Path, names: set[str] | None) -> list[tuple[Path, dict[str, Any]]]:
    manifests: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(fixtures_dir.glob("*/manifest.json")):
        manifest = _read_json(path)
        name = str(manifest.get("name") or path.parent.name)
        if names and name not in names and path.parent.name not in names:
            continue
        manifests.append((path, manifest))
    return manifests


def _clear_templates(api_url: str, timeout: int) -> None:
    _request_json(f"{api_url}/api/templates", method="DELETE", timeout=timeout)


def _upload_templates(api_url: str, templates_dir: Path, timeout: int) -> int:
    template_paths = sorted(templates_dir.glob("*.png"))
    if not template_paths:
        raise RegressionError(f"No template PNG files found in {templates_dir}")
    for template_path in template_paths:
        _multipart_request(
            f"{api_url}/api/templates/upload",
            files={"file": template_path},
            timeout=timeout,
        )
    return len(template_paths)


def _run_fixture(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    api_url: str,
    output_dir: Path,
    timeout: int,
) -> bool:
    name = str(manifest.get("name") or manifest_path.parent.name)
    if not manifest.get("enabled", True):
        reason = manifest.get("blockedReason") or "disabled"
        print(f"SKIP {name}: {reason}")
        return True

    pdf_path = _repo_path(manifest.get("pdfPath"), base=manifest_path.parent)
    templates_dir = _repo_path(manifest.get("templatesDir"), base=manifest_path.parent)
    golden_path = _repo_path(manifest.get("golden"), base=manifest_path.parent)
    if pdf_path is None or not pdf_path.exists():
        raise RegressionError(f"{name}: missing pdfPath {pdf_path}")
    if templates_dir is None or not templates_dir.exists():
        raise RegressionError(f"{name}: missing templatesDir {templates_dir}")
    if golden_path is None or not golden_path.exists():
        raise RegressionError(f"{name}: missing golden {golden_path}")

    print(f"\n=== {name} ===")
    print(f"PDF: {pdf_path.name}")
    print(f"Templates: {templates_dir}")

    preview = _multipart_request(
        f"{api_url}/api/preview",
        files={"file": pdf_path},
        timeout=timeout,
    )
    session_id = str(preview["sessionId"])
    diagnostics = preview.get("pdfDiagnostics") or {}
    recommended = str(diagnostics.get("recommendedProfile") or "")
    expected = str(manifest.get("expectedProfile") or "")
    if expected and recommended != expected:
        raise RegressionError(
            f"{name}: expected recommendedProfile={expected}, got {recommended}"
        )
    print(f"Profile: recommended={recommended} expected={expected or '-'}")

    if manifest.get("extractTemplatesFresh"):
        legend_zone = manifest.get("legendZone")
        if not isinstance(legend_zone, dict):
            raise RegressionError(f"{name}: extractTemplatesFresh requires legendZone")
        extraction_body = {
            "detector_profile": manifest.get("detectorProfile", manifest.get("expectedProfile", "auto")),
            "hidden_layers": manifest.get("hiddenLayers", []),
            "legend_zone": legend_zone,
            "excluded_zones": manifest.get("legendExtractionExcludedZones", []),
        }
        extracted = _request_json(
            f"{api_url}/api/extract-legend?{urllib.parse.urlencode({'session_id': session_id})}",
            method="POST",
            payload=extraction_body,
            timeout=timeout,
        )
        template_count = len(extracted.get("patterns") or [])
        print(
            "Fresh extracted templates: "
            f"{template_count} mask={extracted.get('legendMaskMode', '-')}"
        )
    else:
        _clear_templates(api_url, timeout)
        template_count = _upload_templates(api_url, templates_dir, timeout)
    expected_templates = manifest.get("templateCount")
    if expected_templates is not None and int(expected_templates) != template_count:
        raise RegressionError(
            f"{name}: expected {expected_templates} templates, uploaded {template_count}"
        )
    if not manifest.get("extractTemplatesFresh"):
        print(f"Uploaded templates: {template_count}")

    body = {
        "detector_profile": manifest.get("analyzeProfile", "auto"),
        "include_debug": True,
        "include_image": False,
        "hidden_layers": manifest.get("hiddenLayers", []),
        "excluded_zones": manifest.get("excludedZones", []),
    }
    if manifest.get("excludeLegendFromAnalysis") and manifest.get("legendZone") is not None:
        body["legend_zone"] = manifest["legendZone"]
    if manifest.get("planZone") is not None:
        body["plan_zone"] = manifest["planZone"]

    started = time.perf_counter()
    result = _request_json(
        f"{api_url}/api/analyze?{urllib.parse.urlencode({'session_id': session_id})}",
        method="POST",
        payload=body,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started

    context = result.get("analysisContext") or {}
    used = str(context.get("detectorProfileUsed") or "")
    if expected and used != expected:
        raise RegressionError(f"{name}: expected detectorProfileUsed={expected}, got {used}")

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_id = context.get("analysisId") or uuid.uuid4().hex
    candidate_path = output_dir / f"{name}_{analysis_id}.json"
    candidate_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Analyze: boxes={len(result.get('boxes') or [])} profile={used} elapsed={elapsed:.1f}s")
    print(f"Candidate: {candidate_path}")

    focus = tuple(str(manifest.get("focus", "")).split(","))
    focus = tuple(item.strip() for item in focus if item.strip())
    compare_report = compare_snapshots(
        golden_path=golden_path,
        candidate_path=candidate_path,
        focus_prefixes=focus,
        center_tolerance=float(manifest.get("centerTolerance", 24)),
        size_tolerance=float(manifest.get("sizeTolerance", 0.5)),
    )
    print(compare_report)

    bad_markers = (
        "Missing focus boxes: 0",
        "Extra focus boxes: 0",
        "Class conflicts near golden focus boxes: 0",
    )
    passed = all(marker in compare_report for marker in bad_markers)
    print(f"RESULT {name}: {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ElektroScan golden regression fixtures.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixture", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    selected = set(args.fixture) if args.fixture else None
    manifests = _load_manifests(args.fixtures_dir, selected)
    if not manifests:
        print(f"No manifests found in {args.fixtures_dir}", file=sys.stderr)
        return 2

    failures = 0
    for manifest_path, manifest in manifests:
        try:
            if not _run_fixture(
                manifest_path,
                manifest,
                api_url=api_url,
                output_dir=args.output_dir,
                timeout=args.timeout,
            ):
                failures += 1
        except RegressionError as exc:
            failures += 1
            print(f"FAIL {manifest.get('name') or manifest_path.parent.name}: {exc}")

    if failures:
        print(f"\nRegression finished with {failures} failure(s).")
        return 1
    print("\nRegression finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
