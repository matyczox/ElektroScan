from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = REPO_ROOT / "frontend"

DEFAULT_FIXTURES = [
    "pzu_bydgoszcz_el01_gniazda_color",
    "pzu_bydgoszcz_el02_color",
    "pw_e_01_rev2_color",
    "pw_e_02_rev2_color",
]

EXPECTED_COUNTS = {
    "pzu_bydgoszcz_el01_gniazda_color": "204/204",
    "pzu_bydgoszcz_el02_color": "318/318",
    "pw_e_01_rev2_color": "151/151",
    "pw_e_02_rev2_color": "134/134",
}


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _run_step(name: str, command: list[str], *, cwd: Path) -> tuple[bool, float]:
    printable = " ".join(command)
    print(f"\n=== {name} ===")
    print(f"$ {printable}")
    started = time.perf_counter()
    result = subprocess.run(command, cwd=str(cwd), check=False)
    elapsed = time.perf_counter() - started
    status = "PASS" if result.returncode == 0 else f"FAIL ({result.returncode})"
    print(f"=== {name}: {status} in {elapsed:.1f}s ===")
    return result.returncode == 0, elapsed


def _regression_command(fixtures: list[str], output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "backend" / "tools" / "run_local_golden_regression.py"),
        "--output-dir",
        str(output_dir / "local_regression"),
        "--perf-json",
        str(output_dir / "local_regression_perf.json"),
    ]
    for fixture in fixtures:
        command.extend(["--fixture", fixture])
    return command


def _pytest_command() -> list[str]:
    return [sys.executable, "-m", "pytest", str(REPO_ROOT / "backend" / "tests"), "-q"]


def _frontend_build_command() -> list[str]:
    npm = shutil.which("npm") or "npm"
    return [npm, "run", "build"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local ElektroScan quality gate for detector work."
    )
    parser.add_argument(
        "--fixture",
        action="append",
        help="Fixture to run. Defaults to EL01, EL02, PW-E-01 and PW-E-02.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "backend" / "tests" / "output" / "quality_gate",
        help="Directory for candidate/performance output. This should stay gitignored.",
    )
    parser.add_argument("--skip-backend-tests", action="store_true")
    parser.add_argument("--skip-frontend-build", action="store_true")
    parser.add_argument(
        "--regression-only",
        action="store_true",
        help="Run only golden regression and skip pytest/frontend build.",
    )
    args = parser.parse_args()

    fixtures = args.fixture or DEFAULT_FIXTURES
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("ElektroScan quality gate")
    print(f"Repo: {REPO_ROOT}")
    print(f"Output: {_relative(output_dir)}")
    print("Expected detector counts:")
    for fixture in fixtures:
        print(f"  - {fixture}: {EXPECTED_COUNTS.get(fixture, 'see golden')}")

    steps: list[tuple[str, list[str], Path]] = [
        ("local golden regression", _regression_command(fixtures, output_dir), REPO_ROOT),
    ]
    if not args.regression_only and not args.skip_backend_tests:
        steps.append(("backend tests", _pytest_command(), REPO_ROOT))
    if not args.regression_only and not args.skip_frontend_build:
        steps.append(("frontend build", _frontend_build_command(), FRONTEND_DIR))

    started = time.perf_counter()
    results: list[tuple[str, bool, float]] = []
    for name, command, cwd in steps:
        ok, elapsed = _run_step(name, command, cwd=cwd)
        results.append((name, ok, elapsed))
        if not ok:
            break

    total_elapsed = time.perf_counter() - started
    print("\n=== Quality gate summary ===")
    for name, ok, elapsed in results:
        print(f"{'PASS' if ok else 'FAIL'} {name}: {elapsed:.1f}s")
    print(f"Total: {total_elapsed:.1f}s")
    print(f"Output dir: {_relative(output_dir)}")

    if not all(ok for _name, ok, _elapsed in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
