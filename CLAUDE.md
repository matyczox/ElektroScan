# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ElektroScan — hybrid detector of electrical installation symbols on PDF/image plans. Uses OpenCV template matching, HSV color masks, geometric validation, and HITL (Human In The Loop) corrections. Not a trained ML model — image-based matching with per-symbol template files stored in `backend/templates/`.

Full project documentation is in `openspec/`. Start with `openspec/README.md` if unfamiliar. Key rules: **no hardcoded coordinates**, **no production logic from PDF text layer**, **never commit `backend/analysis_debug/`**.

## Agent team

Use the right tool for the task:

- **Direct answer** — simple questions, quick lookups, one-file edits, explaining code.
- **Spawn an agent** — multi-file changes, diagnosis before any code edit, regression checks, anything that benefits from isolation or specialized context.

| Task type | Agent |
|---|---|
| Wrong detection, missing symbol, strange behavior | `debug-analyst` — diagnose first, never touch code |
| Threshold tuning, mask logic, clustering, promotions | `detector-engineer` |
| CI, Docker, dependencies, linting pipeline | `devops-engineer` |
| UI, canvas, HITL panel, React/TypeScript | `hitl-frontend-dev` |
| New tests, golden snapshots, regression checks | `qa-engineer` |

## Commands

All backend commands run from `backend/` using the Python 3.12 venv (system Python is 3.9 and does not support `@dataclass(slots=True)`).

### Backend

```bash
# Run server
venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Tests
venv/bin/python -m pytest tests/unit/ -v
venv/bin/python -m pytest tests/unit/test_detector_clustering.py -v   # single file
venv/bin/python -m pytest tests/unit/ --cov=core --cov-report=term-missing

# Linting (must all pass before commit)
venv/bin/black --check core/ main.py
venv/bin/isort --check-only core/ main.py
venv/bin/flake8 core/ main.py

# Auto-format
venv/bin/black core/ main.py && venv/bin/isort core/ main.py

# Compile check (quick syntax verification)
python3 -m compileall core/ main.py
```

### Frontend

```bash
# Run dev server
npm run dev -- --host 127.0.0.1

# Tests
npm test                    # vitest run (one-shot)
npm run test:watch          # watch mode
npm run test:coverage

# Linting
npx tsc -p tsconfig.app.json --noEmit
npm run lint
npm run format:check

# Auto-format
npm run format
```

### Docker

```bash
docker compose up -d
docker compose logs -f backend
docker compose cp ./backend/templates/. backend:/app/templates/   # load templates into container
```

### Diagnostic tools (run from `backend/`)

```bash
# Compare two analysis snapshots (golden vs candidate)
venv/bin/python -m tools.compare_analysis_snapshot tests/golden/snapshots/golden.json analysis_debug/latest.json

# Summarize performance of recent snapshots
venv/bin/python -m tools.summarize_analysis_performance analysis_debug/ --latest 3
```

## Architecture

### Detection pipeline (`backend/core/`)

`detect_symbols()` in `detector.py` is the single entry point. It:
1. Renders the legend exclusion zone from PDF text.
2. For each `TemplateInfo`, builds scaled/rotated/mirrored `TemplateVariant` objects.
3. Builds ROI regions from HSV color components in the plan image.
4. Runs `cv2.matchTemplate` per variant within ROIs.
5. Validates each `CandidateHit` with six metrics: `match_score`, `coverage`, `purity`, `context_purity`, `color_similarity`, `verification_score`.
6. Runs family promotions (`detector_promotions.py`): `06→07`, `10→11→12`.
7. Clusters overlapping candidates → keeps one winner per location.
8. Returns `list[DetectionResult]`, each holding grouped `Detection` objects.

When `include_debug=True`, the pipeline also emits `debugCandidates` (typed as `accepted_uncertain`, `rejected_candidate`, `unexplained_component`, `overlap_conflict`, `partial_ghost`) for HITL review in the UI.

### Module responsibilities

| Module | Role |
|---|---|
| `detector.py` | Pipeline orchestration, `detect_symbols()`, `draw_results()` |
| `detector_config.py` | All numeric thresholds and env-var overrides |
| `detector_models.py` | Pure dataclasses: `TemplateInfo`, `TemplateVariant`, `CandidateHit`, `Detection`, `DetectionResult` |
| `detector_masks.py` | HSV masking, `coverage`/`purity`/`context_purity` computation, `content_mask` for text labels |
| `detector_templates.py` | Load templates from disk, build scale/rotation/mirror variants, detect label-like templates |
| `detector_clustering.py` | `_bbox_metrics()` (IoU/IoM/center-dist), NMS prefilter, cluster-then-pick-winner |
| `detector_promotions.py` | Family promotion rules (hardcoded `06/07`, `10/11/12`); do not remove without a general replacement |
| `detector_pdf.py` | PDF text fallback for text-label templates; not the primary detection path |
| `legend_extractor.py` | Render PDF→BGR at 300 DPI, hide layers, extract legend crop |
| `main.py` | FastAPI server; writes snapshots asynchronously via `SNAPSHOT_EXECUTOR` |

### Text labels

Symbols like `TM`, `TSM`, `MSW`, `GSW`, `INT`, `TV` use an image-based content path — **not** a name-to-class map. `detector_masks.py` computes a `content_mask` (letters stripped of frames/lines). Matching weighs `content_score` more than frame match. Do not add a dict like `{"MSW": "05"}`.

### Family promotions — critical invariant

`detector_promotions.py` contains two hardcoded families (`06/07` and `10/11/12`). These protect accuracy on known PDFs. **Do not remove them** until a geometry-driven `core → parent` mechanism is tested in parallel and proven regression-free.

### Frontend state flow

`App.tsx` owns all state: `boxes` (final detections), `debugCandidates`, `manualBoxes`, `patterns` (loaded templates). It passes them down to:
- `CanvasView.tsx` — renders plan image + all box layers on canvas; clicking a box copies the debug payload to clipboard.
- `ResultsPanel.tsx` — list UI for final boxes and HITL candidates.
- `Sidebar.tsx` + `PatternModal.tsx` — upload, legend extraction, analysis trigger, template management.
- `CostPanel.tsx` — read-only count per symbol × editable price → sum (React-only state, not sent to backend).

## Key constraints

- **Session model**: one PDF per `session_id`; results are never cached between requests.
- **DPI**: always 300. Lowering it degrades matching without a full quality test.
- **`analysis_debug/`**: local-only diagnostic snapshots. Never commit them.
- **Env vars**: `ELEKTROSCAN_DETECTOR_SCAN_WORKERS`, `ELEKTROSCAN_DETECTOR_POSTPROCESS_WORKERS`, `ELEKTROSCAN_OPENCV_THREADS` — tune for the host machine.
- **Python version**: requires ≥ 3.10 for `@dataclass(slots=True)`. Use `backend/venv/` (Python 3.12).

## Golden regression cases

These must not break silently (see `openspec/known-issues.md` for full table):

| PDF | bbox | Expected |
|---|---|---|
| `PW-E-02 Rev2.pdf` | `2293,1548,48,31` | symbol `12`, not `11` |
| `PW-E-02 Rev2.pdf` | `2742,975,31,31` | known `09`→`06` error; at minimum HITL/uncertain |
| `PW-E-01 Rev2 (1).pdf` | `1187,1767,46,44` | `08_E_400V` must be detected |

When debugging a wrong detection: copy the debug payload from the UI (click the box), check `frontend_nearby_boxes` and `frontend_nearby_debug_candidates`, then compare against a known-good case before touching any threshold.
