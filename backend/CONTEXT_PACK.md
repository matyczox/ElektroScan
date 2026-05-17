# Backend Context Pack

This file is the short entry point for future debugging and refactors. Keep it
behavior-focused: do not add project-specific detector rules here.

## Invariants

- Detector logic must not use PDF-specific coordinates. Coordinates belong only
  in golden fixtures, sentinels, ROI debug, or manual review notes.
- PDF text in the color path is evidence or a guarded resolver input, not a
  standalone visual symbol detector for pictogram families.
- A color mask is a scan mask. It does not decide the class by itself.
- PZU is a caution baseline. PW-E accepted/caution regressions still protect
  broad behavior.
- Refactors must not update goldens. If boxes change, the refactor is wrong
  until proven otherwise.

## Main Entry Points

- `backend/main.py`: FastAPI route wrappers and app setup.
- `backend/api_analysis_runner.py`: render, detect, format, snapshot analysis.
- `backend/api_rendering.py`: PDF rendering cache, previews, PDF diagnostics.
- `backend/api_workspace.py`: project/session paths, metadata, progress state.
- `backend/api_template_service.py`: template list/upload/crop/update/delete.
- `backend/api_zones.py`: legend/plan/excluded-zone rectangle helpers.
- `backend/analysis_export.py`: export row aggregation and XLSX generation.
- `backend/template_store.py`: template display labels and template payloads.

## Detector Map

- `backend/core/DETECTOR_MAP.md`: detailed detector phase map.
- `backend/core/detector.py`: public color/gray router.
- `backend/core/detector_pipeline.py`: shared orchestration.
- `backend/core/detector_color_resolvers.py`: guarded color-family postprocess.
- `backend/core/detector_clustering.py`: raw/final candidate clustering.
- `backend/core/detector_mask_builders.py`: low-level mask construction.
- `backend/core/detector_shape_metrics.py`: shape, wave, and fragment metrics.
- `backend/core/detector_visualization.py`: drawing debug/result boxes only.

## Golden And Debugging

- `backend/tests/golden/README.md`: accepted vs caution golden policy.
- `backend/DEBUG_PLAYBOOK.md`: command cookbook and manual review flow.
- `backend/tests/golden/pzu_bydgoszcz_el02_color_caution.json`: PZU caution
  snapshot plus manual sentinels.
- `backend/tests/fixtures/pzu_bydgoszcz_el01_gniazda_color/case_pack.json`:
  active diagnostic case pack for the current EL_01 GNIAZDA PDF. It is not a
  golden; use it to generate crops and collect expected/manual-check ROIs
  before changing detector logic.
- `backend/tools/build_pdf_case_report.py`: renders case-pack ROI crops and,
  when given a candidate JSON/templates directory, adds nearby boxes and ROI
  inspector candidates.

## Regression Commands

```powershell
py -3.11 backend\tools\run_local_golden_regression.py --fixture pzu_bydgoszcz_el02_color --fixture pw_e_01_rev2_color --fixture pw_e_02_rev2_color
py -3.11 -m pytest backend\tests -q
npm run build
```

Run `npm run build` from `frontend\`.

Current EL_01 case report:

```powershell
py -3.11 backend\tools\build_pdf_case_report.py backend\tests\fixtures\pzu_bydgoszcz_el01_gniazda_color\case_pack.json --output-dir backend\tests\output\pzu_el01_case_report
```
