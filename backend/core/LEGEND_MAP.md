# Legend Extraction Map

Ten plik jest mapa kontekstu dla ekstrakcji legend. Refactor legendy ma byc
mechaniczny: bez strojenia masek, bez zmiany auto-detekcji i bez aktualizacji
goldenow.

## Publiczne Wejscia

- `legend_extractor.py` zostaje publicznym wrapperem dla:
  - `pdf_to_png`,
  - `get_pdf_layers`,
  - `extract_legend`,
  - `extract_legend_detailed`.
- Testy i frontend moga nadal importowac prywatne helpery z
  `legend_extractor.py`; wrapper importuje je z mniejszych modulow.

## Moduly

- `legend_text.py`: sanitizacja nazw plikow, czyszczenie OCR i tokeny symboli.
- `legend_pdf_render.py`: render PDF, obsluga hidden layers, `pdf_to_png`.
- `legend_models.py`: `ExtractedSymbol` i `LegendExtractionBundle`.
- `legend_visual_code.py`: lekki raster OCR kodow w komorkach legendy.
- `legend_mask_utils.py`: maski HSV/ink/visible-ink dla raster path.
- `legend_extractor.py`: nadal trzyma table/raster/vector orchestration do
  kolejnych etapow rozbijania.

## Invariants

- Ręcznie wybrany rect legendy jest zrodlem prawdy dla ekstrakcji template'ow.
- `extractTemplatesFresh: false` w caution fixture oznacza: uzywamy zapisanych,
  reviewowanych template'ow.
- Refactor nie moze zmienic liczby ani nazw template'ow w PW/PZU regresji.
- Jesli split zmienia golden output, naprawiamy split zamiast aktualizowac golden.
