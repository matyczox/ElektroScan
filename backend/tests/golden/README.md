# Golden Snapshots

Ten katalog trzyma committed snapshoty finalnych boxow. To nie sa lokalne
pliki `backend/analysis_debug/`; te sa tymczasowe i zostaja poza gitem.

## Aktualne Goldeny

- `viking_bronisze_e8_gray_first_pdf_100pct.json` - pierwszy zaakceptowany
  szary PDF: `VIKING-BRONISZE-ELE-Rzuty-E8.pdf`, profil `gray`.

## Porownywanie

Po nowym runie z `include_debug=true` backend zapisze kandydacki snapshot do
`backend/analysis_debug/<analysis_id>.json`. Porownaj go z goldenem:

```powershell
py -3 backend/tools/compare_analysis_snapshot.py `
  backend/tests/golden/viking_bronisze_e8_gray_first_pdf_100pct.json `
  backend/analysis_debug/<analysis_id>.json `
  --focus 01,02,03,04,05,06,07 `
  --center-tolerance 20 `
  --size-tolerance 0.45
```

Jesli wynik sie zmienia, najpierw sprawdz czy to realna poprawa/regresja w
Inspektorze ROI. Nie aktualizuj goldena tylko dlatego, ze progi sie przesunely.

## Zasady

- Nie commituj `backend/analysis_debug/`.
- Nie uzywaj goldena Viking gray jako dowodu, ze wszystkie szare PDF dzialaja.
- Nie mieszaj strojenia gray z kolorowym silnikiem.
