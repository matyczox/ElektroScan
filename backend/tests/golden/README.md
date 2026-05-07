# Golden Snapshots

Ten katalog trzyma committed snapshoty finalnych boxow. To nie sa lokalne
pliki `backend/analysis_debug/`; te sa tymczasowe i zostaja poza gitem.

## Aktualne Goldeny

- `viking_bronisze_e8_gray_first_pdf_100pct.json` - pierwszy zaakceptowany
  szary PDF: `VIKING-BRONISZE-ELE-Rzuty-E8.pdf`, profil `gray`.
- `viking_bronisze_e9_gray_second_pdf_idealny.json` - drugi zaakceptowany
  szary PDF: `VIKING-BRONISZE-ELE-Rzuty-E9.pdf`, profil `gray`; praktycznie
  idealny baseline mimo 2-3 bardzo gesto upakowanych, niejednoznacznych miejsc.

## Porownywanie

Po nowym runie z `include_debug=true` backend zapisze kandydacki snapshot do
`backend/analysis_debug/<analysis_id>.json`. Porownaj go z goldenem:

```powershell
py -3 backend/tools/compare_analysis_snapshot.py `
  backend/tests/golden/viking_bronisze_e8_gray_first_pdf_100pct.json `
  backend/analysis_debug/<analysis_id>.json `
  --focus "01,02,03,04,05,06,07" `
  --center-tolerance 20 `
  --size-tolerance 0.45
```

Drugi szary PDF porownuj analogicznie:

```powershell
py -3 backend/tools/compare_analysis_snapshot.py `
  backend/tests/golden/viking_bronisze_e9_gray_second_pdf_idealny.json `
  backend/analysis_debug/<analysis_id>.json `
  --focus "01,02,03,04,07,08,11,12,13,14,15,16,17" `
  --center-tolerance 24 `
  --size-tolerance 0.50
```

Jesli wynik sie zmienia, najpierw sprawdz czy to realna poprawa/regresja w
Inspektorze ROI. Nie aktualizuj goldena tylko dlatego, ze progi sie przesunely.

## Zasady

- Nie commituj `backend/analysis_debug/`.
- Nie uzywaj goldena Viking gray jako dowodu, ze wszystkie szare PDF dzialaja.
- Nie mieszaj strojenia gray z kolorowym silnikiem.
