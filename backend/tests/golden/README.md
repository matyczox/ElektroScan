# Golden Snapshots

Ten katalog trzyma committed snapshoty finalnych boxow. To nie sa lokalne
pliki `backend/analysis_debug/`; te sa tymczasowe i zostaja poza gitem.

## Aktualne Goldeny

- `viking_bronisze_e8_gray_first_pdf_100pct.json` - pierwszy zaakceptowany
  szary PDF: `VIKING-BRONISZE-ELE-Rzuty-E8.pdf`, profil `gray`.
- `viking_bronisze_e9_gray_second_pdf_idealny.json` - drugi zaakceptowany
  szary PDF: `VIKING-BRONISZE-ELE-Rzuty-E9.pdf`, profil `gray`; praktycznie
  idealny baseline mimo 2-3 bardzo gesto upakowanych, niejednoznacznych miejsc.
- `viking_bronisze_e10_gray_third_pdf_accepted_90_95.json` - trzeci szary PDF:
  `VIKING-BRONISZE-ELE-Rzuty-E10.pdf`, profil `gray`; zaakceptowany jako
  pragmatyczny baseline 90-95%, nie jako overfit do 100%.

## Porownywanie

Najwygodniej odpalic caly runner przez dzialajacy backend:

```powershell
py -3 backend/tools/run_golden_regression.py --api-url http://localhost:8000
```

Runner robi upload PDF, sprawdza klasyfikacje `gray`/`color`, uploaduje zapisane
PNG legendy z fixture i porownuje wynik z goldenem.

Uwaga: E9 uzywa zapisanych stref zakazanych. To jest zamierzone: tekstowe
smieci z bloku opisu/ramki maja byc przykryte layoutem, nie progami detektora.

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

Trzeci szary PDF ma wiecej rotacji skosnych i gesto upakowanych symboli:

```powershell
py -3 backend/tools/compare_analysis_snapshot.py `
  backend/tests/golden/viking_bronisze_e10_gray_third_pdf_accepted_90_95.json `
  backend/analysis_debug/<analysis_id>.json `
  --focus "01,02,03,04,05,06,07,08,09,10,11" `
  --center-tolerance 28 `
  --size-tolerance 0.55
```

Jesli wynik sie zmienia, najpierw sprawdz czy to realna poprawa/regresja w
Inspektorze ROI. Nie aktualizuj goldena tylko dlatego, ze progi sie przesunely.

## Zasady

- Nie commituj `backend/analysis_debug/`.
- Nie uzywaj goldena Viking gray jako dowodu, ze wszystkie szare PDF dzialaja.
- Nie mieszaj strojenia gray z kolorowym silnikiem.
- Nie naprawiaj prawych blokow tytulowych/tekstowych progami detektora. Tam
  docelowo uzywamy strefy planu, strefy zakazanej albo osobnej reguly layoutu.
- Obrazki legendy do testow trzymaj w `backend/tests/fixtures/`, nie w
  `backend/templates/`.
