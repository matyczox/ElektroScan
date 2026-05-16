# Golden Snapshots

Ten katalog trzyma committed snapshoty finalnych boxow. To nie sa lokalne
pliki `backend/analysis_debug/`; te sa tymczasowe i zostaja poza gitem.

## Aktualne Goldeny

- `pw_e_01_rev2_color_demo.json` - zaakceptowany kolorowy baseline PW-E-01,
  profil `color`; release gate dla color path.
- `pw_e_02_rev2_color_caution.json` - zaakceptowany kolorowy baseline PW-E-02,
  profil `color`; release gate dla color path.
- `pzu_bydgoszcz_el02_color_caution.json` - PZU Bydgoszcz EL_02, profil
  `color`; caution baseline z manual sentinels. Snapshot pomaga diagnozowac
  zmiany, ale release blokuje tylko zestaw sentinelowy.
- `viking_bronisze_e8_gray_first_pdf_100pct.json` - pierwszy zaakceptowany
  szary PDF: `VIKING-BRONISZE-ELE-Rzuty-E8.pdf`, profil `gray`.
- `viking_bronisze_e9_gray_second_pdf_idealny.json` - drugi zaakceptowany
  szary PDF: `VIKING-BRONISZE-ELE-Rzuty-E9.pdf`, profil `gray`; praktycznie
  idealny baseline mimo 2-3 bardzo gesto upakowanych, niejednoznacznych miejsc.
- `viking_bronisze_e10_gray_third_pdf_accepted_90_95.json` - trzeci szary PDF:
  `VIKING-BRONISZE-ELE-Rzuty-E10.pdf`, profil `gray`; zaakceptowany jako
  pragmatyczny baseline 90-95%, nie jako overfit do 100%.

## Porownywanie

Lokalne regresje bez UI/backend servera:

```powershell
py -3.11 backend\tools\run_local_golden_regression.py --fixture pzu_bydgoszcz_el02_color --fixture pw_e_01_rev2_color --fixture pw_e_02_rev2_color
```

Oczekiwany stan po stabilizacji color path:

- `pzu_bydgoszcz_el02_color`: `318/318`, wszystkie manual sentinels fixed.
- `pw_e_01_rev2_color`: `151/151`.
- `pw_e_02_rev2_color`: `134/134`.

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
- Nie commituj `backend/tests/output/`.
- PZU traktuj jako caution/sentinel baseline, nie jako automatyczna prawde dla
  wszystkich kolorowych PDF.
- Nie uzywaj goldena Viking gray jako dowodu, ze wszystkie szare PDF dzialaja.
- Nie mieszaj strojenia gray z kolorowym silnikiem.
- Nie naprawiaj prawych blokow tytulowych/tekstowych progami detektora. Tam
  docelowo uzywamy strefy planu, strefy zakazanej albo osobnej reguly layoutu.
- Obrazki legendy do testow trzymaj w `backend/tests/fixtures/`, nie w
  `backend/templates/`.
