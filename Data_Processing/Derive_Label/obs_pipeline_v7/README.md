# v7 ground-NO2 observation pipeline

Consolidated, bug-fixed rebuild of the global surface-NO2 observation labels used
for ML training. Produces **v7** outputs while leaving the existing v6 untouched.

Created 2026-06 after an audit found data-quality bugs in the v6 obs pipeline
(most visibly: a column-mapping bug inflated Japan March-2014 by ~4×). All known
P1/P2 bugs are fixed here and each script's paths are normalized to the canonical
`/path/to/1.project` base.

## Layout
```
obs_pipeline_v7/
  national_networks/   per-country parsers (7 fixed + 5 verified-clean, for reference)
  openaq/              aggregate_openaq.py (+ downloaders for reference)
  combine/             combine_no2_dataset.py, filter_sites.py
  derive_observation_HP.py   sites -> GCHP grid, monthly mean, bias-correct -> NetCDF
  run_all.sh           driver (parsers -> stage -> aggregate -> combine -> filter -> derive -> validate)
  run_all.bsub         LSF submit (compute node, 120GB; heavy steps need RAM)
  validate_v7.py       post-build checks on the v7 label
  README.md
```

## Data flow
```
RAW (national + OpenAQ)
  national parsers ──► /no2_ground/national_networks_v7/*.csv   (staged by run_all.sh)
  openaq aggregate ──► /no2_ground/openaq/data/global/global_monthly_means_v7.csv
        │
        ▼ combine_no2_dataset.py
  combined_global_no2_2005-2023_v7.csv
        ▼ filter_sites.py   (drop US MOBILE + EU traffic; >300; 2024/25)
  combined_global_no2_2005-2023_v7_filtered.csv
        ▼ derive_observation_HP.py  (snap to GeoNO2-v5.13 grid, cell monthly mean, bias-correct)
  TrainingDatasets/Global_NO2_v7/NO2_observation_{corrected,uncorrected}_v7_filtered_v5.13.nc
```

## What was fixed (vs v6)
P1 (corrupted the label):
- **JP**: removed buggy 2014 hardcoded-column special case (March read April's hourly-max). Now name-based for all years.
- **EU**: filter EEA `Validity>=1` (drops the 99999 positive fill → ~53,000 ppb).
- **India**: drop `0<no2<500`, bound years 2016-2023 (raw had 2025 + 1e26 + negatives).
- **Brazil**: normalize unit strings so all `µg/m³` spellings convert; value cap.
- **South Africa**: value cap; fixed `good_sites` NameError.
- **combine**: drop `<=0`; **temporal-spike QC** (>2.5× adjacent-month mean → `flagged_spikes_v7.csv`); 150-300 review log.

P2 (quality/coverage):
- **OpenAQ**: raw-hourly cap 0..1000 ppb (kills 1e26/Turkey extremes); robust unit conversion.
- **NZ**: output column `NO2`→`no2_ppb` (combine requires it); `>0` guard. Keep-all site policy (`>=0`).
- **SA**: keep-all site policy (`>=0`).
- **Mexico**: coverage thresholds 12h→18h/day, 0.3→0.75 month.
- **AU EPA**: no change — output verified plausible ppb.

Verified-clean, NOT re-run (reused from existing `national_networks/` CSVs): **US, CA, CH, AU**.

## Run
Compute node (recommended — heavy):
```
cd .../obs_pipeline_v7 && bsub < run_all.bsub
```
Login node (lighter testing; India/EU/OpenAQ may strain memory):
```
PY=/path/to/miniconda3/envs/python_env/bin/python bash run_all.sh
```
Then check `logs/` and `validate_v7.py` output. The validator asserts: no `<=0`,
no `>300`, no temporal spikes, and Japan 2014-03 back to ~13 ppb.

## After v7 obs is built
Rebuild training data (add channels) from `Global_NO2_v7/`, then retrain → SCV →
maps → analysis. Diff v7 vs v6 (e.g. with the anchor-sensitivity trend scripts)
before committing to the full retrain.
