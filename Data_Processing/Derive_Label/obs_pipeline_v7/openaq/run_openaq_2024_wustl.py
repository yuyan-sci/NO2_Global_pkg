#!/usr/bin/env python3
"""
Run the v7 OpenAQ downloader for 2024 on WUSTL (LSF), writing straight into the
v7 raw dir that aggregate_openaq.py reads from.

Why a wrapper (instead of editing global_openaq_2.py):
  - global_openaq_2.py hardcodes BASE_OUTPUT_DIR = /path/to/openaq,
    which does NOT match aggregate_openaq.py's BASE_INPUT_DIR
    (.../Global_NO2_v7/no2_ground/openaq/raw/). On AWS a sync step bridged them.
  - Here we override BASE_OUTPUT_DIR in-process so 2024 lands exactly where the
    aggregate expects it, no sync/reorg needed.

Non-destructive: global_openaq_2.main() skips any {country}/{year}.csv that
already exists, so the existing 2005-2023 files are untouched.

Usage:
  python run_openaq_2024_wustl.py                 # all non-excluded countries, 2024
  python run_openaq_2024_wustl.py --countries FR  # single-country smoke test
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Use async_openaq_2.py: the corrected v7 downloader. It passes countries_id as
# an int (compatible with the installed openaq SDK 1.1.0; global_openaq_2.py
# passes str(...) which 1.1.0 rejects) AND carries the v7 unit-handling fixes
# (no "*1000 if v<1" heuristic that corrupted clean-air ppb; proper µg/m3 conv).
import async_openaq_2 as g  # noqa: E402

# Aggregate's input dir (aggregate_openaq.py BASE_INPUT_DIR, minus trailing slash).
# Inside the LSF docker container the tree is mounted at /path/to; on the
# login node (smoke tests) only the real /path/to path exists. Pick whichever
# is present so the same wrapper works in both places.
_SUBPATH = "1.project/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/openaq/raw"
_CONTAINER_BASE = os.path.join("/path/to", _SUBPATH)
_REAL_BASE = os.path.join("/path/to", _SUBPATH)
g.BASE_OUTPUT_DIR = _CONTAINER_BASE if os.path.isdir("/path/to") else _REAL_BASE

# Skip India: OpenAQ has no 2024 NO2 (CPCB feed gap Oct 2022-Oct 2025), so its
# ~1474 sensors all return "no measurements" — in the first 2024 run that burned
# ~6h and the buffered fetch helped OOM-kill the job. India ground NO2 for 2024
# comes from neither OpenAQ nor the national network (IN_config capped <=2023).
if hasattr(g, "EXCLUDED_NAMES"):
    g.EXCLUDED_NAMES = set(g.EXCLUDED_NAMES) | {"India"}

# Force 2024; still allow passthrough of --countries (etc.) for smoke tests.
extra = sys.argv[1:]
sys.argv = [sys.argv[0], "--start-year", "2024", "--end-year", "2024"] + extra

if __name__ == "__main__":
    print(f"[run_openaq_2024_wustl] output dir: {g.BASE_OUTPUT_DIR}", flush=True)
    g.main()
