#!/usr/bin/env python3
"""
Download EEA E1a (validated) NO2 HOURLY data for 2024, per country, into
  raw/europe/2024/    (one 2024-ONLY parquet per sampling point)

These 2024-only files are ADDITIVE to the historical flat-dir parquets
(raw/europe/*.parquet hold 2005-2023, no 2024) -> eu_config.py can glob both
with no duplication.

Two EEA endpoints, used adaptively per country:
  * POST /ParquetFile        -> a ZIP of DATE-SLICED (2024-only) parquets. Fast and
                                exact, but has a "Request too large" cap: small
                                countries succeed, big ones return HTTP 400.
  * POST /ParquetFile/urls   -> a text list of per-station blob URLs. Works for any
                                size, but the blobs are FULL-SERIES (2005-2024), so
                                we download each and slice to 2024 locally.
Strategy: try the sliced ZIP first; on the "too large" 400, fall back to
/urls + download-and-slice for that country.

dataset=2 == E1a validated (container airquality-p-e1a), matching the historical
EU data the pipeline uses.

Resumable: a .<CC>.done marker per finished country; existing per-station files
are skipped.

Run:  /path/to/miniconda3/envs/python_env/bin/python download_eu_2024.py [ISO2 ...]
"""
import os
import io
import sys
import json
import time
import zipfile
import urllib.request
import urllib.error

import pandas as pd

API_ZIP = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile"
API_URLS = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/urls"
OUT = ("/path/to/NO2_DL_global/"
       "TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/europe/2024")

COUNTRIES = ['AD', 'AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK',
             'EE', 'ES', 'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IS', 'IT', 'LT',
             'LU', 'LV', 'ME', 'MK', 'MT', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS',
             'SE', 'SI', 'SK', 'TR', 'XK']

BODY = {
    "cities": [],
    "pollutants": ["NO2"],
    "dataset": 2,  # E1a validated
    "dateTimeStart": "2024-01-01T00:00:00Z",
    "dateTimeEnd": "2024-12-31T23:59:59Z",
    "aggregationType": "hour",
}

os.makedirs(OUT, exist_ok=True)


def _post(url, country, retries=3, timeout=900):
    """POST the request body for one country. Returns (status, bytes).
    status in {"ok", "toolarge", "fail"}."""
    body = json.dumps(dict(BODY, countries=[country])).encode()
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return "ok", r.read()
        except urllib.error.HTTPError as e:
            msg = ""
            try:
                msg = e.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            if e.code == 400 and "too large" in msg.lower():
                return "toolarge", None
            print(f"  [{country}] {url.rsplit('/',1)[-1]} attempt {attempt}/{retries} "
                  f"HTTP {e.code}: {msg[:120]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{country}] attempt {attempt}/{retries} failed: {e}", flush=True)
        time.sleep(15)
    return "fail", None


def _write_2024_slice(raw_bytes, basename):
    """Read a (possibly full-series) parquet, keep only 2024, write to OUT.
    Returns True if a non-empty 2024 slice was written."""
    dst = os.path.join(OUT, basename)
    if os.path.exists(dst):
        return True
    df = pd.read_parquet(io.BytesIO(raw_bytes))
    if "Start" not in df.columns or df.empty:
        return False
    df = df[df["Start"].dt.year == 2024]
    if df.empty:
        return False
    df.to_parquet(dst, index=False)
    return True


def _download_blob(url, retries=3, timeout=600):
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print(f"    blob attempt {attempt}/{retries} failed: {e}", flush=True)
            time.sleep(10)
    return None


def do_country(country):
    # Path 1: sliced ZIP (already 2024-only members).
    status, data = _post(API_ZIP, country)
    if status == "ok":
        if not zipfile.is_zipfile(io.BytesIO(data)):
            print(f"[{country}] ZIP endpoint returned no zip (no 2024 data?), "
                  f"len={len(data)}", flush=True)
            return 0
        z = zipfile.ZipFile(io.BytesIO(data))
        n = 0
        for name in z.namelist():
            if name.endswith(".parquet"):
                dst = os.path.join(OUT, os.path.basename(name))
                if not os.path.exists(dst):
                    with open(dst, "wb") as f:
                        f.write(z.read(name))
                n += 1
        print(f"[{country}] sliced-zip: {n} parquet(s)", flush=True)
        return n

    if status == "fail":
        raise RuntimeError(f"[{country}] ZIP request failed after retries")

    # Path 2 (status == "toolarge"): /urls list + download full blobs + slice.
    print(f"[{country}] too large for sliced zip -> using /urls + local slice", flush=True)
    ustatus, ubytes = _post(API_URLS, country)
    if ustatus != "ok":
        raise RuntimeError(f"[{country}] /urls request failed ({ustatus})")
    urls = [ln.strip() for ln in ubytes.decode("utf-8-sig").splitlines()
            if ln.strip() and ln.strip() != "ParquetFileUrl"]
    print(f"[{country}] {len(urls)} station blobs to fetch", flush=True)
    n = 0
    for i, url in enumerate(urls, 1):
        base = os.path.basename(url)
        if os.path.exists(os.path.join(OUT, base)):
            n += 1
            continue
        blob = _download_blob(url)
        if blob is None:
            print(f"    [{country}] blob FAILED: {base}", flush=True)
            continue
        if _write_2024_slice(blob, base):
            n += 1
        if i % 25 == 0:
            print(f"    [{country}] {i}/{len(urls)} processed ({n} kept)", flush=True)
    print(f"[{country}] /urls+slice: {n} parquet(s)", flush=True)
    return n


def main():
    only = [a.upper() for a in sys.argv[1:]]
    countries = [c for c in COUNTRIES if not only or c in only]
    total = 0
    for c in countries:
        marker = os.path.join(OUT, f".{c}.done")
        if os.path.exists(marker):
            print(f"[{c}] already done, skipping", flush=True)
            continue
        try:
            total += do_country(c)
            open(marker, "w").close()
        except Exception as e:  # noqa: BLE001
            print(f"[{c}] ERROR (left un-marked for resume): {e}", flush=True)
    print(f"DONE. 2024 parquet files this run: {total}", flush=True)


if __name__ == "__main__":
    main()
