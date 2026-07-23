"""
Approach-2 prep: TROPOMI fine-structure ratio R_fine(p) = T_1km / boxcar10(T_1km),
where T = mean geophysical NO2 (GeoNO2_trop_GC, 1 km) over the TROPOMI reference
years 2019, 2022, 2023 (all 12 months of each; 2020-2021 COVID years and the
partial 2018 are excluded).  boxcar10 = 0.1deg (~10 km) moving average.

R_fine carries the sub-10 km spatial detail that TROPOMI resolves but OMI cannot;
recon_build.py imprints it onto the boxcar-coarsened OMI geophysical field.

Output (this dir): R_fine.npy  (float32, 13000x36000)
Run inside docker (big-mem; scipy).
"""
import os
import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter

BASE = ("/path/to" if os.path.isdir("/path/to/1.project")
        else "/path/to")
GEOD = f"{BASE}/1.project/NO2_DL_global/input_variables/GeoNO2-v5.13_input"
HERE = os.path.dirname(os.path.abspath(__file__))
BOX = 10   # 0.1 deg ~ 10 km boxcar

# TROPOMI reference: full years 2019, 2022, 2023 (all 12 months of each).
# 2020-2021 (COVID anomaly) and the partial 2018 are intentionally excluded.
REF_YEARS = [2019, 2022, 2023]
months = [(y, m) for y in REF_YEARS for m in range(1, 13)]
print(f"TROPOMI reference years {REF_YEARS}: {len(months)} months "
      f"{months[0]} .. {months[-1]}", flush=True)

acc = None; cnt = None
for y, m in months:
    f = f"{GEOD}/{y}/GeoNO2_trop_GC_001x001_Global_map_{y}{m:02d}.npy"
    a = np.load(f).astype(np.float32)
    # Straight temporal mean (finite months only, negatives INCLUDED) so the
    # spatial positivity gate below reflects the real signed signal.
    ok = np.isfinite(a)
    if acc is None:
        acc = np.zeros_like(a); cnt = np.zeros(a.shape, np.int16)
    acc[ok] += a[ok]; cnt[ok] += 1
    print(f"  + {y}-{m:02d}", flush=True)
T1 = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
del acc, cnt

# boxcar over land+ocean; nan-safe via masked normalisation
mask = np.isfinite(T1).astype(np.float32)
T0 = np.where(np.isfinite(T1), T1, 0.0).astype(np.float32)
num = uniform_filter(T0, size=BOX, mode="nearest")
den = uniform_filter(mask, size=BOX, mode="nearest")
T10 = np.where(den > 0, num / den, np.nan).astype(np.float32)
del T0, mask, num, den

# NOISE FILTER (spatial all-positive, 1x1 AND 10x10): apply the fine-structure
# ratio ONLY where the pixel itself (1x1) and EVERY 1km pixel in its 10x10
# (0.1deg) block are strictly positive. A single negative / non-finite pixel
# anywhere in the block (a sign flip, noisy negative retrieval, or no-data edge)
# disqualifies it, because such a block mixes signs and its boxcar denominator is
# unreliable. Disqualified pixels get R_fine = 1 -> the coarse OMI value passes
# through unchanged and no spurious fine structure / noise is injected.
# minimum_filter over a sentinel (-1 for non-finite) enforces "all pixels in the
# block > 0": block_min > 0  <=>  every pixel in the 10x10 window is finite and >0.
sentinel = np.where(np.isfinite(T1), T1, -1.0).astype(np.float32)
block_min = minimum_filter(sentinel, size=BOX, mode="nearest")
allpos = np.isfinite(T1) & (T1 > 0) & (block_min > 0)
del sentinel, block_min
with np.errstate(invalid="ignore", divide="ignore"):
    ratio = T1 / T10
R = np.where(allpos, ratio, 1.0).astype(np.float32)
# No clipping: the spatial all-positive gate already guarantees every applied
# block has a strictly-positive boxcar denominator, so the ratio stays finite and
# bounded by the real signal; disqualified pixels are exactly R=1.
np.save(os.path.join(HERE, "R_fine.npy"), R)
print(f"Wrote R_fine.npy  median={np.nanmedian(R):.3f} "
      f"p5={np.nanpercentile(R,5):.3f} p95={np.nanpercentile(R,95):.3f}  "
      f"ratio applied on {100*float(np.mean(allpos)):.1f}% of pixels (else R=1)", flush=True)
print("DONE finestructure.")
