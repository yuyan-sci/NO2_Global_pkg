"""
Approach-2 (retrain track), step A: build the DOWNSCALED geophysical NO2 that will
REPLACE the GeoNO2 input channel for OMI-era months, so the training data can be
rebuilt and the model retrained (a genuine anchor-free ML estimate that does NOT
use the 2019 pattern -- it uses OMI large-scale + TROPOMI fine structure).

Per MONTH (matches the training pipeline, which reads monthly GeoNO2):
  OMI months (2005 .. 2018-05):  GeoNO2_ds = R_fine * boxcar10( GeoNO2_OMI(y,m) )
  TROPOMI months (2018-06 .. ):  GeoNO2_ds = GeoNO2(y,m)  (already native 1km)

R_fine (from recon_finestructure.py, spatial all-positive filtered: applied only
where the pixel and its whole 10x10 block are strictly positive) is 1 everywhere
else, so those pixels just get the coarse OMI value -> no injected noise. The
10km-block mean of GeoNO2_ds equals boxcar10(OMI), i.e. the coarse OMI signal is
conserved; R_fine only redistributes within blocks.

Output: input_variables/GeoNO2-v5.13_downscaled/{year}/GeoNO2_trop_GC_001x001_Global_map_{YYYYMM}.npy
        (same filename as the original so a rebuilt training config only needs a
         different GeoNO2 indir). Negatives from OMI are preserved as-is.

Env: RECON_YEARS="2005"  (comma list of years).
     RECON_MONTHS="1"     (comma list of months, 1-12; default all 12) -- lets you
                          build a single month first (e.g. 2005-01) as a smoke test.
Run inside docker (scipy, big-mem).
"""
import os
import numpy as np
from scipy.ndimage import uniform_filter

BASE = ("/path/to" if os.path.isdir("/path/to/1.project")
        else "/path/to")
ROOT = f"{BASE}/1.project/NO2_DL_global"
GEOD = f"{ROOT}/input_variables/GeoNO2-v5.13_input"
OUTD = f"{ROOT}/input_variables/GeoNO2-v5.13_downscaled"
HERE = os.path.dirname(os.path.abspath(__file__))
BOX = 10
YEARS = [int(y) for y in os.environ.get("RECON_YEARS", "2005").split(",")]
MONTHS = [int(m) for m in os.environ.get("RECON_MONTHS", "1,2,3,4,5,6,7,8,9,10,11,12").split(",")]

Rfine = np.load(os.path.join(HERE, "R_fine.npy"))
print(f"R_fine loaded {Rfine.shape}; years={YEARS} months={MONTHS}", flush=True)


def geo(y, m):
    return np.load(f"{GEOD}/{y}/GeoNO2_trop_GC_001x001_Global_map_{y}{m:02d}.npy").astype(np.float32)


def boxcar10(a):
    """10x10-pixel (0.1deg) nan-safe boxcar coarsening; preserves the block mean."""
    mask = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    num = uniform_filter(a0, size=BOX, mode="nearest")
    den = uniform_filter(mask, size=BOX, mode="nearest")
    return np.where(den > 0, num / den, np.nan).astype(np.float32)


def is_omi(y, m):
    return (y < 2018) or (y == 2018 and m <= 5)


for y in YEARS:
    outdir = f"{OUTD}/{y}"
    os.makedirs(outdir, exist_ok=True)
    for m in MONTHS:
        g = geo(y, m)
        if is_omi(y, m):
            ds = (Rfine * boxcar10(g)).astype(np.float32)     # OMI 10km x TROPOMI fine structure
        else:
            ds = g                                            # TROPOMI native 1km, unchanged
        outf = f"{outdir}/GeoNO2_trop_GC_001x001_Global_map_{y}{m:02d}.npy"
        np.save(outf, ds)
        with np.errstate(invalid="ignore"):
            fin = np.isfinite(ds)
            print(f"  {y}-{m:02d} [{'OMI-ds' if is_omi(y,m) else 'TROP-raw'}]: "
                  f"median={np.nanmedian(ds[fin]):.3f} neg%={100*np.mean(ds[fin]<0):.1f}", flush=True)
    print(f"  {y}: wrote {len(MONTHS)} downscaled GeoNO2 -> {outdir}", flush=True)
print("DONE downscaled GeoNO2.", flush=True)
