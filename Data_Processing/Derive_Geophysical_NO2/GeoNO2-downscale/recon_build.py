"""
Build the two anchor-free historical surface-NO2 reconstructions (2005-2018):

  NO2(y,m,p) = [ A_y(p) / A_2019(p) ] * M_2019(m,p)

  M_2019(m,p) = raw 2019 monthly surface-NO2 estimate (NO2_v7_2019{MM}.nc)
  A_2019(p)   = annual mean geophysical NO2 (GeoNO2_trop_GC) for 2019 (TROPOMI)
  A_y(p)      = annual mean for year y, differing by approach:
      approach1 : annual mean of the ANCHOR-FREE (no-anchor) ML map estimate
                  (Map_Estimation/{y}/NO2_v7_{y}{MM}.nc). Does NOT use geophysical
                  NO2 at all; denominator A_2019 = annual mean of the 2019 estimate
                  (= mean of the 12 M_2019 months). By construction the recon's
                  annual mean == the no-anchor estimate's annual mean, but the
                  monthly/fine spatial pattern is inherited from the 2019 estimate.
      approach2 : OMI geophysical large-scale x TROPOMI fine structure (uses
                  GeoNO2_trop_GC); denominator A_2019 = 2019 geophysical annual mean.
                  A_y_ds = R_fine * boxcar10(A_y_raw)  for OMI-era months,
                  raw for TROPOMI-era months (2018-06+; only affects yr 2018).
                  Since boxcar & annual-mean are linear:
                    fully-OMI year -> R_fine * boxcar10(annual_mean_raw)
                    2018 (mixed)   -> [R_fine*boxcar10(sum OMI mo) + sum TROP mo]/12

Era split: OMI <= 2018-05, TROPOMI >= 2018-06.
Outputs: <this>/maps/app{1,2}/{year}/NO2_recon_v7_{y}{MM}.nc   (1 km CF nc)
Run inside docker (big-mem; scipy, netCDF4).
"""
import os
import numpy as np
import netCDF4 as nc
from scipy.ndimage import uniform_filter

BASE = ("/path/to" if os.path.isdir("/path/to/1.project")
        else "/path/to")
ROOT = f"{BASE}/1.project/NO2_DL_global"
GEOD = f"{ROOT}/input_variables/GeoNO2-v5.13_input"
ESTD = f"{ROOT}/Training_Evaluation_Estimation/NO2/v7/Map_Estimation"
HERE = os.path.dirname(os.path.abspath(__file__))
BOX = 10
YEARS = ([int(y) for y in os.environ["RECON_YEARS"].split(",")]
         if os.environ.get("RECON_YEARS") else list(range(2005, 2019)))
A2019_FLOOR = 1e-2      # ppb-equiv floor on the denominator
RATIO_CLIP = 10.0

# grid axes from an estimation map
g0 = nc.Dataset(f"{ESTD}/2019/NO2_v7_201901.nc")
LAT = np.array(g0.variables["lat"][:]); LON = np.array(g0.variables["lon"][:])
g0.close()
ny, nx = len(LAT), len(LON)


def geo(y, m):
    return np.load(f"{GEOD}/{y}/GeoNO2_trop_GC_001x001_Global_map_{y}{m:02d}.npy").astype(np.float32)


def est(y, m):
    """No-anchor ML surface-NO2 estimate for (y, m)."""
    d = nc.Dataset(f"{ESTD}/{y}/NO2_v7_{y}{m:02d}.nc")
    a = np.array(d.variables["NO2"][:], np.float32); d.close()
    return a[0] if a.ndim == 3 else a


def est_annual_mean(y):
    acc = np.zeros((ny, nx), np.float32); cnt = np.zeros((ny, nx), np.int16)
    for m in range(1, 13):
        a = est(y, m); ok = np.isfinite(a) & (a >= 0)
        acc[ok] += a[ok]; cnt[ok] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)


def boxcar(a):
    mask = np.isfinite(a).astype(np.float32)
    a0 = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    num = uniform_filter(a0, size=BOX, mode="nearest")
    den = uniform_filter(mask, size=BOX, mode="nearest")
    return np.where(den > 0, num / den, np.nan).astype(np.float32)


def annual_mean_geo(y):
    acc = np.zeros((ny, nx), np.float32); cnt = np.zeros((ny, nx), np.int16)
    for m in range(1, 13):
        a = geo(y, m); ok = np.isfinite(a) & (a >= 0)
        acc[ok] += a[ok]; cnt[ok] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)


def write_nc(outfile, arr, y, m):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    d = nc.Dataset(outfile, "w", format="NETCDF4")
    d.createDimension("lat", ny); d.createDimension("lon", nx)
    la = d.createVariable("lat", "f4", ("lat",), zlib=True); la[:] = LAT
    lo = d.createVariable("lon", "f4", ("lon",), zlib=True); lo[:] = LON
    v = d.createVariable("NO2", "f4", ("lat", "lon"), zlib=True, complevel=1,
                         fill_value=np.float32(np.nan))
    v[:] = arr; v.units = "ppb"; v.long_name = f"reconstructed surface NO2 {y}-{m:02d}"
    d.close()


print("loading 2019 estimate (12 mo) ...", flush=True)
M2019 = np.stack([est(2019, m) for m in range(1, 13)])        # (12, ny, nx)
# approach-1 denominator: 2019 ML-estimate annual mean (= mean of the 12 months)
A2019_est = np.nanmean(np.where(M2019 >= 0, M2019, np.nan), axis=0).astype(np.float32)
# approach-2 denominator: 2019 geophysical annual mean
A2019_geo = annual_mean_geo(2019)
Rfine = np.load(os.path.join(HERE, "R_fine.npy"))
print("  loaded.", flush=True)


def denom(approach):
    A = A2019_est if approach == 1 else A2019_geo
    return np.where(np.isfinite(A) & (A > A2019_FLOOR), A, np.nan)


def build(approach):
    tag = f"app{approach}"
    A2019d = denom(approach)
    for y in YEARS:
        if approach == 1:
            Ay = est_annual_mean(y)                           # anchor-free ML estimate
        else:
            if y <= 2017:                                     # fully OMI
                Ay = Rfine * boxcar(annual_mean_geo(y))
            else:                                             # 2018 mixed
                sO = np.zeros((ny, nx), np.float32); cO = np.zeros((ny, nx), np.int16)
                sT = np.zeros((ny, nx), np.float32); cT = np.zeros((ny, nx), np.int16)
                for m in range(1, 13):
                    a = geo(y, m); ok = np.isfinite(a) & (a >= 0)
                    if m <= 5:
                        sO[ok] += a[ok]; cO[ok] += 1
                    else:
                        sT[ok] += a[ok]; cT[ok] += 1
                Ay = (Rfine * boxcar(sO) + sT) / 12.0
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.clip(Ay / A2019d, 0.0, RATIO_CLIP)
        for m in range(1, 13):
            out = np.where(np.isfinite(M2019[m - 1]) & np.isfinite(ratio),
                           ratio * M2019[m - 1], np.nan).astype(np.float32)
            write_nc(f"{HERE}/maps/{tag}/{y}/NO2_recon_v7_{y}{m:02d}.nc", out, y, m)
        print(f"  {tag} {y}: ratio med={np.nanmedian(ratio):.3f} written", flush=True)


import sys
approaches = [int(a) for a in sys.argv[1:]] or [1, 2]
for ap in approaches:
    print(f"=== building approach {ap} ===", flush=True)
    build(ap)
print("DONE reconstruction.", flush=True)
