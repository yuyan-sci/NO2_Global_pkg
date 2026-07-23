"""
geono2_v5.13.py — GeoNO2 + SatColNO2 gap-fill script (TROPOMI & OMI).

v5.13: Same bounds as v5.7 [1.5e-17, 1.5e-15] but with CLAMP instead of clip-to-NaN.
  Daily eta_trop values outside bounds are clamped to [eta_min, eta_max] via np.clip
  instead of being set to NaN. Monthly eta_trop used for gap-fill is also clamped.
  This prevents the gap-fill back door where unclipped monthly_eta reintroduces
  extreme values for pixels that were clipped out at the daily level.

  Satellite data source: NO2col-v3 (AMF > 0.5 filter applied at tessellation).

  eta_trop clamping: values outside [ETA_TROP_MIN, ETA_TROP_MAX] clamped to bounds.

Per-month output variables
--------------------------
  filled_SatColNO2_{tot,trop}_{gcshape,raw}  [gap-filled, Tier 1-4]
  filled_GeoNO2_tot   = mean(daily_SatCol_tot_gc × daily_eta_tot)
  filled_GeoNO2_trop  = mean(daily_SatCol_trop_gc × daily_eta_trop)  (clipped)
  filled_GeoNO2_trop_TM5 = mean(daily_SatCol_trop × daily_eta_trop)
  gchp_NO2, gchp_HNO3, gchp_PAN, gchp_alkylnitrates
  gchp_NO2col_tot, gchp_NO2col_trop, gchp_eta_tot, gchp_eta_trop

Gap-fill formula (Tiers 1-3)
-----------------------------
  SatCol fill = (monthly_GCHP_col / ref_GCHP_col) × ref_SAT
  GeoNO2 fill = (monthly_GCHP_col / ref_GCHP_col) × ref_SAT × monthly_eta
  Tier 4 fallback: fill = monthly_GCHP_col  (where yearly_sat also missing)
"""
import os
import gc
import calendar
import argparse
import numpy as np
import xarray as xr
from datetime import datetime
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
res = '1x1km'

_trop_SZA, _trop_QA = 80, 0.75
QCSTR_TROPOMI = 'SZA{}-QA{}'.format(_trop_SZA, int(_trop_QA * 100))

_omi_CF, _omi_SZA, _omi_QA, _omi_RA = 0.5, 80, 0, 0
_omi_SI1, _omi_SI2, _omi_SI3 = 10, 252, 255
QCSTR_OMI = 'CF{:03d}-SZA{}-QA{}-RA{}-SI1{}-SI2{:03d}-SI3{:03d}'.format(
    int(_omi_CF * 100), _omi_SZA, _omi_QA, _omi_RA, _omi_SI1, _omi_SI2, _omi_SI3
)

def get_qcstr(instrument):
    return QCSTR_TROPOMI if instrument == 'TROPOMI' else QCSTR_OMI

# Directories — v5.13: NO2col-v3 (AMF > 0.5 filter)
GCHP_GEO_ROOT   = '/path/to/gchp-v2/forObservation-Geophysical/{year}/'
TROPOMI_COL_DIR = '/path/to/NO2col-v3/TROPOMI/{year}/'
OMI_COL_DIR     = '/path/to/NO2col-v3/OMI_KNMI/{year}/'
OUT_ROOT        = '/path/to/GeoNO2-v5.13/{year}/'

# --slim output mode: set via CLI, reduces output to ML-essential vars only
_SLIM_OUTPUT = False

# ---------------------------------------------------------------------------
# Seasonal eta_trop clamping bounds
# ---------------------------------------------------------------------------
# Same bounds as v5.7 — but clamped instead of clipped to NaN
ETA_TROP_MIN = 1.5e-17           # constant year-round
ETA_TROP_MAX = 1.5e-15           # constant year-round

def get_eta_trop_bounds(month):
    """Return (eta_min, eta_max) — constant bounds for all months."""
    return ETA_TROP_MIN, ETA_TROP_MAX

def get_sat_col_dir(instrument, year):
    base = TROPOMI_COL_DIR if instrument == 'TROPOMI' else OMI_COL_DIR
    return base.format(year=year)

# ---------------------------------------------------------------------------
# Seasonal helpers
# ---------------------------------------------------------------------------
SEASONS = {'DJF': [12,1,2], 'MAM': [3,4,5], 'JJA': [6,7,8], 'SON': [9,10,11]}
SEASON_CYCLE = ['DJF', 'MAM', 'JJA', 'SON']

def get_season(month):
    for name, months in SEASONS.items():
        if month in months:
            return name

def get_candidate_months_seasonal(current_month, num_adjacent_seasons=0):
    season_idx = SEASON_CYCLE.index(get_season(current_month))
    included = {
        SEASON_CYCLE[(season_idx + offset) % 4]
        for offset in range(-num_adjacent_seasons, num_adjacent_seasons + 1)
    }
    return [m for season in SEASON_CYCLE if season in included
            for m in SEASONS[season] if m != current_month]

# ---------------------------------------------------------------------------
# Low-level I/O helpers
# ---------------------------------------------------------------------------

def slice_latitude(ds, lat_min=-60, lat_max=70):
    for coord in ds.coords:
        if coord.lower() in ['lat', 'latitude', 'y']:
            return ds.sel({coord: slice(lat_min, lat_max)})
    return ds

def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

def _find_file(patterns):
    return next((p for p in patterns if os.path.exists(p)), None)

def load_sat_daily(year, month, day, instrument):
    """Load daily satellite file. Returns dict or None."""
    qcstr = get_qcstr(instrument)
    sat_dir = get_sat_col_dir(instrument, year)
    if instrument == 'TROPOMI':
        path = sat_dir + f'daily/Tropomi_Regrid_{year}{month:02d}{day:02d}_{qcstr}.nc'
    else:
        path = sat_dir + f'daily/OMI_KNMI_Regrid_{year}{month:02d}{day:02d}_{qcstr}.nc'
    if not os.path.exists(path):
        return None
    with xr.open_dataset(path, engine='netcdf4') as ds:
        ds = ds.squeeze()
        result = {'lat': ds['lat'].values, 'lon': ds['lon'].values}
        for var in ['NO2_tot', 'NO2_tot_gcshape', 'NO2_trop', 'NO2_trop_gcshape']:
            if var in ds:
                result[var] = ds[var].values.astype('float32')
            else:
                result[var] = None
    return result

def load_gchp_daily(year, month, day):
    """Load daily GCHP surface vars + total/trop NO2col. Returns dict or None."""
    geo_dir = GCHP_GEO_ROOT.format(year=year)

    path_daily  = geo_dir + f'daily/{res}.DailyVars.{year}{month:02d}{day:02d}.nc4'
    path_3h_tot = geo_dir + f'daily/{res}.Hours.13-15.{year}{month:02d}{day:02d}.nc4'

    if not os.path.exists(path_daily) or not os.path.exists(path_3h_tot):
        return None

    result = {}
    with xr.open_dataset(path_daily, engine='netcdf4') as ds:
        ds = slice_latitude(ds.squeeze(), -60, 70)
        result['gchp_NO2'] = ds['gchp_NO2'].values.astype('float32')
        result['gchp_HNO3'] = ds['gchp_HNO3'].values.astype('float32')
        result['gchp_PAN']  = ds['gchp_PAN'].values.astype('float32')
        if 'gchp_alkylnitrates' in ds:
            result['gchp_alkylnitrates'] = ds['gchp_alkylnitrates'].values.astype('float32')
        elif 'gchp_alklnitrates' in ds:
            result['gchp_alkylnitrates'] = ds['gchp_alklnitrates'].values.astype('float32')
        else:
            result['gchp_alkylnitrates'] = np.zeros_like(result['gchp_NO2'])

    with xr.open_dataset(path_3h_tot, engine='netcdf4') as ds:
        ds = slice_latitude(ds.squeeze(), -60, 70)
        result['gchp_NO2col_tot']  = ds['NO2col_tot'].values.astype('float32')
        result['gchp_NO2col_trop'] = ds['NO2col_trop'].values.astype('float32') \
            if 'NO2col_trop' in ds else None

    return result

def load_gchp_monthly(year, month, column_type):
    """Load monthly GCHP NO2col (total or trop). Returns array or None."""
    geo_dir = GCHP_GEO_ROOT.format(year=year)
    if column_type == 'tot':
        path = _find_file([
            geo_dir + f'monthly/{res}.Hours.13-15.{year}{month:02d}.MonMean.nc',
            geo_dir + f'monthly/{res}.Hours.13-15.{year}{month:02d}.nc',
        ])
        var = 'NO2col_tot'
    else:
        path = _find_file([
            geo_dir + f'monthly/{res}.Hours.13-15.{year}{month:02d}.MonMean.nc',
            geo_dir + f'monthly/{res}.Hours.13-15.{year}{month:02d}.nc',
        ])
        var = 'NO2col_trop'
    if path is None:
        return None
    with xr.open_dataset(path, engine='netcdf4') as ds:
        ds = slice_latitude(ds.squeeze(), -60, 70)
        return ds[var].values.astype('float32')

def load_sat_monthly(year, month, instrument, sat_var):
    """Load monthly satellite column (sat_var='NO2_tot_gcshape' or 'NO2_trop_gcshape').
    Returns (array, lat, lon) or None."""
    qcstr   = get_qcstr(instrument)
    sat_dir = get_sat_col_dir(instrument, year)
    if instrument == 'TROPOMI':
        path = _find_file([
            sat_dir + f'monthly/Tropomi_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc',
            sat_dir + f'monthly/Tropomi_Regrid_{year}{month:02d}_{qcstr}.nc',
        ])
    else:
        path = _find_file([
            sat_dir + f'monthly/OMI_KNMI_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc',
            sat_dir + f'monthly/OMI_KNMI_Regrid_{year}{month:02d}_{qcstr}.nc',
        ])
    if path is None:
        return None
    with xr.open_dataset(path, engine='netcdf4') as ds:
        ds = ds.squeeze()
        if sat_var not in ds:
            # Fallback: try related names
            fallback = {'NO2_trop_gcshape': 'NO2_trop', 'NO2_tot_gcshape': 'NO2_tot'}
            sat_var = fallback.get(sat_var, sat_var)
        if sat_var not in ds:
            return None
        return ds[sat_var].values.astype('float32'), ds['lat'].values, ds['lon'].values

def load_sat_yearly(year, instrument, sat_var):
    """Load yearly satellite column. Returns (array, lat, lon) or None."""
    qcstr   = get_qcstr(instrument)
    sat_dir = get_sat_col_dir(instrument, year)
    if instrument == 'TROPOMI':
        path = _find_file([
            sat_dir + f'yearly/Tropomi_Regrid_{year}_{qcstr}.nc',
            sat_dir + f'yearly/TROPOMI_OMI_Regrid_{year}.nc',
        ])
    else:
        path = _find_file([
            sat_dir + f'yearly/OMI_KNMI_Regrid_{year}_{qcstr}.nc',
        ])
    if path is None:
        return None
    with xr.open_dataset(path, engine='netcdf4') as ds:
        ds = ds.squeeze()
        fb = {'NO2_trop_gcshape': 'NO2_trop', 'NO2_tot_gcshape': 'NO2_tot'}
        v  = sat_var if sat_var in ds else fb.get(sat_var)
        if v is None or v not in ds:
            return None
        return ds[v].values.astype('float32'), ds['lat'].values, ds['lon'].values

def load_yearly_data(year, instrument):
    """Load yearly refs used as gap-fill fallback. Returns dict."""
    geo_dir = GCHP_GEO_ROOT.format(year=year)
    gchp_yearly_path = geo_dir + f'yearly/{res}.Hours.13-15.{year}.AnnualMean.nc'

    with xr.open_dataset(gchp_yearly_path, engine='netcdf4') as ds:
        yearly_gchp_tot = slice_latitude(ds.squeeze(), -60, 70)['NO2col_tot'].values.astype('float32')
        yearly_gchp_trop = slice_latitude(ds.squeeze(), -60, 70)['NO2col_trop'].values.astype('float32')

    result_tot      = load_sat_yearly(year, instrument, 'NO2_tot_gcshape')
    result_trop     = load_sat_yearly(year, instrument, 'NO2_trop_gcshape')
    result_tot_raw  = load_sat_yearly(year, instrument, 'NO2_tot')
    result_trop_raw = load_sat_yearly(year, instrument, 'NO2_trop')

    return {
        'gchp_tot':      yearly_gchp_tot,
        'gchp_trop':     yearly_gchp_trop,
        'sat_tot':       result_tot[0]      if result_tot      else None,
        'sat_trop':      result_trop[0]     if result_trop     else None,
        'sat_tot_raw':   result_tot_raw[0]  if result_tot_raw  else None,
        'sat_trop_raw':  result_trop_raw[0] if result_trop_raw else None,
    }

# ---------------------------------------------------------------------------
# Per-day computation
# ---------------------------------------------------------------------------

def process_single_day(year, month, day, instrument):
    """Load daily satellite + GCHP, compute daily eta and daily GeoNO2.

    Returns dict with all daily variables or None if data missing.
    """
    sat = load_sat_daily(year, month, day, instrument)
    if sat is None:
        return None
    gchp = load_gchp_daily(year, month, day)
    if gchp is None:
        return None

    sfc = gchp['gchp_NO2']
    col_tot  = gchp['gchp_NO2col_tot']
    col_trop = gchp['gchp_NO2col_trop']

    # --- daily eta ---
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_eta_tot = np.where(
            col_tot > 0, sfc / col_tot, np.nan
        ).astype('float32')
        daily_eta_trop = np.where(
            col_trop > 0, sfc / col_trop, np.nan
        ).astype('float32') if col_trop is not None else None

    # --- clamp eta_trop to bounds (no NaN, values pushed to nearest bound) ---
    if daily_eta_trop is not None:
        eta_min, eta_max = get_eta_trop_bounds(month)
        valid = np.isfinite(daily_eta_trop)
        daily_eta_trop = np.where(
            valid, np.clip(daily_eta_trop, eta_min, eta_max), np.nan
        ).astype('float32')

    # --- daily GeoNO2 = daily_SatCol × daily_eta ---
    # GeoNO2_tot: SatCol_tot_gcshape × eta_tot
    sat_tot_gc = sat['NO2_tot_gcshape']
    daily_GeoNO2_tot = None
    if sat_tot_gc is not None:
        daily_GeoNO2_tot = np.where(
            np.isfinite(sat_tot_gc) & np.isfinite(daily_eta_tot),
            sat_tot_gc * daily_eta_tot, np.nan
        ).astype('float32')

    # GeoNO2_trop: SatCol_trop_gcshape × eta_trop
    sat_trop_gc = sat['NO2_trop_gcshape']
    daily_GeoNO2_trop = None
    if sat_trop_gc is not None and daily_eta_trop is not None:
        daily_GeoNO2_trop = np.where(
            np.isfinite(sat_trop_gc) & np.isfinite(daily_eta_trop),
            sat_trop_gc * daily_eta_trop, np.nan
        ).astype('float32')

    # GeoNO2_trop_TM5: SatCol_trop (raw / TM5 AMF) × eta_trop
    sat_trop_raw = sat['NO2_trop']
    daily_GeoNO2_trop_TM5 = None
    if sat_trop_raw is not None and daily_eta_trop is not None:
        daily_GeoNO2_trop_TM5 = np.where(
            np.isfinite(sat_trop_raw) & np.isfinite(daily_eta_trop),
            sat_trop_raw * daily_eta_trop, np.nan
        ).astype('float32')

    return {
        'lat': sat['lat'],
        'lon': sat['lon'],
        'SatColNO2_tot':           sat['NO2_tot'],
        'SatColNO2_trop':          sat['NO2_trop'],
        'SatColNO2_tot_gcshape':   sat['NO2_tot_gcshape'],
        'SatColNO2_trop_gcshape':  sat['NO2_trop_gcshape'],
        'gchp_NO2':                sfc,
        'gchp_HNO3':               gchp['gchp_HNO3'],
        'gchp_PAN':                gchp['gchp_PAN'],
        'gchp_alkylnitrates':      gchp['gchp_alkylnitrates'],
        'gchp_NO2col_tot':         col_tot,
        'gchp_NO2col_trop':        col_trop,
        'GeoNO2_tot':              daily_GeoNO2_tot,
        'GeoNO2_trop':             daily_GeoNO2_trop,
        'GeoNO2_trop_TM5':        daily_GeoNO2_trop_TM5,
    }

# ---------------------------------------------------------------------------
# Reference calculation for gap-filling
# ---------------------------------------------------------------------------

def calculate_reference(year, current_month, instrument, candidate_months,
                        column_type, sat_var=None):
    """
    Average monthly GCHP col and satellite col over candidate_months,
    at pixels where the satellite had valid observations.

    column_type: 'tot' or 'trop'
    sat_var: satellite variable name to load (default: NO2_{column_type}_gcshape)
    Returns (ref_gchp, ref_sat, months_used).
    """
    if sat_var is None:
        sat_var = 'NO2_tot_gcshape' if column_type == 'tot' else 'NO2_trop_gcshape'
    print(f"    [{column_type}] Candidate months: {candidate_months}")

    gchp_sum    = None
    sat_sum     = None
    valid_count = None
    months_used = []

    for month in candidate_months:
        sat_result = load_sat_monthly(year, month, instrument, sat_var)
        if sat_result is None:
            continue
        gchp_mon = load_gchp_monthly(year, month, column_type)
        if gchp_mon is None:
            continue

        sat_mon, _, _ = sat_result
        valid_mask = (sat_mon > 0) & np.isfinite(sat_mon) & np.isfinite(gchp_mon)
        if not valid_mask.any():
            continue

        if gchp_sum is None:
            gchp_sum    = np.zeros_like(gchp_mon, dtype='float64')
            sat_sum     = np.zeros_like(sat_mon,  dtype='float64')
            valid_count = np.zeros_like(gchp_mon, dtype='int32')

        gchp_sum    = np.where(valid_mask, gchp_sum    + gchp_mon.astype('float64'), gchp_sum)
        sat_sum     = np.where(valid_mask, sat_sum     + sat_mon.astype('float64'),  sat_sum)
        valid_count = np.where(valid_mask, valid_count + 1, valid_count)
        months_used.append(month)
        gc.collect()

    if not months_used:
        return None, None, []

    with np.errstate(divide='ignore', invalid='ignore'):
        ref_gchp = np.where(valid_count > 0, (gchp_sum / valid_count).astype('float32'), np.nan)
        ref_sat  = np.where(valid_count > 0, (sat_sum  / valid_count).astype('float32'), np.nan)

    print(f"    [{column_type}] Reference from {len(months_used)} months: {months_used}")
    return ref_gchp, ref_sat, months_used

# ---------------------------------------------------------------------------
# Unified gap-fill (GeoNO2 or SatColNO2)
# ---------------------------------------------------------------------------

def fill_missing(year, current_month, instrument, column_type,
                 monthly_missing, monthly_gchp_col, yearly_sat,
                 surface_ratio=None, sat_var=None):
    """
    Pixel-level unbiased gap-fill with seasonal borrowing cascade.

    SatColNO2 fill (surface_ratio=None):
        Tiers 1-3: fill = (monthly_GCHP_col / ref_GCHP_col) x ref_SAT
        Tier 4:    fill = monthly_GCHP_col  (where yearly_sat also missing)

    GeoNO2 fill (surface_ratio=monthly_eta):
        Tiers 1-3: fill = (monthly_GCHP_col / ref_GCHP_col) x ref_SAT x monthly_eta
        Tier 4:    fill = monthly_GCHP_col  (where yearly_sat also missing)

    sat_var: satellite variable for calculate_reference (default: NO2_{column_type}_gcshape)
    """
    fallback_tiers = [
        (f'Tier 1 -- same season ({get_season(current_month)})',
         get_candidate_months_seasonal(current_month, 0), 1),
        (f'Tier 2 -- same + adjacent seasons ({get_season(current_month)} +/- 1)',
         get_candidate_months_seasonal(current_month, 1), 2),
        ('Tier 3 -- all months in year',
         [m for m in range(1, 13) if m != current_month], 1),
    ]

    sat_missing = ~np.isfinite(monthly_missing)
    fill_values = np.full(monthly_missing.shape, np.nan, dtype='float32')
    total       = monthly_missing.size
    n_orig_miss = int(sat_missing.sum())

    for label, candidates, min_months in fallback_tiers:
        still_needed = sat_missing & np.isnan(fill_values)
        if not still_needed.any():
            break
        print(f"  {label}: {int(still_needed.sum()):,} pixels")

        rg, rs, months_used = calculate_reference(
            year, current_month, instrument, candidates, column_type, sat_var)

        if rg is None or len(months_used) < min_months:
            print(f"    Insufficient data -- skipping tier")
            continue

        # SAT reference: seasonal avg with yearly fallback per pixel
        nan_rs  = np.isnan(rs)
        sat_ref = np.where(nan_rs & (yearly_sat is not None),
                           yearly_sat if yearly_sat is not None else np.nan,
                           rs).astype('float32')

        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where((rg > 0) & np.isfinite(monthly_gchp_col),
                             monthly_gchp_col / rg, np.nan).astype('float32')

        fill_this = scale * sat_ref
        if surface_ratio is not None:
            fill_this = fill_this * surface_ratio
        fill_this = fill_this.astype('float32')

        applied    = still_needed & np.isfinite(fill_this)
        fill_values = np.where(applied, fill_this, fill_values)
        n_still    = int(np.sum(sat_missing & np.isnan(fill_values)))
        print(f"    -> filled {int(applied.sum()):,}; {n_still:,} still missing")

    # Tier 4 (GCHP-direct) -- fill all remaining pixels where GCHP is available
    # For GeoNO2 (surface_ratio != None): multiply by eta to convert column → surface ppb
    still_needed = sat_missing & np.isnan(fill_values)
    if still_needed.any():
        fb = monthly_gchp_col.copy()
        if surface_ratio is not None:
            fb = (fb * surface_ratio).astype('float32')
        fb_mask = still_needed & np.isfinite(fb)
        fill_values = np.where(fb_mask, fb, fill_values)
        n_fb = int(fb_mask.sum())
        if n_fb:
            print(f"  Tier 4 -- GCHP direct: filled {n_fb:,} pixels")

    result = np.where(sat_missing, fill_values, monthly_missing).astype('float32')

    n_filled = int((sat_missing & np.isfinite(fill_values)).sum())
    n_still  = int((~np.isfinite(result)).sum())
    print(f"  Fill summary [{column_type}]: orig_missing={n_orig_miss:,}  "
          f"filled={n_filled:,}  still_missing={n_still:,}  total={total:,}")
    if n_still > 0:
        print(f"  [WARN] {n_still:,} pixels still NaN after Tier 4 -- "
              f"GCHP monthly file may have missing values at those pixels")
    return result

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PLOT_VARS = [
    'filled_GeoNO2_tot',            'filled_GeoNO2_trop',
    'filled_GeoNO2_trop_TM5',
    'gap_SatColNO2_tot_gcshape',    'filled_SatColNO2_tot_gcshape',
    'gap_SatColNO2_trop_gcshape',   'filled_SatColNO2_trop_gcshape',
    'gap_SatColNO2_tot',            'filled_SatColNO2_tot',
    'gap_SatColNO2_trop',           'filled_SatColNO2_trop',
    'gchp_NO2', 'gchp_eta_tot', 'gchp_eta_trop',
]
PLOT_LABELS = {
    'filled_GeoNO2_tot':             'GeoNO2 total (filled)',
    'filled_GeoNO2_trop':            'GeoNO2 trop (filled)',
    'filled_GeoNO2_trop_TM5':       'GeoNO2 trop TM5 (filled)',
    'gap_SatColNO2_tot_gcshape':     'SatColNO2 tot gcshape (before fill)',
    'filled_SatColNO2_tot_gcshape':  'SatColNO2 tot gcshape (filled)',
    'gap_SatColNO2_trop_gcshape':    'SatColNO2 trop gcshape (before fill)',
    'filled_SatColNO2_trop_gcshape': 'SatColNO2 trop gcshape (filled)',
    'gap_SatColNO2_tot':             'SatColNO2 total (before fill)',
    'filled_SatColNO2_tot':          'SatColNO2 total (filled)',
    'gap_SatColNO2_trop':            'SatColNO2 trop (before fill)',
    'filled_SatColNO2_trop':         'SatColNO2 trop (filled)',
    'gchp_NO2':                      'GCHP surface NO\u2082',
    'gchp_eta_tot':                  'GCHP \u03b7 total',
    'gchp_eta_trop':                 'GCHP \u03b7 trop',
}
VMAX = {k: 15 for k in PLOT_LABELS}
for _v in ['gap_SatColNO2_tot_gcshape', 'filled_SatColNO2_tot_gcshape',
           'gap_SatColNO2_trop_gcshape', 'filled_SatColNO2_trop_gcshape',
           'gap_SatColNO2_tot', 'filled_SatColNO2_tot',
           'gap_SatColNO2_trop', 'filled_SatColNO2_trop']:
    VMAX[_v] = 1e16
for _v in ['gchp_eta_tot', 'gchp_eta_trop']:
    VMAX[_v] = 1e-15

# Fixed 6x3 plot layout
PLOT_LAYOUT = [
    ['gap_SatColNO2_tot_gcshape',    'gap_SatColNO2_trop_gcshape',    'gchp_NO2'                       ],
    ['filled_SatColNO2_tot_gcshape', 'filled_SatColNO2_trop_gcshape', 'filled_GeoNO2_tot'              ],
    ['gap_SatColNO2_tot',            'gap_SatColNO2_trop',            'filled_GeoNO2_trop'             ],
    ['filled_SatColNO2_tot',         'filled_SatColNO2_trop',         'filled_GeoNO2_trop_TM5'        ],
    ['filled_GeoNO2_trop',           'filled_GeoNO2_trop_TM5',        'gchp_eta_tot'                   ],
    [None,                           None,                            'gchp_eta_trop'                  ],
]

def plot_geono2_monthly(outpath, year, month, instrument):
    try:
        with xr.open_dataset(outpath, engine='netcdf4') as ds:
            lats = ds['lat'].values
            lons = ds['lon'].values
            nrows, ncols = len(PLOT_LAYOUT), len(PLOT_LAYOUT[0])
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(ncols * 7, nrows * 3),
                subplot_kw={'projection': ccrs.PlateCarree()},
                squeeze=False,
            )
            ext = [float(lons.min()), float(lons.max()),
                   float(lats.min()), float(lats.max())]
            for r, row_vars in enumerate(PLOT_LAYOUT):
                for c, var in enumerate(row_vars):
                    ax = axes[r, c]
                    if var is None or var not in ds.data_vars:
                        ax.set_visible(False)
                        continue
                    v = ds[var].values.squeeze()
                    if v.size == 0 or np.all(np.isnan(v)):
                        ax.set_visible(False)
                        continue
                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax.add_feature(cfeature.BORDERS,   linewidth=0.3)
                    ax.set_extent(ext, crs=ccrs.PlateCarree())
                    mesh = ax.pcolormesh(lons, lats, v,
                                         transform=ccrs.PlateCarree(),
                                         cmap='RdYlBu_r', vmin=0,
                                         vmax=VMAX.get(var, 15))
                    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                                        pad=0.02, fraction=0.03, shrink=0.8)
                    cbar.set_label(PLOT_LABELS.get(var, var), fontsize=8)
                    ax.set_title(f"{instrument}  {year}-{month:02d}\n"
                                 f"{PLOT_LABELS.get(var, var)}", fontsize=9, pad=4)
                    ax.gridlines(draw_labels=(c == 0), alpha=0.3, linewidth=0.4)
        fig.suptitle(f"{instrument}  {year}-{month:02d}", fontsize=13, y=1.01)
        fig.subplots_adjust(hspace=0.45, wspace=0.08)
        png_path = outpath.replace('.nc', '.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved plot: {png_path}", flush=True)
    except Exception as e:
        print(f"  [WARN] Plot failed: {e}", flush=True)

# ---------------------------------------------------------------------------
# Monthly streaming processor
# ---------------------------------------------------------------------------

# Variables accumulated day-by-day (includes daily GeoNO2)
_DAY_VARS = [
    'SatColNO2_tot', 'SatColNO2_trop',
    'SatColNO2_tot_gcshape', 'SatColNO2_trop_gcshape',
    'gchp_NO2', 'gchp_HNO3', 'gchp_PAN', 'gchp_alkylnitrates',
    'gchp_NO2col_tot', 'gchp_NO2col_trop',
    'GeoNO2_tot', 'GeoNO2_trop', 'GeoNO2_trop_TM5',
]


def process_monthly_streaming(year, month, instrument, yearly_refs, make_plots=True):
    print(f"\n[{instrument}] Processing {year}-{month:02d}...")

    eta_min, eta_max = get_eta_trop_bounds(month)
    print(f"  eta_trop bounds for month {month}: [{eta_min:.3e}, {eta_max:.3e}]")

    days = get_days_in_month(year, month)
    monthly_sums  = {}
    valid_counts  = {}
    valid_day_cnt = 0
    lat_out = lon_out = None

    for day in range(1, days + 1):
        result = process_single_day(year, month, day, instrument)
        if result is None:
            print(f"  Day {day:02d}: missing")
            continue

        if valid_day_cnt == 0:
            lat_out = result['lat']
            lon_out = result['lon']
            for var in _DAY_VARS:
                v = result.get(var)
                if v is not None:
                    monthly_sums[var]  = np.zeros_like(v, dtype='float64')
                    valid_counts[var]  = np.zeros_like(v, dtype='int32')

        for var in list(monthly_sums.keys()):
            v = result.get(var)
            if v is None:
                continue
            mask = ~np.isnan(v)
            monthly_sums[var]  = np.where(mask, monthly_sums[var]  + v.astype('float64'), monthly_sums[var])
            valid_counts[var]  = np.where(mask, valid_counts[var]  + 1,                   valid_counts[var])

        valid_day_cnt += 1
        del result
        gc.collect()
        print(f"  Day {day:02d}: ok")

    if valid_day_cnt == 0:
        print(f"  No valid days for {year}-{month:02d}")
        return False

    print(f"  Computing monthly means from {valid_day_cnt} days...")
    means = {}
    for var, s in monthly_sums.items():
        vc = valid_counts[var]
        means[var] = np.where(vc > 0, (s / vc).astype('float32'), np.nan)

    # Monthly GCHP columns for gap-fill scaling
    monthly_gchp_tot  = load_gchp_monthly(year, month, 'tot')
    monthly_gchp_trop = load_gchp_monthly(year, month, 'trop')

    # ---- Gap-fill SatColNO2_tot_gcshape ------------------------------------
    if 'SatColNO2_tot_gcshape' in means and monthly_gchp_tot is not None:
        print("  Gap-filling SatColNO2_tot_gcshape...")
        filled_SatColNO2_tot_gcshape = fill_missing(
            year, month, instrument, 'tot',
            means['SatColNO2_tot_gcshape'], monthly_gchp_tot,
            yearly_refs.get('sat_tot'),
        )
    else:
        filled_SatColNO2_tot_gcshape = means.get('SatColNO2_tot_gcshape')

    # ---- Gap-fill SatColNO2_trop_gcshape -----------------------------------
    if 'SatColNO2_trop_gcshape' in means and monthly_gchp_trop is not None:
        print("  Gap-filling SatColNO2_trop_gcshape...")
        filled_SatColNO2_trop_gcshape = fill_missing(
            year, month, instrument, 'trop',
            means['SatColNO2_trop_gcshape'], monthly_gchp_trop,
            yearly_refs.get('sat_trop'),
        )
    else:
        filled_SatColNO2_trop_gcshape = means.get('SatColNO2_trop_gcshape')

    # ---- Gap-fill SatColNO2_tot (raw) --------------------------------------
    if 'SatColNO2_tot' in means and monthly_gchp_tot is not None:
        print("  Gap-filling SatColNO2_tot (raw)...")
        filled_SatColNO2_tot = fill_missing(
            year, month, instrument, 'tot',
            means['SatColNO2_tot'], monthly_gchp_tot,
            yearly_refs.get('sat_tot_raw'), sat_var='NO2_tot',
        )
    else:
        filled_SatColNO2_tot = means.get('SatColNO2_tot')

    # ---- Gap-fill SatColNO2_trop (raw) -------------------------------------
    if 'SatColNO2_trop' in means and monthly_gchp_trop is not None:
        print("  Gap-filling SatColNO2_trop (raw)...")
        filled_SatColNO2_trop = fill_missing(
            year, month, instrument, 'trop',
            means['SatColNO2_trop'], monthly_gchp_trop,
            yearly_refs.get('sat_trop_raw'), sat_var='NO2_trop',
        )
    else:
        filled_SatColNO2_trop = means.get('SatColNO2_trop')

    # ---- Monthly eta (from monthly means) for gap-fill only ----------------
    # v5.9: GeoNO2 comes from daily computation, but gap-fill still needs
    # monthly_eta to convert filled SatCol -> surface ppb.
    # Monthly eta is CLAMPED to the same bounds used for daily eta.
    mean_sfc      = means.get('gchp_NO2')
    mean_col_tot  = means.get('gchp_NO2col_tot')
    mean_col_trop = means.get('gchp_NO2col_trop')

    with np.errstate(divide='ignore', invalid='ignore'):
        monthly_eta_tot = (np.where(mean_col_tot > 0, mean_sfc / mean_col_tot, 0).astype('float32')
                           if (mean_sfc is not None and mean_col_tot is not None) else None)
        monthly_eta_trop_raw = (np.where(mean_col_trop > 0, mean_sfc / mean_col_trop, 0).astype('float32')
                                if (mean_sfc is not None and mean_col_trop is not None) else None)

    # Clamp monthly eta_trop to same bounds (closes the gap-fill back door)
    monthly_eta_trop = None
    monthly_eta_trop_clamped = None
    if monthly_eta_trop_raw is not None:
        monthly_eta_trop = monthly_eta_trop_raw
        monthly_eta_trop_clamped = np.clip(monthly_eta_trop_raw, eta_min, eta_max).astype('float32')
        n_clamped = int(((monthly_eta_trop_raw != monthly_eta_trop_clamped) &
                         (monthly_eta_trop_raw > 0)).sum())
        n_valid = int((monthly_eta_trop_raw > 0).sum())
        print(f"  Monthly eta_trop: clamped {n_clamped:,}/{n_valid:,} pixels "
              f"({100*n_clamped/n_valid:.1f}%) to [{eta_min:.3e}, {eta_max:.3e}]")

    # ---- Gap-fill GeoNO2_tot (daily-averaged) ------------------------------
    # Monthly mean GeoNO2 comes from averaging daily GeoNO2 values.
    # Gap-fill uses: fill = (GCHP/ref_GCHP) x ref_SAT x monthly_eta
    mean_GeoNO2_tot = means.get('GeoNO2_tot')
    if mean_GeoNO2_tot is not None and monthly_gchp_tot is not None and monthly_eta_tot is not None:
        print("  Gap-filling GeoNO2_tot...")
        filled_GeoNO2_tot = fill_missing(
            year, month, instrument, 'tot',
            mean_GeoNO2_tot, monthly_gchp_tot,
            yearly_refs.get('sat_tot'),
            surface_ratio=monthly_eta_tot,
        )
    else:
        filled_GeoNO2_tot = mean_GeoNO2_tot

    # ---- Gap-fill GeoNO2_trop (daily-averaged, eta-clamped) ----------------
    mean_GeoNO2_trop = means.get('GeoNO2_trop')
    if mean_GeoNO2_trop is not None and monthly_gchp_trop is not None and monthly_eta_trop_clamped is not None:
        print("  Gap-filling GeoNO2_trop (using clamped monthly eta)...")
        filled_GeoNO2_trop = fill_missing(
            year, month, instrument, 'trop',
            mean_GeoNO2_trop, monthly_gchp_trop,
            yearly_refs.get('sat_trop'),
            surface_ratio=monthly_eta_trop_clamped,
        )
    else:
        filled_GeoNO2_trop = mean_GeoNO2_trop

    # ---- Gap-fill GeoNO2_trop_TM5 (daily-averaged, eta-clamped) -----------
    mean_GeoNO2_trop_TM5 = means.get('GeoNO2_trop_TM5')
    if mean_GeoNO2_trop_TM5 is not None and monthly_gchp_trop is not None and monthly_eta_trop_clamped is not None:
        print("  Gap-filling GeoNO2_trop_TM5 (using clamped monthly eta)...")
        filled_GeoNO2_trop_TM5 = fill_missing(
            year, month, instrument, 'trop',
            mean_GeoNO2_trop_TM5, monthly_gchp_trop,
            yearly_refs.get('sat_trop_raw'),
            surface_ratio=monthly_eta_trop_clamped, sat_var='NO2_trop',
        )
    else:
        filled_GeoNO2_trop_TM5 = mean_GeoNO2_trop_TM5

    # ---- Build output dataset -----------------------------------------------
    save_vars = {}

    def _add(name, arr):
        if arr is not None:
            save_vars[name] = (['lat', 'lon'], arr)

    # SatColNO2: gap (pre-fill monthly mean) + filled
    _add('gap_SatColNO2_tot_gcshape',    means.get('SatColNO2_tot_gcshape'))
    _add('filled_SatColNO2_tot_gcshape', filled_SatColNO2_tot_gcshape)
    _add('gap_SatColNO2_trop_gcshape',   means.get('SatColNO2_trop_gcshape'))
    _add('filled_SatColNO2_trop_gcshape',filled_SatColNO2_trop_gcshape)
    _add('gap_SatColNO2_tot',            means.get('SatColNO2_tot'))
    _add('filled_SatColNO2_tot',         filled_SatColNO2_tot)
    _add('gap_SatColNO2_trop',           means.get('SatColNO2_trop'))
    _add('filled_SatColNO2_trop',        filled_SatColNO2_trop)
    # GeoNO2: gap (pre-fill daily mean) + filled
    _add('gap_GeoNO2_tot',              mean_GeoNO2_tot)
    _add('gap_GeoNO2_trop',             mean_GeoNO2_trop)
    _add('gap_GeoNO2_trop_TM5',        mean_GeoNO2_trop_TM5)
    _add('filled_GeoNO2_tot',            filled_GeoNO2_tot)
    _add('filled_GeoNO2_trop',           filled_GeoNO2_trop)
    _add('filled_GeoNO2_trop_TM5',      filled_GeoNO2_trop_TM5)
    # GCHP auxiliary vars
    _add('gchp_NO2',          means.get('gchp_NO2'))
    _add('gchp_HNO3',         means.get('gchp_HNO3'))
    _add('gchp_PAN',          means.get('gchp_PAN'))
    _add('gchp_alkylnitrates',means.get('gchp_alkylnitrates'))
    _add('gchp_NO2col_tot',   means.get('gchp_NO2col_tot'))
    _add('gchp_NO2col_trop',  means.get('gchp_NO2col_trop'))
    _add('gchp_eta_tot',      monthly_eta_tot)
    _add('gchp_eta_trop',     monthly_eta_trop)
    _add('gchp_eta_trop_clamped', monthly_eta_trop_clamped)

    # --slim: keep only ML-essential variables
    if _SLIM_OUTPUT:
        _SLIM_VARS = {
            'filled_GeoNO2_trop', 'filled_SatColNO2_trop_gcshape',
            'filled_SatColNO2_trop',
            'gchp_NO2', 'gchp_alkylnitrates', 'gchp_HNO3', 'gchp_PAN',
        }
        n_before = len(save_vars)
        save_vars = {k: v for k, v in save_vars.items() if k in _SLIM_VARS}
        print(f"  --slim: kept {len(save_vars)}/{n_before} variables")

    ds_out = xr.Dataset(
        save_vars,
        coords={
            'lat':  lat_out,
            'lon':  lon_out,
            'time': datetime(year, month, 15),
        },
        attrs={
            'title':       f'GeoNO2 {instrument} {year}-{month:02d}',
            'version':     'v5.13 — daily GeoNO2 + eta_trop clamping (v5.7 bounds)',
            'instrument':  instrument,
            'qc_filter':   get_qcstr(instrument),
            'sat_source':  'NO2col-v3 (AMF > 0.5)',
            'eta_trop_min': f'{eta_min:.3e}',
            'eta_trop_max': f'{eta_max:.3e}',
            'days_in_month': days,
            'days_averaged': valid_day_cnt,
            'created':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    )

    out_dir = OUT_ROOT.format(year=year)
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f'{res}.GeoNO2.{year}{month:02d}.MonMean.nc')
    enc = {v: {'zlib': True, 'complevel': 4, 'shuffle': True, 'dtype': 'float32'}
           for v in ds_out.data_vars}
    ds_out.to_netcdf(outpath, encoding=enc)
    print(f"  Saved: {outpath} ({os.path.getsize(outpath)/1024**2:.1f} MB)")

    del monthly_sums, valid_counts, means, ds_out
    gc.collect()

    if make_plots:
        plot_geono2_monthly(outpath, year, month, instrument)

    return True

# ---------------------------------------------------------------------------
# Yearly average
# ---------------------------------------------------------------------------

def process_yearly_average(year, instrument):
    print(f"\n=== Annual average {instrument} {year} ===")
    monthly_dir = OUT_ROOT.format(year=year)
    sums   = {}
    counts = {}
    lat_c  = lon_c = None
    months_found = []

    for month in range(1, 13):
        fpath = os.path.join(monthly_dir,
                             f'{res}.GeoNO2.{year}{month:02d}.MonMean.nc')
        if not os.path.exists(fpath):
            continue
        with xr.open_dataset(fpath, engine='netcdf4') as ds:
            if lat_c is None:
                lat_c = ds['lat'].values
                lon_c = ds['lon'].values
                for v in ds.data_vars:
                    sums[v]   = np.zeros_like(ds[v].values, dtype='float64')
                    counts[v] = np.zeros_like(ds[v].values, dtype='int32')
            for v in ds.data_vars:
                d = ds[v].values
                m = ~np.isnan(d)
                sums[v]   = np.where(m, sums[v] + d.astype('float64'), sums[v])
                counts[v] = np.where(m, counts[v] + 1, counts[v])
        months_found.append(month)
        gc.collect()

    if not months_found:
        print("  No monthly files found.")
        return False

    annual = {v: np.where(counts[v] > 0, (sums[v] / counts[v]).astype('float32'), np.nan)
              for v in sums}
    ds_ann = xr.Dataset(
        {v: (['lat', 'lon'], arr) for v, arr in annual.items()},
        coords={'lat': lat_c, 'lon': lon_c, 'time': datetime(year, 7, 1)},
        attrs={'title': f'GeoNO2 {instrument} {year} annual mean',
               'months': str(months_found)}
    )
    enc = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'} for v in ds_ann.data_vars}
    ann_path = os.path.join(monthly_dir, f'{res}.GeoNO2.{year}.AnnualMean.nc')
    ds_ann.to_netcdf(ann_path, encoding=enc)
    print(f"  Annual mean saved: {ann_path}")
    return True

# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def process_single_month(year, month, instrument, make_plots=True):
    print(f"\n=== {instrument} {year}-{month:02d} ===")
    try:
        yearly_refs = load_yearly_data(year, instrument)
    except Exception as e:
        print(f"  [ERROR] Could not load yearly refs: {e}")
        yearly_refs = {}
    return process_monthly_streaming(year, month, instrument, yearly_refs, make_plots)


def process_year(year, instrument, make_plots=True):
    print(f"\n=== {instrument} {year} ===")
    try:
        yearly_refs = load_yearly_data(year, instrument)
    except Exception as e:
        print(f"  [ERROR] Could not load yearly refs: {e}")
        yearly_refs = {}

    n_ok = 0
    for month in range(1, 13):
        if process_monthly_streaming(year, month, instrument, yearly_refs, make_plots):
            n_ok += 1
        gc.collect()

    if n_ok > 0:
        process_yearly_average(year, instrument)

    print(f"\n=== {instrument} {year}: {n_ok}/12 months processed ===")
    return n_ok > 0

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='GeoNO2 v5.13 (daily GeoNO2 + eta_trop clamping)')
    parser.add_argument('year',   type=int)
    parser.add_argument('--month', type=int, choices=range(1, 13))
    parser.add_argument('--instrument', choices=['TROPOMI', 'OMI', 'both'], default='both')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip processing; regenerate plots from existing NetCDF files')
    parser.add_argument('--slim', action='store_true',
                        help='Output only ML-essential vars (saves ~70%% disk)')
    args = parser.parse_args()

    make_plots   = not args.no_plot
    instruments  = ['TROPOMI', 'OMI'] if args.instrument == 'both' else [args.instrument]

    global _SLIM_OUTPUT
    _SLIM_OUTPUT = args.slim

    print(f"geono2_v5.13  year={args.year}  instrument={args.instrument}"
          f"{'  [SLIM OUTPUT]' if _SLIM_OUTPUT else ''}")
    print(f"  daily GeoNO2 + eta_trop clamping")
    print(f"  sat source: NO2col-v3 (AMF > 0.5)")
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}  "
          f"Job: {os.getenv('LSB_JOBID', 'not_in_lsf')}")
    print(f"Start: {datetime.now():%Y-%m-%d %H:%M:%S}")

    if args.plot_only:
        monthly_dir = OUT_ROOT.format(year=args.year)
        months = [args.month] if args.month else range(1, 13)
        for inst in instruments:
            for month in months:
                outpath = os.path.join(monthly_dir,
                                       f'{res}.GeoNO2.{args.year}{month:02d}.MonMean.nc')
                if os.path.exists(outpath):
                    plot_geono2_monthly(outpath, args.year, month, inst)
                else:
                    print(f"  [SKIP] Not found: {outpath}")
    else:
        for inst in instruments:
            if args.month:
                process_single_month(args.year, args.month, inst, make_plots)
            else:
                process_year(args.year, inst, make_plots)

    print("\nDone.")


if __name__ == '__main__':
    main()
