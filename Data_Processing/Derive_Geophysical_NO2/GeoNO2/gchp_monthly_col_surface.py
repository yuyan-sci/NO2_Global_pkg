#!/usr/bin/env python3
"""
Monthly average of GCHP NO2 column (NO2col) and surface NO2 (gchp_NO2),
plus their ratio NO2col / gchp_NO2.

Saves a compressed NetCDF and produces a 3-panel map figure.

Usage:
    python gchp_monthly_col_surface.py <year> <month>
    python gchp_monthly_col_surface.py 2019 7
"""
import os
import sys
import gc
import calendar
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── CONFIG ──────────────────────────────────────────────────────────────────
res      = '1x1km'
LAT_MIN  = -60
LAT_MAX  =  70

gchp_base = '/path/to/gchp/forObservation-Geophysical/'
out_base  = '/path/to/gchp/gchp_col_surface/'

# Downsampling factor for plotting (the 0.01° grid is too large to render directly)
PLOT_STRIDE = 20   # keep every Nth pixel → ~0.2° effective resolution for the figure

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def get_days_in_month(year, month):
    return calendar.monthrange(year, month)[1]


def slice_lat(ds, lat_min=LAT_MIN, lat_max=LAT_MAX):
    for coord in ds.coords:
        if coord.lower() in ('lat', 'latitude', 'y'):
            return ds.sel({coord: slice(lat_min, lat_max)})
    return ds


def load_daily(year, month, day, gchp_dir):
    """
    Load one day's gchp_NO2 (surface layer, DailyVars file) and
    NO2col (tropospheric column, 13-15 h mean file).
    Returns (no2col, no2_sfc, lat, lon) as float32 arrays, or None on failure.
    """
    daily_path   = gchp_dir + f'daily/{res}.DailyVars.{year}{month:02d}{day:02d}.nc4'
    col_path     = gchp_dir + f'daily/{res}.Hours.13-15.{year}{month:02d}{day:02d}.nc4'

    if not os.path.exists(daily_path) or not os.path.exists(col_path):
        return None

    try:
        with xr.open_dataset(daily_path, engine='netcdf4') as ds:
            ds = slice_lat(ds.squeeze())
            no2_sfc = ds['gchp_NO2'].values.astype('float32')
            lat     = ds['lat'].values if 'lat' in ds.coords else ds['latitude'].values
            lon     = ds['lon'].values if 'lon' in ds.coords else ds['longitude'].values

        with xr.open_dataset(col_path, engine='netcdf4') as ds:
            ds = slice_lat(ds.squeeze())
            no2col  = ds['NO2col'].values.astype('float32')

        return no2col, no2_sfc, lat, lon

    except Exception as e:
        print(f"    [WARN] day {day:02d}: {e}")
        return None


# ─── MONTHLY AVERAGE ─────────────────────────────────────────────────────────
def monthly_average(year, month):
    gchp_dir     = gchp_base + f'{year}/'
    days_in_month = get_days_in_month(year, month)

    col_sum   = None
    sfc_sum   = None
    col_count = None
    sfc_count = None
    lat = lon = None
    valid_days = 0

    print(f"Computing monthly average for {year}-{month:02d} "
          f"({days_in_month} days)...")

    for day in range(1, days_in_month + 1):
        print(f"  day {day:02d}/{days_in_month}...", end=' ', flush=True)
        result = load_daily(year, month, day, gchp_dir)

        if result is None:
            print("missing")
            continue

        no2col, no2_sfc, lat, lon = result

        if col_sum is None:
            col_sum   = np.zeros_like(no2col,   dtype='float64')
            sfc_sum   = np.zeros_like(no2_sfc,  dtype='float64')
            col_count = np.zeros_like(no2col,   dtype='int32')
            sfc_count = np.zeros_like(no2_sfc,  dtype='int32')

        m_col = ~np.isnan(no2col)
        m_sfc = ~np.isnan(no2_sfc)
        col_sum[m_col]   += no2col[m_col].astype('float64')
        sfc_sum[m_sfc]   += no2_sfc[m_sfc].astype('float64')
        col_count[m_col] += 1
        sfc_count[m_sfc] += 1

        valid_days += 1
        print("ok")

        gc.collect()

    if valid_days == 0:
        raise RuntimeError(f"No valid daily files found for {year}-{month:02d}")

    print(f"  Averaging over {valid_days} valid days...")

    with np.errstate(divide='ignore', invalid='ignore'):
        no2col_mean = np.where(col_count > 0, (col_sum / col_count).astype('float32'), np.nan)
        no2sfc_mean = np.where(sfc_count > 0, (sfc_sum / sfc_count).astype('float32'), np.nan)
        ratio       = np.where(
            (no2sfc_mean > 0) & (~np.isnan(no2sfc_mean)) & (~np.isnan(no2col_mean)),
            no2col_mean / no2sfc_mean,
            np.nan
        ).astype('float32')

    return no2col_mean, no2sfc_mean, ratio, lat, lon, valid_days


# ─── SAVE ────────────────────────────────────────────────────────────────────
def save_nc(no2col, no2sfc, ratio, lat, lon, year, month, valid_days, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    outpath = out_dir + f'{res}.GCHP_column_surface_ratio.{year}{month:02d}.MonMean.nc'

    ds = xr.Dataset(
        {
            'gchp_column_NO2': (['lat', 'lon'], no2col,
                       {'units': 'molec cm-2',
                        'long_name': 'GCHP Tropospheric NO2 Column (13-15 h mean)'}),
            'gchp_surface_NO2': (['lat', 'lon'], no2sfc,
                         {'units': 'ppb',
                          'long_name': 'GCHP Surface-Layer NO2 (daily mean)'}),
            'gchp_column_to_surface_ratio': (['lat', 'lon'], ratio,
                                   {'units': '1',
                                    'long_name': 'Ratio gchp_column_NO2 / gchp_surface_NO2 (column-to-surface)'}),
        },
        coords={'lat': lat, 'lon': lon, 'time': datetime(year, month, 15)},
        attrs={
            'title': f'GCHP monthly column and surface NO2 ratio (column-to-surface) — {year}-{month:02d}',
            'source': f'GCHP {res} DailyVars + Hours.13-15',
            'latitude_range': f'{LAT_MIN} to {LAT_MAX}',
            'valid_days': valid_days,
            'created': datetime.now().isoformat(),
        }
    )

    enc = {v: {'zlib': True, 'complevel': 4, 'shuffle': True, 'dtype': 'float32'}
           for v in ds.data_vars}
    enc['lat'] = {'_FillValue': None}
    enc['lon'] = {'_FillValue': None}

    ds.to_netcdf(outpath, encoding=enc)
    size_mb = os.path.getsize(outpath) / 1e6
    print(f"  Saved: {outpath} ({size_mb:.1f} MB)")
    return outpath


# ─── PLOT ────────────────────────────────────────────────────────────────────
def plot_maps(no2col, no2sfc, ratio, lat, lon, year, month, out_dir):
    """
    3-panel figure: gchp_column_NO2 | gchp_surface_NO2 | gchp_column_to_surface_ratio.
    Data are downsampled by PLOT_STRIDE before rendering.
    """
    # Downsample for plotting
    s = PLOT_STRIDE
    lat_p   = lat[::s]
    lon_p   = lon[::s]
    col_p   = no2col[::s, ::s]
    sfc_p   = no2sfc[::s, ::s]
    ratio_p = ratio[::s, ::s]

    month_name = datetime(year, month, 1).strftime('%B %Y')

    # Colour scale: use 2nd–98th percentile to clip outliers
    def pct_lim(arr, lo=2, hi=98):
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return 0, 1
        return np.percentile(valid, lo), np.percentile(valid, hi)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             subplot_kw={'aspect': 'auto'})
    fig.suptitle(f'GCHP monthly mean — {month_name}', fontsize=14, fontweight='bold')

    panels = [
        (col_p,   'GCHP column NO2 (gchp_column_NO2)', 'molec cm$^{-2}$', 'RdYlBu'),
        (sfc_p,   'GCHP surface NO2 (gchp_surface_NO2)', 'molec cm$^{-2}$', 'RdYlBu'),
        (ratio_p, 'Ratio gchp_column_NO2 / gchp_surface_NO2 (column-to-surface)', '(dimensionless)', 'RdYlBu'),
    ]

    for ax, (data, title, unit, cmap) in zip(axes, panels):
        vmin, vmax = pct_lim(data)
        im = ax.pcolormesh(lon_p, lat_p, data,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           shading='auto', rasterized=True)
        cb = fig.colorbar(im, ax=ax, orientation='vertical',
                          pad=0.01, fraction=0.015)
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Longitude (°)', fontsize=9)
        ax.set_ylabel('Latitude (°)',  fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlim(lon_p[0], lon_p[-1])
        ax.set_ylim(lat_p[0], lat_p[-1])
        ax.axhline(0, color='k', linewidth=0.3, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    figpath = out_dir + f'{res}.GCHP_column_surface_ratio.{year}{month:02d}.png'
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure: {figpath}")
    return figpath


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Monthly mean of GCHP NO2col and surface NO2, with map plots.'
    )
    parser.add_argument('year',  type=int, help='Year  (e.g. 2019)')
    parser.add_argument('month', type=int, choices=range(1, 13),
                        help='Month (1-12)')
    args = parser.parse_args()

    year, month = args.year, args.month
    out_dir = out_base + f'{year}/'

    print(f"{'='*60}")
    print(f" GCHP monthly NO2col / surface NO2  —  {year}-{month:02d}")
    print(f"{'='*60}")
    t0 = datetime.now()

    no2col, no2sfc, ratio, lat, lon, valid_days = monthly_average(year, month)

    # Quick statistics
    for name, arr in [('NO2col', no2col), ('gchp_NO2', no2sfc), ('ratio', ratio)]:
        valid = arr[~np.isnan(arr)]
        if valid.size:
            print(f"  {name:25s}  mean={np.mean(valid):.3e}  "
                  f"min={np.min(valid):.3e}  max={np.max(valid):.3e}")

    nc_path  = save_nc(no2col, no2sfc, ratio, lat, lon, year, month, valid_days, out_dir)
    fig_path = plot_maps(no2col, no2sfc, ratio, lat, lon, year, month, out_dir)

    elapsed = datetime.now() - t0
    print(f"\nDone in {elapsed}.")
    print(f"  NetCDF : {nc_path}")
    print(f"  Figure : {fig_path}")


if __name__ == '__main__':
    main()
