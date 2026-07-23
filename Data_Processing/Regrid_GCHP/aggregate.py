import os
import xarray as xr
import argparse
import sys
from datetime import datetime
import calendar
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required for batch/headless jobs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gc
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def print_memory(label=""):
    """Print current memory usage if psutil is available"""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        avail_gb = mem.available / (1024**3)
        pct = mem.percent
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"  [{timestamp}] {label}: {used_gb:.1f}GB used, {avail_gb:.1f}GB free ({pct:.1f}%)", flush=True)

def slice_latitude(ds, lat_min=-60, lat_max=70):
    """
    Slice dataset to specified latitude range
    """
    # Find the latitude coordinate name (could be 'lat', 'latitude', etc.)
    lat_coord = None
    for coord in ds.coords:
        if coord.lower() in ['lat', 'latitude', 'y']:
            lat_coord = coord
            break
    
    if lat_coord is None:
        print("[WARN] No latitude coordinate found, returning original dataset", flush=True)
        return ds
    
    print(f"  Slicing latitude from {lat_min} to {lat_max} degrees using coordinate '{lat_coord}'", flush=True)
    
    # Slice the dataset
    lat_slice = ds.sel({lat_coord: slice(lat_min, lat_max)})
    
    # Add slicing info to attributes
    lat_slice.attrs.update({
        'latitude_slice': f'{lat_min} to {lat_max} degrees',
        'original_lat_range': f'{float(ds[lat_coord].min().values):.2f} to {float(ds[lat_coord].max().values):.2f}'
    })
    
    print(f"  Latitude range after slicing: {float(lat_slice[lat_coord].min().values):.2f} to {float(lat_slice[lat_coord].max().values):.2f}", flush=True)
    
    return lat_slice

def average_daily_to_monthly(year, month, lat_min=-60, lat_max=70):
    """
    Aggregate daily GCHP NO₂ data into monthly means with latitude slicing.
    High-performance version with adaptive chunking and memory management.
    """
    base_dir = f'/path/to/gchp-v2/forObservation-Geophysical/{year}/'
    daily_dir = os.path.join(base_dir, "daily")
    monthly_dir = os.path.join(base_dir, "monthly")
    os.makedirs(monthly_dir, exist_ok=True)
    
    # Get number of days in the month
    days_in_month = calendar.monthrange(year, month)[1]
    
    ds_list = []
    valid_days = 0
    
    print(f"Processing {year}-{month:02d} ({days_in_month} days)", flush=True)
    print(f"Process ID: {os.getpid()}", flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print_memory("Initial")
    
    # Determine optimal chunk size based on available memory
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024**3)
        # Use larger chunks if we have plenty of memory
        if avail_gb > 200:
            chunk_size = 2000
            print(f"  Using large chunks ({chunk_size}×{chunk_size}) for optimal speed", flush=True)
        else:
            chunk_size = 1000
            print(f"  Using moderate chunks ({chunk_size}×{chunk_size}) for safety", flush=True)
    else:
        chunk_size = 1000
        print(f"  Using default chunks ({chunk_size}×{chunk_size})", flush=True)
    
    start_load = datetime.now()
    
    for day in range(1, days_in_month + 1):
        fname = f"1x1km.Hours.13-15.{year}{month:02d}{day:02d}.nc4"
        fpath = os.path.join(daily_dir, fname)
        
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing daily file: {fname}", flush=True)
            continue
            
        try:
            # Use native chunking from the file to avoid chunk misalignment issues
            try:
                ds = xr.open_dataset(fpath, chunks="auto")
            except (ValueError, ImportError):
                # If dask not available, open without chunking
                ds = xr.open_dataset(fpath)
            
            # Apply latitude slicing
            ds = slice_latitude(ds, lat_min, lat_max)
            
            # Ensure there's a 'day' dimension
            ds = ds.expand_dims(day=[day])
            ds_list.append(ds)
            valid_days += 1
            print(f"  ✓ Day {day:02d} loaded", flush=True)
        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {str(e)}", flush=True)
            continue
   
    if not ds_list:
        print(f"[ERROR] No daily data found for {year}-{month:02d}", flush=True)
        return False
    
    print_memory(f"Loaded {valid_days} days")
    load_duration = (datetime.now() - start_load).total_seconds()
    
    print(f"  Averaging {valid_days}/{days_in_month} days...", flush=True)
    
    try:
        # Concatenate along the 'day' axis
        start_concat = datetime.now()
        ds_all = xr.concat(ds_list, dim="day")
        print_memory("After concatenation")
        concat_duration = (datetime.now() - start_concat).total_seconds()
        print(f"  Concatenation completed in {concat_duration:.1f}s", flush=True)
        
        # Clear the original list to free memory
        ds_list = None
        gc.collect()
        
        # Compute mean — .load() materialises into RAM before writing to avoid
        # interleaved compute+I/O during to_netcdf()
        start_mean = datetime.now()
        monthly_mean = ds_all.mean(dim="day", skipna=True).load()
        print_memory("After mean computation")
        mean_duration = (datetime.now() - start_mean).total_seconds()
        print(f"  Mean computation completed in {mean_duration:.1f}s", flush=True)
        
        # Clear concatenated dataset
        ds_all = None
        gc.collect()
        print_memory("After cleanup before writing")
        
        # Add metadata
        monthly_mean.attrs.update({
            'title': f'GCHP monthly mean 3 Hours NO2 col for {year}-{month:02d}',
            'source': 'GCHP c180 daily output',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'days_averaged': valid_days,
            'total_days_in_month': days_in_month,
            'processed_by_pid': os.getpid(),
            'latitude_slice': f'{lat_min} to {lat_max} degrees',
            'processing_time_seconds': load_duration + concat_duration + mean_duration
        })
        
        # Save monthly mean with compression
        out_fname = f"1x1km.Hours.13-15.{year}{month:02d}.MonMean.nc"
        out_path = os.path.join(monthly_dir, out_fname)
        
        print(f"  Writing monthly mean to {out_path}", flush=True)
        
        encoding = {var: {"zlib": True, "complevel": 4, "shuffle": True, "fletcher32": True}
                    for var in monthly_mean.data_vars}
        
        monthly_mean.to_netcdf(out_path, encoding=encoding)
        
        # Print file size info
        file_size = os.path.getsize(out_path) / (1024 * 1024)  # MB
        print(f"  ✓ Successfully created monthly average for {year}-{month:02d}", flush=True)
        print(f"  File size: {file_size:.2f} MB", flush=True)
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to create monthly average for {year}-{month:02d}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def average_monthly_to_yearly(year, lat_min=-60, lat_max=70):
    """
    Aggregate pre-computed monthly GCHP NO₂ means into a single yearly mean with latitude slicing.
    """
    base_dir = f'/path/to/gchp-v2/forObservation-Geophysical/{year}/'
    monthly_dir = os.path.join(base_dir, "monthly")
    yearly_dir = os.path.join(base_dir, "yearly")
    os.makedirs(yearly_dir, exist_ok=True)
    
    ds_list = []
    valid_months = 0
    
    print(f"Creating yearly average for {year}")
    print(f"Process ID: {os.getpid()}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for month in range(1, 13):
        # Look for the latitude-sliced monthly files first, then fall back to original
        fname = f"1x1km.Hours.13-15.{year}{month:02d}.MonMean.nc"
        
        fpath = os.path.join(monthly_dir, fname)
        
        if not os.path.exists(fpath):
            print(f"  [WARN] Missing monthly file: {fname}")
            continue
            
        try:
            # Use native chunking from the file
            try:
                ds = xr.open_dataset(fpath, chunks="auto")
            except (ValueError, ImportError):
                # If dask not available, open without chunking
                ds = xr.open_dataset(fpath)
            
            # Ensure there's a 'month' dimension
            ds = ds.expand_dims(month=[month])
            ds_list.append(ds)
            valid_months += 1
            print(f"  ✓ Month {month:02d}", flush=True)
        except Exception as e:
            print(f"  [ERROR] Failed to load {os.path.basename(fpath)}: {str(e)}", flush=True)
            continue
    
    if not ds_list:
        print(f"[ERROR] No monthly data found for year {year}")
        return False
    
    print(f"  Averaging {valid_months}/12 months...")

    try:
        ds_all      = xr.concat(ds_list, dim="month")
        yearly_mean = ds_all.mean(dim="month", skipna=True).load()  # materialise before closing sources

        yearly_mean.attrs.update({
            'title':            f'Yearly mean GCHP for {year}',
            'source':           'GCHP monthly averages',
            'created':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'months_averaged':  valid_months,
            'total_months':     12,
            'processed_by_pid': os.getpid(),
            'latitude_slice':   f'{lat_min} to {lat_max} degrees',
        })

        # Close all source datasets and release HDF5 handles before writing
        ds_all.close()
        for ds in ds_list:
            ds.close()
        ds_list.clear()
        gc.collect()

        out_fname = f"1x1km.Hours.13-15.{year}.AnnualMean.nc"
        out_path  = os.path.join(yearly_dir, out_fname)
        print(f"  Writing yearly mean to {out_path}")

        encoding = {v: {'zlib': True, 'complevel': 4, 'shuffle': True}
                    for v in yearly_mean.data_vars}
        yearly_mean.to_netcdf(out_path, encoding=encoding)
        yearly_mean.close()
        gc.collect()

        file_size = os.path.getsize(out_path) / 1024**2
        print(f"  ✓ Successfully created yearly average for {year}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create yearly average for {year}: {str(e)}")
        for ds in ds_list:
            try: ds.close()
            except: pass
        gc.collect()
        return False

# Variables to plot from forObservation-Geophysical monthly/yearly files
GEO_PLOT_VARS = ['NO2col_tot', 'NO2col_trop']

# Fixed colour-scale limits per variable (fall back to 98th-percentile if not listed)
VMAX_MAP = {
    'NO2col_tot':  2e16,   # mol/m²  (total column)
    'NO2col_trop': 1e16,   # mol/m²  (tropospheric column)
}

def _plot_nc(nc_path, title, out_png, plot_vars):
    """Open a NetCDF file, plot it, then explicitly close and GC to prevent memory/HDF issues."""
    with xr.open_dataset(nc_path, engine='netcdf4') as ds:
        plot_map(ds, title, out_png, plot_vars=plot_vars)
    gc.collect()

def plot_map(ds, title, out_png, plot_vars=None):
    """
    Plot one panel per requested variable using the dataset's own lat/lon coordinates.
    plot_vars: list of variable names to plot; defaults to all data_vars containing 'NO2'.
    """
    # Find lat/lon coordinate names
    lat_coord = next((c for c in ds.coords if c.lower() in ('lat', 'latitude', 'y')), None)
    lon_coord = next((c for c in ds.coords if c.lower() in ('lon', 'longitude', 'x')), None)
    if lat_coord is None or lon_coord is None:
        print("[WARN] lat/lon coordinates not found in dataset, skipping plot")
        return

    lats = ds[lat_coord].values
    lons = ds[lon_coord].values

    # Decide which variables to draw
    if plot_vars is None:
        plot_vars = [v for v in ds.data_vars if 'NO2' in v.upper()]
    plot_vars = [v for v in plot_vars if v in ds.data_vars]
    if not plot_vars:
        print(f"[WARN] None of the requested plot variables found in dataset, skipping plot")
        return

    n = len(plot_vars)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n),
                             subplot_kw={'projection': ccrs.PlateCarree()},
                             squeeze=False)

    lat_min_val, lat_max_val = float(lats.min()), float(lats.max())
    lon_min_val, lon_max_val = float(lons.min()), float(lons.max())

    for i, var in enumerate(plot_vars):
        ax = axes[i, 0]
        v = ds[var].values
        if v.ndim != 2:
            v = v.squeeze()

        if v.size == 0 or np.all(np.isnan(v)):
            print(f"  [WARN] {var} has no valid data, skipping panel")
            ax.set_visible(False)
            continue

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3)
        ax.set_extent([lon_min_val, lon_max_val, lat_min_val, lat_max_val],
                      crs=ccrs.PlateCarree())

        if var in VMAX_MAP:
            vmax = VMAX_MAP[var]
        else:
            vmax = float(np.nanpercentile(v, 98))
            vmax = vmax if vmax > 0 else 1.0

        mesh = ax.pcolormesh(lons, lats, v,
                             transform=ccrs.PlateCarree(),
                             cmap='RdYlBu_r', vmin=0, vmax=vmax)
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                            pad=0.02, fraction=0.03, shrink=0.75)
        cbar.set_label(var)
        ax.set_title(title, fontsize=13, pad=6)
        ax.gridlines(draw_labels=True, alpha=0.3, linewidth=0.4)

    # Reserve space: top for title (on ax), bottom for colorbar
    fig.subplots_adjust(top=0.96, bottom=0.14, left=0.04, right=0.96)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot to {out_png}", flush=True)

def plot_existing_files(year, month=None, yearly_only=False):
    """
    Generate PNG plots from already-existing NetCDF files without re-running averaging.
    yearly_only=True : only plot the annual-mean files (skip all monthly files).
    month=N          : only plot that specific monthly file (ignored when yearly_only).
    otherwise        : plot all monthly files + yearly file.
    """
    geo_dir = f'/path/to/gchp-v2/forObservation-Geophysical/{year}/'
    plotted = 0

    if not yearly_only:
        months = [month] if month else list(range(1, 13))
        for m in months:
            nc = os.path.join(geo_dir, 'monthly', f"1x1km.Hours.13-15.{year}{m:02d}.MonMean.nc")
            if os.path.exists(nc):
                try:
                    _plot_nc(nc, f"Geo {year}-{m:02d}",
                             nc.replace('.nc', '.png'), plot_vars=GEO_PLOT_VARS)
                    plotted += 1
                except Exception as e:
                    print(f"  [WARN] monthly {m:02d} plot failed: {e}")
            else:
                print(f"  [WARN] not found: {nc}")

    if yearly_only or not month:
        nc = os.path.join(geo_dir, 'yearly', f"1x1km.Hours.13-15.{year}.AnnualMean.nc")
        if os.path.exists(nc):
            try:
                _plot_nc(nc, f"Geo {year} Annual",
                         nc.replace('.nc', '.png'), plot_vars=GEO_PLOT_VARS)
                plotted += 1
            except Exception as e:
                print(f"  [WARN] yearly plot failed: {e}")
        else:
            print(f"  [WARN] not found: {nc}")

    print(f"  Plot-only: {plotted} PNG(s) generated")
    return plotted > 0

def process_full_workflow(year, lat_min=-60, lat_max=70, make_plots=True):
    """Complete workflow: process all months then yearly average."""
    print(f"\n{'='*60}")
    print(f"FULL WORKFLOW FOR YEAR {year}")
    print(f"Latitude range: {lat_min} to {lat_max} degrees")
    print(f"{'='*60}")

    monthly_ok = 0
    for month in range(1, 13):
        print(f"\n--- Month {month:02d} ---")
        if average_daily_to_monthly(year, month, lat_min, lat_max):
            monthly_ok += 1
            if make_plots:
                mdir  = os.path.join(f'/path/to/gchp-v2/forObservation-Geophysical/{year}/', 'monthly')
                mpath = os.path.join(mdir, f"1x1km.Hours.13-15.{year}{month:02d}.MonMean.nc")
                if os.path.exists(mpath):
                    try:
                        _plot_nc(mpath, f"Geo {year}-{month:02d}",
                                 mpath.replace('.nc', '.png'), plot_vars=GEO_PLOT_VARS)
                    except Exception as e:
                        print(f"  [WARN] plot failed: {e}")

    print(f"\n  monthly: {monthly_ok}/12 months processed")
    if monthly_ok > 0:
        print(f"\n--- Yearly average ---")
        average_monthly_to_yearly(year, lat_min, lat_max)

    return True

def main():
    """Main processing function with command line arguments"""
    parser = argparse.ArgumentParser(
        description='Aggregate GCHP NO2 daily → monthly → yearly (forObservation-Geophysical)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aggregate.py 2023 --month 1          # monthly mean for Jan 2023
  aggregate.py 2023 --yearly-only      # yearly mean from existing monthly files
  aggregate.py 2023 --full-workflow    # all months + yearly
""")
    parser.add_argument('year', type=int, help='Year to process (e.g., 2023)')
    parser.add_argument('--month', type=int, metavar='MONTH', choices=range(1, 13),
                        help='Process specific month only (1-12)')
    parser.add_argument('--yearly-only', action='store_true',
                        help='Only create yearly average from existing monthly files')
    parser.add_argument('--full-workflow', action='store_true',
                        help='Process all months then create yearly average')
    parser.add_argument('--lat-min', type=float, default=-60,
                        help='Minimum latitude for slicing (default: -60)')
    parser.add_argument('--lat-max', type=float, default=70,
                        help='Maximum latitude for slicing (default: 70)')
    parser.add_argument('--no-plot', action='store_true',
                        help="Don't make PNG plots")
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate PNG plots from existing NetCDF files (no averaging)')
    args = parser.parse_args()

    year       = args.year
    lat_min    = args.lat_min
    lat_max    = args.lat_max
    make_plots = not args.no_plot

    print(f"Processing year {year}")
    print(f"Latitude slice: {lat_min} to {lat_max}")
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}")
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}")

    try:
        start_time = datetime.now()
        success = False

        if args.plot_only:
            success = plot_existing_files(year, args.month, yearly_only=args.yearly_only)

        elif args.full_workflow:
            success = process_full_workflow(year, lat_min, lat_max, make_plots)

        elif args.yearly_only:
            success = average_monthly_to_yearly(year, lat_min, lat_max)
            if success and make_plots:
                ydir  = os.path.join(f'/path/to/gchp-v2/forObservation-Geophysical/{year}/', 'yearly')
                ypath = os.path.join(ydir, f"1x1km.Hours.13-15.{year}.AnnualMean.nc")
                if os.path.exists(ypath):
                    try:
                        _plot_nc(ypath, f"Geo {year} Annual",
                                 ypath.replace('.nc', '.png'), plot_vars=GEO_PLOT_VARS)
                    except Exception as e:
                        print(f"  [WARN] yearly plot failed: {e}")

        elif args.month:
            success = average_daily_to_monthly(year, args.month, lat_min, lat_max)
            if success and make_plots:
                mdir  = os.path.join(f'/path/to/gchp-v2/forObservation-Geophysical/{year}/', 'monthly')
                mpath = os.path.join(mdir, f"1x1km.Hours.13-15.{year}{args.month:02d}.MonMean.nc")
                if os.path.exists(mpath):
                    try:
                        _plot_nc(mpath, f"Geo {year}-{args.month:02d}",
                                 mpath.replace('.nc', '.png'), plot_vars=GEO_PLOT_VARS)
                    except Exception as e:
                        print(f"  [WARN] monthly plot failed: {e}")

        else:
            print("Error: please specify one of --month N | --yearly-only | --full-workflow")
            sys.exit(1)

        duration = datetime.now() - start_time
        if success:
            print(f"\n{'='*60}\n✓ PROCESSING COMPLETED SUCCESSFULLY\nTotal time: {duration}\n{'='*60}")
            sys.exit(0)
        else:
            print(f"\n✗ Processing failed for year {year}")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⚠ Processing interrupted for year {year}")
        sys.exit(2)
    except Exception as e:
        print(f"\n✗ Unexpected error: {str(e)}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()