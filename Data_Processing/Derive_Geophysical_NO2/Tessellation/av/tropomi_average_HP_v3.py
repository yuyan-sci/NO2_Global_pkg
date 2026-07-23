# v3: Reads from NO2col-v3/TROPOMI (with AMF > 0.5 filter applied at tessellation)
import os
import xarray as xr
import argparse
import sys
from datetime import datetime
import calendar
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import psutil
import gc
import time
import warnings

def check_memory():
    """Monitor memory usage"""
    mem = psutil.virtual_memory()
    return {
        'used_gb': mem.used / 1e9,
        'available_gb': mem.available / 1e9,
        'percent': mem.percent
    }

def log_memory(step, flush_output=True):
    """Log memory usage with timestamp"""
    mem = check_memory()
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"  [{timestamp}] {step}: {mem['used_gb']:.1f}GB used, {mem['available_gb']:.1f}GB free ({mem['percent']:.1f}%)")
    if flush_output:
        sys.stdout.flush()

def force_cleanup():
    """Force garbage collection and memory cleanup"""
    gc.collect()
    time.sleep(0.1)  # Brief pause for system cleanup

def average_daily_to_monthly_optimized(year, month):
    """
    Optimized daily to monthly aggregation with memory monitoring
    """
    sza_max, QAlim = 80, 0.75
    qcstr = 'SZA{}-QA{}'.format(sza_max, int(QAlim * 100))

    base_dir = f'/path/to/NO2col-v3/TROPOMI/{year}'
    daily_dir = os.path.join(base_dir, "daily")
    monthly_dir = os.path.join(base_dir, "monthly")
    os.makedirs(monthly_dir, exist_ok=True)

    # Get number of days in the month
    days_in_month = calendar.monthrange(year, month)[1]

    print(f"Processing {year}-{month:02d} ({days_in_month} days)", flush=True)
    print(f"Process ID: {os.getpid()}", flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    log_memory("Initial")

    chunk_size = 200  # used only for output NetCDF chunk encoding
    vars_to_avg = ["NO2_trop", "NO2_trop_gcshape", "NO2_tot", "NO2_tot_gcshape"]
    sum_arr = None        # accumulated sum (float64) per variable
    cnt_arr = None        # valid-pixel count (int32) per variable
    first_valid_path = None
    valid_days = 0

    for day in range(1, days_in_month + 1):
        fname = f"Tropomi_Regrid_{year}{month:02d}{day:02d}_{qcstr}.nc"
        fpath = os.path.join(daily_dir, fname)

        if not os.path.exists(fpath):
            print(f"  [WARN] Missing daily file: {fname}", flush=True)
            continue

        try:
            with xr.open_dataset(fpath) as ds:
                if sum_arr is None:
                    first_valid_path = fpath
                    sum_arr = {}
                    cnt_arr = {}
                    for v in vars_to_avg:
                        arr = ds[v].values.astype(np.float64)
                        valid = ~np.isnan(arr)
                        sum_arr[v] = np.where(valid, arr, 0.0)
                        cnt_arr[v] = valid.astype(np.int32)
                else:
                    for v in vars_to_avg:
                        arr = ds[v].values.astype(np.float64)
                        valid = ~np.isnan(arr)
                        sum_arr[v] += np.where(valid, arr, 0.0)
                        cnt_arr[v] += valid.astype(np.int32)
            valid_days += 1
            print(f"  ✓ Day {day:02d} loaded & accumulated", flush=True)

        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {str(e)}", flush=True)
            continue

    if sum_arr is None:
        print(f"[ERROR] No daily data found for {year}-{month:02d}", flush=True)
        return False

    log_memory(f"Loaded {valid_days} days")
    print(f"  Computing mean over {valid_days}/{days_in_month} days...", flush=True)

    try:
        mean_start = time.time()
        with xr.open_dataset(first_valid_path) as ds_ref:
            mean_vars = {}
            for v in vars_to_avg:
                with np.errstate(invalid='ignore'):
                    mean_arr = np.where(
                        cnt_arr[v] > 0, sum_arr[v] / cnt_arr[v], np.nan
                    ).astype(np.float32)
                mean_vars[v] = xr.DataArray(mean_arr, dims=ds_ref[v].dims, attrs=ds_ref[v].attrs)
            monthly_mean = xr.Dataset(mean_vars, coords=ds_ref.coords)

        del sum_arr, cnt_arr
        force_cleanup()
        mean_time = time.time() - mean_start
        log_memory("After mean computation")
        print(f"  Mean computed in {mean_time:.1f}s", flush=True)

        monthly_mean.attrs.update({
            'title': f'Monthly mean TROPOMI NO2 for {year}-{month:02d}',
            'source': 'TROPOMI daily data',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'days_averaged': valid_days,
            'total_days_in_month': days_in_month,
            'quality_control': qcstr,
            'processed_by_pid': os.getpid(),
            'mean_time_seconds': mean_time
        })

        out_fname = f"Tropomi_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc"
        out_path = os.path.join(monthly_dir, out_fname)

        encoding = {var: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (chunk_size, chunk_size)
        } for var in monthly_mean.data_vars}

        print(f"  Writing monthly mean to {out_path}...", flush=True)
        write_start = time.time()
        monthly_mean.to_netcdf(out_path, encoding=encoding)
        write_time = time.time() - write_start
        output_size = os.path.getsize(out_path) / 1e6

        print(f"  ✓ File written in {write_time:.1f}s ({output_size:.1f} MB)", flush=True)
        print(f"  ✓ Successfully created monthly average for {year}-{month:02d}", flush=True)
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create monthly average for {year}-{month:02d}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def average_monthly_to_yearly_optimized(year):
    """
    Optimized monthly to yearly aggregation with comprehensive safety features
    """
    sza_max, QAlim = 80, 0.75
    qcstr = 'SZA{}-QA{}'.format(sza_max, int(QAlim * 100))

    base_dir = f'/path/to/NO2col-v3/TROPOMI/{year}'
    monthly_dir = os.path.join(base_dir, "monthly")
    yearly_dir = os.path.join(base_dir, "yearly")
    os.makedirs(yearly_dir, exist_ok=True)

    print(f"Creating yearly average for {year}", flush=True)
    print(f"Process ID: {os.getpid()}", flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Available CPU cores: {psutil.cpu_count()}", flush=True)
    print(f"Total system memory: {psutil.virtual_memory().total / 1e9:.1f} GB", flush=True)
    log_memory("Initial")

    # Check and report monthly files first
    monthly_files = []
    total_input_size = 0

    for month in range(1, 13):
        fname = f"Tropomi_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc"
        fpath = os.path.join(monthly_dir, fname)

        if os.path.exists(fpath):
            file_size = os.path.getsize(fpath) / 1e6  # MB
            monthly_files.append((month, fpath, file_size))
            total_input_size += file_size
            print(f"  Found month {month:02d}: {file_size:.1f} MB", flush=True)
        else:
            print(f"  [WARN] Missing monthly file: {fname}", flush=True)

    if not monthly_files:
        print(f"[ERROR] No monthly data found for year {year}", flush=True)
        return False

    print(f"  Total input data: {total_input_size:.1f} MB ({total_input_size/1000:.1f} GB)", flush=True)
    print(f"  Will process {len(monthly_files)}/12 months", flush=True)

    chunk_size = 200  # used only for output NetCDF chunk encoding
    vars_to_avg = ["NO2_trop", "NO2_trop_gcshape", "NO2_tot", "NO2_tot_gcshape"]
    sum_arr = None        # accumulated sum (float64) per variable
    cnt_arr = None        # valid-pixel count (int32) per variable
    first_valid_path = None
    valid_months = 0

    print("  Loading and accumulating monthly files...", flush=True)

    for month, fpath, file_size in monthly_files:
        try:
            with xr.open_dataset(fpath) as ds:
                if sum_arr is None:
                    first_valid_path = fpath
                    sum_arr = {}
                    cnt_arr = {}
                    for v in vars_to_avg:
                        arr = ds[v].values.astype(np.float64)
                        valid = ~np.isnan(arr)
                        sum_arr[v] = np.where(valid, arr, 0.0)
                        cnt_arr[v] = valid.astype(np.int32)
                else:
                    for v in vars_to_avg:
                        arr = ds[v].values.astype(np.float64)
                        valid = ~np.isnan(arr)
                        sum_arr[v] += np.where(valid, arr, 0.0)
                        cnt_arr[v] += valid.astype(np.int32)
            valid_months += 1
            mem_after = check_memory()
            print(f"    ✓ Month {month:02d} loaded & accumulated (mem: {mem_after['percent']:.1f}%)", flush=True)

        except Exception as e:
            print(f"    [ERROR] Failed to load month {month:02d}: {str(e)}", flush=True)
            continue

    if sum_arr is None:
        print(f"[ERROR] No monthly data could be loaded for year {year}", flush=True)
        return False

    log_memory(f"Loaded {valid_months} monthly files")

    try:
        mean_start = time.time()
        with xr.open_dataset(first_valid_path) as ds_ref:
            mean_vars = {}
            for v in vars_to_avg:
                with np.errstate(invalid='ignore'):
                    mean_arr = np.where(
                        cnt_arr[v] > 0, sum_arr[v] / cnt_arr[v], np.nan
                    ).astype(np.float32)
                mean_vars[v] = xr.DataArray(mean_arr, dims=ds_ref[v].dims, attrs=ds_ref[v].attrs)
            yearly_mean = xr.Dataset(mean_vars, coords=ds_ref.coords)
            yearly_mean.attrs['year'] = str(year)

        del sum_arr, cnt_arr
        force_cleanup()
        mean_time = time.time() - mean_start
        log_memory("After mean computation")
        print(f"  Yearly mean computed in {mean_time:.1f}s", flush=True)

        out_fname = f"Tropomi_Regrid_{year}_{qcstr}.nc"
        out_path = os.path.join(yearly_dir, out_fname)

        return write_yearly_output(yearly_mean, out_path, total_input_size, chunk_size)

    except Exception as e:
        print(f"  [ERROR] Failed to create yearly average for {year}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def write_yearly_output(yearly_mean, out_path, total_input_size, chunk_size):
    """Write yearly output with optimized settings"""
    print(f"  Writing yearly mean to {out_path}", flush=True)
    write_start = time.time()

    # Add comprehensive metadata
    yearly_mean.attrs.update({
        'title': f'Yearly mean TROPOMI NO2 for {yearly_mean.attrs.get("year", "unknown")}',
        'source': 'TROPOMI monthly averages',
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processed_by_pid': os.getpid(),
        'processing_host': os.uname().nodename,
        'chunk_size_used': chunk_size,
        'total_input_size_mb': total_input_size
    })

    # Memory-adaptive compression
    mem_before_write = check_memory()
    if mem_before_write['available_gb'] < 30:
        complevel = 0  # No compression
        print("    Using no compression due to low memory", flush=True)
    elif mem_before_write['available_gb'] < 60:
        complevel = 1  # Minimal compression
        print("    Using minimal compression", flush=True)
    else:
        complevel = 2  # Light compression
        print("    Using light compression for good balance", flush=True)

    # Optimized encoding
    encoding = {}
    for var in yearly_mean.data_vars:
        encoding[var] = {
            "zlib": complevel > 0,
            "complevel": complevel,
            "shuffle": True,
            "chunksizes": (chunk_size, chunk_size)
        }

    try:
        yearly_mean.to_netcdf(out_path, encoding=encoding)

        write_time = time.time() - write_start
        output_size = os.path.getsize(out_path) / 1e6
        compression_ratio = total_input_size / output_size if output_size > 0 else 0

        print(f"  ✓ File written in {write_time:.1f}s", flush=True)
        print(f"  ✓ Output file size: {output_size:.1f} MB", flush=True)
        print(f"  ✓ Compression ratio: {compression_ratio:.1f}x", flush=True)
        print(f"  ✓ Successfully created yearly average", flush=True)
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

        return True

    except Exception as e:
        print(f"  [ERROR] Failed to write output file: {str(e)}", flush=True)
        return False

def plot_tropomi(ds, title, out_png):
    """Plot TROPOMI NO2 data with error handling"""
    try:
        # Load grid arrays
        x = np.load('/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLON_global_MAP.npy')
        y = np.load('/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLAT_global_MAP.npy')
    except FileNotFoundError:
        print("[WARN] Grid coordinate files not found, skipping plot", flush=True)
        return

    vmax_by_var = {"NO2_trop": 1e16, "NO2_trop_gcshape": 1e16, "NO2_tot": 2e16, "NO2_tot_gcshape": 2e16}

    try:
        fig, axes = plt.subplots(4, 1, figsize=(16, 20),
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        vars_no2 = [v for v in ds.data_vars if 'NO2' in v]

        for ax, var in zip(axes, vars_no2):
            v = ds[var].values
            if v.size == 0 or np.all(np.isnan(v)):
                print(f"[WARN] variable {var} has no valid data for plot '{title}'; skipping", flush=True)
                continue
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())

            mesh = ax.pcolormesh(x, y, v,
                                 transform=ccrs.PlateCarree(),
                                 cmap='RdYlBu_r',
                                 vmin=0, vmax=vmax_by_var[var])
            cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                                pad=0.05, fraction=0.05)
            cbar.set_label(var)
            ax.set_title(f"{title}: {var}", pad=10)
        plt.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Plot saved: {out_png}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to create plot: {str(e)}", flush=True)

def main():
    """Main processing function with comprehensive error handling"""
    parser = argparse.ArgumentParser(description='Process TROPOMI NO2 data averaging (Optimized & Safe)')
    parser.add_argument('year', type=int, help='Year to process (e.g., 2019)')
    parser.add_argument('--month', type=int, metavar='MONTH', choices=range(1, 13),
                       help='Process specific month only (1-12)')
    parser.add_argument('--yearly-only', action='store_true',
                       help='Only create yearly average from existing monthly files')
    parser.add_argument('--no-plot', action='store_true',
                        help="Don't make PNG plots")
    parser.add_argument('--plot-only', action='store_true',
                        help='Only create PNG plot from existing output file (skip processing)')
    args = parser.parse_args()
    year = args.year

    # Enable unbuffered output for real-time monitoring
    sys.stdout.reconfigure(line_buffering=True)

    print(f"=== TROPOMI NO2 PROCESSING (OPTIMIZED & SAFE) ===", flush=True)
    print(f"Processing year {year}", flush=True)
    print(f"Hostname: {os.getenv('HOSTNAME', 'unknown')}", flush=True)
    print(f"Job ID: {os.getenv('LSB_JOBID', 'not_in_lsf')}", flush=True)
    print(f"Python version: {sys.version.split()[0]}", flush=True)
    print(f"Script start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # Suppress xarray chunking warnings for cleaner output
    warnings.filterwarnings("ignore", message=".*chunks separate.*")

    # Set up quality control string
    sza_max, QAlim = 80, 0.75
    qcstr = 'SZA{}-QA{}'.format(sza_max, int(QAlim * 100))

    base_dir = f'/path/to/NO2col-v3/TROPOMI/{year}'
    monthly_dir = os.path.join(base_dir, "monthly")
    yearly_dir = os.path.join(base_dir, "yearly")

    try:
        start_time = datetime.now()
        success = False

        if args.plot_only:
            # Only create plot from an existing output file
            if args.no_plot:
                print("Error: --plot-only and --no-plot are mutually exclusive", flush=True)
                sys.exit(1)
            if args.month:
                nc_path = os.path.join(monthly_dir, f"Tropomi_Regrid_{year}{args.month:02d}_Monthly_{qcstr}.nc")
                label = f"{year}-{args.month:02d}"
                out_png = os.path.join(monthly_dir, f"Tropomi_Regrid_{year}{args.month:02d}_{qcstr}.png")
            elif args.yearly_only:
                nc_path = os.path.join(yearly_dir, f"Tropomi_Regrid_{year}_{qcstr}.nc")
                label = f"{year} Yearly"
                out_png = os.path.join(yearly_dir, f"Tropomi_Regrid_{year}_{qcstr}.png")
            else:
                print("Error: --plot-only requires either --month N or --yearly-only", flush=True)
                sys.exit(1)
            if not os.path.exists(nc_path):
                print(f"[ERROR] NetCDF file not found: {nc_path}", flush=True)
                sys.exit(1)
            print(f"Creating plot from existing file: {nc_path}", flush=True)
            ds_plot = xr.open_dataset(nc_path)
            plot_tropomi(ds_plot, label, out_png)
            ds_plot.close()
            success = True

        elif args.yearly_only:
            # Only create yearly average
            success = average_monthly_to_yearly_optimized(year)

            # Create plot if requested
            if success and not args.no_plot:
                yearly_path = os.path.join(yearly_dir, f"Tropomi_Regrid_{year}_{qcstr}.nc")
                if os.path.exists(yearly_path):
                    print("Creating visualization plot...", flush=True)
                    ds_y = xr.open_dataset(yearly_path)
                    out_png = os.path.join(yearly_dir, f"Tropomi_Regrid_{year}_{qcstr}.png")
                    plot_tropomi(ds_y, f"{year} Yearly", out_png)
                    ds_y.close()

        elif args.month:
            # Process specific month only
            success = average_daily_to_monthly_optimized(year, args.month)

            # Create plot if requested
            if success and not args.no_plot:
                monthly_path = os.path.join(monthly_dir, f"Tropomi_Regrid_{year}{args.month:02d}_Monthly_{qcstr}.nc")
                if os.path.exists(monthly_path):
                    print(f"Creating visualization plot for {year}-{args.month:02d}...", flush=True)
                    ds_m = xr.open_dataset(monthly_path)
                    out_png = os.path.join(monthly_dir, f"Tropomi_Regrid_{year}{args.month:02d}_{qcstr}.png")
                    plot_tropomi(ds_m, f"{year}-{args.month:02d}", out_png)
                    ds_m.close()
        else:
            print("Error: Please specify either --month N or --yearly-only", flush=True)
            print("For parallel processing, use separate jobs for each month", flush=True)
            sys.exit(1)

        end_time = datetime.now()
        duration = end_time - start_time

        if success:
            if args.plot_only:
                print(f"\n✓ Successfully created plot", flush=True)
            elif args.month:
                print(f"\n✓ Successfully completed monthly processing for {year}-{args.month:02d}", flush=True)
            else:
                print(f"\n✓ Successfully completed yearly processing for {year}", flush=True)
            print(f"Total processing time: {duration}", flush=True)
            log_memory("Final")
            sys.exit(0)
        else:
            print(f"\n✗ Failed to process {year}", flush=True)
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n⚠ Processing interrupted for year {year}", flush=True)
        sys.exit(2)
    except Exception as e:
        print(f"\n✗ Unexpected error processing year {year}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
