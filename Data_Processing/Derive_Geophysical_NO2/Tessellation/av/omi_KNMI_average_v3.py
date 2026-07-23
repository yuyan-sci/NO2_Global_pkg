# v3: Reads from NO2col-v3/OMI_KNMI (with AMF > 0.5 filter applied at tessellation)
import os
import xarray as xr
import argparse
import sys
from datetime import datetime
import calendar
import numpy as np
import psutil
import gc
import time
import warnings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_QC = dict(
    CloudFraction_max=0.5, sza_max=80, QAFlag=0, RowAnomalyFlag=0,
    SnowIceFlag1=10, SnowIceFlag2=252, SnowIceFlag3=255,
)

_NO2_VARS  = ["NO2_trop","NO2_trop_gcshape","NO2_tot","NO2_tot_gcshape"]
_CHUNK     = 200   # spatial chunk size for NetCDF output encoding

# ---------------------------------------------------------------------------
# Memory utilities  (same pattern as tropomi_average_HP.py)
# ---------------------------------------------------------------------------

def check_memory():
    mem = psutil.virtual_memory()
    return {
        "used_gb":      mem.used      / 1e9,
        "available_gb": mem.available / 1e9,
        "percent":      mem.percent,
    }

def log_memory(step):
    mem = check_memory()
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {step}: {mem['used_gb']:.1f} GB used, "
          f"{mem['available_gb']:.1f} GB free ({mem['percent']:.1f}%)", flush=True)

def force_cleanup():
    gc.collect()
    time.sleep(0.1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qcstr(params: dict) -> str:
    if "SnowIceFlag3" in params:
        return "CF{:03d}-SZA{}-QA{}-RA{}-SI1{}-SI2{:03d}-SI3{:03d}".format(
            int(params["CloudFraction_max"] * 100),
            params["sza_max"], params["QAFlag"], params["RowAnomalyFlag"],
            params["SnowIceFlag1"], params["SnowIceFlag2"], params["SnowIceFlag3"],
        )
    return "CF{:03d}-SZA{}-QA{}-RA{}-SA{:03d}".format(
        int(params["CloudFraction_max"] * 100),
        params["sza_max"], params["QAFlag"], params["RowAnomalyFlag"],
        int(params["SurfaceAlbedo"] * 100),
    )


def _accumulate(sum_arr, cnt_arr, ds):
    """Add one dataset's values into the running sum/count dicts."""
    for v in _NO2_VARS:
        arr   = ds[v].values.astype(np.float64)
        valid = ~np.isnan(arr)
        sum_arr[v] += np.where(valid, arr, 0.0)
        cnt_arr[v] += valid.astype(np.int32)


def _build_mean_dataset(sum_arr, cnt_arr, ref_path: str) -> xr.Dataset:
    """Divide accumulated sums by counts, restore coords from *ref_path*."""
    with xr.open_dataset(ref_path) as ds_ref:
        mean_vars = {}
        for v in _NO2_VARS:
            with np.errstate(invalid="ignore"):
                mean_arr = np.where(
                    cnt_arr[v] > 0,
                    sum_arr[v] / cnt_arr[v],
                    np.nan,
                ).astype(np.float32)
            mean_vars[v] = xr.DataArray(mean_arr, dims=ds_ref[v].dims, attrs=ds_ref[v].attrs)
        return xr.Dataset(mean_vars, coords=ds_ref.coords)


def _write_nc(ds: xr.Dataset, out_path: str, total_input_mb: float = 0.0) -> bool:
    """Write *ds* with memory-adaptive compression and chunksizes."""
    mem = check_memory()
    if mem["available_gb"] < 30:
        complevel = 0
        print("    Using no compression (low memory)", flush=True)
    elif mem["available_gb"] < 60:
        complevel = 1
        print("    Using minimal compression", flush=True)
    else:
        complevel = 4
        print("    Using standard compression (complevel=4)", flush=True)

    encoding = {v: {
        "zlib":       complevel > 0,
        "complevel":  complevel,
        "shuffle":    True,
        "chunksizes": (_CHUNK, _CHUNK),
    } for v in ds.data_vars}

    write_start = time.time()
    ds.to_netcdf(out_path, encoding=encoding)
    write_time  = time.time() - write_start
    output_mb   = os.path.getsize(out_path) / 1e6

    print(f"  ✓ Written in {write_time:.1f}s ({output_mb:.1f} MB)", flush=True)
    if total_input_mb > 0:
        ratio = total_input_mb / output_mb if output_mb > 0 else 0
        print(f"  ✓ Compression ratio: {ratio:.1f}×", flush=True)
    return True

# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def average_daily_to_monthly(year: int, month: int) -> bool:
    """Aggregate daily OMI-KNMI NO₂ data into monthly means."""
    qcstr = _make_qcstr(_QC)

    base_dir    = "/path/to/NO2col-v3/OMI_KNMI/"
    daily_dir   = os.path.join(base_dir, f"{year}/daily")
    monthly_dir = os.path.join(base_dir, f"{year}/monthly")
    os.makedirs(monthly_dir, exist_ok=True)

    days_in_month = calendar.monthrange(year, month)[1]

    print(f"Processing {year}-{month:02d} ({days_in_month} days)", flush=True)
    print(f"Process ID: {os.getpid()}", flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    log_memory("Initial")

    sum_arr, cnt_arr = None, None
    first_valid_path = None
    valid_days       = 0

    for day in range(1, days_in_month + 1):
        fname = f"OMI_KNMI_Regrid_{year}{month:02d}{day:02d}_{qcstr}.nc"
        fpath = os.path.join(daily_dir, fname)

        if not os.path.exists(fpath):
            print(f"  [WARN] Missing daily file: {fname}", flush=True)
            continue

        try:
            with xr.open_dataset(fpath) as ds:
                if sum_arr is None:
                    first_valid_path = fpath
                    sum_arr = {v: np.where(~np.isnan(ds[v].values), ds[v].values.astype(np.float64), 0.0) for v in _NO2_VARS}
                    cnt_arr = {v: (~np.isnan(ds[v].values)).astype(np.int32) for v in _NO2_VARS}
                else:
                    _accumulate(sum_arr, cnt_arr, ds)
            valid_days += 1
            print(f"  ✓ Day {day:02d} accumulated", flush=True)
        except Exception as e:
            print(f"  [ERROR] Failed to load {fname}: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue

    if sum_arr is None:
        print(f"[ERROR] No daily data found for {year}-{month:02d}", flush=True)
        return False

    log_memory(f"Loaded {valid_days} days")
    print(f"  Computing mean over {valid_days}/{days_in_month} days...", flush=True)

    try:
        mean_start   = time.time()
        monthly_mean = _build_mean_dataset(sum_arr, cnt_arr, first_valid_path)
        del sum_arr, cnt_arr
        force_cleanup()
        mean_time = time.time() - mean_start
        log_memory("After mean computation")
        print(f"  Mean computed in {mean_time:.1f}s", flush=True)

        monthly_mean.attrs.update({
            "title":               f"Monthly mean OMI-KNMI NO2 for {year}-{month:02d}",
            "source":              "OMI-KNMI daily data",
            "created":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "days_averaged":       valid_days,
            "total_days_in_month": days_in_month,
            "quality_control":     qcstr,
            "processed_by_pid":    os.getpid(),
            "mean_time_seconds":   mean_time,
        })

        out_fname = f"OMI_KNMI_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc"
        out_path  = os.path.join(monthly_dir, out_fname)
        print(f"  Writing monthly mean to {out_path}...", flush=True)

        _write_nc(monthly_mean, out_path)
        print(f"  ✓ Successfully created monthly average for {year}-{month:02d}", flush=True)
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create monthly average for {year}-{month:02d}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return False


def average_monthly_to_yearly(year: int) -> bool:
    """Aggregate pre-computed monthly OMI-KNMI NO₂ means into a single yearly mean."""
    qcstr = _make_qcstr(_QC)

    base_dir    = "/path/to/NO2col-v3/OMI_KNMI/"
    monthly_dir = os.path.join(base_dir, f"{year}/monthly")
    yearly_dir  = os.path.join(base_dir, f"{year}/yearly")
    os.makedirs(yearly_dir, exist_ok=True)

    print(f"Creating yearly average for {year}", flush=True)
    print(f"Process ID: {os.getpid()}", flush=True)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Available CPU cores: {psutil.cpu_count()}", flush=True)
    print(f"Total system memory: {psutil.virtual_memory().total / 1e9:.1f} GB", flush=True)
    log_memory("Initial")

    # Inventory monthly files first
    monthly_files    = []
    total_input_mb   = 0.0
    for month in range(1, 13):
        fname = f"OMI_KNMI_Regrid_{year}{month:02d}_Monthly_{qcstr}.nc"
        fpath = os.path.join(monthly_dir, fname)
        if os.path.exists(fpath):
            mb = os.path.getsize(fpath) / 1e6
            monthly_files.append((month, fpath, mb))
            total_input_mb += mb
            print(f"  Found month {month:02d}: {mb:.1f} MB", flush=True)
        else:
            print(f"  [WARN] Missing monthly file: {fname}", flush=True)

    if not monthly_files:
        print(f"[ERROR] No monthly data found for year {year}", flush=True)
        return False

    print(f"  Total input: {total_input_mb:.1f} MB across {len(monthly_files)}/12 months", flush=True)

    sum_arr, cnt_arr = None, None
    first_valid_path = None
    valid_months     = 0

    for month, fpath, mb in monthly_files:
        try:
            with xr.open_dataset(fpath) as ds:
                if sum_arr is None:
                    first_valid_path = fpath
                    sum_arr = {v: np.where(~np.isnan(ds[v].values), ds[v].values.astype(np.float64), 0.0) for v in _NO2_VARS}
                    cnt_arr = {v: (~np.isnan(ds[v].values)).astype(np.int32) for v in _NO2_VARS}
                else:
                    _accumulate(sum_arr, cnt_arr, ds)
            valid_months += 1
            mem = check_memory()
            print(f"  ✓ Month {month:02d} accumulated (mem: {mem['percent']:.1f}%)", flush=True)
        except Exception as e:
            print(f"  [ERROR] Failed to load month {month:02d}: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue

    if sum_arr is None:
        print(f"[ERROR] No monthly data could be loaded for year {year}", flush=True)
        return False

    log_memory(f"Loaded {valid_months} months")

    try:
        mean_start  = time.time()
        yearly_mean = _build_mean_dataset(sum_arr, cnt_arr, first_valid_path)
        yearly_mean.attrs["year"] = str(year)
        del sum_arr, cnt_arr
        force_cleanup()
        mean_time = time.time() - mean_start
        log_memory("After mean computation")
        print(f"  Yearly mean computed in {mean_time:.1f}s", flush=True)

        yearly_mean.attrs.update({
            "title":              f"Yearly mean OMI-KNMI NO2 for {year}",
            "source":             "OMI-KNMI monthly averages",
            "created":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "months_averaged":    valid_months,
            "total_months":       12,
            "quality_control":    qcstr,
            "processed_by_pid":   os.getpid(),
            "processing_host":    os.uname().nodename,
            "mean_time_seconds":  mean_time,
            "total_input_size_mb": total_input_mb,
        })

        out_fname = f"OMI_KNMI_Regrid_{year}_{qcstr}.nc"
        out_path  = os.path.join(yearly_dir, out_fname)
        print(f"  Writing yearly mean to {out_path}...", flush=True)

        _write_nc(yearly_mean, out_path, total_input_mb)
        print(f"  ✓ Successfully created yearly average for {year}", flush=True)
        print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to create yearly average for {year}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return False

# ---------------------------------------------------------------------------
# Plotting (lazy import — only loaded when actually needed)
# ---------------------------------------------------------------------------

def plot_omi(ds, title: str, out_png: str) -> None:
    """Plot OMI-KNMI NO2 data."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    try:
        x = np.load(
            "/path/to/NO2_DL_global_2019/NO2_global_pkg"
            "/input_variables/tSATLON_global_MAP.npy"
        )
        y = np.load(
            "/path/to/NO2_DL_global_2019/NO2_global_pkg"
            "/input_variables/tSATLAT_global_MAP.npy"
        )
    except FileNotFoundError:
        print("[WARN] Grid coordinate files not found, skipping plot", flush=True)
        return

    vmax_by_var = {"NO2_trop": 1e16, "NO2_trop_gcshape": 1e16, "NO2_tot": 2e16, "NO2_tot_gcshape": 2e16}

    try:
        fig, axes = plt.subplots(
            4, 1, figsize= (16, 20),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        for ax, var in zip(axes, _NO2_VARS):
            v = ds[var].values
            if v.size == 0 or np.all(np.isnan(v)):
                print(f"[WARN] {var} has no valid data for '{title}'; skipping", flush=True)
                continue
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS,   linewidth=0.5)
            ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
            mesh = ax.pcolormesh(x, y, v, transform=ccrs.PlateCarree(),
                                 cmap="RdYlBu_r", vmin=0, vmax=vmax_by_var[var])
            cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
            cbar.set_label(var)
            ax.set_title(f"{title}: {var}", pad=10)
        plt.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"Plot saved: {out_png}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to create plot: {e}", flush=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Process OMI-KNMI NO2 data averaging")
    parser.add_argument("year",    type=int, help="Year to process (e.g., 2019)")
    parser.add_argument("--month", type=int, metavar="MONTH", choices=range(1, 13),
                        help="Process specific month only (1-12)")
    parser.add_argument("--yearly-only", action="store_true",
                        help="Only create yearly average from existing monthly files")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate PNG plots from existing output files (no averaging)")
    parser.add_argument("--no-plot", action="store_true", help="Don't make PNG plots")
    args = parser.parse_args()
    year = args.year

    sys.stdout.reconfigure(line_buffering=True)
    warnings.filterwarnings("ignore", message=".*chunks separate.*")

    print("=== OMI-KNMI NO2 PROCESSING ===", flush=True)
    print(f"Processing year {year}", flush=True)
    print(f"Hostname:       {os.getenv('HOSTNAME', 'unknown')}", flush=True)
    print(f"Job ID:         {os.getenv('LSB_JOBID', 'not_in_lsf')}", flush=True)
    print(f"Python version: {sys.version.split()[0]}", flush=True)
    print(f"Script start:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    qcstr = _make_qcstr(_QC)

    base_dir    = "/path/to/NO2col-v3/OMI_KNMI/"
    monthly_dir = os.path.join(base_dir, f"{year}/monthly")
    yearly_dir  = os.path.join(base_dir, f"{year}/yearly")

    try:
        start_time = datetime.now()
        success    = False

        if args.plot_only:
            # ── Plot-only: read existing output files and render PNGs ──────
            if not args.month and not args.yearly_only:
                print("Error: --plot-only requires --month N or --yearly-only", flush=True)
                sys.exit(1)
            if args.yearly_only:
                yearly_path = os.path.join(yearly_dir, f"OMI_KNMI_Regrid_{year}_{qcstr}.nc")
                if not os.path.exists(yearly_path):
                    print(f"[ERROR] Yearly file not found: {yearly_path}", flush=True)
                    sys.exit(1)
                print(f"Plotting from {yearly_path}...", flush=True)
                ds_y    = xr.open_dataset(yearly_path)
                out_png = os.path.join(yearly_dir, f"OMI_KNMI_Regrid_{year}_{qcstr}.png")
                plot_omi(ds_y, f"{year} Yearly", out_png)
                ds_y.close()
            else:
                monthly_path = os.path.join(
                    monthly_dir,
                    f"OMI_KNMI_Regrid_{year}{args.month:02d}_Monthly_{qcstr}.nc",
                )
                if not os.path.exists(monthly_path):
                    print(f"[ERROR] Monthly file not found: {monthly_path}", flush=True)
                    sys.exit(1)
                print(f"Plotting from {monthly_path}...", flush=True)
                ds_m    = xr.open_dataset(monthly_path)
                out_png = os.path.join(
                    monthly_dir,
                    f"OMI_KNMI_Regrid_{year}{args.month:02d}_{qcstr}.png",
                )
                plot_omi(ds_m, f"{year}-{args.month:02d}", out_png)
                ds_m.close()
            success = True

        elif args.yearly_only:
            success = average_monthly_to_yearly(year)
            if success and not args.no_plot:
                yearly_path = os.path.join(yearly_dir, f"OMI_KNMI_Regrid_{year}_{qcstr}.nc")
                if os.path.exists(yearly_path):
                    print("Creating visualization plot...", flush=True)
                    ds_y = xr.open_dataset(yearly_path)
                    out_png = os.path.join(yearly_dir, f"OMI_KNMI_Regrid_{year}_{qcstr}.png")
                    plot_omi(ds_y, f"{year} Yearly", out_png)
                    ds_y.close()

        elif args.month:
            success = average_daily_to_monthly(year, args.month)
            if success and not args.no_plot:
                monthly_path = os.path.join(
                    monthly_dir,
                    f"OMI_KNMI_Regrid_{year}{args.month:02d}_Monthly_{qcstr}.nc",
                )
                if os.path.exists(monthly_path):
                    print(f"Creating visualization plot for {year}-{args.month:02d}...", flush=True)
                    ds_m = xr.open_dataset(monthly_path)
                    out_png = os.path.join(
                        monthly_dir,
                        f"OMI_KNMI_Regrid_{year}{args.month:02d}_{qcstr}.png",
                    )
                    plot_omi(ds_m, f"{year}-{args.month:02d}", out_png)
                    ds_m.close()
        else:
            print("Error: Please specify --month N, --yearly-only, or --plot-only", flush=True)
            print("For parallel processing, use separate jobs for each month", flush=True)
            sys.exit(1)

        duration = datetime.now() - start_time

        if success:
            label = f"{year}-{args.month:02d}" if args.month else str(year)
            mode  = "monthly" if args.month else "yearly"
            print(f"\n✓ Successfully completed {mode} processing for {label}", flush=True)
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
        print(f"\n✗ Unexpected error processing year {year}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
