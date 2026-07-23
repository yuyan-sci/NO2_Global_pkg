#!/usr/bin/env python3
import sys
print("DEBUG: Script starting...", flush=True)
import os
import shutil
import tempfile
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import psutil
from Tess_func import (
    read_TROPOMI,
    write_tessellation_input_grid_file,
    load_and_save_TROPOMI_to_nc
)

def print_status(stage):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[{stage}] - Memory: {mem:.2f} GB")

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
# spatial grid
max_lat, min_lat, max_lon, min_lon = 70, -60, 180, -180
latres, lonres = 0.01, 0.01
out_lat_edges = np.arange(min_lat, max_lat + latres, latres)
out_lon_edges = np.arange(min_lon, max_lon + lonres, lonres)
tlat = out_lat_edges[:-1] + latres / 2
tlon = out_lon_edges[:-1] + lonres / 2

# QC thresholds
# CloudFraction_max, sza_max, QAlim = 0.1, 75, 0.75
# qcstr = 'CF{:03d}-SZA{}-QA{}'.format(
#     int(CloudFraction_max * 100),
#     sza_max,
#     int(QAlim * 100)
# )

sza_max, QAlim = 80, 0.75
qcstr = 'SZA{}-QA{}'.format(
    sza_max,
    int(QAlim * 100)
)

Na = 6.023e23  # Avogadro's number
MwAir = 28.97 # unit is g/mol

AMF_TROP_MIN = 0.5  # exclude pixels where amf_trop_gcshape < this value

# directories
gchp_dir           = '/path/to/gchp-v2/forTessellation/{year}/'
tess_code_dir      = '/path/to/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/gfortran_0p025_global/'
tropomi_l2_in_dir  = '/my-projects/1.project/TROPOMI_L2_V2_NO2_2018-2023/'
tess_temp_dir      = '/path/to/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/temp/'

os.makedirs(tess_temp_dir, exist_ok=True)

tess_var_str = [
    'tropomi_NO2_trop',
    'tropomi_NO2_trop_GC',
    'tropomi_NO2_total',
    'tropomi_NO2_total_GC'
]

# ─── PER-DAY WORKER ──────────────────────────────────────────────────────────────
def process_single_day(year, month, day):
    out_dir            = f'/path/to/NO2col-v3/TROPOMI/{year}/daily/'
    os.makedirs(out_dir, exist_ok=True)
    label = f"{year:04d}{month:02d}{day:02d}"
    scratch = tempfile.mkdtemp(prefix=f"tess_{label}_", dir=tess_temp_dir)

    try:
        # 1) Copy & rename the executable into scratch
        exe_src = os.path.join(tess_code_dir, 'tessellate_ifx')
        exe_dst = os.path.join(scratch, f"tess_ifx_{label}")
        shutil.copy(exe_src, exe_dst)
        os.chmod(exe_dst, 0o755)

        # 2) Prepare file paths
        flabel   = f"{label}_{qcstr}"
        tess_in  = os.path.join(scratch, f"tessellate_input_{flabel}.dat")
        tess_out = os.path.join(scratch, f"tessellate_output_{flabel}.dat")
        setup_f  = os.path.join(scratch, f"tessellate_setup_{flabel}.dat")
        nc_out   = os.path.join(out_dir,  f"Tropomi_Regrid_{flabel}.nc")

        # 3) Load that day's GCHP file
        gc_file = (
            gchp_dir.format(year=year)
            + f'daily/01x01.Hours.13-15.{year}{month:02d}{day:02d}.nc4'
        )
        print(f"Loading GCHP file: {gc_file}", flush=True)
        ds = xr.open_dataset(gc_file, engine='netcdf4')
        gc_lat = ds['lat'].astype('float32').values
        gc_lon = ds['lon'].astype('float32').values
        P_GC      = ds['Met_PMIDDRY'].astype('float32').values
        no2_gc    = ds['SpeciesConcVV_NO2'].astype('float32').values
        a         = ds['Met_AIRDEN'].astype('float32').values* 1e3 # unit is g/m3
        b         = ds['Met_BXHEIGHT'].astype('float32').values
        troppt_gc = ds['Met_TROPPT'].astype('float32').values  # tropopause pressure (same units as Met_PMIDDRY)
        ds.close()

        partial_column = no2_gc * a * b * (1e-4 / MwAir) * Na # unit is molec cm-2

        # 4) Loop over all TROPOMI orbits that day
        tess_files = []
        yyyymmdd = label
        patterns = [f"S5P_RPRO_L2__NO2____{yyyymmdd}T",f"S5P_OFFL_L2__NO2____{yyyymmdd}T"]

        if not os.path.exists(tropomi_l2_in_dir):
            print(f"ERROR: TROPOMI input directory not found: {tropomi_l2_in_dir}", flush=True)
            return

        print(f"Searching for TROPOMI files in {tropomi_l2_in_dir}", flush=True)
        files = os.listdir(tropomi_l2_in_dir)
        print(f"Found {len(files)} files in {tropomi_l2_in_dir}", flush=True)

        # Initialize aggregate statistics counters
        processed_count = 0
        total_pixels_all = 0
        passed_pixels_all = 0
        cf_pass_all = 0
        sza_pass_all = 0
        qa_pass_all = 0
        lat_pass_all = 0
        lon_pass_all = 0

        for i, fname in enumerate(files, start=1):
            # Check if filename matches any of the patterns
            if not any(pattern in fname for pattern in patterns):
                continue

            # Skip hidden files
            if fname.startswith("._") or fname.startswith("."):
                continue

            processed_count += 1

            # Try to read the file, skip if corrupted
            try:
                trop = read_TROPOMI(os.path.join(tropomi_l2_in_dir, fname))
            except Exception as e:
                print(f"WARNING: Skipping corrupted file {fname}: {e}", flush=True)
                continue
            nscan, npix = trop['no2_tot_sc'].shape[1:]
            scanIndex   = np.broadcast_to(
                np.arange(nscan)[None,:,None],
                (1, nscan, npix)
            )
            groundIndex = np.broadcast_to(
                np.arange(npix)[None,None,:],
                (1, nscan, npix)
            )

            good = (
                (trop['sza'] < sza_max) &
                (trop['QualityFlag'] > QAlim) &
                (trop['Latitude'] >= min_lat) &
                (trop['Latitude'] <= max_lat) &
                (trop['Longitude']>= min_lon) &
                (trop['Longitude']<= max_lon)
            )

            # Accumulate filtering statistics
            total_pixels = good.size
            passed_pixels = np.sum(good)

            total_pixels_all += total_pixels
            passed_pixels_all += passed_pixels
            sza_pass_all += np.sum(trop['sza'] < sza_max)
            qa_pass_all += np.sum(trop['QualityFlag'] > QAlim)
            lat_pass_all += np.sum((trop['Latitude'] >= min_lat) & (trop['Latitude'] <= max_lat))
            lon_pass_all += np.sum((trop['Longitude'] >= min_lon) & (trop['Longitude'] <= max_lon))

            if not good.any():
                continue

            # compute GC-shaped total column
            no2_tot_gcshape = np.full(trop['no2_tot_vc'].shape, np.nan)
            no2_trop_gcshape = np.full(trop['no2_trop_vc'].shape, np.nan)
            amf_trop_gc_pixel = np.full(trop['AMFtrop'].shape, np.nan)
            valid = np.where(good[0])
            lat_idx = np.round(
                np.interp(trop['Latitude'][good], gc_lat, np.arange(len(gc_lat)))
            ).astype(int)
            lon_idx = np.round(
                np.interp(trop['Longitude'][good], gc_lon, np.arange(len(gc_lon)))
            ).astype(int)

            n_diag = 0   # number of diagnostic pixels printed per orbit
            for idx, (j, k) in enumerate(zip(*valid)):
                si, gi = int(scanIndex[0, j, k]), int(groundIndex[0, j, k])
                p_gc       = P_GC[:, lat_idx[idx], lon_idx[idx]]
                no2prof_gc = partial_column[:, lat_idx[idx], lon_idx[idx]]
                p_tm5      = np.ma.filled(np.array(trop['p'][0, si, gi, :]), np.nan)

                if np.isnan(p_gc).any() or np.isnan(no2prof_gc).any() or np.isnan(p_tm5).any():
                    if n_diag < 3:
                        print(f"  [DIAG idx={idx}] SKIP — NaN in pressure/profile: "
                              f"p_gc_nan={np.isnan(p_gc).sum()} "
                              f"prof_nan={np.isnan(no2prof_gc).sum()} "
                              f"p_tm5_nan={np.isnan(p_tm5).sum()}", flush=True)
                        n_diag += 1
                    continue

                interp_f = interp1d(p_gc, no2prof_gc, bounds_error=False, fill_value='extrapolate')
                prof_tm5_tot = interp_f(p_tm5)

                sc_tot = trop['no2_tot_sc'][0, si, gi]
                AMFtot = trop['AMFtot'][0, si, gi]
                AvKtot    = trop['AvKtot'][0, si, gi, :]
                amf_tot_gcshape = AMFtot * np.sum(prof_tm5_tot * AvKtot) / np.sum(prof_tm5_tot)

                if amf_tot_gcshape and not np.isnan(amf_tot_gcshape):
                    no2_tot_gcshape[0, j, k] = sc_tot / amf_tot_gcshape

                # GC tropopause pressure for this pixel
                troppt_val = troppt_gc[lat_idx[idx], lon_idx[idx]]

                sc_trop = trop['no2_trop_sc'][0, si, gi]
                AMFtrop = trop['AMFtrop'][0, si, gi]
                AvKtrop_raw = trop['AvKtrop'][0, si, gi, :]
                AvKtrop = np.where(p_tm5 < troppt_val, 0.0, AvKtrop_raw)  # zero stratospheric layers
                # use GC profile truncated at tropopause for tropospheric AMF
                no2prof_gc_trop = np.where(p_gc >= troppt_val, no2prof_gc, 0.0)
                sum_trop_gc = np.sum(no2prof_gc_trop)
                if sum_trop_gc > 0:
                    interp_f_trop = interp1d(p_gc, no2prof_gc_trop, bounds_error=False, fill_value=0.0)
                    prof_tm5_trop = np.where(p_tm5 >= troppt_val, interp_f_trop(p_tm5), 0.0)
                    sum_prof_trop = np.sum(prof_tm5_trop)
                    if sum_prof_trop > 0:
                        amf_trop_gcshape = AMFtrop * np.sum(prof_tm5_trop * AvKtrop) / sum_prof_trop
                        amf_trop_gc_pixel[0, j, k] = amf_trop_gcshape
                        if amf_trop_gcshape and not np.isnan(amf_trop_gcshape):
                            no2_trop_gcshape[0, j, k] = sc_trop / amf_trop_gcshape

                if n_diag < 3:
                    print(f"    sum(prof*AvKtot)/sum(prof)={np.sum(prof_tm5_tot * AvKtot)/np.sum(prof_tm5_tot) if np.sum(prof_tm5_tot) else 'div0':.4f}  "
                          f"AMFtot*sum(prof*AvKtot)/sum(prof)="
                          f"{AMFtot*np.sum(prof_tm5_tot * AvKtot)/np.sum(prof_tm5_tot) if np.sum(prof_tm5_tot) else 'div0':.4f}", flush=True)
                    print(f"    troppt_val={troppt_val:.1f} hPa", flush=True)
                    print(f"    p_tm5 layers below tropopause (trop): {np.sum(p_tm5 >= troppt_val)} / {len(p_tm5)}", flush=True)
                    print(f"    p_gc  layers below tropopause (trop): {np.sum(p_gc  >= troppt_val)} / {len(p_gc)}", flush=True)

                    # AvKtrop before zeroing
                    print(f"    AvKtrop (raw):  sum={np.nansum(AvKtrop_raw):.4f}  "
                        f"min={np.nanmin(AvKtrop_raw):.4f}  max={np.nanmax(AvKtrop_raw):.4f}", flush=True)
                    # AvKtrop after zeroing strat layers (what's actually used)
                    print(f"    AvKtrop (used): sum={np.nansum(AvKtrop):.4f}  "
                        f"strat layers zeroed={np.sum(p_tm5 < troppt_val)}", flush=True)

                    print(f"    no2prof_gc_trop: sum={no2prof_gc_trop.sum():.3e}  "
                        f"nonzero layers={np.sum(no2prof_gc_trop > 0)}", flush=True)

                    if sum_trop_gc > 0:
                        print(f"    prof_tm5_trop: sum={prof_tm5_trop.sum():.3e}  "
                            f"nonzero={np.sum(prof_tm5_trop > 0)}", flush=True)
                        print(f"    sum(prof_tm5_trop * AvKtrop)={np.sum(prof_tm5_trop * AvKtrop):.3e}", flush=True)
                        if sum_prof_trop > 0:
                            ratio = np.sum(prof_tm5_trop * AvKtrop) / np.sum(prof_tm5_trop)
                            print(f"    amf_trop_gcshape = AMFtrop({AMFtrop:.4f}) * ratio({ratio:.4f}) = {AMFtrop*ratio:.4f}", flush=True)
                            print(f"    sc_trop={sc_trop:.3e}  => no2_trop_gcshape = {sc_trop/(AMFtrop*ratio) if AMFtrop*ratio else 'div0':.3e}", flush=True)
                    n_diag += 1

            # AMF quality filter: exclude all vars where amf_trop_gcshape < AMF_TROP_MIN
            amf_bad = ~np.isfinite(amf_trop_gc_pixel) | (amf_trop_gc_pixel < AMF_TROP_MIN)
            no2_tot_gcshape[amf_bad] = np.nan
            no2_trop_gcshape[amf_bad] = np.nan
            trop['no2_trop_vc'] = np.where(amf_bad, np.nan, trop['no2_trop_vc'])
            trop['no2_tot_vc'] = np.where(amf_bad, np.nan, trop['no2_tot_vc'])
            n_qc_pass = int(good.sum())
            n_amf_bad = int(amf_bad[good].sum())
            n_remain = n_qc_pass - n_amf_bad
            pct = 100 * n_amf_bad / n_qc_pass if n_qc_pass > 0 else 0
            print(f"    AMF filter (amf_trop_gcshape < {AMF_TROP_MIN}): "
                  f"removed {n_amf_bad:,} / {n_qc_pass:,} pixels ({pct:.1f}%), "
                  f"{n_remain:,} remaining", flush=True)

            # Replace NaN with 0 for all data variables before writing.
            # The Fortran tessellation binary cannot parse the string "nan"; if it
            # encounters one in the middle of a row it advances to the next token
            # (a corner longitude, -180..180) and reads that as the data value.
            # no2_trop and no2_tot_vc come from netCDF4 masked arrays and can carry
            # fill-value NaN even for quality-flagged pixels.  no2_gcshape is NaN
            # wherever the GCHP profile interpolation failed.
            # load_and_save_TROPOMI_to_nc already converts 0 → NaN, so 0 is the
            # correct missing-value sentinel for the tessellation binary.
            trop_vals   = np.where(np.isnan(trop['no2_trop_vc'][good]),   0.0, trop['no2_trop_vc'][good])
            tot_vals    = np.where(np.isnan(trop['no2_tot_vc'][good]), 0.0, trop['no2_tot_vc'][good])
            trop_gcshape_vals = no2_trop_gcshape[good].copy()
            trop_gcshape_vals[np.isnan(trop_gcshape_vals)] = 0.0
            tot_gcshape_vals = no2_tot_gcshape[good].copy()
            tot_gcshape_vals[np.isnan(tot_gcshape_vals)] = 0.0

            # stack into Nx12 array: 8 corner coords + 4 data vars
            arr = np.vstack([
                trop['CornerLongitude'][good,0], trop['CornerLatitude'][good,0],
                trop['CornerLongitude'][good,1], trop['CornerLatitude'][good,1],
                trop['CornerLongitude'][good,2], trop['CornerLatitude'][good,2],
                trop['CornerLongitude'][good,3], trop['CornerLatitude'][good,3],
                trop_vals,
                trop_gcshape_vals,
                tot_vals,
                tot_gcshape_vals
            ]).T

            orbit_dat = os.path.join(scratch, f"orbit_{i:02d}.dat")
            np.savetxt(
                orbit_dat,
                arr,
                fmt='%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %15.5E %15.5E %15.5E %15.5E'
            )
            tess_files.append(orbit_dat)

        if not tess_files:
            print("No matching TROPOMI files found for this day.", flush=True)
            return  # nothing to do this day

        # Print comprehensive filtering statistics
        print(f"\n{'='*70}", flush=True)
        print(f"FILTERING STATISTICS SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Files in directory:             {len(files):>10,}", flush=True)
        print(f"Files processed (matched):      {processed_count:>10,}", flush=True)
        print(f"Orbits with data after filter:  {len(tess_files):>10,}", flush=True)
        print(f"\nPIXEL STATISTICS:", flush=True)
        print(f"Total pixels:                   {total_pixels_all:>10,} (100.0%)", flush=True)
        print(f"Pixels passing ALL filters:     {passed_pixels_all:>10,} ({100*passed_pixels_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"Pixels filtered out:            {total_pixels_all-passed_pixels_all:>10,} ({100*(total_pixels_all-passed_pixels_all)/total_pixels_all:5.1f}%)", flush=True)
        print(f"\nINDIVIDUAL FILTER PERFORMANCE:", flush=True)
        print(f"SZA < {sza_max}°:                    {sza_pass_all:>10,} ({100*sza_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"QualityFlag > {QAlim}:               {qa_pass_all:>10,} ({100*qa_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"Latitude in bounds:             {lat_pass_all:>10,} ({100*lat_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"Longitude in bounds:            {lon_pass_all:>10,} ({100*lon_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"{'='*70}", flush=True)

        print_status(f"Finished AMF correction for {len(tess_files)} orbits")

        # 5) Build the master input for the Fortran code
        with open(tess_in, 'w') as fid:
            fid.write(f"{len(tess_var_str)}  4\n")
            for fdat in tess_files:
                data = np.loadtxt(fdat)
                if data.ndim < 2:
                    continue
                n_corners = 8  # 4 corner points x (lon, lat)
                fmt = ' '.join(['%10.4f'] * n_corners + ['%15.5E'] * (data.shape[1] - n_corners))
                np.savetxt(fid, data, fmt=fmt)
                # os.remove(fdat)

        # write the grid file
        write_tessellation_input_grid_file(
            setup_f, tess_in, tess_out, out_lat_edges, out_lon_edges
        )

        print_status("Finished building impute for Tessellation")

        # 6) run the Fortran binary
        subprocess.run([exe_dst, setup_f], cwd=scratch, check=True)
        print_status("Finished Tessellation")
        # os.remove(setup_f)

        # 7) convert to netCDF
        load_and_save_TROPOMI_to_nc(tess_out, nc_out, len(tess_var_str), tlat, tlon)
        # os.remove(tess_out)
        print_status("Daily Tessellation saved")

        os.system(f"rm -rf {scratch}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise

    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ─── DISPATCH ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mon',  type=int, required=True)
    parser.add_argument('--day',  type=int, required=True)
    args = parser.parse_args()

    process_single_day(args.year, args.mon, args.day)
