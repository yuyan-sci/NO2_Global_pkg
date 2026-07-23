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
    read_OMI_KNMI,
    write_tessellation_input_grid_file,
    load_and_save_OMI_KNMI_to_nc,
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
CloudFraction_max, sza_max, QAFlag, RowAnomalyFlag, SnowIceFlag1, SnowIceFlag2, SnowIceFlag3 = 0.5, 80, 0, 0, 10, 252, 255
qcstr = 'CF{:03d}-SZA{}-QA{}-RA{}-SI1{}-SI2{:03d}-SI3{:03d}'.format(
    int(CloudFraction_max * 100),
    sza_max,
    QAFlag ,
    RowAnomalyFlag,
    SnowIceFlag1,
    SnowIceFlag2,
    SnowIceFlag3
)

# CloudFraction_max, sza_max, QAFlag, RowAnomalyFlag, SurfaceAlbedo = 0.5, 80, 0, 0, 0.3
# qcstr = 'CF{:03d}-SZA{}-QA{}-RA{}-SA{:03d}'.format(
#     int(CloudFraction_max * 100),
#     sza_max,
#     QAFlag ,
#     RowAnomalyFlag,
#     int(SurfaceAlbedo * 100)
# )

Na = 6.023e23  # Avogadro’s number
MwAir = 28.97 # unit is g/mol

# directories
gchp_dir           = '/path/to/gchp-v2/forTessellation/{year}/'
tess_code_dir      = '/path/to/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/gfortran_0p025_global/'
omi_l2_base_dir    = '/my-projects/1.project/OMI_KNMI/'
tess_temp_dir      = '/path/to/NO2_DL_global/NO2_global_pkg/Data_Processing/Derive_Geophysical_NO2/Tessellation/temp/'

os.makedirs(tess_temp_dir, exist_ok=True)

tess_var_str = [
    'omi_NO2_trop',
    'omi_NO2_trop_GC',
    'omi_NO2_total',
    'omi_NO2_total_GC',
]

# ─── PER-DAY WORKER ──────────────────────────────────────────────────────────────
def process_single_day(year, month, day):
    out_dir            = f'/path/to/NO2col-v2/OMI_KNMI/{year}/daily/'
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
        nc_out   = os.path.join(out_dir,  f"OMI_KNMI_Regrid_{flabel}.nc")

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
        troppt_gc = ds['Met_TROPPT'].astype('float32').values
        ds.close()

        partial_column = no2_gc * a * b * (1e-4 / MwAir) * Na # unit is molec cm-2

        # 4) Loop over all OMI-KNMI orbits that day
        tess_files = []
        yyyymmdd = label
        pattern = f"QA4ECV_L2_NO2_OMI_{yyyymmdd}T"

        # Each day is packed in a tar archive: {base}/{year}/omi_no2_qa4ecv_{yyyymmdd}.tar
        tar_path = os.path.join(omi_l2_base_dir, str(year),
                                f"omi_no2_qa4ecv_{yyyymmdd}.tar")
        if not os.path.exists(tar_path):
            print(f"ERROR: OMI-KNMI tar archive not found: {tar_path}", flush=True)
            return

        extract_dir = os.path.join(scratch, 'nc_files')
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Extracting {tar_path} -> {extract_dir}", flush=True)
        subprocess.run(['tar', '-xf', tar_path, '-C', extract_dir], check=True)

        files = os.listdir(extract_dir)
        print(f"Found {len(files)} files in tar archive", flush=True)
        
        # Initialize aggregate statistics counters
        processed_count = 0
        total_pixels_all = 0
        passed_pixels_all = 0
        cf_pass_all = 0
        sza_pass_all = 0
        qa_pass_all = 0
        ra_pass_all = 0
        si1_pass_all = 0
        si2_pass_all = 0
        si3_pass_all = 0
        lat_pass_all = 0
        lon_pass_all = 0

        for i, fname in enumerate(files, start=1):
            # Check if filename matches the QA4ECV pattern for this day
            if pattern not in fname:
                continue

            # Skip hidden files
            if fname.startswith("._") or fname.startswith("."):
                continue

            processed_count += 1

            # Try to read the file, skip if corrupted
            try:
                omi = read_OMI_KNMI(os.path.join(extract_dir, fname))
            except Exception as e:
                print(f"WARNING: Skipping corrupted file {fname}: {e}", flush=True)
                continue
            nscan, npix = omi['no2_tot_sc'].shape[1:]
            scanIndex   = np.broadcast_to(
                np.arange(nscan)[None, :, None], (1, nscan, npix)
            )
            groundIndex = np.broadcast_to(
                np.arange(npix)[None, None, :], (1, nscan, npix)
            )

            # ── Shape diagnostics (printed for first orbit only) ──────────────
            if processed_count == 1:
                print(f"\n[SHAPE DIAG] File: {fname}", flush=True)
                shape_vars = [
                    # key,              expected shape
                    ('no2_tot_sc',      f'(1, {nscan}, {npix})'),
                    ('no2_tot_vc',      f'(1, {nscan}, {npix})'),
                    ('no2_trop_vc',     f'(1, {nscan}, {npix})'),
                    ('AMFtot',          f'(1, {nscan}, {npix})'),
                    ('AvKtot',          f'(1, {nscan}, {npix}, nlev)'),
                    ('p',               f'(1, {nscan}, {npix}, nlev)'),
                    ('Latitude',        f'(1, {nscan}, {npix})'),
                    ('Longitude',       f'(1, {nscan}, {npix})'),
                    ('CF',              f'(1, {nscan}, {npix})'),
                    ('sza',             f'(1, {nscan}, {npix})'),
                    ('QualityFlag',     f'(1, {nscan}, {npix})'),
                    ('RowAnomalyFlag',  f'(1, {nscan}, {npix})'),
                    ('SnowIceFlag',     f'(1, {nscan}, {npix})'),
                    ('CornerLatitude',  f'(1, {nscan}, {npix}, 4)'),
                    ('CornerLongitude', f'(1, {nscan}, {npix}, 4)'),
                ]
                for key, expected in shape_vars:
                    if key in omi:
                        actual = np.shape(omi[key])
                        ok = '✓' if actual[0] == 1 else '✗ TIME!=1'
                        print(f"  {key:<20} shape={str(actual):<30} expected~{expected}  {ok}",
                              flush=True)
                    else:
                        print(f"  {key:<20} *** MISSING from omi dict ***", flush=True)
                print(f"  GCHP P_GC shape: {P_GC.shape}   (nlev, nlat, nlon)", flush=True)
                print(f"  partial_column shape: {partial_column.shape}", flush=True)
                print(flush=True)
            # ─────────────────────────────────────────────────────────────────

            good = (
                (omi['CF'] <= CloudFraction_max) &
                (omi['sza'] < sza_max) &
                (omi['QualityFlag'] == QAFlag) &
                (omi['RowAnomalyFlag'] == RowAnomalyFlag) &
                ((omi['SnowIceFlag'] <= SnowIceFlag1) | (omi['SnowIceFlag'] == SnowIceFlag2) | (omi['SnowIceFlag'] == SnowIceFlag3)) &
                (omi['Latitude'] >= min_lat) &
                (omi['Latitude'] <= max_lat) &
                (omi['Longitude']>= min_lon) &
                (omi['Longitude']<= max_lon)
            )

            # Accumulate filtering statistics
            total_pixels = good.size
            passed_pixels = np.sum(good)
            
            total_pixels_all += total_pixels
            passed_pixels_all += passed_pixels
            cf_pass_all += np.sum(omi['CF'] <= CloudFraction_max)
            sza_pass_all += np.sum(omi['sza'] < sza_max)
            qa_pass_all += np.sum(omi['QualityFlag'] == QAFlag)
            ra_pass_all += np.sum(omi['RowAnomalyFlag'] == RowAnomalyFlag)
            si1_pass_all += np.sum((omi['SnowIceFlag'] <= SnowIceFlag1))
            si2_pass_all += np.sum(omi['SnowIceFlag'] == SnowIceFlag2)
            si3_pass_all += np.sum(omi['SnowIceFlag'] == SnowIceFlag3)
            lat_pass_all += np.sum((omi['Latitude'] >= min_lat) & (omi['Latitude'] <= max_lat))
            lon_pass_all += np.sum((omi['Longitude'] >= min_lon) & (omi['Longitude'] <= max_lon))

            if not good.any():
                continue

            # compute GC-shaped total column
            # valid gives (scanline_idx, ground_pixel_idx) for all passing pixels
            no2_tot_gcshape = np.full(omi['no2_tot_vc'].shape, np.nan)
            no2_trop_gcshape = np.full(omi['no2_trop_vc'].shape, np.nan)
            valid = np.where(good[0])  # good[0]: (nscan, npix)
            lat_idx = np.round(
                np.interp(omi['Latitude'][good], gc_lat, np.arange(len(gc_lat)))
            ).astype(int)
            lon_idx = np.round(
                np.interp(omi['Longitude'][good], gc_lon, np.arange(len(gc_lon)))
            ).astype(int)

            n_diag = 0   # number of diagnostic pixels printed per orbit
            for idx, (j, k) in enumerate(zip(*valid)):
                si, gi = int(scanIndex[0, j, k]), int(groundIndex[0, j, k])
                p_gc       = P_GC[:, lat_idx[idx], lon_idx[idx]]
                no2prof_gc = partial_column[:, lat_idx[idx], lon_idx[idx]]
                p_tm5      = np.ma.filled(np.array(omi['p'][0, si, gi, :]), np.nan)

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

                # Fill masked values from netCDF4 masked arrays before arithmetic
                sc_tot = omi['no2_tot_sc'][0, si, gi]
                AMFtot = omi['AMFtot'][0, si, gi]
                AvKtot   = omi['AvKtot'][0, si, gi, :]
                
                amf_tot_gcshape = AMFtot * np.sum(prof_tm5_tot * AvKtot) / np.sum(prof_tm5_tot)
                if amf_tot_gcshape and not np.isnan(amf_tot_gcshape):
                    no2_tot_gcshape[0, j, k] = sc_tot / amf_tot_gcshape

                # tropospheric GC-shaped column: GC profile truncated at tropopause
                troppt_val   = troppt_gc[lat_idx[idx], lon_idx[idx]]
                
                sc_trop      = omi['no2_trop_sc'][0, si, gi]
                AMFtrop      = omi['AMFtrop'][0, si, gi]
                AvKtrop_raw      = omi['AvKtrop'][0, si, gi, :]
                AvKtrop      = np.where(p_tm5 < troppt_val, 0.0, AvKtrop_raw)  # zero stratospheric layers

                prof_gc_trop = np.where(p_gc >= troppt_val, no2prof_gc, 0.0)
                sum_trop_gc = np.sum(prof_gc_trop)
                if sum_trop_gc > 0:
                    interp_trop   = interp1d(p_gc, prof_gc_trop,bounds_error=False, fill_value=0.0)
                    prof_tm5_trop = np.where(p_tm5 >= troppt_val, interp_trop(p_tm5), 0.0)
                    sum_prof_trop = np.sum(prof_tm5_trop)
                    if sum_prof_trop > 0:
                        amf_trop_gcshape = AMFtrop * np.sum(prof_tm5_trop * AvKtrop) / sum_prof_trop
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
                    
                    print(f"    prof_gc_trop: sum={prof_gc_trop.sum():.3e}  "
                        f"nonzero layers={np.sum(prof_gc_trop > 0)}", flush=True)
                    
                    if sum_trop_gc > 0:
                        print(f"    prof_tm5_trop: sum={prof_tm5_trop.sum():.3e}  "
                            f"nonzero={np.sum(prof_tm5_trop > 0)}", flush=True)
                        print(f"    sum(prof_tm5_trop * AvKtrop)={np.sum(prof_tm5_trop * AvKtrop):.3e}", flush=True)
                        if sum_prof_trop > 0:
                            ratio = np.sum(prof_tm5_trop * AvKtrop) / np.sum(prof_tm5_trop)
                            print(f"    amf_trop_gcshape = AMFtrop({AMFtrop:.4f}) * ratio({ratio:.4f}) = {AMFtrop*ratio:.4f}", flush=True)
                            print(f"    sc_trop={sc_trop:.3e}  => no2_trop_gcshape = {sc_trop/(AMFtrop*ratio) if AMFtrop*ratio else 'div0':.3e}", flush=True)        
                    n_diag += 1
                               
            no2_trop_vals    = np.where(np.isnan(omi['no2_trop_vc'][good]), 0.0, omi['no2_trop_vc'][good])
            no2_trop_gc_vals = no2_trop_gcshape[good].copy(); no2_trop_gc_vals[np.isnan(no2_trop_gc_vals)] = 0.0
            no2_tot_vals     = np.where(np.isnan(omi['no2_tot_vc'][good]), 0.0, omi['no2_tot_vc'][good])
            no2_tot_gc_vals  = no2_tot_gcshape[good].copy();  no2_tot_gc_vals[np.isnan(no2_tot_gc_vals)]  = 0.0

            # stack into Nx12 array: 8 corner coords + 4 data vars
            arr = np.vstack([
                omi['CornerLongitude'][good,0], omi['CornerLatitude'][good,0],
                omi['CornerLongitude'][good,1], omi['CornerLatitude'][good,1],
                omi['CornerLongitude'][good,2], omi['CornerLatitude'][good,2],
                omi['CornerLongitude'][good,3], omi['CornerLatitude'][good,3],
                no2_trop_vals,
                no2_trop_gc_vals,
                no2_tot_vals,
                no2_tot_gc_vals,
            ]).T

            orbit_dat = os.path.join(scratch, f"orbit_{i:02d}.dat")
            np.savetxt(
                orbit_dat,
                arr,
                fmt='%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %15.5E %15.5E %15.5E %15.5E'
            )
            tess_files.append(orbit_dat)

        if not tess_files:
            print("No matching OMI-KNMI files found for this day.", flush=True)
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
        print(f"Cloud Fraction <= {CloudFraction_max}:        {cf_pass_all:>10,} ({100*cf_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"SZA < {sza_max}°:                    {sza_pass_all:>10,} ({100*sza_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"QualityFlag == {QAFlag}:               {qa_pass_all:>10,} ({100*qa_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"RowAnomalyFlag == {RowAnomalyFlag}:               {ra_pass_all:>10,} ({100*ra_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"SnowIceFlag <= {SnowIceFlag1}:               {si1_pass_all:>10,} ({100*si1_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"SnowIceFlag == {SnowIceFlag2}:               {si2_pass_all:>10,} ({100*si2_pass_all/total_pixels_all:5.1f}%)", flush=True)
        print(f"SnowIceFlag == {SnowIceFlag3}:               {si3_pass_all:>10,} ({100*si3_pass_all/total_pixels_all:5.1f}%)", flush=True)
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
                
                n_corners = 8
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
        load_and_save_OMI_KNMI_to_nc(tess_out, nc_out, len(tess_var_str), tlat, tlon)
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
