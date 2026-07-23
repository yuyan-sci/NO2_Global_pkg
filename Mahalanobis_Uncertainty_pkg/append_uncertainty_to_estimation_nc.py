"""Phase U6 post-processing.

For each monthly estimation NetCDF in a given year, copy it to a release-named
file and append an NO2_uncertainty variable (absolute uncertainty in ppb) with
CF/ACDD-compliant attributes, _FillValue=-999, compression and chunking.

The absolute uncertainty .npy is the output of Phase U5/U6
(Get_absolute_uncertainty_map). It is larger than the estimation map (has a
halo padding), so we crop it back to the estimation shape before writing.

Non-destructive: source .nc is never modified. Output goes to a release/
subdirectory with the publication filename
    NO2_surface_monthly_0p01deg_{YYYYMM}.nc
"""

import argparse
import os
import shutil
from datetime import datetime

import numpy as np
import netCDF4 as nc

# Make sibling packages importable when running from Mahalanobis_Uncertainty/
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from map_uncertainty_func.iostream import load_absolute_uncertainty_map

# Paths match map_uncertainty_func.utils and data_func.utils
EST_ROOT = '/path/to/NO2_DL_global/Training_Evaluation_Estimation/'

MONTH_STR = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


def _center_crop(big: np.ndarray, target_shape):
    """Reverse the symmetric center padding used in Get_absolute_uncertainty_map."""
    bh, bw = big.shape
    th, tw = target_shape
    pad_h = (bh - th) // 2
    pad_w = (bw - tw) // 2
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f'uncertainty map shape {big.shape} smaller than estimation {target_shape}')
    return big[pad_h:pad_h + th, pad_w:pad_w + tw]


def _append_uncertainty_variable(nc_path: str, unc_2d: np.ndarray, species: str):
    """Open NetCDF in append mode and add {SPECIES}_uncertainty variable."""
    with nc.Dataset(nc_path, 'a') as ds:
        varname = f'{species}_uncertainty'
        if varname in ds.variables:
            # replace in place: netCDF4 cannot delete a variable, so overwrite data only
            ds.variables[varname][:] = unc_2d
            return

        lat_size = ds.dimensions['lat'].size
        lon_size = ds.dimensions['lon'].size
        chunk_lat = min(512, lat_size)
        chunk_lon = min(512, lon_size)

        # NaN fill so ocean / no-data cells are real missing values.
        unc = ds.createVariable(
            varname, 'f4', ('lat', 'lon'),
            fill_value=np.float32(np.nan),
            zlib=True, complevel=4, shuffle=True,
            chunksizes=(chunk_lat, chunk_lon),
        )
        unc.standard_name = f'{species.lower()}_absolute_uncertainty'
        unc.long_name     = f'Absolute uncertainty (Mahalanobis-distance based) for Monthly {species} [ppb]'
        unc.units         = 'ppb'
        unc.coordinates   = 'lat lon'
        unc.grid_mapping  = 'crs'
        unc[:] = unc_2d

        # Stamp history
        stamp = datetime.utcnow().strftime('%Y-%m-%d')
        prev = getattr(ds, 'history', '')
        ds.history = (prev + ' ; ' if prev else '') + f'{stamp} appended {varname} via append_uncertainty_to_estimation_nc.py'


def process_month(year: int, month_idx: int,
                  species: str, version: str, special_name: str,
                  obs_version: str, nearby_sites_number: int,
                  release_subdir: str, dry_run: bool,
                  source: str = 'forced_slope_unity'):
    mm = MONTH_STR[month_idx]

    # Published release .nc is built from the ForcedSlopeUnity map by default
    # (set --source map_estimation to target the plain Map_Estimation files).
    if source == 'forced_slope_unity':
        src_dir = f'{EST_ROOT}{species}/{version}/ForcedSlopeUnity_Map_Estimation/{year}/'
        src_base = f'{species}_{version}_{year}{mm}_ForcedSlopeUnity'
    elif source == 'map_estimation':
        src_dir = f'{EST_ROOT}{species}/{version}/Map_Estimation/{year}/'
        src_base = f'{species}_{version}_{year}{mm}{special_name}'
    else:
        raise ValueError(f'Unknown --source: {source!r}')

    landfill_nc = f'{src_dir}{src_base}_landfill.nc'
    original_nc = f'{src_dir}{src_base}.nc'
    src_nc = landfill_nc if os.path.isfile(landfill_nc) else original_nc

    out_dir = f'{src_dir}{release_subdir}/'
    out_nc  = f'{out_dir}{species}_surface_monthly_0p01deg_{year}{mm}.nc'

    if not os.path.isfile(src_nc):
        print(f'  [skip] missing estimation file: {src_nc}')
        return

    # Load uncertainty .npy (shape = padded lat x padded lon)
    try:
        unc_big = load_absolute_uncertainty_map(
            species=species, version=version, special_name=special_name,
            YYYY=year, MM=mm,
            obs_version=obs_version, nearby_sites_number=nearby_sites_number,
        )
    except FileNotFoundError as e:
        print(f'  [skip] missing uncertainty npy for {year}{mm}: {e}')
        return

    # Pull estimation shape
    with nc.Dataset(src_nc, 'r') as ds:
        est_shape = ds.variables[species].shape
        est_data  = ds.variables[species][:]

    unc_cropped = _center_crop(unc_big, est_shape).astype('f4')

    # Mirror the estimation's NaN mask onto the uncertainty, so the two
    # variables in the release file have exactly the same ocean / no-data
    # footprint.
    if np.ma.isMaskedArray(est_data):
        est_arr = est_data.filled(np.nan)
    else:
        est_arr = np.asarray(est_data, dtype=np.float32)
    unc_cropped = np.where(np.isnan(est_arr), np.nan, unc_cropped)

    src_kind = 'landfill' if src_nc == landfill_nc else 'original'
    print(f'  {year}-{mm}: src={src_kind}  est={est_shape}  unc_npy={unc_big.shape}  -> {out_nc}')

    if dry_run:
        return

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    shutil.copyfile(src_nc, out_nc)
    _append_uncertainty_variable(out_nc, unc_cropped, species)


def parse_args():
    p = argparse.ArgumentParser(description='Phase U6 post-processing: append NO2_uncertainty to estimation .nc files.')
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--species', type=str, default='NO2')
    p.add_argument('--version', type=str, default='v5')
    p.add_argument('--special-name', type=str, default='')
    p.add_argument('--obs-version', type=str, default=None,
                   help="Obs version string. Defaults to data_func.utils.Obs_version if not given.")
    p.add_argument('--nearby-sites-number', type=int, default=30)
    p.add_argument('--months', type=str, default='1,2,3,4,5,6,7,8,9,10,11,12',
                   help='Comma-separated months (1-12).')
    p.add_argument('--release-subdir', type=str, default='release',
                   help='Subdirectory (under the per-year source dir) to write released files.')
    p.add_argument('--source', type=str, default='forced_slope_unity',
                   choices=['forced_slope_unity', 'map_estimation'],
                   help='Which estimation .nc to copy from. Published dataset uses forced_slope_unity.')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.obs_version is None:
        from data_func.utils import Obs_version as _Obs_version
        args.obs_version = _Obs_version

    months = [int(m) - 1 for m in args.months.split(',') if m.strip() != '']
    print(f'[U6-append] year={args.year}  species={args.species}  version={args.version}  '
          f'obs_version={args.obs_version}  sites={args.nearby_sites_number}  months={[m+1 for m in months]}')

    for m in months:
        process_month(
            year=args.year, month_idx=m,
            species=args.species, version=args.version,
            special_name=args.special_name,
            obs_version=args.obs_version,
            nearby_sites_number=args.nearby_sites_number,
            release_subdir=args.release_subdir,
            dry_run=args.dry_run,
            source=args.source,
        )


if __name__ == '__main__':
    main()
