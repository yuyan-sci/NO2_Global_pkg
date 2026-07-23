"""Build a strict Natural Earth land mask on the pipeline's (13000, 36000)
global 0.01° grid, and cache it as a NumPy .npy for fast reuse.

The mask is a boolean array:  True  = land pixel,  False = ocean / missing.

Why: the old land mask (NewLandMask-0.01.mat, sum of MASKp1..MASKp7) leaks
into ocean pixels, so estimation code predicts over water and the published
map has values in the ocean. Natural Earth 10 m land polygons give a strict
coastline on the same 0.01° grid, which the pipeline's Estimation,
Uncertainty and plotting steps already use.

Run once (takes a few minutes and ~a few GB of transient memory):
    python3 Estimation_pkg/build_natural_earth_land_mask.py

Output:
    /path/to/NO2_DL_global/input_variables/mask/
        Natural_Earth_land_mask_10m_0p01deg.npy
"""
import argparse
import os

import numpy as np
import regionmask
from cartopy.io import shapereader as shpreader


INPUT_VARS = '/path/to/NO2_DL_global/input_variables/'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resolution', type=str, default='10m',
                   choices=['10m', '50m', '110m'],
                   help='Natural Earth resolution (default 10m, strictest).')
    p.add_argument('--lat-npy', type=str,
                   default=os.path.join(INPUT_VARS, 'tSATLAT_global.npy'))
    p.add_argument('--lon-npy', type=str,
                   default=os.path.join(INPUT_VARS, 'tSATLON_global.npy'))
    p.add_argument('--out', type=str,
                   default=os.path.join(INPUT_VARS, 'mask',
                                         'Natural_Earth_land_mask_10m_0p01deg.npy'))
    return p.parse_args()


def main():
    args = parse_args()

    lat = np.load(args.lat_npy)
    lon = np.load(args.lon_npy)
    print(f'Grid:  lat {lat.shape} [{lat[0]:.3f} .. {lat[-1]:.3f}]')
    print(f'       lon {lon.shape} [{lon[0]:.3f} .. {lon[-1]:.3f}]')

    print(f'Loading Natural Earth {args.resolution} physical/land polygons ...')
    shp = shpreader.natural_earth(resolution=args.resolution,
                                  category='physical', name='land')
    geoms = list(shpreader.Reader(shp).geometries())
    regions = regionmask.Regions(outlines=geoms)
    print(f'  {len(geoms)} land polygons')

    # regionmask.mask() returns an xarray with region id (or NaN for ocean).
    # We only care about the NaN/not-NaN split.
    print('Rasterising land polygons onto the 0.01° grid ...')
    # regionmask builds a (ny, nx) 2D mask without us having to materialise a
    # meshgrid, which saves memory on the (13000, 36000) grid.
    m = regions.mask(lon, lat).values
    is_land = ~np.isnan(m)
    del m

    print(f'Land pixels: {is_land.sum():,d}  ({100.0 * is_land.mean():.2f}% of grid)')

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, is_land)
    print(f'Saved -> {args.out}   ({is_land.nbytes / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
