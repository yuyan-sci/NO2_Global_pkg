"""
main_nearest_neighbour.py

This script is used to regrid the GCHP data to the nearest neighbour grid.
 files generaed by this script are forTessellation and forObservation-Geophysical
 forTessellation contains:
    NO2 profile of at 0.1° resolution at local solar hours 13-15
    ['SpeciesConcVV_NO2','Met_PMIDDRY','Met_BXHEIGHT','Met_AIRDEN', 'Met_TROPPT']
 forObservation-Geophysical contains:
    the daily GCHP surface NO2, PAN, HNO3, and alkylnitrates at 1 km resolution
    the GCHP NO2 Total and Tropospheric column data at 1 km resolution
"""

#!/usr/bin/env python3
import os
import argparse
import datetime
import numpy as np
import xarray as xr
import sparselt.esmf
import sparselt.xr
import xesmf as xe
import dask
import psutil
import gc


# — Configure dask for bounded memory usage —
dask.config.set({'array.chunk-size': '500MB'})

gchp_version = 'gchp-v2'
# — User settings & constants —
IN_ROOT        = '/fs2/yuanjian.z/archive/c180/longterm_v2/OutputDir'
MET_ROOT       = '/ExtData/GEOS_C180/GEOS_IT'
WEIGHT_FILE    = '/path/to/supportData/gridinfo/c180_to_1800x3600_weights.nc'
OUT_TESS_ROOT  = f'/path/to/{gchp_version}/forTessellation'
OUT_GEO_ROOT   = f'/path/to/{gchp_version}/forObservation-Geophysical'
LOCAL_HOURS    = [13, 14, 15]   # desired local solar hours (satellite overpass ~13:30 LST)
KEEP_DIMS      = {'lev','nf','Ydim','Xdim'}
DIM_ORDER      = ('lev','nf','Ydim','Xdim','time')
CRES_01        = '01x01'
NA             = 6.022e23    # molecules/mol
MwAir          = 28.97       # g/mol
# Chunking for optional fallback
CHUNK_LAT      = 1000        
CHUNK_LON      = 1000
# Nearest-neighbor repeat factor: fine grid (18000×36000) / coarse grid (1800×3600)
FINE_FACTOR    = 10

def print_memory_usage(stage=""):
    mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[{stage}] RSS = {mb:.1f} MB", flush=True)

# — Nearest-neighbour helpers (parallel structure to bilinear helpers) —
def get_nearestneighbour_Index(fine_Lat, fine_Lon, coarse_Lat, coarse_Lon):
    """
    Compute 1-based floor indices mapping each fine-grid point to its
    nearest coarse-grid cell. Only floor indices are needed (no ceil, dx, dy).
    """
    delta_x   = coarse_Lat[1] - coarse_Lat[0]
    delta_y   = coarse_Lon[1] - coarse_Lon[0]
    lat_floor = np.floor((fine_Lat - coarse_Lat[0]) / delta_x).astype(int) + 1
    lon_floor = np.floor((fine_Lon - coarse_Lon[0]) / delta_y).astype(int) + 1
    lat_floor = np.clip(lat_floor, 1, len(coarse_Lat))
    lon_floor = np.clip(lon_floor, 1, len(coarse_Lon))
    return lat_floor, lon_floor

def get_Concentration4NearestNeighbour(init_mapdata, lat_floor_array, lon_floor_array, ix):
    """
    Return the coarse values at the nearest-neighbour cell for row ix.
    Mirrors get_Concentration4Interpolation but only needs the floor-floor corner.
    """
    return init_mapdata[lat_floor_array[ix] - 1, lon_floor_array[:] - 1]

def Nearestneighbour_GC_concentration(Cxfyf):
    """
    Nearest-neighbour 'interpolation': the value is simply the nearest cell.
    Mirrors Bilinearinterpolate_GC_concentration in signature and call site.
    """
    return Cxfyf


def init_worker():
    global transform
    global lon_01, lat_01, lon_km, lat_km
    global lat_floor, lon_floor

    print_memory_usage("init start")
    # load ESMF weights once
    transform = sparselt.esmf.load_weights(
        WEIGHT_FILE,
        input_dims =[('nf','Ydim','Xdim'), (6,180,180)],
        output_dims=[('lat','lon'), (1800,3600)]
    )
    # define coarse (0.1°) and fine (1 km) grids
    lon_01 = np.round(np.linspace(-179.995, 179.995, 3600), 5)
    lat_01 = np.round(np.linspace( -89.995,  89.995, 1800), 5)
    lon_km = np.round(np.linspace(-179.995, 179.995, 36000), 5)
    lat_km = np.round(np.linspace( -89.995,  89.995, 18000), 5)

    lat_floor, lon_floor = get_nearestneighbour_Index(lat_km, lon_km, lat_01, lon_01)

    print_memory_usage("init done")

def process_day(year, mon, day):
    # prepare output dirs
    src_dir = os.path.join(IN_ROOT, str(year), f"{mon:02d}")
    tessD = os.path.join(OUT_TESS_ROOT, str(year),'daily'); os.makedirs(tessD, exist_ok=True)
    geoD  = os.path.join(OUT_GEO_ROOT,  str(year),'daily'); os.makedirs(geoD,  exist_ok=True)

    print_memory_usage("start process_day")

    # ── A) local-time 3-h avg → 0.1° ──
    # UTC offset per 0.1° longitude: round(lon/15), range -12..+12
    UTC_OFF = np.round(lon_01 / 15).astype(int)          # shape (3600,)

    # Raw UTC hours (no mod): 13-12=1 .. 15-(-12)=27
    # raw_h <= 23 → same day; raw_h >= 24 → next calendar day (raw_h - 24)
    needed_raw = sorted({lh - int(off)
                         for off in np.unique(UTC_OFF)
                         for lh in LOCAL_HOURS})

    base_date = datetime.date(year, mon, day)
    next_date = base_date + datetime.timedelta(days=1)

    # Stream one hour at a time: accumulate running sum, free each regridded dataset immediately.
    # This keeps peak memory ~1 regridded dataset instead of 27 simultaneously.
    utc_off_unique = np.unique(UTC_OFF)
    off_masks = {off: (UTC_OFF == off) for off in utc_off_unique}
    h_to_offsets = {raw_h: [off for off in utc_off_unique if (raw_h + int(off)) in LOCAL_HOURS]
                    for raw_h in needed_raw}

    local_sum  = None
    ref_dims   = None
    ref_coords = None

    for raw_h in needed_raw:
        offsets = h_to_offsets[raw_h]
        if not offsets:
            continue
        if raw_h <= 23:
            file_date, h = base_date, raw_h
        else:
            file_date, h = next_date, raw_h - 24
        fp = os.path.join(IN_ROOT, str(file_date.year),
                          f"{file_date.month:02d}",
                          f"GEOSChem.ACAGNO2Hourly."
                          f"{file_date.year}{file_date.month:02d}{file_date.day:02d}_{h:02d}00z.nc4")
        if not os.path.exists(fp):
            fp = os.path.join(src_dir,
                              f"GEOSChem.ACAGNO2Hourly.{year}{mon:02d}{day:02d}_2300z.nc4")
        ds = xr.open_dataset(fp).squeeze()
        ds = ds.drop_dims([d for d in ds.dims if d not in KEEP_DIMS], errors="ignore")
        for v in ds.data_vars:
            if ds[v].dtype == "float64":
                ds[v] = ds[v].astype("float32")
        ds = ds.transpose(*DIM_ORDER, missing_dims="ignore")
        ds_regrid = sparselt.xr.apply(transform, ds).assign_coords(lon=lon_01, lat=lat_01)
        ds.close()
        if local_sum is None:
            local_sum  = {var: np.zeros(ds_regrid[var].shape, dtype=np.float32)
                          for var in ds_regrid.data_vars}
            ref_dims   = {var: ds_regrid[var].dims for var in ds_regrid.data_vars}
            ref_coords = ds_regrid.coords
        for var in ds_regrid.data_vars:
            arr = ds_regrid[var].values
            for off in offsets:
                local_sum[var][..., off_masks[off]] += arr[..., off_masks[off]]
        del ds_regrid
        gc.collect()
        print_memory_usage(f"after regrid hour {raw_h}")

    # Each longitude zone received exactly len(LOCAL_HOURS) contributions.
    local_avg = {var: (ref_dims[var], local_sum[var] / len(LOCAL_HOURS)) for var in local_sum}
    d01 = xr.Dataset(local_avg, coords=ref_coords)
    del local_sum
    gc.collect()
    print_memory_usage("after 0.1° regrid")

    # ── A.1) Load TROPPT from GEOS-IT met with proper local-time conversion ──
    # Apply the same UTC-offset logic as NO2: for each longitude zone select
    # the UTC hour that corresponds to local solar hours 13-15.
    met       = os.path.join(MET_ROOT, str(year), f'{mon:02d}')
    fp_met    = os.path.join(met, f'GEOSIT.{year}{mon:02d}{day:02d}.A1.C180.nc')
    fp_met_nd = os.path.join(MET_ROOT, str(next_date.year), f'{next_date.month:02d}',
                              f'GEOSIT.{next_date.year}{next_date.month:02d}{next_date.day:02d}.A1.C180.nc')
    ds_met    = xr.open_dataset(fp_met)
    ds_met_nd = xr.open_dataset(fp_met_nd) if os.path.exists(fp_met_nd) else None

    troppt_sum = None
    for raw_h in needed_raw:
        offsets = h_to_offsets[raw_h]
        if not offsets:
            continue
        if raw_h <= 23:
            troppt_h = ds_met[['TROPPT']].isel(time=raw_h)
        else:
            if ds_met_nd is None:
                continue
            troppt_h = ds_met_nd[['TROPPT']].isel(time=raw_h - 24)
        troppt_h = troppt_h.drop_dims(
            [d for d in troppt_h.dims if d not in KEEP_DIMS], errors='ignore'
        ).transpose(*DIM_ORDER, missing_dims='ignore')
        troppt_rg = sparselt.xr.apply(transform, troppt_h).assign_coords(lon=lon_01, lat=lat_01)
        if troppt_sum is None:
            troppt_sum = np.zeros(troppt_rg['TROPPT'].shape, dtype=np.float32)
        for off in offsets:
            troppt_sum[..., off_masks[off]] += troppt_rg['TROPPT'].values[..., off_masks[off]]
        del troppt_rg
        gc.collect()

    ds_met.close()
    if ds_met_nd is not None:
        ds_met_nd.close()

    d01_met = xr.Dataset(
        {'TROPPT': (['lat', 'lon'], troppt_sum / len(LOCAL_HOURS))},
        coords={'lat': lat_01, 'lon': lon_01}
    )
    print_memory_usage("after 0.1° regrid TROPPT")

    # ── A.2) save five vars at 0.1° (including Met_TROPPT for trop column computation) ──
    out01  = d01[['SpeciesConcVV_NO2','Met_PMIDDRY','Met_BXHEIGHT','Met_AIRDEN']]
    out01['Met_TROPPT'] = d01_met['TROPPT']
    path01 = os.path.join(tessD, f'{CRES_01}.Hours.13-15.{year}{mon:02d}{day:02d}.nc4')
    out01.to_netcdf(path01, encoding={v:{'zlib':True,'complevel':4} for v in out01.data_vars})
    print_memory_usage("after save 0.1°")

    # ── B) compute total & tropospheric NO2col @0.1°, then nearest-neighbour → 1 km ──
    AirDen = d01['Met_AIRDEN'] * 1e3
    BoxH   = d01['Met_BXHEIGHT']
    conc   = d01['SpeciesConcVV_NO2'] * AirDen / MwAir

    # B.1) Total column
    no2col = (conc * BoxH).sum('lev') * 1e-4 * NA
    coarse = no2col.values.astype(np.float32)
    fine_NO2 = coarse[lat_floor - 1][:, lon_floor - 1]

    # B.2) Tropospheric column — levels where mid-level pressure >= tropopause pressure
    trop_mask   = d01['Met_PMIDDRY'] >= d01_met['TROPPT']
    no2col_trop = (conc * BoxH).where(trop_mask).sum('lev') * 1e-4 * NA
    coarse_trop = no2col_trop.values.astype(np.float32)
    fine_trop   = coarse_trop[lat_floor - 1][:, lon_floor - 1]

    # Save both columns in one file
    ds_col = xr.Dataset(
        {'NO2col_tot': (['lat', 'lon'], fine_NO2),
         'NO2col_trop': (['lat', 'lon'], fine_trop)},
        coords={'lat': lat_km, 'lon': lon_km}
    )
    path1km = os.path.join(geoD, f'1x1km.Hours.13-15.{year}{mon:02d}{day:02d}.nc4')
    ds_col.to_netcdf(path1km, encoding={v: {'zlib': True, 'complevel': 4} for v in ds_col.data_vars})
    print_memory_usage("after save 1km NO2col_tot + NO2col_trop")
    
    
    # C) daily geophysical regrid: 0.1° then 1 km via helper
    fpD = os.path.join(src_dir, f'GEOSChem.ACAGGasDaily.{year}{mon:02d}{day:02d}_0000z.nc4')
    dD  = xr.open_dataset(fpD).squeeze()
    dD  = dD.drop_dims([d for d in dD.dims if d not in KEEP_DIMS], errors='ignore')
    d01D = sparselt.xr.apply(transform, dD).assign_coords(lon=lon_01, lat=lat_01)
    print_memory_usage("after 0.1° geophy")
    # build surface-level variables
    geo = xr.Dataset({
        'gchp_NO2':           d01D['SpeciesConcVV_NO2'].isel(lev=0)*1e9,
        'gchp_PAN':           d01D['SpeciesConcVV_PAN'].isel(lev=0)*1e9,
        'gchp_HNO3':          d01D['SpeciesConcVV_HNO3'].isel(lev=0)*1e9,
        'gchp_alkylnitrates': (d01D['SpeciesConcVV_BUTN']+d01D['SpeciesConcVV_NPRNO3']).isel(lev=0)*1e9,
    })
    # ── C.1) nearest-neighbour upsample to 1 km and save ──
    others = {}
    for var in geo.data_vars:
        coarse2D = geo[var].values.astype(np.float32)
        others[var] = (['lat', 'lon'], coarse2D[lat_floor - 1][:, lon_floor - 1])
        del coarse2D
        gc.collect()
    ds_geo = xr.Dataset(others, coords={'lat': lat_km, 'lon': lon_km})
    gpath = os.path.join(geoD, f'1x1km.DailyVars.{year}{mon:02d}{day:02d}.nc4')
    ds_geo.to_netcdf(gpath, encoding={v:{'zlib':True,'complevel':4} for v in ds_geo.data_vars})
    print_memory_usage("after daily geophy interpolation")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--mon',  type=int, required=True)
    parser.add_argument('--day',  type=int, required=True)
    args = parser.parse_args()

    init_worker()
    process_day(args.year, args.mon, args.day)
    # plot_multiple_geophy(args.year, f"{args.mon:02d}", f"{args.day:02d}", 
    #                      vars_to_plot = ['gchp_NH3','gchp_O3','gchp_OH'])
