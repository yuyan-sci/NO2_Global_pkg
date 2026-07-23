import numpy as np
import netCDF4 as nc
import mat73 as mat
import os
import scipy.io as scio
from ISA_pkg.utils import ISA_indir, ISA_input_outdir


def load_Coarse_LatLon(YYYY):
    InDir = ISA_indir + 'EstISA_{}.nc'.format(YYYY)
    dataset = nc.Dataset(InDir)
    Coarse_lat = dataset.variables['lat'][:].data
    Coarse_lon = dataset.variables['lon'][:].data
    delta_x = np.abs(Coarse_lat[2]-Coarse_lat[1])
    delta_y = np.abs(Coarse_lon[2]-Coarse_lon[1])
    
    print('Coarse_lat.shape',Coarse_lat.shape)
    print('Coarse_lon.shape',Coarse_lon.shape)
    print('delta_x',delta_x)
    print('delta_y',delta_y)
    return Coarse_lat, Coarse_lon, delta_x, delta_y

def load_global_GeoLatLon():
    indir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/'
    lat_infile = indir + 'tSATLAT_global.npy'
    lon_infile = indir + 'tSATLON_global.npy'
    global_GeoLAT = np.load(lat_infile)
    global_GeoLON = np.load(lon_infile)
    return global_GeoLAT, global_GeoLON

def load_global_GeoLatLon_Map():
    indir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/'
    lat_infile = indir + 'tSATLAT_global_MAP.npy'
    lon_infile = indir + 'tSATLON_global_MAP.npy'
    global_GeoLAT_Map = np.load(lat_infile)
    global_GeoLON_Map = np.load(lon_infile)
    return global_GeoLAT_Map, global_GeoLON_Map


def load_ISA_data(YYYY):
    InDir = ISA_indir + 'EstISA_{}.nc'.format(YYYY)
    data = nc.Dataset(InDir)
    ISA = data.variables['ISA'][:]
    return ISA

def load_ISA_interpolate_indices(indir):
    lat_nearest_index_file = indir + 'lat_nearest_index.npy'
    lon_nearest_index_file = indir + 'lon_nearest_index.npy'
    lat_floor_index_file   = indir + 'lat_floor_index.npy'
    lon_floor_index_file   = indir + 'lon_floor_index.npy'
    lat_ceil_index_file    = indir + 'lat_ceil_index.npy'
    lon_ceil_index_file    = indir + 'lon_ceil_index.npy'
    dx_file = indir + 'distance_x.npy'
    dy_file = indir + 'distance_y.npy'

    lat_nearest_array = np.load(lat_nearest_index_file)
    lon_nearest_array = np.load(lon_nearest_index_file)
    lat_floor_array   = np.load(lat_floor_index_file)
    lon_floor_array   = np.load(lon_floor_index_file)
    lat_ceil_array    = np.load(lat_ceil_index_file)
    lon_ceil_array    = np.load(lon_ceil_index_file)
    dx = np.load(dx_file)
    dy = np.load(dy_file)
    return lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy

def load_ISA_interpolated_mapdata(YYYY):
    indir = ISA_input_outdir + '{}/'.format(YYYY)
    infile = indir + 'ISA_NOAA_001x001_global_{}.npy'.format(YYYY)
    mapdata = np.load(infile)
    return mapdata

def save_ISA_interpolate_indices(outdir,lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy):
    os.makedirs(outdir, exist_ok=True) 
    lat_nearest_index_file = outdir + 'lat_nearest_index.npy'
    lon_nearest_index_file = outdir + 'lon_nearest_index.npy'
    lat_floor_index_file   = outdir + 'lat_floor_index.npy'
    lon_floor_index_file   = outdir + 'lon_floor_index.npy'
    lat_ceil_index_file    = outdir + 'lat_ceil_index.npy'
    lon_ceil_index_file    = outdir + 'lon_ceil_index.npy'
    dx_file = outdir + 'distance_x.npy'
    dy_file = outdir + 'distance_y.npy'

    np.save(lat_nearest_index_file, lat_nearest_array)
    np.save(lon_nearest_index_file, lon_nearest_array)
    np.save(lat_floor_index_file, lat_floor_array)
    np.save(lon_floor_index_file, lon_floor_array)
    np.save(lat_ceil_index_file, lat_ceil_array)
    np.save(lon_ceil_index_file, lon_ceil_array)
    np.save(dx_file,dx)
    np.save(dy_file,dy)
    return

def save_ISA_interpolated_mapdata(mapdata,YYYY):
    outdir = ISA_input_outdir + '{}/'.format(YYYY)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'ISA_NOAA_001x001_global_{}.npy'.format(YYYY)
    print(outfile)
    mapdata_float64 = np.array(mapdata, dtype=np.float64)
    np.save(outfile, mapdata_float64)
    return
