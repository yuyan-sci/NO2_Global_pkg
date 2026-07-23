import numpy as np
import netCDF4 as nc
import mat73 as mat
import os
import scipy.io as scio
from GFED_input_pkg.utils import Offline_GFED_outdir

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

def load_Coarse_LatLon():
    infile = '/ExtData/HEMCO/GFED4/v2023-03/2019/GFED4_gen.025x025.201901.nc'
    dataset = nc.Dataset(infile)
    Coarse_lat = dataset.variables['lat'][:]
    Coarse_lon = dataset.variables['lon'][:]
    return Coarse_lat, Coarse_lon

def load_OFFLINE_GFED4_dataset(indir,YEAR,MONTH):
    infile = indir + '{}/GFED4_gen.025x025.{}{}.nc'.format(YEAR,YEAR,MONTH)
    dataset = nc.Dataset(infile)
    return dataset

def load_GFED4_interpolate_indices(indir):
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



def load_GFED_interpolated_mapdata(nametag,YEAR,MONTH):
    indir = Offline_GFED_outdir + '{}/'.format(YEAR)
    infile = indir + 'GFED4-{}_001x001_{}{}.npy'.format(nametag,YEAR,MONTH)
    mapdata = np.load(infile)
    return mapdata



def save_GFED_interpolate_indices(outdir,lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
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


def save_GFED_interpolated_mapdata(mapdata,nametag,YEAR,MONTH):
    outdir = Offline_GFED_outdir + '{}/'.format(YEAR)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    outfile = outdir + 'GFED4-{}_001x001_{}{}.npy'.format(nametag,YEAR,MONTH)
    print(outfile)
    np.save(outfile,mapdata)
    
    return

