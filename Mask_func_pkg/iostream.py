import numpy as np
import netCDF4 as nc
import scipy.io as scio
import mat73 as mat
import os
from Mask_func_pkg.utils import cropped_data_indir

def load_mask_index_files():
    indir = '/path/to/supportData/NA_Masks/mask_index_files/'
    LANDigIND_0p01_infile = indir + 'LANDigIND_0p01.npy'
    LANDigLAT_0p01_infile = indir + 'LANDigLAT_0p01.npy'
    LANDigLON_0p01_infile = indir + 'LANDigLON_0p01.npy'
    LANDigIND_0p01 = np.load(LANDigIND_0p01_infile)
    LANDigLAT_0p01 = np.load(LANDigLAT_0p01_infile)
    LANDigLON_0p01 = np.load(LANDigLON_0p01_infile)
    return LANDigIND_0p01, LANDigLAT_0p01, LANDigLON_0p01

def load_GL_GeoLatLon():
    indir = '/path/to/NO2_DL_global/NO2_global_pkg/input_variables/'
    lat_infile = indir + 'tSATLAT_global.npy'
    lon_infile = indir + 'tSATLON_global.npy'
    GL_GeoLAT = np.load(lat_infile)
    GL_GeoLON = np.load(lon_infile)
    return GL_GeoLAT, GL_GeoLON


def load_cropped_mask_map(Continent_Name:str,region_type_name:str):
    indir = cropped_data_indir
    infile = indir + '{}.nc'.format(Continent_Name)
    data = nc.Dataset(infile)
    cropped_mask_data = data[region_type_name.lower()][:]
    Lat = data['lat'][:]
    Lon = data['lon'][:]
    return cropped_mask_data, Lat, Lon