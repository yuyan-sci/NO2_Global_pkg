import numpy as np
import netCDF4 as nc
import os 
from LandCover_nearest_distances_pkg.utils import LandCover_buffer_outdir, LandCover_nearest_distance_outdir


def save_buffer_map(mapdata,nametag,buffer,YEAR,Area):
    outdir = LandCover_buffer_outdir + '{}/'.format(nametag)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}-MCD12C1_Buffer-{}-forEachPixel_001x001_{}_{}.npy'.format(nametag,buffer,Area,YEAR)
    print(outfile)
    
    # Convert to float64 before saving
    mapdata_float64 = np.array(mapdata, dtype=np.float64)
    # Remove dtype argument from save
    np.save(outfile, mapdata_float64)
    return

def save_landcover_nearest_pixels(mapdata,nametag,YEAR,Area):
    outdir = LandCover_nearest_distance_outdir + '{}/'.format(nametag)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}-MCD12C1_forEachPixel_001x001_{}_{}.npy'.format(nametag,Area,YEAR)
    print(outfile)
    
    # Convert to float64 before saving
    mapdata_float64 = np.array(mapdata, dtype=np.float64)
    # Remove dtype argument from save
    np.save(outfile, mapdata_float64)
    return

def load_landcover_nearest_pixels(nametag,YEAR,Area):
    indir = LandCover_nearest_distance_outdir + '{}/'.format(nametag)
    infile = indir + '{}-MCD12C1_forEachPixel_001x001_{}_{}.npy'.format(nametag,Area,YEAR)
    if not os.path.isfile(infile):
        print('The file {} does not exist!'.format(infile))
        return None
    
    landcover_nearest_distance_map = np.load(infile)
    return landcover_nearest_distance_map

def load_Global_GeoLatLon():
    indir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/'
    lat_infile = indir + 'tSATLAT_global.npy'
    lon_infile = indir + 'tSATLON_global.npy'
    Global_GeoLAT = np.load(lat_infile)
    Global_GeoLON = np.load(lon_infile)
    return Global_GeoLAT, Global_GeoLON


def load_Global_GeoLatLon_Map():
    indir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/'
    lat_infile = indir + 'tSATLAT_global_MAP.npy'
    lon_infile = indir + 'tSATLON_global_MAP.npy'
    Global_GeoLAT_MAP = np.load(lat_infile)
    Global_GeoLON_MAP = np.load(lon_infile)
    return Global_GeoLAT_MAP, Global_GeoLON_MAP