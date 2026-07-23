import numpy as np
import netCDF4 as nc
import os

def load_init_Global_Population_density(YEAR):
    indir = '/path/to/supportData/pop/'
    infile = indir + 'WorldPopGrid-{}-0.01-new-GPWv4.nc'.format(YEAR)
    MapData = nc.Dataset(infile)
    Population_data = MapData.variables['Population'][:]
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    return Population_data, lat, lon


def load_cropped_interpolated_PopulationMap(YEAR):
    indir  = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/Population_input/'
    if not os.path.isdir(indir):
        os.makedirs(indir)
    infile = indir + 'WorldPopGrid-{}-0.01.npy'.format(YEAR)
    TempMap = np.load(infile)
    return TempMap

def save_cropped_interpolated_PopulationMap(Population_MapData, YEAR):
    outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/Population_input/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'WorldPopGrid-{}-0.01.npy'.format(YEAR)
    np.save(outfile, Population_MapData.data)
    return