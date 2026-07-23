import netCDF4 as nc
import numpy as np


indir = '/ExtData/HEMCO/GFED4/v2023-03/'

Offline_GFED_outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/GFED4_Emissions_input/'

delta_x = 0.25
delta_y = 0.25

def get_certain_emissions(data,keyword):
    map_data = data.variables[keyword][:]
    map_data = np.squeeze(map_data)
    return map_data