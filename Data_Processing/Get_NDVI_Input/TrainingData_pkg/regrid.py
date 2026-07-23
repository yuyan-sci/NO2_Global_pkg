import numpy as np
import netCDF4 as nc
import scipy.io as io
from scipy import interpolate
from scipy.io import savemat


def get_nearest_site_index(sitelon, sitelat, lon_grid, lat_grid):
    '''
    func: get the index of stations on the grids map
    inputs:
        sitelon, sitelat: stations location，eg:[42.353,110.137] 0th dim:lon 1st dim:lat
        lon_grid: grids longitude
        lat_grid: grids latitude
    return:
        index: [index_lat,index_lon]
    '''
    # step1: get the spatial resolution; Default: the latitude and longitude have the same resolution
    det = lon_grid[1] - lon_grid[0]
    # step2:

    lon_min = np.min(lon_grid)
    lat_min = np.min(lat_grid)

    index_lon = np.round((sitelon - lon_min) / det)
    index_lat = np.round((sitelat - lat_min) / det)

    index_lon = index_lon.astype(int)
    index_lat = index_lat.astype(int)

    return index_lon, index_lat


def regrid_map_nearest_index(FINAL_LAT, FINAL_LON,INI_LAT_MIN, INI_LON_MIN, INI_LAT_Delta, INI_LON_Delta):
    """This function is used to convert one map to anther map (regrid).

    Args:
        FINAL_LAT (_type_): Final latitude
        FINAL_LON (_type_): Finall longitude
        INI_LAT_MIN (_type_): Initial latitude
        INI_LON_MIN (_type_): Initial longitude
        INI_LAT_Delta (_type_): Initial latitude
        INI_LON_Delta (_type_): Initial longitude

    Returns:
        _type_: _description_
    """
    lat_index = np.zeros((len(FINAL_LAT)),dtype = np.float)
    lon_index = np.zeros((len(FINAL_LON)),dtype = np.float)

    for ix in range(len(FINAL_LAT)):
        lat_index[ix] = np.round((FINAL_LAT[ix] - INI_LAT_MIN) / INI_LAT_Delta)

    for iy in range(len(FINAL_LON)):

        lon_index[iy] = np.round((FINAL_LON[iy] - INI_LON_MIN) / INI_LON_Delta)

    #lon_index[np.where(lon_index == 144)] = 143
    #lat_index[np.where(lat_index == 91)] = 90
    lat_index = lat_index.astype(int)
    lon_index = lon_index.astype(int)
    return lat_index, 


