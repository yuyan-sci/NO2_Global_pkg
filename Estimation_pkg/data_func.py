import numpy as np
import os
import netCDF4 as nc
import mat73 as mat
from Uncertainty_pkg.iostream import load_pixels_nearest_sites_distances_map
from Estimation_pkg.iostream import load_map_data
from Estimation_pkg.utils import *
from Training_pkg.Statistic_Func import linear_regression,regress2, Cal_RMSE, Cal_NRMSE,Cal_PWM_rRMSE,Calculate_PWA_PM25

from Training_pkg.utils import *

def crop_mapdata(init_map,extent):
    bottom_lat = extent[0]
    top_lat    = extent[1]
    left_lon   = extent[2]
    right_lon  = extent[3]

    lat_start_index = int((bottom_lat - 10.005)* 100)
    lon_start_index = int((left_lon + 169.995) * 100 )
    lat_end_index = int((top_lat - 10.005) * 100 )
    lon_end_index = int((right_lon + 169.995)*100 )
    cropped_mapdata = init_map[lat_start_index:lat_end_index+1,lon_start_index:lon_end_index+1]
    return cropped_mapdata

def get_extent_index(extent)->np.array:
    '''
    :param extent:
        The range of the input. [Bottom_Lat, Up_Lat, Left_Lon, Right_Lon]
    :return:
        lat_index, lon_index
    '''
    indir = '/path/to/NO2_DL_global/input_variables/'
    lat_infile = indir + 'tSATLAT_global.npy'
    lon_infile = indir + 'tSATLON_global.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)
    lat_index = np.where((SATLAT >= extent[0])&(SATLAT<=extent[1]))
    lon_index = np.where((SATLON >= extent[2])&(SATLON<=extent[3]))
    lat_index = np.squeeze(np.array(lat_index))
    lon_index = np.squeeze(np.array(lon_index))
    return lat_index,lon_index

def get_GL_extent_index(extent)->np.array:
    '''
    :param extent:
        The range of the input. [Bottom_Lat, Up_Lat, Left_Lon, Right_Lon]
    :return:
        lat_index, lon_index
    '''
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')
    lat_index = np.where((SATLAT >= extent[0])&(SATLAT<=extent[1]))
    lon_index = np.where((SATLON >= extent[2])&(SATLON<=extent[3]))
    lat_index = np.squeeze(np.array(lat_index))
    lon_index = np.squeeze(np.array(lon_index))
    return lat_index,lon_index

def get_landtype(YYYY, extent) -> np.array:
    """Return a land mask (0 = ocean, 1 = land) subsetted to `extent`.

    Primary source: strict Natural Earth 10 m land mask cached on the
    pipeline's (13000, 36000) 0.01 deg grid at
        /path/to/NO2_DL_global/input_variables/mask/
            Natural_Earth_land_mask_10m_0p01deg.npy
    built once by Estimation_pkg/build_natural_earth_land_mask.py.

    Fallback: legacy NewLandMask-0.01.mat (union of MASKp1..MASKp7). That
    mask leaked into ocean at coastlines so the estimation predicted over
    water; it is kept here only as a transition safety net. Once the strict
    .npy is built on any given host, all new estimation runs use it.
    """
    strict_mask_npy = ('/path/to/NO2_DL_global/input_variables/'
                       'mask/Natural_Earth_land_mask_10m_0p01deg.npy')
    if os.path.exists(strict_mask_npy):
        landtype_full = np.load(strict_mask_npy, mmap_mode='r')
    else:
        landtype_infile = '/path/to/supportData/mask/Land_Ocean_Mask/NewLandMask-0.01.mat'
        LandMask = mat.loadmat(landtype_infile)
        landtype_full = (LandMask['MASKp1'] + LandMask['MASKp2'] + LandMask['MASKp3']
                         + LandMask['MASKp4'] + LandMask['MASKp5'] + LandMask['MASKp6']
                         + LandMask['MASKp7'])

    lat_index, lon_index = get_GL_extent_index(extent=extent)
    output = np.asarray(landtype_full[np.ix_(lat_index, lon_index)], dtype=int)
    return output

def Get_coefficient_map():
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/'.format(species,version)
    outfile = outdir + '{}_coefficient_startDistance_{}km.npy'.format(species,Coefficient_start_distance)
    if os.path.exists(outfile):
        coefficient = np.load(outfile)
    else:
        nearest_site_distance_forEachPixel = load_pixels_nearest_sites_distances_map()
        coefficient = (nearest_site_distance_forEachPixel - Coefficient_start_distance)/(nearest_site_distance_forEachPixel+1.0)
        coefficient[np.where(coefficient<0.0)]=0.0
        coefficient = np.square(coefficient)
        np.save(outfile,coefficient)
    return coefficient

def Combine_CNN_GeophysicalSpecies(CNN_Species,coefficient,
                                YYYY,MM):
   GeophysicalSpecies = load_map_data('Geo{}'.format(species),YYYY=YYYY,MM=MM)
   Cropped_GeophysicalSpecies = crop_mapdata(init_map=GeophysicalSpecies,extent=Extent)
   Combined_Species = (1.0-coefficient)*CNN_Species + coefficient * Cropped_GeophysicalSpecies
   return Combined_Species

def Estimation_ForcedSlopeUnity_Func(train_final_data,train_obs_data,train_area_index,endyear,beginyear,month_index):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ForcedSlopeUnity_Dictionary_forEstimation = {'slope':{}, 'offset':{}}
    for iyear in range((endyear - beginyear + 1)):
        ForcedSlopeUnity_Dictionary_forEstimation['slope'][str(beginyear+iyear)] = {}
        ForcedSlopeUnity_Dictionary_forEstimation['offset'][str(beginyear+iyear)] = {}
        for imonth in range(len(month_index)):        
            temp_train_final_data = train_final_data[(iyear*len(month_index)+imonth)*len(train_area_index):(iyear*len(month_index)+imonth+1)*len(train_area_index)]
            temp_train_obs_data   = train_obs_data[(iyear*len(month_index)+imonth)*len(train_area_index):(iyear*len(month_index)+imonth+1)*len(train_area_index)]
            temp_regression_dic = regress2(_x=temp_train_obs_data,_y=temp_train_final_data,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
            temp_offset,temp_slope = temp_regression_dic['intercept'], temp_regression_dic['slope']
            ForcedSlopeUnity_Dictionary_forEstimation['slope'][str(beginyear+iyear)][MONTH[month_index[imonth]]] = temp_slope
            ForcedSlopeUnity_Dictionary_forEstimation['offset'][str(beginyear+iyear)][MONTH[month_index[imonth]]] = temp_offset
    return ForcedSlopeUnity_Dictionary_forEstimation
