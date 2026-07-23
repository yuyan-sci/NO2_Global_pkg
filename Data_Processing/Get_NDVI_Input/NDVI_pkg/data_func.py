import numpy as np
import netCDF4 as nc
from NDVI_pkg.iostream import *
from NDVI_pkg.utils import *
from TrainingData_pkg.interpolation import *

def get_NDVI_indices(YYYY,MM):
    for imonth in range(len(MM)):
        global_GeoLAT, global_GeoLON = load_global_GeoLatLon()
        global_GeoLAT_Map, global_GeoLON_Map = load_global_GeoLatLon_Map()
        GC_LAT, GC_LON, delta_x, delta_y = load_Coarse_LatLon(YYYY,MM[imonth])
        lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy = get_BilinearInterpolate_Index(fine_Lat=global_GeoLAT,fine_Lon=global_GeoLON,fine_Lat_map=global_GeoLAT_Map,
                                                                                                                                            fine_Lon_map=global_GeoLON_Map,coarse_Lat=GC_LAT,coarse_Lon=GC_LON)
        index_dir = f'/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/NDVI_input/{YYYY}-{MM[imonth]}/'
        save_NDVI_interpolate_indices(outdir=index_dir,lat_nearest_array=lat_nearest_array,lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,
                                   lon_nearest_array=lon_nearest_array,lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,dx=dx,dy=dy)

    return


def get_Concentration4Interpolation(init_mapdata,lat_ceil_array,lat_floor_array,lon_ceil_array,lon_floor_array,ix):
    Cxfyf = init_mapdata[lat_floor_array[ix]-1,lon_floor_array[:]-1]
    Cxfyc = init_mapdata[lat_floor_array[ix]-1,lon_ceil_array[:]-1]
    Cxcyf = init_mapdata[lat_ceil_array[ix]-1,lon_floor_array[:]-1]
    Cxcyc = init_mapdata[lat_ceil_array[ix]-1,lon_ceil_array[:]-1]
    return Cxcyc,Cxcyf,Cxfyc,Cxfyf


def interpolate_NDVI_map(GeoLAT,GeoLON,lat_ceil_array,lat_floor_array,lon_ceil_array,lon_floor_array,dx,dy,delta_x,delta_y,YEAR,MM,Area):

    init_total_NDVI = load_NDVI_data(YYYY=YEAR,MM=MM)
    interpolated_monthly_data = np.zeros((len(GeoLAT),len(GeoLON)),dtype=np.float128)
    for ix in range(len(GeoLAT)):
        Cxcyc,Cxcyf,Cxfyc,Cxfyf = get_Concentration4Interpolation(init_mapdata=init_total_NDVI[:,:],lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,
                                                                    lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,ix=ix)
        interpolated_monthly_data[ix,:] = Bilinearinterpolate_GC_concentraion(Cxfyf=Cxfyf,Cxfyc=Cxfyc,Cxcyf=Cxcyf,Cxcyc=Cxcyc,delta_x=delta_x,delta_y=delta_y,dx=dx[ix,:],dy=dy[ix,:])
    save_NDVI_interpolated_mapdata(interpolated_monthly_data,Area, YEAR, MM)    
    return
