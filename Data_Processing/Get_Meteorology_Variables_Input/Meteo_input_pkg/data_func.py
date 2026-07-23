import numpy as np
import netCDF4 as nc
from TrainingData_pkg.interpolation import *
from Meteo_input_pkg.iostream import * 
from Meteo_input_pkg.utils import *


def get_meteorology_interpolate_indices():
    global_GeoLAT, global_GeoLON = load_global_GeoLatLon()
    global_GeoLAT_Map, global_GeoLON_Map = load_global_GeoLatLon_Map()
    GC_LAT, GC_LON = load_global_GC_LatLon()
    lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy = get_BilinearInterpolate_Index(fine_Lat=global_GeoLAT,fine_Lon=global_GeoLON,fine_Lat_map=global_GeoLAT_Map,
                                                                                                                                            fine_Lon_map=global_GeoLON_Map,coarse_Lat=GC_LAT,coarse_Lon=GC_LON)
    save_meteo_interpolate_indices(outdir=meteorology_mapdata_outdir,lat_nearest_array=lat_nearest_array,lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,
                                   lon_nearest_array=lon_nearest_array,lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,dx=dx,dy=dy)
    return


def get_Concentration4Interpolation(init_mapdata,lat_ceil_array,lat_floor_array,lon_ceil_array,lon_floor_array,ix):
    Cxfyf = init_mapdata[lat_floor_array[ix]-1,lon_floor_array[:]-1]
    Cxfyc = init_mapdata[lat_floor_array[ix]-1,lon_ceil_array[:]-1]
    Cxcyf = init_mapdata[lat_ceil_array[ix]-1,lon_floor_array[:]-1]
    Cxcyc = init_mapdata[lat_ceil_array[ix]-1,lon_ceil_array[:]-1]
    return Cxcyc,Cxcyf,Cxfyc,Cxfyf


def interpolate_meteorology_map(init_mapdata,GeoLAT,GeoLON,lat_ceil_array,lat_floor_array,lon_ceil_array,lon_floor_array,dx,dy,tagname,YEAR,MONTH,Area):
    final_mapdata = np.zeros((len(GeoLAT),len(GeoLON)),dtype=np.float64)
    for ix in range(len(GeoLAT)):
        Cxcyc,Cxcyf,Cxfyc,Cxfyf = get_Concentration4Interpolation(init_mapdata,lat_ceil_array,lat_floor_array,lon_ceil_array,lon_floor_array,ix)
        final_mapdata[ix,:] = Bilinearinterpolate_GC_concentraion(Cxcyc=Cxcyc,Cxcyf=Cxcyf,Cxfyc=Cxfyc,Cxfyf=Cxfyf,delta_x=delta_x,delta_y=delta_y,dx=dx[ix,:],dy=dy[ix,:])
    save_meteo_mapdata(outdir=meteorology_mapdata_outdir,map_data=final_mapdata,tagname=tagname,YYYY=YEAR,MM=MONTH,Area=Area)
    
    return