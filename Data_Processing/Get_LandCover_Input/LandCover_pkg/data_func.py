import numpy as np
from LandCover_pkg.iostream import *
from TrainingData_pkg.interpolation import *

def get_LandCover_indices():
    Global_GeoLAT, Global_GeoLON = load_Global_GeoLatLon()
    Global_GeoLAT_Map, Global_GeoLON_Map = load_Global_GeoLatLon_Map()
    GC_LAT, GC_LON = load_Coarse_LatLon()
    lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy = get_BilinearInterpolate_Index(fine_Lat=Global_GeoLAT,fine_Lon=Global_GeoLON,fine_Lat_map=Global_GeoLAT_Map,
                                                                                                                                            fine_Lon_map=Global_GeoLON_Map,coarse_Lat=GC_LAT,coarse_Lon=GC_LON)
    save_LandCover_interpolate_indices(outdir=LandCover_outdir,lat_nearest_array=lat_nearest_array,lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,
                                   lon_nearest_array=lon_nearest_array,lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,dx=dx,dy=dy)
    return

def get_Concentration4Interpolation(init_mapdata,lat_ceil_array,lat_floor_array,lon_ceil_array,lon_floor_array,ix):
    Cxfyf = init_mapdata[lat_floor_array[ix]-1,lon_floor_array[:]-1]
    Cxfyc = init_mapdata[lat_floor_array[ix]-1,lon_ceil_array[:]-1]
    Cxcyf = init_mapdata[lat_ceil_array[ix]-1,lon_floor_array[:]-1]
    Cxcyc = init_mapdata[lat_ceil_array[ix]-1,lon_ceil_array[:]-1]
    return Cxcyc,Cxcyf,Cxfyc,Cxfyf

def interpolate_LandCover_Input(GC_LAT, GC_LON, GeoLAT,GeoLON,lat_ceil_array,lat_floor_array,lon_ceil_array,lon_floor_array,dx,dy,YEAR,index,nametag,Area):
    init_mapdata_years = np.zeros((len(GC_LAT),len(GC_LON)),dtype=np.float64)

    init_data = load_LandCover_init_mapdata(YYYY=YEAR)
    init_mapdata_years = get_LandCover_Percentage_Variables(data=init_data)[:,:,index]
    # init_mapdata_years = np.flip(init_mapdata_years,axis=0)
    interpolated_monthly_data = np.zeros((len(GeoLAT),len(GeoLON)),dtype=np.float128)
    for ix in range(len(GeoLAT)):
        Cxcyc,Cxcyf,Cxfyc,Cxfyf = get_Concentration4Interpolation(init_mapdata=init_mapdata_years, lat_ceil_array=lat_ceil_array, lat_floor_array=lat_floor_array,
                                                                    lon_ceil_array=lon_ceil_array, lon_floor_array=lon_floor_array, ix=ix)
        interpolated_monthly_data[ix,:] = Bilinearinterpolate_GC_concentraion(Cxfyf=Cxfyf,Cxfyc=Cxfyc,Cxcyf=Cxcyf,Cxcyc=Cxcyc,delta_x=delta_x,delta_y=delta_y,dx=dx[ix,:],dy=dy[ix,:])
    
    save_LandCover_mapdata(interpolated_monthly_data,nametag,YEAR,Area)
    return
