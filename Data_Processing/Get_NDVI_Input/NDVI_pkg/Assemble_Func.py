from NDVI_pkg.data_func import *


def calculate_save_NDVI_interpolate_indices(YYYY,MM):
    get_NDVI_indices(YYYY=YYYY,MM=MM)
    return

def interpolate_NDVI_mapdata(YYYY,MM, Area):
    GC_LAT, GC_LON, delta_x, delta_y = load_Coarse_LatLon(YYYY=YYYY,MM=MM)
    GEOLAT, GEOLON = load_global_GeoLatLon()
    index_dir = f'/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/NDVI_input/{YYYY}-{MM}/'
    lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy = load_NDVI_interpolate_indices(indir=index_dir)
    interpolate_NDVI_map(GeoLAT=GEOLAT,GeoLON=GEOLON,lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,
                                dx=dx,dy=dy,delta_x=delta_x,delta_y=delta_y,YEAR=YYYY,MM=MM,Area=Area)
    return
