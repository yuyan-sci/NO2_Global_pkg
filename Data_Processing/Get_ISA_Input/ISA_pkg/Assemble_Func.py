from ISA_pkg.data_func import *
from ISA_pkg.utils import ISA_input_outdir
from ISA_pkg.iostream import *

def calculate_save_ISA_interpolate_indices(YYYY):
    get_ISA_indices(YYYY=YYYY)
    return

def interpolate_ISA_mapdata(YYYY):
    GC_LAT, GC_LON, delta_x, delta_y = load_Coarse_LatLon(YYYY=YYYY)
    GEOLAT, GEOLON = load_global_GeoLatLon()
    lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy = load_ISA_interpolate_indices(indir=ISA_input_outdir)
    interpolate_ISA_map(GeoLAT=GEOLAT,GeoLON=GEOLON,lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,
                                dx=dx,dy=dy,delta_x=delta_x,delta_y=delta_y,YEAR=YYYY)
    return
