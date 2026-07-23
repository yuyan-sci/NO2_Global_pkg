from LandCover_pkg.data_func import *

def calculate_save_LandCover_interpolate_indices():
    get_LandCover_indices()
    return

def interpolate_LandCover_mapdata(YYYY, nametags, Area, index):
    GC_LAT, GC_LON = load_Coarse_LatLon()
    GEOLAT, GEOLON = load_Global_GeoLatLon()
    lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy = load_LandCover_interpolate_indices(indir=LandCover_outdir)
    
    for i in range(len(index)):
        nametag = nametags[i]
        print('{}'.format(nametag))
        interpolate_LandCover_Input(GC_LAT=GC_LAT,GC_LON=GC_LON,GeoLAT=GEOLAT,GeoLON=GEOLON,lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,
                                       dx=dx,dy=dy,YEAR=YYYY,index=index[i],nametag=nametag,Area=Area)
    return
