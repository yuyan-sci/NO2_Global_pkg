import geopandas as gpd
import numpy as np
from OSM_pkg.iostream import *
import time
from scipy.interpolate import NearestNDInterpolator



def calculate_distance_forArray(site_lat:np.float32,site_lon:np.float32,
                                SATLAT_MAP:np.array,SATLON_MAP:np.array,r=6371.01):
    site_pos1 = site_lat * np.pi / 180.0
    site_pos2 = site_lon * np.pi / 180.0
    other_sites_pos1_array = SATLAT_MAP * np.pi / 180.0
    other_sites_pos2_array = SATLON_MAP * np.pi / 180.0
    dist_map = r * np.arccos(np.sin(site_pos1)*np.sin(other_sites_pos1_array)+np.cos(site_pos1)*np.cos(other_sites_pos1_array)*np.cos(site_pos2-other_sites_pos2_array))
    return dist_map

def get_nearest_RoadDensitypixels_distance_for_each_pixel(RoadDensityMap, RoadDensity_Pixels_Lat,RoadDensity_Pixels_Lon,init_nametag,YEAR,Area):
    SATLAT,SATLON = load_global_GeoLatLon()
    Global_GeoLAT_MAP, Global_GeoLON_MAP = load_global_GeoLatLon_Map()
    interp_start = time.time()
    # interp = NearestNDInterpolator(list(zip(sites_lat,sites_lon)),sites_index)
    # nearest_index_map = interp(tSATLAT_map,tSATLON_map)
    interp_lat = NearestNDInterpolator(list(zip(RoadDensity_Pixels_Lat,RoadDensity_Pixels_Lon)),RoadDensity_Pixels_Lat)
    interp_lon = NearestNDInterpolator(list(zip(RoadDensity_Pixels_Lat,RoadDensity_Pixels_Lon)),RoadDensity_Pixels_Lon)
    nearest_lat_map = interp_lat(Global_GeoLAT_MAP,Global_GeoLON_MAP)
    nearest_lon_map = interp_lon(Global_GeoLAT_MAP,Global_GeoLON_MAP)

    interp_end   = time.time()

    interp_total = interp_end - interp_start
    print('Finish the nearest interpolation! Time costs:',interp_total,' seconds')
    nearest_distance_map = np.full(Global_GeoLAT_MAP.shape, np.nan, dtype=np.float64) 

    for ix in range(len(SATLAT)):
        if ix % 100 == 0: 
            print(f'Processing latitude row {ix}/{len(SATLAT)} ({np.round(100*(ix/len(SATLAT)),2)}%)...' )

        start_time = time.time()
        row_distances = calculate_distance_forArray(nearest_lat_map[ix,:],
                                                    nearest_lon_map[ix,:],
                                                    Global_GeoLAT_MAP[ix,:],
                                                    Global_GeoLON_MAP[ix,:])
        nearest_distance_map[ix,:] = row_distances
        end_time = time.time()
        Get_distance_forOneLatitude_time = end_time - start_time

    print("Setting distance to 0 for pixels with initial road density > 0...")
    original_road_pixels = RoadDensityMap > 0
    nearest_distance_map[original_road_pixels] = 0.0
    print(f"Number of pixels set to 0: {np.sum(original_road_pixels)}")

    print("Checking for remaining NaN values...")
    nan_count = np.sum(np.isnan(nearest_distance_map))
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain in the distance map. Consider handling them.")
        # Option: Fill NaNs with a large value
        # nearest_distance_map = np.nan_to_num(nearest_distance_map, nan=9999.0)
        # print("NaN values replaced with 9999.0")


    nametag = init_nametag + '_NearestDistances'
    save_RoadDensity_nearest_pixels(mapdata=nearest_distance_map,nametag=nametag,YEAR=YEAR,Area=Area)
    return 



def get_lat_lon_grids(gdf, resolution):
    print(f"Starting get_lat_lon_grids with {len(gdf)} features, resolution={resolution}")
    
    lat_min, lat_max = gdf.bounds.miny.min(), gdf.bounds.maxy.max()
    lon_min, lon_max = gdf.bounds.minx.min(), gdf.bounds.maxx.max()
    resolution = 0.01  # Set grid resolution in degrees (adjust as needed)

    if np.isnan(lat_min):
        lat_min = -69.995
        lat_max = 69.995
        lon_min = -179.995
        lon_max = 179.995
    else:
        lat_min = max(-69.995,(np.floor(lat_min) + 0.5*resolution))
        lat_max = min(69.995, (np.ceil(lat_max)  - 0.5*resolution))
        lon_min = max(-179.995, (np.floor(lon_min) + 0.5*resolution))
        lon_max = min(179.995, (np.ceil(lon_max)  - 0.5*resolution))
    print(lat_min,lat_max,lon_min,lon_max)
    n_lat = round((lat_max - lat_min) / resolution) +1
    n_lon = round((lon_max - lon_min) / resolution) +1
    print('nlat: {}. nlon: {}'.format(n_lat,n_lon))
    lat_values = np.linspace(lat_min, lat_max, n_lat)
    lon_values = np.linspace(lon_min, lon_max, n_lon)
    
    print(f"Completed get_lat_lon_grids: lat_values={len(lat_values)}, lon_values={len(lon_values)}")
    return lat_values, lon_values