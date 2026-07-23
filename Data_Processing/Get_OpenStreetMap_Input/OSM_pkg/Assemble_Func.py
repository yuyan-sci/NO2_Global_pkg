from OSM_pkg.data_func import get_lat_lon_grids,get_nearest_RoadDensitypixels_distance_for_each_pixel
from OSM_pkg.gpdDate2Raster_func import get_transform, calculate_lengths_in_grid
from OSM_pkg.iostream import *

def get_log(YEAR):
    
    # motorway_density_map = load_global_road_map(YEAR=YEAR,entry='motorway')
    # motorway_density_log_map = np.log(motorway_density_map+1)
    # save_global_road_map(YEAR=YEAR,entry='motorway_log',RoadDensityMap=motorway_density_log_map)
    
    # trunk_density_map = load_global_road_map(YEAR=YEAR,entry='trunk')
    # trunk_density_log_map = np.log(trunk_density_map+1)
    # save_global_road_map(YEAR=YEAR,entry='trunk_log',RoadDensityMap=trunk_density_log_map)
    
    # primary_density_map = load_global_road_map(YEAR=YEAR,entry='primary')
    # primary_density_log_map = np.log(primary_density_map+1)
    # save_global_road_map(YEAR=YEAR,entry='primary_log',RoadDensityMap=primary_density_log_map)
    
    # secondary_density_map = load_global_road_map(YEAR=YEAR,entry='secondary')
    # secondary_density_log_map = np.log(secondary_density_map+1)
    # save_global_road_map(YEAR=YEAR,entry='secondary_log',RoadDensityMap=secondary_density_log_map)
    
    # tertiary_density_map = load_global_road_map(YEAR=YEAR,entry='tertiary')
    # tertiary_density_log_map = np.log(tertiary_density_map+1)
    # save_global_road_map(YEAR=YEAR,entry='tertiary_log',RoadDensityMap=tertiary_density_log_map)
    
    # unclassified_density_map = load_global_road_map(YEAR=YEAR,entry='unclassified')
    # unclassified_density_log_map = np.log(unclassified_density_map+1)
    # save_global_road_map(YEAR=YEAR,entry='unclassified_log',RoadDensityMap=unclassified_density_log_map)
    
    # residential_density_map = load_global_road_map(YEAR=YEAR,entry='residential')
    # residential_density_log_map = np.log(residential_density_map+1)
    # save_global_road_map(YEAR=YEAR,entry='residential_log',RoadDensityMap=residential_density_log_map)
    
    major_roads_density_map = load_global_road_map(YEAR=YEAR,entry='major_roads')
    major_roads_density_log_map = np.log(major_roads_density_map+1)
    save_global_road_map(YEAR=YEAR,entry='major_roads_log',RoadDensityMap=major_roads_density_log_map)
    
    major_roads_new_density_map = load_global_road_map(YEAR=YEAR,entry='major_roads_new')
    major_roads_new_density_log_map = np.log(major_roads_new_density_map+1)
    save_global_road_map(YEAR='2025',entry='major_roads_new_log',RoadDensityMap=major_roads_new_density_log_map)
    
    minor_roads_density_map = load_global_road_map(YEAR='2025',entry='minor_roads')
    minor_roads_density_log_map = np.log(minor_roads_density_map+1)
    save_global_road_map(YEAR='2025',entry='minor_roads_log',RoadDensityMap=minor_roads_density_log_map)
    
    minor_roads_new_density_map = load_global_road_map(YEAR='2025',entry='minor_roads_new')
    minor_roads_new_density_log_map = np.log(minor_roads_new_density_map+1)
    save_global_road_map(YEAR='2025',entry='minor_roads_new_log',RoadDensityMap=minor_roads_new_density_log_map)
    
    print('Finish the log road density map for year {}!'.format(YEAR))
    return

def add_buffer(buffer):
    for entry in ['major_roads', 'minor_roads', 'major_roads_new', 'minor_roads_new']:
        RoadDensity_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=entry,YEAR='2025',Area='Global')
        RoadDensity_buffer_map = np.where(RoadDensity_nearest_distances_map>buffer,0,1)
        save_global_buffer_map(RoadDensityMap=RoadDensity_buffer_map,nametag=entry,buffer=buffer,YEAR='2025',Area='Global')
    return

def derive_aggregated_new_RoadVariablesMap(YEAR):
    major_roads = ['primary', 'secondary','trunk']
    minor_roads = ['tertiary', 'residential']
    for index, road_type in enumerate(major_roads):
        print('Processing the road type: {}'.format(road_type))
        if index == 0:
            major_roads_density_map = load_global_road_map(YEAR='2025',entry=road_type)
            major_roads_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=road_type+'_NearestDistances',YEAR='2025',Area='Global')
        else:
            major_roads_density_map += load_global_road_map(YEAR='2025',entry=road_type)
            temp_road_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=road_type+'_NearestDistances',YEAR='2025',Area='Global')
            major_roads_nearest_distances_map = np.minimum(major_roads_nearest_distances_map,temp_road_nearest_distances_map)
    for index, road_type in enumerate(minor_roads):
        if index == 0:
            minor_roads_density_map = load_global_road_map(YEAR='2025',entry=road_type)
            minor_roads_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=road_type+'_NearestDistances',YEAR='2025',Area='Global')
        else:
            minor_roads_density_map += load_global_road_map(YEAR='2025',entry=road_type)
            temp_road_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=road_type+'_NearestDistances',YEAR='2025',Area='Global')
            minor_roads_nearest_distances_map = np.minimum(minor_roads_nearest_distances_map,temp_road_nearest_distances_map)
            

    print('Processing the year: {}'.format(YEAR))
    save_global_road_map(YEAR=YEAR,entry='major_roads_new',RoadDensityMap=major_roads_density_map)
    save_RoadDensity_nearest_pixels(mapdata=major_roads_nearest_distances_map,nametag='major_roads_new',YEAR=YEAR,Area='Global')
    save_global_road_map(YEAR=YEAR,entry='minor_roads_new',RoadDensityMap=minor_roads_density_map)
    save_RoadDensity_nearest_pixels(mapdata=minor_roads_nearest_distances_map,nametag='minor_roads_new',YEAR=YEAR,Area='Global')
    print('Finish the aggregated road variables map for year {}!'.format(YEAR))
    return

def derive_aggregated_RoadVariablesMap(YEAR):
    major_roads = ['motorway', 'trunk', 'primary', 'secondary']
    minor_roads = ['tertiary', 'unclassified', 'residential']
    for index, road_type in enumerate(major_roads):
        print('Processing the road type: {}'.format(road_type))
        if index == 0:
            major_roads_density_map = load_global_road_map(YEAR='2025',entry=road_type)
            major_roads_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=road_type+'_NearestDistances',YEAR='2025',Area='Global')
        else:
            major_roads_density_map += load_global_road_map(YEAR='2025',entry=road_type)
            temp_road_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=road_type+'_NearestDistances',YEAR='2025',Area='Global')
            major_roads_nearest_distances_map = np.minimum(major_roads_nearest_distances_map,temp_road_nearest_distances_map)
    for index, road_type in enumerate(minor_roads):
        if index == 0:
            minor_roads_density_map = load_global_road_map(YEAR='2025',entry=road_type)
            minor_roads_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=road_type+'_NearestDistances',YEAR='2025',Area='Global')
        else:
            minor_roads_density_map += load_global_road_map(YEAR='2025',entry=road_type)
            temp_road_nearest_distances_map = load_RoadDensity_nearest_pixels(nametag=road_type+'_NearestDistances',YEAR='2025',Area='Global')
            minor_roads_nearest_distances_map = np.minimum(minor_roads_nearest_distances_map,temp_road_nearest_distances_map)
            

    print('Processing the year: {}'.format(YEAR))
    save_global_road_map(YEAR=YEAR,entry='major_roads',RoadDensityMap=major_roads_density_map)
    save_RoadDensity_nearest_pixels(mapdata=major_roads_nearest_distances_map,nametag='major_roads',YEAR=YEAR,Area='Global')
    save_global_road_map(YEAR=YEAR,entry='minor_roads',RoadDensityMap=minor_roads_density_map)
    save_RoadDensity_nearest_pixels(mapdata=minor_roads_nearest_distances_map,nametag='minor_roads',YEAR=YEAR,Area='Global')
    print('Finish the aggregated road variables map for year {}!'.format(YEAR))
    return

def derive_NearestPixels_RoadDensity(YEAR,entry_name):
    RoadDensityMap = load_global_road_map(YEAR,entry_name)
    SATLAT,SATLON = load_global_GeoLatLon()
    RoadDensityMap_Pixels_index = np.where(RoadDensityMap>0)
    RoadDensityMap_Pixels_Lat = SATLAT[RoadDensityMap_Pixels_index[0]]
    RoadDensityMap_Pixels_Lon = SATLON[RoadDensityMap_Pixels_index[1]]
    get_nearest_RoadDensitypixels_distance_for_each_pixel(RoadDensityMap=RoadDensityMap,
                                                        RoadDensity_Pixels_Lat=RoadDensityMap_Pixels_Lat,
                                                        RoadDensity_Pixels_Lon=RoadDensityMap_Pixels_Lon,
                                                        init_nametag=entry_name,YEAR=YEAR,Area='Global')
    return 


def derive_raster_RoadDensityMap_forRegions(Continent, Region_List, OSM_YEAR, entries_list, head_index='type', resolution=0.01, use_osm_file=False):
    for iregion in Region_List:
        init_data = load_road_shpfile(YEAR=OSM_YEAR, Continent=Continent, Region=iregion)
        for ientry in entries_list:
            print('Region: {}, Entry: {}'.format(iregion,ientry))
            road_data = get_certain_entries(init_data=init_data, head_index=head_index, entry=ientry)
            if road_data is None:
                print(f"No {ientry} entry found for {iregion} in {Continent}. Skipping.")
                continue
            
            lat_values, lon_values = get_lat_lon_grids(gdf=road_data, resolution=resolution)
            print('lat_values: {}, \n lon_values: {}'.format(len(lat_values),len(lon_values)))
            transform = get_transform(lat_min=np.min(lat_values),lon_min=np.min(lon_values),resolution=resolution)
            RoadDensityMap = calculate_lengths_in_grid(gdf=road_data,resolution=resolution,n_lat=len(lat_values),n_lon=len(lon_values),
                                                        transform=transform)
            save_Continent_RoadMapDensity(YEAR=OSM_YEAR,entry=ientry,Continent=Continent,Region=iregion,RoadDensityMap=RoadDensityMap,lat_values=lat_values,lon_values=lon_values)
    return


def derive_raster_RoadDensityMap_forGlobal(Continent_List, Region_List, entries_list, YEAR, use_osm_file=False):
    for ientry in entries_list:
        # Load the consistent Global lat/lon grids
        SATLAT, SATLON = load_global_GeoLatLon()
        Global_map = np.zeros((len(SATLAT), len(SATLON)), dtype=np.float64)    
        
        for iContinent in Continent_List:
            for iRegion in Region_List[iContinent]:
                print('Entry: {}, Continent: {}, Region: {}'.format(ientry, iContinent, iRegion))
                # Skip regions with missing files
                result = load_Continent_RoadMapDensity(YEAR=YEAR, entry=ientry, Continent=iContinent, Region=iRegion)
                if result is None:
                    print(f"Skipping region {iRegion} due to missing file")
                    continue
                RoadDensityMap, lat_values, lon_values = result
                
                # Find the closest indices in the Global grid for the regional grid start points
                lat_start_idx = np.abs(SATLAT - lat_values[0]).argmin()
                lon_start_idx = np.abs(SATLON - lon_values[0]).argmin()
            
                # Ensure we don't exceed array bounds
                lat_end_idx = min(lat_start_idx + len(lat_values), len(SATLAT))
                lon_end_idx = min(lon_start_idx + len(lon_values), len(SATLON))
                map_lat_size = lat_end_idx - lat_start_idx
                map_lon_size = lon_end_idx - lon_start_idx
                
                # Add the regional map to the Global map, ensuring we only use the overlapping parts
                Global_map[lat_start_idx:lat_end_idx, lon_start_idx:lon_end_idx] += RoadDensityMap[:map_lat_size, :map_lon_size]
            
        save_global_road_map(YEAR=YEAR, entry=ientry, RoadDensityMap=Global_map)
    return