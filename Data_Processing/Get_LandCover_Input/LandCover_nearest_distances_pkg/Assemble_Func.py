import numpy as np
from LandCover_nearest_distances_pkg.data_func import get_nearest_landcoverpixels_distance_for_each_pixel
from LandCover_nearest_distances_pkg.iostream import *
from LandCover_pkg.iostream import load_LandCover_mapdata, save_LandCover_mapdata

def add_buffer(buffer, YEAR):
    for enrty in ['forests','shrublands','croplands','Urban_Builtup_Lands','Water_Bodies']:
        nearest_distances_map = load_landcover_nearest_pixels(nametag=enrty,YEAR=YEAR,Area='Global')
        nearest_distances_map = np.where(nearest_distances_map>buffer,0,1)
        save_buffer_map(mapdata=nearest_distances_map,nametag=enrty,buffer=buffer,YEAR=YEAR,Area='Global')
    return


def derive_aggregated_LandTypeVariablesMap(YEAR):
    forests = ['Deciduous-Broadleaf-Forests', 'Evergreen-Broadleaf-Forests', 'Evergreen-Needleleaf-Forests', 'Mixed-Forests']
    shrublands = ['Closed-Shrublands', 'Open-Shrublands', 'Woody-Shrublands']
    croplands = ['Cropland-Natural-Vegetation-Mosaics', 'Croplands']
    Urban_Builtup_Lands = ['Urban-Builtup-Lands']
    Water_Bodies = ['Water-Bodies']
    for index, land_type in enumerate(forests):
        print('Processing the land type: {}'.format(land_type))
        if index == 0:
            forests_density_map = load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            forests_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
        else:
            forests_density_map += load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            temp_forests_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
            forests_nearest_distances_map = np.minimum(forests_nearest_distances_map,temp_forests_nearest_distances_map)
    for index, land_type in enumerate(shrublands):
        if index == 0:
            shrublands_density_map = load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            shrublands_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
        else:
            shrublands_density_map += load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            temp_shrublands_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
            shrublands_nearest_distances_map = np.minimum(shrublands_nearest_distances_map,temp_shrublands_nearest_distances_map)
    for index, land_type in enumerate(croplands):
        if index == 0:
            croplands_density_map = load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            croplands_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
        else:
            croplands_density_map += load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            temp_croplands_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
            croplands_nearest_distances_map = np.minimum(croplands_nearest_distances_map,temp_croplands_nearest_distances_map)
    for index, land_type in enumerate(Urban_Builtup_Lands):
        if index == 0:
            Urban_Builtup_Lands_density_map = load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            Urban_Builtup_Lands_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
        else:
            Urban_Builtup_Lands_density_map += load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            temp_Urban_Builtup_Lands_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
            Urban_Builtup_Lands_nearest_distances_map = np.minimum(Urban_Builtup_Lands_nearest_distances_map,temp_Urban_Builtup_Lands_nearest_distances_map)
    for index, land_type in enumerate(Water_Bodies):
        if index == 0:
            Water_Bodies_density_map = load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            Water_Bodies_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
        else:
            Water_Bodies_density_map += load_LandCover_mapdata(nametag=land_type,YEAR=YEAR,Area='Global')
            temp_Water_Bodies_nearest_distances_map = load_landcover_nearest_pixels(nametag=land_type+'_NearestDistances',YEAR=YEAR,Area='Global')
            Water_Bodies_nearest_distances_map = np.minimum(Water_Bodies_nearest_distances_map,temp_Water_Bodies_nearest_distances_map)
            

    print('Processing the year: {}'.format(YEAR))
    save_LandCover_mapdata(mapdata=forests_density_map,nametag='forests',YEAR=YEAR,Area='Global')
    save_landcover_nearest_pixels(mapdata=forests_nearest_distances_map,nametag='forests',YEAR=YEAR,Area='Global')
    save_LandCover_mapdata(mapdata=shrublands_density_map,nametag='shrublands',YEAR=YEAR,Area='Global')
    save_landcover_nearest_pixels(mapdata=shrublands_nearest_distances_map,nametag='shrublands',YEAR=YEAR,Area='Global')
    save_LandCover_mapdata(mapdata=croplands_density_map,nametag='croplands',YEAR=YEAR,Area='Global')
    save_landcover_nearest_pixels(mapdata=croplands_nearest_distances_map,nametag='croplands',YEAR=YEAR,Area='Global')
    save_LandCover_mapdata(mapdata=Urban_Builtup_Lands_density_map,nametag='Urban_Builtup_Lands',YEAR=YEAR,Area='Global')
    save_landcover_nearest_pixels(mapdata=Urban_Builtup_Lands_nearest_distances_map,nametag='Urban_Builtup_Lands',YEAR=YEAR,Area='Global')
    save_LandCover_mapdata(mapdata=Water_Bodies_density_map,nametag='Water_Bodies',YEAR=YEAR,Area='Global')
    save_landcover_nearest_pixels(mapdata=Water_Bodies_nearest_distances_map,nametag='Water_Bodies',YEAR=YEAR,Area='Global')
    print('Finish the aggregated land type variables map for year {}!'.format(YEAR))
    return

def Calculate_save_LandCover_NearestDistances_forEachPixel(init_nametags,Area, YEAR):
    for inametag in init_nametags:
        print('Nametag: {}, YEAR: {}'.format(inametag,YEAR))
        mapdata = load_LandCover_mapdata(nametag=inametag,YEAR=YEAR,Area=Area)
        # Use the 2D map loader
        SATLAT, SATLON = load_Global_GeoLatLon_Map()
        LandCover_index = np.where(mapdata == 100)
        # Correct indexing to get 1D arrays of coordinates at specific indices
        LandCover_Lat = SATLAT[LandCover_index]
        LandCover_Lon = SATLON[LandCover_index]
        get_nearest_landcoverpixels_distance_for_each_pixel(landcover_lats=LandCover_Lat, landcover_lons=LandCover_Lon,
                                                            init_nametag=inametag, YYYY=YEAR, Area=Area)
    return