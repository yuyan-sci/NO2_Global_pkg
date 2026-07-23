import netCDF4 as nc
import geopandas as gpd
import os
import numpy as np
# from OSM_pkg.utils import *

def load_global_buffer_map(nametag,buffer,YEAR,Area):
    RoadDensity_nearest_distance_buffer_dir ='/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/OpenStreetMap_RoadDensity_Buffer_forEachPixels_input/'+ '{}/'.format(nametag) 
    infile = RoadDensity_nearest_distance_buffer_dir + 'OpenStreetMap-{}-Buffer-{}-forEachPixel_001x001_{}_{}.npy'.format(nametag,buffer,Area,YEAR)
    if not os.path.isfile(infile):
        print('The file {} does not exist!'.format(infile))
        return None
    
    RoadDensity_nearest_distance_buffer_map = np.load(infile)
    return RoadDensity_nearest_distance_buffer_map
   
def save_global_buffer_map(RoadDensityMap,nametag,buffer,YEAR,Area):
    outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/OpenStreetMap_RoadDensity_Buffer_forEachPixels_input/'+ '{}/'.format(nametag)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'OpenStreetMap-{}-Buffer-{}-forEachPixel_001x001_{}_{}.npy'.format(nametag,buffer,Area,YEAR)
    print(outfile)
    np.save(outfile, RoadDensityMap)
    return

def save_RoadDensity_nearest_pixels(mapdata,nametag,YEAR,Area):
    RoadDensity_nearest_distance_outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/OpenStreetMap_RoadDensity_NearestDistances_forEachPixels_input/'
    outdir = RoadDensity_nearest_distance_outdir + '{}/'.format(nametag)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'OpenStreetMap-{}-NearestDistanceforEachPixel_001x001_{}_{}.npy'.format(nametag,Area,YEAR)
    print(outfile)
    np.save(outfile,mapdata)
    return

def load_RoadDensity_nearest_pixels(nametag,YEAR,Area):
    RoadDensity_nearest_distance_outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/OpenStreetMap_RoadDensity_NearestDistances_forEachPixels_input/'
    indir = RoadDensity_nearest_distance_outdir + '{}/'.format(nametag)
    infile = indir + 'OpenStreetMap-{}-NearestDistanceforEachPixel_001x001_{}_{}.npy'.format(nametag,Area,YEAR)
    if not os.path.isfile(infile):
        print('The file {} does not exist!'.format(infile))
        return None
    
    RoadDensity_nearest_distance_map = np.load(infile)
    return RoadDensity_nearest_distance_map

def load_global_GeoLatLon():
    indir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/'
    lat_infile = indir + 'tSATLAT_global.npy'
    lon_infile = indir + 'tSATLON_global.npy'
    global_GeoLAT = np.load(lat_infile)
    global_GeoLON = np.load(lon_infile)
    return global_GeoLAT, global_GeoLON

def load_global_GeoLatLon_Map():
    indir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/'
    lat_infile = indir + 'tSATLAT_global_MAP.npy'
    lon_infile = indir + 'tSATLON_global_MAP.npy'
    global_GeoLAT_MAP = np.load(lat_infile)
    global_GeoLON_MAP = np.load(lon_infile)
    return global_GeoLAT_MAP, global_GeoLON_MAP

def load_road_shpfile(YEAR,Continent,Region):
    indir = '/path/to/OpenStreetMap/geofabrik_shapefiles/{}/{}/{}-latest-free.shp/'.format(YEAR,Continent,Region)
    infile = indir + 'gis_osm_roads_free_1.shp'
    init_data = gpd.read_file(infile)
    return init_data

def get_certain_entries(init_data,head_index,entry):
    road_data    = init_data[init_data[head_index]==entry]
    if road_data.empty:
        print(f"Warning: The entry '{entry}' does not exist. Skipping this entry.")
        return None
    return road_data

def load_Continent_RoadMapDensity(YEAR,entry,Continent,Region):
    indir = '/path/to/OpenStreetMap/raster_ncfiles_global/{}/{}/{}/'.format(YEAR,entry, Continent)
    infile = indir + 'OpenStreetMap-{}-RoadDensityMap_{}_{}.nc'.format(entry,Region,YEAR)
    if not os.path.isfile(infile):
        print('The file {} does not exist!'.format(infile))
        return None
    else:
        MapData = nc.Dataset(infile)
        lat = MapData.variables['lat'][:]
        lon = MapData.variables['lon'][:]
        RoadDensityMap = MapData.variables[entry][:]
        RoadDensityMap = np.array(RoadDensityMap)
        return RoadDensityMap, lat, lon

def save_Continent_RoadMapDensity(YEAR,entry,Continent,Region,RoadDensityMap,lat_values,lon_values):
    outdir = '/path/to/OpenStreetMap/raster_ncfiles_global/{}/{}/{}/'.format(YEAR,entry, Continent)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'OpenStreetMap-{}-RoadDensityMap_{}_{}.nc'.format(entry,Region,YEAR)
    lat_size = len(lat_values)
    lon_size = len(lon_values)
    lat_delta = 0.01
    lon_delta = 0.01

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Open Street Map - {} Road Denstity (km/grid) over {} Area.'.format(entry, Region)
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    RoadDensity = MapData.createVariable(entry,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = lat_values
    longitudes[:] = lon_values
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    RoadDensity.units = 'km/grid'
    RoadDensity.long_name = 'Open Street Map {} Road Density[km/grid]'.format(entry)
    RoadDensity[:] = RoadDensityMap
    return

def save_global_road_map(YEAR,entry,RoadDensityMap):
    outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/OpenStreetMap_RoadDensity_input/' + '{}/'.format(YEAR)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'OpenStreetMap-Global-{}-RoadDensityMap_{}.npy'.format(entry,YEAR)
    print(outfile)
    np.save(outfile, RoadDensityMap)
    return

def load_global_road_map(YEAR,entry):
    indir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/OpenStreetMap_RoadDensity_input/' + '{}/'.format(YEAR)
    infile = indir + 'OpenStreetMap-Global-{}-RoadDensityMap_{}.npy'.format(entry,YEAR)
    RoadDensityMap = np.load(infile)
    return RoadDensityMap