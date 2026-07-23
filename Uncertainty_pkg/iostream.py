import csv 
import numpy as np
import netCDF4 as nc
import os
from Training_pkg.utils import *
from Estimation_pkg.utils import *
from Evaluation_pkg.utils import *
from Uncertainty_pkg.utils import *

def save_data_for_LOWESS_calculation(total_obs_data,total_final_data,total_nearbysites_distances_data,total_nearest_distances_data,radius,nchannels,width,height):
    outdir = Uncertainty_outdir + '{}/{}/Uncertainty_Results/Data_for_LOWESS_calculation/'.format(species,version) 
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    obs_data_outfile = outdir + 'BLISCO_obs_data_{}_{}_{}-folds_{}-SeedsNumbers_radius-{}km_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}.npy'.format(
                                    version, species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,radius,Max_distances_for_Bins,Number_of_Bins,
                                    nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,
                                    width,height)  
    final_data_outfile = outdir + 'BLISCO_final_data_{}_{}_{}-folds_{}-SeedsNumbers_radius-{}km_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}.npy'.format(
                                    version, species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,radius,Max_distances_for_Bins,Number_of_Bins,
                                    nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,
                                    width,height)
    nearby_distances_outfile = outdir + 'BLISCO_nearby_distances_data_{}_{}_{}-folds_{}-SeedsNumbers_radius-{}km_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}.npy'.format(
                                    version, species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,radius,Max_distances_for_Bins,Number_of_Bins,
                                    nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,
                                   width,height)
    nearest_distances_outfile = outdir + 'BLISCO_nearest_distances_data_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}.npy'.format(
                                    version, species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,
                                    nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,
                                    width,height)
    np.save(obs_data_outfile,total_obs_data)
    np.save(final_data_outfile,total_final_data)
    np.save(nearby_distances_outfile,total_nearbysites_distances_data)
    np.save(nearest_distances_outfile,total_nearest_distances_data)
    return

def load_data_for_LOWESS_calculation(radius,nchannels,width,height):
    indir = Uncertainty_outdir + '{}/{}/Uncertainty_Results/Data_for_LOWESS_calculation/'.format(species,version) 
    obs_data_infile = indir + 'BLISCO_obs_data_{}_{}_{}-folds_{}-SeedsNumbers_radius-{}km_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}.npy'.format(
                                    version, species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,radius,Max_distances_for_Bins,Number_of_Bins,
                                    nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,
                                    width,height)  
    final_data_infile = indir + 'BLISCO_final_data_{}_{}_{}-folds_{}-SeedsNumbers_radius-{}km_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}.npy'.format(
                                    version, species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,radius,Max_distances_for_Bins,Number_of_Bins,
                                    nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,
                                    width,height)
    nearby_distances_infile = indir + 'BLISCO_nearby_distances_data_{}_{}_{}-folds_{}-SeedsNumbers_radius-{}km_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}.npy'.format(
                                    version, species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,radius,Max_distances_for_Bins,Number_of_Bins,
                                    nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,
                                    width,height)
    nearest_distances_infile = indir + 'BLISCO_nearest_distances_data_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}.npy'.format(
                                    version, species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,
                                    nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,
                                    width,height)
    total_obs_data = np.load(obs_data_infile,allow_pickle=True).item()
    total_final_data = np.load(final_data_infile,allow_pickle=True).item()
    total_nearbysites_distances_data = np.load(nearby_distances_infile,allow_pickle=True).item()
    total_nearest_distances_data = np.load(nearest_distances_infile,allow_pickle=True).item()
    return total_obs_data,total_final_data,total_nearbysites_distances_data,total_nearest_distances_data

def save_LOWESS_values_bins(LOWESS_values_dic, rRMSE_dic, bins, nchannels,width,height):
    outdir = Uncertainty_outdir + '{}/{}/Uncertainty_Results/LOWESS_values_bins/'.format(species,version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    LOWESS_values_outfile = outdir + 'BLISCO_LOWESS_values_frac-{}_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}{}.npy'.format(LOWESS_frac,version,species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,width,height,special_name)
    rRMSE_outfile = outdir + 'BLISCO_rRMSE_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}{}.npy'.format(version,species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,width,height,special_name)
    bins_outfile = outdir + 'BLISCO_bins_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}{}.npy'.format(version,species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,width,height,special_name)
    np.save(LOWESS_values_outfile,LOWESS_values_dic)
    np.save(rRMSE_outfile,rRMSE_dic)
    np.save(bins_outfile,bins)
    return 

def load_LOWESS_values_bins(nchannels,width,height):
    indir = Uncertainty_outdir + '{}/{}/Uncertainty_Results/LOWESS_values_bins/'.format(species,version)

    LOWESS_values_infile = indir + 'BLISCO_LOWESS_values_frac-{}_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}{}.npy'.format(LOWESS_frac,version,species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,width,height,special_name)
    rRMSE_infile = indir + 'BLISCO_rRMSE_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}{}.npy'.format(version,species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,width,height,special_name)
    bins_infile = indir + 'BLISCO_bins_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}{}.npy'.format(version,species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannels,width,height,special_name)
    LOWESS_values = np.load(LOWESS_values_infile,allow_pickle=True).item()
    rRMSE         = np.load(rRMSE_infile,allow_pickle=True).item()
    bins_array    = np.load(bins_infile)
    return LOWESS_values,rRMSE,bins_array

def load_BLCO_rRMSE(nchannel,width,height,Number_ClusterSeeds):
    start_year = ''
    end_year   = ''
    Region_list = ['North America'] + BLCO_additional_test_regions + ['None']# None is for the recording convinence.
    rRMSE = np.zeros((len(Region_list),17,len(Uncertainty_Buffer_radii_forUncertainty)))
    rRMSE_std = np.zeros((len(Region_list),17,len(Uncertainty_Buffer_radii_forUncertainty)))
    typeName = '{}-bias'.format(species)
    R2    = np.zeros((len(Region_list),17,len(Uncertainty_Buffer_radii_forUncertainty)))
    R2_std= np.zeros((len(Region_list),17,len(Uncertainty_Buffer_radii_forUncertainty)))
    GeoR2 = np.zeros((len(Region_list),17,len(Uncertainty_Buffer_radii_forUncertainty)))
    for iradius in range(len(Uncertainty_Buffer_radii_forUncertainty)):
        indir = Uncertainty_outdir + '{}/{}/Results/results-SelfIsolated_BLCOCV/statistical_indicators/{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}/'.format(species, version,Uncertainty_Buffer_radii_forUncertainty[iradius],BLCO_kfold,Number_ClusterSeeds,typeName,species,version,nchannel,width,height,special_name)
        infile = indir + 'SelfIsolated_BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(start_year,end_year,Uncertainty_Buffer_radii_forUncertainty[iradius],BLCO_kfold,Number_ClusterSeeds,typeName,species,version,nchannel,width,height,special_name)
        print(infile)
        with open(infile, newline='') as f:
            reader = csv.reader(f)
            count = 0
            Region_index = 0
            for row in reader:
                if row[0] == 'Area: {} ; Time Period: {} - {}'.format(Region_list[Region_index],start_year,end_year):
                        print(row)
                        Region_index += 1
                        count = 0
                if count >= 1: 
                    for i in range(len(row)):
                        if row[i] == '\n NRMSE -  Avg: ':
                            rRMSE[Region_index-1,count-1,iradius] = row[i+1]
                            rRMSE_std[Region_index-1,count-1,iradius] = row[i+7]
                        if row[i] == '\n Test R2 - Avg: ':
                            R2[Region_index-1,count-1,iradius] = row[i+1]
                            R2_std[Region_index-1,count-1,iradius] = row[i+7]
                        if row[i] == '\n Geophysical R2 - Avg: ':
                            GeoR2[Region_index-1,count-1,iradius] = row[i+1]
                count += 1
    return rRMSE,rRMSE_std, R2,R2_std, GeoR2

def load_rRMSE_map_data( YYYY, MM:str, version:str, special_name):
    indir = Uncertainty_outdir + '{}/{}/Uncertainty_Results/rRMSE_Map/{}/'.format(species,version,YYYY)
    infile = indir + 'rRMSE_Map_{}_{}_{}{}{}.nc'.format(species,version,YYYY,MM,special_name)
    MapData = nc.Dataset(infile)
    SPECIES_Map = MapData.variables[species][:]
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = np.array(SPECIES_Map)
    print('Type of SPECIES_MAP: {}'.format(type(SPECIES_Map)))
    return SPECIES_Map, lat, lon

def load_pixels_nearest_sites_distances_map():
    indir =  Uncertainty_outdir + '{}/{}/Results/Pixels2sites_distances/{}/'.format(species,version,YYYY)
    infile = indir + '{}_nearest_site_distances_forEachPixel.nc'.format(species)
    MapData = nc.Dataset(infile)
    Distance_Map = MapData.variables['Distance'][:]
    Distance_Map = np.array(Distance_Map)
    return Distance_Map

def load_pixels_nearby_sites_distances_map(YYYY,MM):
    indir =  Uncertainty_outdir + '{}/{}/Results/Pixels2sites_distances/{}/'.format(species,version,YYYY)
    infile = indir + '{}_nearby_site_distances_forEachPixel_{}-mode_{}Number_{}{}.nc'.format(species,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,YYYY,MM)
    MapData = nc.Dataset(infile)
    Distance_Map = MapData.variables['Distance'][:]
    Distance_Map = np.array(Distance_Map)
    return Distance_Map

def load_absolute_uncertainty_map_data(YYYY:str, MM:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/Uncertainty_Results/Absolute-Uncertainty_Map/{}/'.format(species,version,YYYY)
    infile = indir + 'AbsoluteUncertainty_{}_{}_{}{}{}.nc'.format(species,version,YYYY,MM,special_name)
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables[species][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon

def load_GeoLatLon():
    indir = '/path/to/NO2_DL_global/input_variables/'
    lat_infile = indir + 'tSATLAT_global.npy'
    lon_infile = indir + 'tSATLON_global.npy'
    GeoLAT = np.load(lat_infile)
    GeoLON = np.load(lon_infile)
    return GeoLAT, GeoLON

def load_GeoLatLon_Map():
    indir = '/path/to/NO2_DL_global/input_variables/'
    lat_infile = indir + 'tSATLAT_global_MAP.npy'
    lon_infile = indir + 'tSATLON_global_MAP.npy'
    GeoLAT_MAP = np.load(lat_infile)
    GeoLON_MAP = np.load(lon_infile)
    return GeoLAT_MAP, GeoLON_MAP


def save_nearest_site_distances_forEachPixel(nearest_distance_map,extent_lat,extent_lon):
    outdir = Uncertainty_outdir + '{}/{}/Results/Pixels2sites_distances/{}/'.format(species,version,YYYY)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}_nearest_site_distances_forEachPixel.nc'.format(species)
    
    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Nearset distance for each pixel from {} sites'.format(species)
    MapData.CONTACT = 'Yu Yan <yany1@wustl.edu>'

    lat = MapData.createDimension("lat",len(extent_lat))
    lon = MapData.createDimension("lon",len(extent_lon))
    Distance = MapData.createVariable('Distance','f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = extent_lat
    longitudes[:] = extent_lon
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    Distance.units = 'km'
    Distance[:] = nearest_distance_map
    return

def save_nearby_site_distances_forEachPixel(nearby_distance_map,extent_lat,extent_lon,YYYY,MM):
    outdir = Uncertainty_outdir + '{}/{}/Results/Pixels2sites_distances/{}/'.format(species,version,YYYY)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}_nearby_site_distances_forEachPixel_{}-mode_{}Number_{}{}.nc'.format(species,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,YYYY,MM)
    
    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'Nearset distance for each pixel from {} sites under {} mode with {} Number'.format(species,nearby_sites_distances_mode,number_of_nearby_sites_forAverage)
    MapData.CONTACT = 'Yu Yan <yany1@wustl.edu>'

    lat = MapData.createDimension("lat",len(extent_lat))
    lon = MapData.createDimension("lon",len(extent_lon))
    Distance = MapData.createVariable('Distance','f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = extent_lat
    longitudes[:] = extent_lon
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    Distance.units = 'km'
    Distance[:] = nearby_distance_map
    return

def save_rRMSE_uncertainty_Map(Map_rRMSE:np.array,YYYY, MM:str,):

    outdir = Uncertainty_outdir + '{}/{}/Uncertainty_Results/rRMSE_Map/{}/'.format(species,version,YYYY)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'rRMSE_Map_{}_{}_{}{}{}.nc'.format(species,version,YYYY,MM,special_name)
    lat_size = Map_rRMSE.shape[0]
    lon_size = Map_rRMSE.shape[1]
    lat_delta = (Extent[1]-Extent[0])/lat_size
    lon_delta = (Extent[3]-Extent[2])/lon_size

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'LightGBM Global Monthly {} rRMSE Map.'.format(species)
    MapData.CONTACT = 'Yu Yan <yany1@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}'.format(MM)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    rRMSE = MapData.createVariable(species,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = np.arange(Extent[0],Extent[1],lat_delta)
    longitudes[:] = np.arange(Extent[2],Extent[3],lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    rRMSE.units = 'unitless'
    rRMSE.long_name = 'LightGBM derived Monthly {} rRMSE'.format(species)
    rRMSE[:] = Map_rRMSE
    return

def save_absolute_uncertainty_data(final_data:np.array, YYYY:str, MM:str):
    outdir = Estimation_outdir + '{}/{}/Uncertainty_Results/Absolute-Uncertainty_Map/{}/'.format(species,version,YYYY)
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + 'AbsoluteUncertainty_{}_{}_{}{}{}.nc'.format(species,version,YYYY,MM,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = (Extent[1]-Extent[0])/lat_size
    lon_delta = (Extent[3]-Extent[2])/lon_size

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'LightGBM Global Monthly {} Absolute Uncertainty Estimation.'.format(species)
    MapData.CONTACT = 'Yu Yan <yany1@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}/{}'.format(MM,YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    NO2 = MapData.createVariable(species,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = np.arange(Extent[0],Extent[1],lat_delta)
    longitudes[:] = np.arange(Extent[2],Extent[3],lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    NO2.units = 'ppb'
    NO2.long_name = 'LightGBM derived Monthly {} absolute Uncertainty [ppb]'.format(species)
    NO2[:] = final_data
    return
