import numpy as np
import netCDF4 as nc
import os
import mat73 as mat
from data_func.utils import Resampled_Training_BLISCO_data_outdir
from map_uncertainty_func.utils import Estimation_outdir


def save_absoulute_uncertainty_map(absolute_uncertainty_map,species,version,special_name,YYYY,MM,obs_version,nearby_sites_number):
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Absolute_Uncertainty_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir + 'RawObs-{}_pixels_nearby_{}_sites_absolute_uncertainty_map_{}{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM,special_name)
    print(f'Saving absolute uncertainty map to {outfile}')
    np.save(outfile, absolute_uncertainty_map)
    return

def load_absolute_uncertainty_map(species,version,special_name,YYYY,MM,obs_version,nearby_sites_number):
    indir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Absolute_Uncertainty_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    infile = indir + 'RawObs-{}_pixels_nearby_{}_sites_absolute_uncertainty_map_{}{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM,special_name)
    absolute_uncertainty_map = np.load(infile)
    return absolute_uncertainty_map

def load_estimation_map_data(YYYY:str, MM:str,SPECIES:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    base = '{}_{}_{}{}{}'.format(SPECIES,version,YYYY,MM,special_name)
    landfill = indir + base + '_landfill.nc'
    original = indir + base + '.nc'
    infile = landfill if os.path.exists(landfill) else original
    print(f'load_estimation_map_data: reading {infile}')
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables[SPECIES][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon

def save_rRMSE_map(rRMSE_uncertainty_map,species,version,YYYY,MM,obs_version,nearby_sites_number,special_name=''):
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/rRMSE_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir + 'RawObs-{}_pixels_nearby_{}_sites_rRMSE_uncertainty_map_{}{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM,special_name)
    np.save(outfile, rRMSE_uncertainty_map)
    return
def load_rRMSE_map(species,version,YYYY,MM,obs_version,nearby_sites_number,special_name=''):
    indir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/rRMSE_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    infile = indir + 'RawObs-{}_pixels_nearby_{}_sites_rRMSE_uncertainty_map_{}{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM,special_name)
    rRMSE_uncertainty_map = np.load(infile)
    return rRMSE_uncertainty_map

def load_bins_LOWESS_values(species,version,special_name,nearby_sites_number,nchannel,startyear,endyear):
    """Load LOWESS curve values written by U4.

    Keyed by ``nchannel`` and the ``startyear-endyear`` period so the 26ch
    (2019-2023) and 27ch (2005-2018) relationships don't clobber each other.

    Prefer a ``.npz`` (portable, no pickle) if present; fall back to the
    legacy ``.npy`` (pickled dict) only when needed.
    """
    indir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Mahalanobis_distance_LOWESS_values/'.format(species,version)
    base = '{}_nearby_sites-{}_{}-{}_{}Channel{}_Mahalanobis_distance_LOWESS_values'.format(
        species,nearby_sites_number,startyear,endyear,nchannel,special_name)
    npz_path = indir + base + '.npz'
    npy_path = indir + base + '.npy'
    if os.path.exists(npz_path):
        with np.load(npz_path) as d:
            Mahalanobis_distance_bin_centers = d['Mahalanobis_distance_bin_centers']
            WINTER_LOWESS_values = d['WINTER_LOWESS_values']
            SPRING_LOWESS_values = d['SPRING_LOWESS_values']
            SUMMER_LOWESS_values = d['SUMMER_LOWESS_values']
            AUTUMN_LOWESS_values = d['AUTUMN_LOWESS_values']
            ALL_LOWESS_values = d['ALL_LOWESS_values']
    else:
        bins_LOWESS_values = np.load(npy_path,allow_pickle=True).item()
        Mahalanobis_distance_bin_centers = bins_LOWESS_values['Mahalanobis_distance_bin_centers']
        WINTER_LOWESS_values = bins_LOWESS_values['WINTER_LOWESS_values']
        SPRING_LOWESS_values = bins_LOWESS_values['SPRING_LOWESS_values']
        SUMMER_LOWESS_values = bins_LOWESS_values['SUMMER_LOWESS_values']
        AUTUMN_LOWESS_values = bins_LOWESS_values['AUTUMN_LOWESS_values']
        ALL_LOWESS_values = bins_LOWESS_values['ALL_LOWESS_values']

    return Mahalanobis_distance_bin_centers,WINTER_LOWESS_values,SPRING_LOWESS_values,SUMMER_LOWESS_values,AUTUMN_LOWESS_values,ALL_LOWESS_values

def save_mahalanobis_distance_map(mahalanobis_distance_map,species,version,YYYY,MM,obs_version,nearby_sites_number):
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Mahalanobis_distance_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir + 'RawObs-{}_pixels_nearby_{}_sites_mahalanobis_distance_map_{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM)
    np.save(outfile, mahalanobis_distance_map)
    return

def load_mahalanobis_distance_map(species,version,YYYY,MM,obs_version,nearby_sites_number):
    indir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Mahalanobis_distance_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    infile = indir + 'RawObs-{}_pixels_nearby_{}_sites_mahalanobis_distance_map_{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM)
    mahalanobis_distance_map = np.load(infile)
    return mahalanobis_distance_map

def get_nearby_sites_index_map_path(species,version,YYYY,MM,obs_version,nearby_sites_number):
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Nearby_sites_indices/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir + 'RawObs-{}_pixels_nearby_{}_sites_index_map_{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM)
    return outfile

def save_pixel_nearby_sites_index_map(nearby_sites_training_data_indices,species,version,YYYY,MM,obs_version,nearby_sites_number):
    outfile = get_nearby_sites_index_map_path(species,version,YYYY,MM,obs_version,nearby_sites_number)
    np.save(outfile, nearby_sites_training_data_indices)
    return

def load_pixels_nearest_sites_indices_map(species,version,YYYY,MM,obs_version,nearby_sites_number):
    indir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Nearby_sites_indices/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    infile = indir + 'RawObs-{}_pixels_nearby_{}_sites_index_map_{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM)
    nearby_sites_training_data_indices = np.load(infile, mmap_mode='r')
    return nearby_sites_training_data_indices

def save_local_reference_map(local_reference_for_channels_map,species,version,YYYY,MM,obs_version,nearby_sites_number):
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Local_reference_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir + 'RawObs-{}_pixels_nearby_{}_sites_local_reference_map_{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM)
    np.save(outfile, local_reference_for_channels_map)
    return

def load_local_reference_map(species,version,YYYY,MM,obs_version,nearby_sites_number):
    indir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/Local_reference_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    infile = indir + 'RawObs-{}_pixels_nearby_{}_sites_local_reference_map_{}{}.npy'.format(obs_version,nearby_sites_number,YYYY,MM)
    local_reference_for_channels_map = np.load(infile,allow_pickle=True).item()
    return local_reference_for_channels_map

def load_mapdata(infile):
    mapdata = np.load(infile)
    if len(np.where(np.isnan(mapdata)==True)):
        mapdata[np.where(np.isnan(mapdata)==True)] = np.mean(mapdata[np.where(np.isnan(mapdata)==False)])
    return mapdata

def get_landtype(extent)->np.array:
    Mask_indir = '/path/to/supportData/Global_Masks/'

    Africa_data = nc.Dataset(Mask_indir+'Africa.nc')
    Asia_data        = nc.Dataset(Mask_indir+'Asia.nc')
    Europe_data        = nc.Dataset(Mask_indir+'Europe.nc')
    North_America_data        = nc.Dataset(Mask_indir+'North_America.nc')
    Oceania_data        = nc.Dataset(Mask_indir+'Oceania_Australia.nc')
    South_America_data        = nc.Dataset(Mask_indir+'South_America.nc')
    
    Africa_mask = np.array(Africa_data['continent_mask'][:])
    Asia_mask        = np.array(Asia_data['continent_mask'][:])
    Europe_mask        = np.array(Europe_data['continent_mask'][:])
    North_America_mask        = np.array(North_America_data['continent_mask'][:])
    Oceania_Australia_mask        = np.array(Oceania_data['continent_mask'][:])
    South_America_mask        = np.array(South_America_data['continent_mask'][:])
    landtype = Africa_mask + Asia_mask + Europe_mask + North_America_mask + Oceania_Australia_mask + South_America_mask
    lat_index,lon_index = get_GL_extent_index(extent=extent)
    
    output = np.zeros((len(lat_index),len(lon_index)), dtype=int)

    for ix in range(len(lat_index)):
        output[ix,:] = landtype[lat_index[ix],lon_index]
    return output

def get_GL_extent_index(extent)->np.array:
    '''
    :param extent:
        The range of the input. [Bottom_Lat, Up_Lat, Left_Lon, Right_Lon]
    :return:
        lat_index, lon_index
    '''
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')
    lat_index = np.where((SATLAT >= extent[0])&(SATLAT<=extent[1]))
    lon_index = np.where((SATLON >= extent[2])&(SATLON<=extent[3]))
    lat_index = np.squeeze(np.array(lat_index))
    lon_index = np.squeeze(np.array(lon_index))
    return lat_index,lon_index