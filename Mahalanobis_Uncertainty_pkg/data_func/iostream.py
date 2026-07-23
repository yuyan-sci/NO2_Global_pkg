import numpy as np
import netCDF4 as nc
import os
from data_func.utils import Resampled_Training_BLISCO_data_outdir 

def load_TrainingVariables(nametags):
    # Training .nc holding the model channels. Env-overridable so a v7.2 run reads
    # the v7 29-channel file (which has 'GeoNO2-v2') without disturbing the v6 default.
    training_infile = os.environ.get(
        'MAHAL_TRAINING_NC',
        '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v6/'
        'v6_filtered_TrainingData_30channels_5x5_200501-202312.nc')
    data = nc.Dataset(training_infile,'r')
    width = np.array(data.variables['width'][:])[0]
    height = np.array(data.variables['height'][:])[0]
    total_number = np.array(data.variables['Total_number'])[0]
    sitesnumber = np.array(data.variables['sites_number'])[0]
    start_YYYY  = np.array(data.variables['start_YYYY'])[0]
    TrainingDatasets = np.zeros((total_number,len(nametags),width,height),dtype=np.float64)
    for i in range(len(nametags)):
        TrainingDatasets[:,i,:,:] = np.array(data.variables[nametags[i]][:,:,:])
    return width, height, sitesnumber,start_YYYY, TrainingDatasets

def load_RawObs_training_data(channel_lists):
    """
    Load training data consistent with the NO2 pipeline (netCDF), and return
    center-pixel values for each channel (same behavior as the previous NPZ approach).

    Args:
        channel_lists: list of channel names to load
    Returns:
        dict[channel] -> 1D array of length Total_number (center pixel values)
    """
    # Reuse the shared Training_pkg loader to avoid duplicating file logic.
    width_nc, height_nc, sitesnumber, start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_lists)

    center_w = int((width_nc - 1) / 2)
    center_h = int((height_nc - 1) / 2)

    total_data = {}
    for i, ch in enumerate(channel_lists):
        # TrainingDatasets shape: (Total_number, C, W, H)
        total_data[ch] = TrainingDatasets[:, i, center_w, center_h]

    return total_data

def load_month_based_BLISCO_data_recording(species, version, typeName, beginyear, endyear,
                                           nchannel, special_name, width, height,
                                           buffer_radius_list, BLCO_kfold, BLCO_seeds_number):

    indir = '/path/to/NO2_DL_global/Training_Evaluation_Estimation/' + '{}/{}/Results/results-SelfIsolated_BLCO_DataRecording/'.format(species, version)
    BLISCO_obs_data_recording = {}
    BLISCO_final_data_recording = {}
    BLISCO_test_sites_index_recording = {}
    BLISCO_train_sites_index_recording = {}
    BLISCO_training_obs_data_recording = {}
    for buffer_radius in buffer_radius_list:
        BLISCO_obs_data_recording[buffer_radius] = {}
        BLISCO_final_data_recording[buffer_radius] = {}
        BLISCO_test_sites_index_recording[buffer_radius] = {}
        BLISCO_train_sites_index_recording[buffer_radius] = {}
        BLISCO_training_obs_data_recording[buffer_radius] = {}
        obs_data_infile =  indir + '{}-{}-Obs-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
        final_data_infile = indir + '{}-{}-Final-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
        test_sites_index_recording_infile =  indir + '{}-{}-test_sites_index_recording-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
        train_sites_index_recording_infile = indir + '{}-{}-train_sites_index_recording-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
        training_obs_data_infile = indir + '{}-{}-training_obs_data-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
        obs_data = np.load(obs_data_infile,allow_pickle=True).item()
        final_data = np.load(final_data_infile,allow_pickle=True).item()
        test_sites_index_recording = np.load(test_sites_index_recording_infile, allow_pickle=True).item()
        train_sites_index_recording = np.load(train_sites_index_recording_infile, allow_pickle=True).item()
        training_obs_data = np.load(training_obs_data_infile, allow_pickle=True).item()
        BLISCO_obs_data_recording[buffer_radius] = obs_data
        BLISCO_final_data_recording[buffer_radius] = final_data
        BLISCO_test_sites_index_recording[buffer_radius] = test_sites_index_recording
        BLISCO_train_sites_index_recording[buffer_radius] = train_sites_index_recording
        BLISCO_training_obs_data_recording[buffer_radius] = training_obs_data
    return BLISCO_obs_data_recording, BLISCO_final_data_recording, BLISCO_test_sites_index_recording, BLISCO_train_sites_index_recording, BLISCO_training_obs_data_recording


def save_training_and_BLISCO_data(resampled_RawObs_testing_site_input_data,resampled_RawObs_training_site_input_data,
                                  BLISCO_obs_data_recording, BLISCO_final_data_recording,
                                  species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height):
    """
    Save training and BLISCO data to a .npy file.

    Parameters:
    data (numpy.ndarray): The training data to be saved.
    outfile (str): The path to the output .npy file.
    """
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/data/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # Save the data
    resampled_training_data_outfile = outdir + '{}-{}-Resampled_RawObs_Training_and_BLISCOData_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, startyear, endyear, width, height, nchannel, special_name)
    np.save(resampled_training_data_outfile, {
        'resampled_RawObs_testing_site_input_data': resampled_RawObs_testing_site_input_data,
        'resampled_RawObs_training_site_input_data': resampled_RawObs_training_site_input_data,
        'BLISCO_obs_data_recording': BLISCO_obs_data_recording,
        'BLISCO_final_data_recording': BLISCO_final_data_recording
    })
    
def load_training_and_BLISCO_data(species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height):
    """
    Load training and BLISCO data from a .npy file.

    Parameters:
    infile (str): The path to the input .npy file.

    Returns:
    dict: The loaded training and BLISCO data.
    """
    infile = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/data/'.format(species, version) + '{}-{}-Resampled_RawObs_Training_and_BLISCOData_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, startyear, endyear, width, height, nchannel, special_name)
    data = np.load(infile, allow_pickle=True).item()
    return data

def save_BLISCO_data(BLISCO_obs_data_recording, BLISCO_final_data_recording,
                                  species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height):
    """
    Save BLISCO data to a .npy file.
    Parameters:
    data (numpy.ndarray): The BLISCO data to be saved.
    outfile (str): The path to the output .npy file.
    """
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/data/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    # Save the data
    BLISCO_data_outfile = outdir + '{}-{}-BLISCOData_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, startyear,
                                                                                            endyear, width, height, nchannel, special_name)
    np.save(BLISCO_data_outfile, {
        'BLISCO_obs_data_recording': BLISCO_obs_data_recording,
        'BLISCO_final_data_recording': BLISCO_final_data_recording
    })  
    return

def load_BLISCO_data(species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height):
    """
    Load BLISCO data from a .npy file.
    Parameters:
    infile (str): The path to the input .npy file.
    Returns:
    dict: The loaded BLISCO data.
    """
    infile = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/data/'.format(species, version) + '{}-{}-BLISCOData_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, startyear,
                                                                                            endyear, width, height, nchannel, special_name)
    data = np.load(infile, allow_pickle=True).item()
    return data

def save_mahalanobis_distance_data(EachMonth_EachYear_Martix_Mahalanobis_distance_recording, EachMonth_AllYear_Martix_Mahalanobis_distance_recording, AllMonth_AllYear_Martix_Mahalanobis_distance_recording,
                                   species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height, nearby_sites_number):
    """
    Save Mahalanobis distance recording data to a .npy file.

    Parameters:
    mahalanobis_distance_recording (dict): The Mahalanobis distance recording data to be saved.
    outfile (str): The path to the output .npy file.
    """
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/data/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir + '{}-{}-{}_sites_nearby-MahalanobisDistance_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, nearby_sites_number, startyear, endyear, width, height, nchannel, special_name)
    np.save(outfile, {
        'EachMonth_EachYear_Martix_Mahalanobis_distance_recording': EachMonth_EachYear_Martix_Mahalanobis_distance_recording,
        'EachMonth_AllYear_Martix_Mahalanobis_distance_recording': EachMonth_AllYear_Martix_Mahalanobis_distance_recording,
        'AllMonth_AllYear_Martix_Mahalanobis_distance_recording': AllMonth_AllYear_Martix_Mahalanobis_distance_recording
    })

    return

def load_mahalanobis_distance_data(species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height, nearby_sites_number):
    """
    Load Mahalanobis distance recording data from a .npy file.

    Parameters:
    infile (str): The path to the input .npy file.

    Returns:
    dict: The loaded Mahalanobis distance recording data.
    """
    infile = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/data/'.format(species, version) + '{}-{}-{}_sites_nearby-MahalanobisDistance_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, nearby_sites_number, startyear, endyear, width, height, nchannel, special_name)
    data = np.load(infile, allow_pickle=True).item()
    return data

def save_nearby_sites_index_reference_data(nearby_sites_index_dict, nearby_sites_channel_reference_dict,
                                           species, version, typeName, startyear, endyear,
                                           nchannel, special_name, width, height, nearby_sites_number):
    """
    Save nearby sites index and channel reference data to a .npy file.
    Parameters:
    data (numpy.ndarray): The nearby sites index and channel reference data to be saved.
    outfile (str): The path to the output .npy file.
    """
    outdir = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/data/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir + '{}-{}-{}_sites_nearby-NearbySitesIndexAndChannelReference_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, nearby_sites_number, startyear, endyear, width, height, nchannel, special_name)
    np.save(outfile, {
        'nearby_sites_index_dict': nearby_sites_index_dict,
        'nearby_sites_channel_reference_dict': nearby_sites_channel_reference_dict
    })
    return

def load_nearby_sites_index_reference_data(species, version, typeName, startyear, endyear,
                                             nchannel, special_name, width, height, nearby_sites_number):
     """
     Load nearby sites index and channel reference data from a .npy file.
     Parameters:
     infile (str): The path to the input .npy file.
     Returns:
     dict: The loaded nearby sites index and channel reference data.
     """
     infile = Resampled_Training_BLISCO_data_outdir + '{}/{}/Mahalanobis_Uncertainty/data/'.format(species, version) + '{}-{}-{}_sites_nearby-NearbySitesIndexAndChannelReference_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, nearby_sites_number, startyear, endyear, width, height, nchannel, special_name)
     data = np.load(infile, allow_pickle=True).item()
     return data
 
def load_RawObservation(speices):
    # Ground-monitor obs used to seed nearby-site references. Env-overridable so a
    # v7.2 run points at the v7 obs without disturbing the v5/v6 default.
    infile = os.environ.get(
        'MAHAL_RAWOBS_NC',
        '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v6/'
        'NO2_observation_corrected_v6_filtered_v5.13.nc')
    dataset = nc.Dataset(infile)
    obsdata = dataset.variables[speices][:]
    obs_lat = dataset.variables['latitude'][:]
    obs_lon = dataset.variables['longitude'][:]
    return obsdata, obs_lat, obs_lon

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
