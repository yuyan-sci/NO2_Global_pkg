import numpy as np
import os
import netCDF4 as nc

from data_func.iostream import load_TrainingVariables,save_nearby_sites_index_reference_data,load_nearby_sites_index_reference_data,load_RawObs_training_data, load_month_based_BLISCO_data_recording,save_training_and_BLISCO_data,load_training_and_BLISCO_data,save_mahalanobis_distance_data,save_BLISCO_data, load_BLISCO_data
from data_func.calculation import calculate_covariance_matrix, invert_matrix, calculate_mahalanobis_distance,get_mean_std_dic_for_channels
from data_func.utils import neighbors_haversine_indices


def derive_corresponding_training_data_BLISCO_data(species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height,
                                        buffer_radius_list, BLCO_kfold, BLCO_seeds_number,
                                        channel_lists, desire_year_list):
    '''
    Desired_year_list: must be string, list of years to process, e.g., ['2000', '2001', ..., '2023']
    '''
    
    MONTH_lists = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    # Load Raw Observation Training Data (center pixel) and training meta
    RawObs_training_data = load_RawObs_training_data(channel_lists)
    width_nc, height_nc, sites_number, start_YYYY_training, _ = load_TrainingVariables(channel_lists)
    total_samples = RawObs_training_data[channel_lists[0]].shape[0]
    months_available = total_samples // sites_number  # total months in training cube

    resampled_RawObs_training_site_input_data = {}
    resampled_RawObs_testing_site_input_data = {}
    for channel in channel_lists:
        resampled_RawObs_training_site_input_data[channel] = {}
        resampled_RawObs_testing_site_input_data[channel] = {}
        for buffer_radius in buffer_radius_list:
            resampled_RawObs_training_site_input_data[channel][buffer_radius] = {}
            resampled_RawObs_testing_site_input_data[channel][buffer_radius] = {}
            for iyear in desire_year_list:
                resampled_RawObs_training_site_input_data[channel][buffer_radius][iyear] = {}
                resampled_RawObs_testing_site_input_data[channel][buffer_radius][iyear] = {}
                for imonth, MONTH in enumerate(MONTH_lists):
                    resampled_RawObs_training_site_input_data[channel][buffer_radius][iyear][MONTH] = {}
                    resampled_RawObs_testing_site_input_data[channel][buffer_radius][iyear][MONTH] = {}
                    for ifold in range(BLCO_kfold):
                        resampled_RawObs_training_site_input_data[channel][buffer_radius][iyear][MONTH][ifold] = np.full(sites_number,np.nan,dtype=np.float32)
                        resampled_RawObs_testing_site_input_data[channel][buffer_radius][iyear][MONTH][ifold] = np.full(sites_number,np.nan,dtype=np.float32)

    # Load BLISCO Data Recording
    BLISCO_obs_data_recording, BLISCO_final_data_recording, BLISCO_test_sites_index_recording, BLISCO_train_sites_index_recording, BLISCO_training_obs_data_recording = load_month_based_BLISCO_data_recording(
        species, version, typeName, startyear, endyear,
        nchannel, special_name, width, height,
        buffer_radius_list, BLCO_kfold, BLCO_seeds_number)

    for buffer_radius in buffer_radius_list:
        for iyear in desire_year_list:
            for imonth, MONTH in enumerate(MONTH_lists):
                print(f"Resampling Processing {species} - {MONTH} {iyear} - Buffer Radius: {buffer_radius}km")
                # Here you would add the code to process each month/year combination
                # For example, you might want to extract the relevant data from
                # RawObs_training_data and obs_data_recording for this month/year
                # and then perform your analysis or calculations.
                
                for ifold in range(BLCO_kfold):
                    valid_obs_index = np.where(~np.isnan(BLISCO_obs_data_recording[buffer_radius][str(iyear)][MONTH][ifold*sites_number:(ifold+1)*sites_number]))[0]
                    valid_train_obs_index = np.where(~np.isnan(BLISCO_training_obs_data_recording[buffer_radius][str(iyear)][MONTH][ifold*sites_number:(ifold+1)*sites_number]))[0]
                    ## The sites index recording just record the initial indices before removing NaN values based on observation data.
                    ## So here we need to get the non-NaN indices, but not the real indices used in that month.
                    test_site_temp_index = BLISCO_test_sites_index_recording[buffer_radius][str(iyear)][MONTH][valid_obs_index+ifold*sites_number]
                    train_site_temp_index = BLISCO_train_sites_index_recording[buffer_radius][str(iyear)][MONTH][valid_train_obs_index+ifold*sites_number]
                    for channel in channel_lists:
                        # Compute global month index relative to training start year
                        month_idx_global = (int(iyear) - int(start_YYYY_training)) * 12 + imonth
                        if month_idx_global < 0 or month_idx_global >= months_available:
                            # Skip if requested year/month is outside training coverage
                            continue
                        base_offset = month_idx_global * sites_number
                        resampled_RawObs_testing_site_input_data[channel][buffer_radius][str(iyear)][MONTH][ifold][valid_obs_index] = RawObs_training_data[channel][base_offset + test_site_temp_index.astype(int)]
                        resampled_RawObs_training_site_input_data[channel][buffer_radius][str(iyear)][MONTH][ifold][valid_train_obs_index] = RawObs_training_data[channel][base_offset + train_site_temp_index.astype(int)]
    save_training_and_BLISCO_data(resampled_RawObs_testing_site_input_data,resampled_RawObs_training_site_input_data,
                                    BLISCO_obs_data_recording, BLISCO_final_data_recording,
                                    species, version, typeName, startyear, endyear,
                                          nchannel, special_name, width, height)



def derive_local_reference_for_channels(
    channel_lists, species, version, typeName, startyear, endyear,
    nchannel, special_name, width, height, buffer_radius_list, BLCO_kfold,
    desire_year_list, mode='mean', nearby_sites_number=5,
):
    """
    Accelerated version:
      - Vectorized distance & neighbor search
      - No per-site loops
      - One neighbor computation reused across all channels
    """
    assert mode in ('mean', 'median')
    reducer = np.nanmean if mode == 'mean' else np.nanmedian
    MONTH_lists = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    data = load_training_and_BLISCO_data(
        species, version, typeName, startyear, endyear,
        nchannel, special_name, width, height
    )
    resampled_RawObs_testing_site_input_data  = data['resampled_RawObs_testing_site_input_data']
    resampled_RawObs_training_site_input_data = data['resampled_RawObs_training_site_input_data']
    BLISCO_obs_data_recording = data['BLISCO_obs_data_recording']
    BLISCO_final_data_recording = data['BLISCO_final_data_recording']

    valid_BLISCO_obs_data_recording = {}
    valid_BLISCO_final_data_recording = {}

                    
    
    # Using the lat/lon from training files, not directly from BLISCO results.
    test_lat_rec  = resampled_RawObs_testing_site_input_data['lat'] 
    test_lon_rec  = resampled_RawObs_testing_site_input_data['lon']
    
    train_lat_rec = resampled_RawObs_training_site_input_data['lat']
    train_lon_rec = resampled_RawObs_training_site_input_data['lon']
    
    sites_number = load_TrainingVariables(channel_lists)[2]
    
    nearby_sites_index_dict = {}
    nearby_sites_channel_reference_dict = {}

    for buffer_radius in buffer_radius_list:
        nearby_sites_index_dict[buffer_radius] = {}
        nearby_sites_channel_reference_dict[buffer_radius] = {}
        valid_BLISCO_obs_data_recording[buffer_radius] = {}
        valid_BLISCO_final_data_recording[buffer_radius] = {}
        for year in desire_year_list:
            nearby_sites_index_dict[buffer_radius][year] = {}
            nearby_sites_channel_reference_dict[buffer_radius][year] = {}
            valid_BLISCO_obs_data_recording[buffer_radius][year] = {}
            valid_BLISCO_final_data_recording[buffer_radius][year] = {}
            for month in MONTH_lists:
                nearby_sites_index_dict[buffer_radius][year][month] = {}
                nearby_sites_channel_reference_dict[buffer_radius][year][month] = {}
                valid_BLISCO_obs_data_recording[buffer_radius][year][month] = np.full(sites_number,np.nan,dtype=np.float32)
                valid_BLISCO_final_data_recording[buffer_radius][year][month] = np.full(sites_number,np.nan,dtype=np.float32)
                print(f"Deriving local reference - Buffer Radius: {buffer_radius}km, Year: {year}, Month: {month}")
                for ifold in range(BLCO_kfold):
                    # Coordinates
                    test_lat  = np.asarray(test_lat_rec[buffer_radius][year][month][ifold])
                    test_lon  = np.asarray(test_lon_rec[buffer_radius][year][month][ifold])
                    test_valid_index = np.where(~np.isnan(test_lat))[0]
                    train_lat = np.asarray(train_lat_rec[buffer_radius][year][month][ifold])
                    train_lon = np.asarray(train_lon_rec[buffer_radius][year][month][ifold])
                    train_valid_index = np.where(~np.isnan(train_lat))[0]
                    print(f"  Fold {ifold+1}/{BLCO_kfold}: {len(test_lat[test_valid_index])} test sites, {len(train_lat[train_valid_index])} training sites")
                    # Vectorized neighbor indices
                    if len(test_lat[test_valid_index]) > 0 and len(train_lat[train_valid_index]) >= nearby_sites_number:
                        idx = neighbors_haversine_indices(
                            train_lat[train_valid_index], train_lon[train_valid_index], test_lat[test_valid_index], test_lon[test_valid_index], nearby_sites_number
                        )
                        nearby_sites_index_dict[buffer_radius][year][month][ifold] = np.full((sites_number, nearby_sites_number), np.nan, dtype=np.int64)
                        nearby_sites_index_dict[buffer_radius][year][month][ifold][test_valid_index,:] = idx
                        valid_index = np.where(~np.isnan(BLISCO_obs_data_recording[buffer_radius][year][month][ifold*sites_number:(ifold+1)*sites_number]))[0]
                        valid_BLISCO_obs_data_recording[buffer_radius][year][month][valid_index] = BLISCO_obs_data_recording[buffer_radius][year][month][ifold*sites_number:(ifold+1)*sites_number][valid_index]
                        valid_BLISCO_final_data_recording[buffer_radius][year][month][valid_index] = BLISCO_final_data_recording[buffer_radius][year][month][ifold*sites_number:(ifold+1)*sites_number][valid_index]
                
                        # Compute local reference for each channel
                        nearby_sites_channel_reference_dict[buffer_radius][year][month][ifold] = {}
                        for channel in channel_lists:
                            train_vals = np.asarray(
                                resampled_RawObs_training_site_input_data[channel][buffer_radius][year][month][ifold][train_valid_index]
                            )
                            gathered = train_vals[idx] if idx.size else np.empty((sites_number, 0))
                            ref = reducer(gathered, axis=1) if gathered.size else np.full(sites_number, np.nan)
                            nearby_sites_channel_reference_dict[buffer_radius][year][month][ifold][channel] = np.full(sites_number, np.nan, dtype=np.float32)
                            nearby_sites_channel_reference_dict[buffer_radius][year][month][ifold][channel][test_valid_index] = ref
                    else:
                        nearby_sites_index_dict[buffer_radius][year][month][ifold] = np.empty((0, nearby_sites_number), dtype=np.int64)
                        nearby_sites_channel_reference_dict[buffer_radius][year][month][ifold] = {}
                        for channel in channel_lists:
                            nearby_sites_channel_reference_dict[buffer_radius][year][month][ifold][channel] = np.array([], dtype=np.float32)
    save_BLISCO_data(valid_BLISCO_obs_data_recording, valid_BLISCO_final_data_recording,
                    species, version, typeName, startyear, endyear,
                          nchannel, special_name, width, height)
    save_nearby_sites_index_reference_data(
        nearby_sites_index_dict, nearby_sites_channel_reference_dict,
        species, version, typeName, startyear, endyear,
        nchannel, special_name, width, height, nearby_sites_number
    )
    return nearby_sites_index_dict, nearby_sites_channel_reference_dict

def Assemble_Mahalanobis_distance_data(species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height,
                                        buffer_radius_list, BLCO_kfold,
                                        channel_lists, desire_year_list, nearby_sites_number=5,
                                        ):

    MONTH_lists = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    ## Load resampled training and BLISCO data
    nearby_data = load_nearby_sites_index_reference_data(
        species, version, typeName, startyear, endyear,
        nchannel, special_name, width, height, nearby_sites_number
    )
    sites_number = load_TrainingVariables(channel_lists)[2]
    nearby_sites_index_dict = nearby_data['nearby_sites_index_dict']
    nearby_sites_channel_reference_dict = nearby_data['nearby_sites_channel_reference_dict']
    
    data = load_training_and_BLISCO_data(species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height)
    RawObs_training_data = load_RawObs_training_data(channel_lists)
    ## Convert RawObs_training_data into shape of (N_training_sites, N_channels) from dictionary to numpy array
    total_channels_training_site_data_list = []
    for channel in channel_lists:
        total_channels_training_site_data_list.append(RawObs_training_data[channel])
    total_channels_training_site_data_list = np.stack(total_channels_training_site_data_list, axis=1)
    covariance_matrix = calculate_covariance_matrix(total_channels_training_site_data_list)
    inverted_covariance_matrix = invert_matrix(covariance_matrix)
    resampled_RawObs_testing_site_input_data = data['resampled_RawObs_testing_site_input_data']
    resampled_RawObs_training_site_input_data = data['resampled_RawObs_training_site_input_data']
    
    
    
    AllMonth_AllYear_Martix_Mahalanobis_distance_recording = {}
    EachMonth_AllYear_Martix_Mahalanobis_distance_recording = {}
    EachMonth_EachYear_Martix_Mahalanobis_distance_recording = {}
    
    
    for buffer_radius in buffer_radius_list:
        AllMonth_AllYear_Martix_Mahalanobis_distance_recording[buffer_radius] = {}
        EachMonth_AllYear_Martix_Mahalanobis_distance_recording[buffer_radius] = {}
        EachMonth_EachYear_Martix_Mahalanobis_distance_recording[buffer_radius] = {}
        for iyear in desire_year_list:
            AllMonth_AllYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear] = {}
            EachMonth_AllYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear] = {}
            EachMonth_EachYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear] = {}
            for imonth, MONTH in enumerate(MONTH_lists):
                AllMonth_AllYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear][MONTH] = np.full(sites_number,np.nan,dtype=np.float32)
                EachMonth_AllYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear][MONTH] = np.full(sites_number,np.nan,dtype=np.float32)
                EachMonth_EachYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear][MONTH] = np.full(sites_number,np.nan,dtype=np.float32)
    ## Calculate the Covariance Matrix and its Inverse Matrix from Training Data, and the mean vector for each buffer radius and each fold
    
    ## All Month All Year Martix Mahalanobis Distance Calculation
    
    for buffer_radius in buffer_radius_list:
        print(f"Processing Mahalanobis Distance Calculation for Buffer Radius: {buffer_radius}km")
        ## First Get the training data across all months and all years in shape of (N_training_sites, N_channels)
        for ifold in range(BLCO_kfold):
            print(f"{species} Processing Fold: {ifold + 1}/{BLCO_kfold} for All Month All Year")
            total_channels_training_site_data_list = []
            """
            for channel in channel_lists:
                temp_channel_training_site_data_list = []
                for iyear in desire_year_list:
                    for imonth, MONTH in enumerate(MONTH_lists):
                        temp_channel_training_site_data_list.append(resampled_RawObs_training_site_input_data[channel][buffer_radius][iyear][MONTH][ifold])
                total_channels_training_site_data_list.append(np.concatenate(temp_channel_training_site_data_list, axis=0))
            
            ### Global Covariance Matrix, but use localized mean vector
            total_channels_training_site_data_list = np.stack(total_channels_training_site_data_list, axis=1)  # Shape: (N_training_sites, N_channels)
            covariance_matrix = calculate_covariance_matrix(total_channels_training_site_data_list)
            inverted_covariance_matrix = invert_matrix(covariance_matrix)
            """
            
            for iyear in desire_year_list:
                for imonth, MONTH in enumerate(MONTH_lists):
                    temp_channel_testing_site_data_list = []
                    temp_mean_vector_channel_list = []
                    for channel in channel_lists:
                        temp_channel_testing_site_data_list.append(resampled_RawObs_testing_site_input_data[channel][buffer_radius][iyear][MONTH][ifold])
                        temp_mean_vector_channel_list.append(nearby_sites_channel_reference_dict[buffer_radius][iyear][MONTH][ifold][channel])
                    total_channels_testing_site_data = np.stack(temp_channel_testing_site_data_list, axis=1)  # Shape: (N_testing_sites, N_channels)
                    temp_mean_vector_channel_list = np.stack(temp_mean_vector_channel_list, axis=1)  # Shape: (N_testing_sites, N_channels)
                    valid_index = np.where(~np.isnan(temp_mean_vector_channel_list[:,0]))[0]
                    temp_mahalanobis_distance = calculate_mahalanobis_distance(total_channels_testing_site_data[valid_index], temp_mean_vector_channel_list[valid_index], inverted_covariance_matrix)
                    
                    AllMonth_AllYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear][MONTH][valid_index] = temp_mahalanobis_distance
        ## Each Month All Year Martix Mahalanobis Distance Calculation
        for imonth, MONTH in enumerate(MONTH_lists):
            print(f"Processing Month: {MONTH} for Each Month All Year")
            ## First Get the training data across all years in shape of (N_training_sites, N_channels)
            for ifold in range(BLCO_kfold):
                total_channels_training_site_data_list = []
                
                #for channel in channel_lists:
                #    temp_channel_training_site_data_list = []
                #    for iyear in desire_year_list:
                #        temp_channel_training_site_data_list.append(resampled_RawObs_training_site_input_data[channel][buffer_radius][iyear][MONTH][ifold])
                    
                #    total_channels_training_site_data_list.append(np.concatenate(temp_channel_training_site_data_list, axis=0))
                #total_channels_training_site_data_list = np.stack(total_channels_training_site_data_list, axis=1)  # Shape: (N_training_sites, N_channels)
                #covariance_matrix = calculate_covariance_matrix(total_channels_training_site_data_list)
                #inverted_covariance_matrix = invert_matrix(covariance_matrix)
                
                for iyear in desire_year_list:
                    temp_channel_testing_site_data_list = []
                    temp_mean_vector_channel_list = []
                    for channel in channel_lists:
                        temp_channel_testing_site_data_list.append(resampled_RawObs_testing_site_input_data[channel][buffer_radius][iyear][MONTH][ifold])
                        temp_mean_vector_channel_list.append(nearby_sites_channel_reference_dict[buffer_radius][iyear][MONTH][ifold][channel])
                    total_channels_testing_site_data = np.stack(temp_channel_testing_site_data_list, axis=1)  # Shape: (N_testing_sites, N_channels)
                    temp_mean_vector_channel_list = np.stack(temp_mean_vector_channel_list, axis=1)  # Shape: (N_testing_sites, N_channels)
                    valid_index = np.where(~np.isnan(temp_mean_vector_channel_list[:,0]))[0]
                    temp_mahalanobis_distance = calculate_mahalanobis_distance(total_channels_testing_site_data[valid_index], temp_mean_vector_channel_list[valid_index], inverted_covariance_matrix)
                    EachMonth_AllYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear][MONTH][valid_index] = temp_mahalanobis_distance
        
        ## Each Month Each Year Martix Mahalanobis Distance Calculation
        for iyear in desire_year_list:
            for imonth, MONTH in enumerate(MONTH_lists):
                print(f"Processing {MONTH} {iyear} for Each Month Each Year")
                ## First Get the training data for this month and year in shape of (N_training_sites, N_channels)
                for ifold in range(BLCO_kfold):
                    total_channels_training_site_data_list = []
                    #for channel in channel_lists:
                    #    total_channels_training_site_data_list.append(resampled_RawObs_training_site_input_data[channel][buffer_radius][iyear][MONTH][ifold])
                    #total_channels_training_site_data_list = np.stack(total_channels_training_site_data_list, axis=1)  # Shape: (N_training_sites, N_channels)
                    #covariance_matrix = calculate_covariance_matrix(total_channels_training_site_data_list)
                    #inverted_covariance_matrix = invert_matrix(covariance_matrix)
                    temp_mean_vector_channel_list = []
                    temp_channel_testing_site_data_list = []
                    for channel in channel_lists:
                        temp_mean_vector_channel_list.append(nearby_sites_channel_reference_dict[buffer_radius][iyear][MONTH][ifold][channel])
                        temp_channel_testing_site_data_list.append(resampled_RawObs_testing_site_input_data[channel][buffer_radius][iyear][MONTH][ifold])
                    total_channels_testing_site_data = np.stack(temp_channel_testing_site_data_list, axis=1)  # Shape: (N_testing_sites, N_channels)
                    temp_mean_vector_channel_list = np.stack(temp_mean_vector_channel_list, axis=1)  # Shape: (N_testing_sites, N_channels)
                    valid_index = np.where(~np.isnan(temp_mean_vector_channel_list[:,0]))[0]
                    temp_mahalanobis_distance = calculate_mahalanobis_distance(total_channels_testing_site_data[valid_index], temp_mean_vector_channel_list[valid_index], inverted_covariance_matrix)
                    EachMonth_EachYear_Martix_Mahalanobis_distance_recording[buffer_radius][iyear][MONTH][valid_index] = temp_mahalanobis_distance
    save_mahalanobis_distance_data(EachMonth_EachYear_Martix_Mahalanobis_distance_recording, EachMonth_AllYear_Martix_Mahalanobis_distance_recording, AllMonth_AllYear_Martix_Mahalanobis_distance_recording,
                                   species, version, typeName, startyear, endyear,
                                        nchannel, special_name, width, height, nearby_sites_number)
    return