import toml
import numpy as np
import time
import math

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.environ.get('NO2_CONFIG_PATH', os.path.join(os.path.dirname(current_dir), 'config.toml'))
cfg = toml.load(config_path)

############################ Spatial Cross-Validation ################################
Spatial_CrossValidation_Switch = cfg['Spatial-CrossValidation']['Spatial_CrossValidation_Switch'] # On/Off for Spatial Crosss Validation
Spatial_CV_LossAccuracy_plot_Switch = cfg['Spatial-CrossValidation']['Spatial_CV_LossAccuracy_plot_Switch']
regression_plot_switch   = cfg['Spatial-CrossValidation']['Visualization_Settings']['regression_plot_switch']
#######################################################################################
# training Settings
Spatial_Trainning_Settings = cfg['Spatial-CrossValidation']['Training_Settings']
Spatial_CV_test_only_Switch = Spatial_Trainning_Settings['Spatial_CV_test_only_Switch']
training_region_filter_Switch = Spatial_Trainning_Settings['training_region_filter_Switch']
training_regions = Spatial_Trainning_Settings['training_regions']
stratified_kfold_Switch = Spatial_Trainning_Settings['stratified_kfold_Switch']
kfold = Spatial_Trainning_Settings['kfold']
repeats = Spatial_Trainning_Settings['repeats']
beginyears = Spatial_Trainning_Settings['beginyears']
endyears = Spatial_Trainning_Settings['endyears']
training_months = Spatial_Trainning_Settings['training_months']
test_beginyears = Spatial_Trainning_Settings['test_beginyears']
test_endyears = Spatial_Trainning_Settings['test_endyears']
additional_test_regions = Spatial_Trainning_Settings['additional_test_regions']
######################################################################################
LassoNet_Stability_Selection_Settings = cfg['LassoNet_Stability_Selection-Settings']
LassoNet_Stability_Selection_Switch = LassoNet_Stability_Selection_Settings['LassoNet_Stability_Selection_Switch']
TRAIN_MODELS = LassoNet_Stability_Selection_Settings['TRAIN_MODELS']
SAVE_MODELS = LassoNet_Stability_Selection_Settings['SAVE_MODELS']
GENERATE_PLOTS = LassoNet_Stability_Selection_Settings['GENERATE_PLOTS']
FORCE_RETRAIN = LassoNet_Stability_Selection_Settings['FORCE_RETRAIN']
LassoNet_beginyears = LassoNet_Stability_Selection_Settings['beginyears']
LassoNet_endyears = LassoNet_Stability_Selection_Settings['endyears']
LassoNet_batch_size = LassoNet_Stability_Selection_Settings['batch_size']
LassoNet_hidden_layers = LassoNet_Stability_Selection_Settings['hidden_layers']
LassoNet_channel_names = LassoNet_Stability_Selection_Settings['channel_names']
#######################################################################################
Hyperparameters_Search_Validation_Switch = cfg['Hyperparameters_Search_Validation-Settings']['Hyperparameters_Search_Validation_Switch']
HSV_Apply_wandb_sweep_Switch = cfg['Hyperparameters_Search_Validation-Settings']['HSV_Apply_wandb_sweep_Switch']
Use_recorded_data_to_show_validation_results = cfg['Hyperparameters_Search_Validation-Settings']['Use_recorded_data_to_show_validation_results']
wandb_sweep_count = cfg['Hyperparameters_Search_Validation-Settings']['wandb_sweep_count']

HSV_TrainingSettings = cfg['Hyperparameters_Search_Validation-Settings']['Training-Settings']
HSV_Spatial_splitting_Switch = HSV_TrainingSettings['Spatial_splitting_Switch']
HSV_Spatial_splitting_begindates = HSV_TrainingSettings['Spatial_splitting_begindates']
HSV_Spatial_splitting_enddates = HSV_TrainingSettings['Spatial_splitting_enddates']
HSV_Spatial_splitting_training_portion = HSV_TrainingSettings['Spatial_splitting_training_portion']
HSV_Spatial_splitting_validation_portion = HSV_TrainingSettings['Spatial_splitting_validation_portion']
HSV_Temporal_splitting_Switch = HSV_TrainingSettings['Temporal_splitting_Switch']
HSV_Temporal_splitting_training_begindates = HSV_TrainingSettings['Temporal_splitting_training_begindates']
HSV_Temporal_splitting_training_enddates = HSV_TrainingSettings['Temporal_splitting_training_enddates']
HSV_Temporal_splitting_validation_begindates = HSV_TrainingSettings['Temporal_splitting_validation_begindates']
HSV_Temporal_splitting_validation_enddates = HSV_TrainingSettings['Temporal_splitting_validation_enddates']

#######################################################################################
# Forced Slope Unity Settings
ForcedSlopeUnityTable = cfg['Spatial-CrossValidation']['Forced-Slope-Unity']

ForcedSlopeUnity = ForcedSlopeUnityTable['ForcedSlopeUnity']
EachMonthForcedSlopeUnity = ForcedSlopeUnityTable['EachMonthForcedSlopeUnity']

#######################################################################################
# Visualization Settings

every_point_begin_years = cfg['Spatial-CrossValidation']['Visualization_Settings']['every_point_begin_years']
every_point_end_years = cfg['Spatial-CrossValidation']['Visualization_Settings']['every_point_end_years']

#######################################################################################
# SHAP Values Analysis Settings

SHAP_Analysis_Settings = cfg['Spatial-CrossValidation']['SHAP_Analysis_Settings']
SHAP_Analysis_switch   = SHAP_Analysis_Settings['SHAP_Analysis_switch']
SHAP_Analysis_Calculation_Switch = SHAP_Analysis_Settings['SHAP_Analysis_Calculation_Switch']
SHAP_Analysis_visualization_Switch = SHAP_Analysis_Settings['SHAP_Analysis_visualization_Switch']
SHAP_Analysis_background_number = SHAP_Analysis_Settings['SHAP_Analysis_background_number']
SHAP_Analysis_test_number = SHAP_Analysis_Settings['SHAP_Analysis_test_number']
SHAP_Analysis_plot_type = SHAP_Analysis_Settings['SHAP_Analysis_plot_type']
#######################################################################################
# Training file Path
results_dir = cfg['Pathway']['Results-dir'] 

txt_outdir = results_dir['txt_outdir']

#######################################################################################
# Fixed number Spatial CV Training Settings
FixNumber_Spatial_CrossValidation_Switch = cfg['FixNumber-SpatialCrossValidation']['FixNumber_CrossValidation_Switch']

FixNumber_Spatial_Settings = cfg['FixNumber-SpatialCrossValidation']['TrainingSettings']
Fixnumber_Spatial_CV_test_only_Switch = FixNumber_Spatial_Settings['Spatial_CV_test_only_Switch']
Fixnumber_kfold  = FixNumber_Spatial_Settings['kfold']
Fixnumber_repeats = FixNumber_Spatial_Settings['repeats']
Fixnumber_beginyears = FixNumber_Spatial_Settings['beginyears']
Fixnumber_endyears   = FixNumber_Spatial_Settings['endyears']
Fixnumber_training_months = FixNumber_Spatial_Settings['training_months']
Fixnumber_test_beginyears = FixNumber_Spatial_Settings['test_beginyears'] 
Fixnumber_test_endyears   = FixNumber_Spatial_Settings['test_endyears'] 
Fixnumber_additional_test_regions = FixNumber_Spatial_Settings['additional_test_regions']
Fixednumber_test_sites   = FixNumber_Spatial_Settings['fixednumber_test_sites']
Fixednumber_train_sites  = FixNumber_Spatial_Settings['fixednumber_train_sites']

################################## BLOO Cross-Validation ################################

BLOO_CrossValidation_Switch = cfg['BLOO-CrossValidation']['BLOO_CrossValidation_Switch']
BLOO_Buffer_size = cfg['BLOO-CrossValidation']['Buffer_size']

#######################################################################################
# BLOO Training Settings

BLOO_TrainingSettings = cfg['BLOO-CrossValidation']['TrainingSettings']
BLOO_Spatial_CV_test_only_Switch = BLOO_TrainingSettings['Spatial_CV_test_only_Switch']
BLOO_kfold   = BLOO_TrainingSettings['kfold']
BLOO_repeats = BLOO_TrainingSettings['repeats']
BLOO_beginyears = BLOO_TrainingSettings['beginyears']
BLOO_endyears   = BLOO_TrainingSettings['endyears']
BLOO_training_months = BLOO_TrainingSettings['training_months']
BLOO_test_beginyears = BLOO_TrainingSettings['test_beginyears']
BLOO_test_endyears   = BLOO_TrainingSettings['test_endyears']
BLOO_additional_test_regions = BLOO_TrainingSettings['additional_test_regions']


################################## BLCO Cross-Validation ################################

BLCO_CrossValidation_Switch = cfg['BLCO-CrossValidation']['BLCO_CrossValidation_Switch']
utilize_self_isolated_sites = cfg['BLCO-CrossValidation']['Utilize_SelfIsolated_Sites_BLCO_Switch']
BLCO_Buffer_size = cfg['BLCO-CrossValidation']['Buffer_size']

#######################################################################################
# BLCO Training Settings

BLCO_TrainingSettings = cfg['BLCO-CrossValidation']['TrainingSettings']
BLCO_Spatial_CV_test_only_Switch = BLCO_TrainingSettings['Spatial_CV_test_only_Switch']
BLCO_seeds_number   =   BLCO_TrainingSettings['seeds_number']
BLCO_kfold   = BLCO_TrainingSettings['kfold']
BLCO_repeats = BLCO_TrainingSettings['repeats']
BLCO_beginyears = BLCO_TrainingSettings['beginyears']
BLCO_endyears   = BLCO_TrainingSettings['endyears']
BLCO_training_months = BLCO_TrainingSettings['training_months']
BLCO_test_beginyears = BLCO_TrainingSettings['test_beginyears']
BLCO_test_endyears   = BLCO_TrainingSettings['test_endyears']
BLCO_additional_test_regions = BLCO_TrainingSettings['additional_test_regions']

#######################################################################################
# BLCO Visualiztion Settings
BLCO_Visualization_Settings = cfg['BLCO-CrossValidation']['visualization_Settings']
Test_Train_Buffers_Distributions_plot = BLCO_Visualization_Settings['Test_Train_Buffers_Distributions_plot']
BLCO_plot_extent = BLCO_Visualization_Settings['plot_extent']
################################## Sensitivity Test Cross-Validation ################################

Sensitivity_Test_Settings = cfg['Sensitivity_Test-Settings']
Sensitivity_Test_Switch = Sensitivity_Test_Settings['Sensitivity_Test_Switch']
Sensitivity_plot_Switch = Sensitivity_Test_Settings['Sensitivity_plot_Switch']

Sensitivity_Test_Training_Settings = Sensitivity_Test_Settings['Training_Settings']
Sensitivity_Test_Spatial_CV_test_only_Switch = Sensitivity_Test_Training_Settings['Spatial_CV_test_only_Switch']
Sensitivity_Test_kfold             = Sensitivity_Test_Training_Settings['kfold']
Sensitivity_Test_repeats           = Sensitivity_Test_Training_Settings['repeats']
Sensitivity_Test_beginyears        = Sensitivity_Test_Training_Settings['beginyears']
Sensitivity_Test_endyears          = Sensitivity_Test_Training_Settings['endyears']
Sensitivity_Test_training_months   = Sensitivity_Test_Training_Settings['training_months']         
Sensitivity_Test_test_beginyears   = Sensitivity_Test_Training_Settings['test_beginyears']
Sensitivity_Test_test_endyears     = Sensitivity_Test_Training_Settings['test_endyears']
Sensitivity_Test_additional_test_regions = Sensitivity_Test_Training_Settings['additional_test_regions']
Exclude_Variables_Sensitivity_Test_Switch             = Sensitivity_Test_Training_Settings['Exclude_Variables_Sensitivity_Test_Switch']
Exclude_Variables_Sensitivity_Test_Variables          = Sensitivity_Test_Training_Settings['Exclude_Variables_Sensitivity_Test_Variables']
Include_Variables_Sensitivity_Test_Switch             = Sensitivity_Test_Training_Settings['Include_Variables_Sensitivity_Test_Switch']
Include_Variables_Sensitivity_Test_Variables          = Sensitivity_Test_Training_Settings['Include_Variables_Sensitivity_Test_Variables']

#######################################################################################


def get_nearest_point_index(sitelon, sitelat, lon_grid, lat_grid):
    '''
    func: get the index of stations on the grids map
    inputs:
        sitelon, sitelat: stations location, eg:[42.353,110.137] 0th dim:lat 1st dim:lat
        lon_grid: grids longitude
        lat_grid: grids latitude
    return:
        index: [index_lat,index_lon]
    '''
    # step1: get the spatial resolution; Default: the latitude and longitude have the same resolution
    det = 0.01
    # step2:
    lon_min = np.min(lon_grid)
    lat_min = np.min(lat_grid)
    index_lon = np.round((sitelon - lon_min) / det)
    index_lat = np.round((sitelat - lat_min) / det)
    index_lon = index_lon.astype(int)
    index_lat = index_lat.astype(int)
    print('site_lat: {}, \n lat_min: {}'.format(sitelat, lat_min))
    return index_lon,index_lat


def Get_typeName(bias, normalize_bias, normalize_species, absolute_species, log_species, species):
    if bias == True:
        typeName = '{}-bias'.format(species)
    elif normalize_bias:
        typeName = 'Normalized-{}-bias'.format(species)
    elif normalize_species == True:
        typeName = 'Normaized-{}'.format(species)
    elif absolute_species == True:
        typeName = 'Absolute-{}'.format(species)
    elif log_species == True:
        typeName = 'Log-{}'.format(species)
    return  typeName

def initialize_AVD_SHAPValues_DataRecording(beginyear:int,endyear:int):
   # MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    shap_values_values = {}
    shap_values_base   = {}
    shap_values_data   = {}
    for iyear in range(endyear-beginyear+1): 
        shap_values_values[str(beginyear+iyear)] = {}
        shap_values_base[str(beginyear+iyear)] = {}
        shap_values_data[str(beginyear+iyear)] = {}
        #for imonth in MONTH:
        #    shap_values_values[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
        #    shap_values_base[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
        #    shap_values_data[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
    return shap_values_values,shap_values_base,shap_values_data

def initialize_BLCO_SitesFoldsRecording(beginyear:int,endyear:int):
    test_sites_index_recording = {}
    train_sites_index_recording = {}
    excluded_sites_index_recording = {}
    testsites2trainsites_nearest_distances = {}
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    for iyear in range(endyear-beginyear+1):
        test_sites_index_recording[str(beginyear+iyear)] = {}
        train_sites_index_recording[str(beginyear+iyear)] = {}
        excluded_sites_index_recording[str(beginyear+iyear)] = {}
        testsites2trainsites_nearest_distances[str(beginyear+iyear)] = {}
        for imonth in MONTH:
            test_sites_index_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
            train_sites_index_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
            excluded_sites_index_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
            testsites2trainsites_nearest_distances[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
    return test_sites_index_recording, train_sites_index_recording, excluded_sites_index_recording, testsites2trainsites_nearest_distances


def initialize_AVD_DataRecording(beginyear:int,endyear:int):
    """This is used to return data recording dict. dict = {  {Year : {Month : np.array() }}}

    Args:
        Area_beginyears (dict): _description_
        endyear (int): _description_

    Returns:
        _type_: _description_
    """
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
    
    final_data_recording = {}
    obs_data_recording = {}
    geo_data_recording = {}
    testing_population_data_recording  = {}
    training_final_data_recording = {}
    training_obs_data_recording = {}
    training_dataForSlope_recording = {}

    
    for iyear in range(endyear-beginyear+1): 
            print(str(beginyear+iyear))   
            final_data_recording[str(beginyear+iyear)] = {}
            obs_data_recording[str(beginyear+iyear)] = {}
            geo_data_recording[str(beginyear+iyear)] = {}
            testing_population_data_recording[str(beginyear+iyear)] = {}
            training_final_data_recording[str(beginyear+iyear)] = {}
            training_obs_data_recording[str(beginyear+iyear)] = {}
            training_dataForSlope_recording[str(beginyear+iyear)] = {}

            for imonth in MONTH:
                final_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                obs_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                geo_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                testing_population_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_final_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_obs_data_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)
                training_dataForSlope_recording[str(beginyear+iyear)][imonth] = np.array([],dtype=np.float64)

    return final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording

def initialize_AVD_CV_dict(test_beginyear:int,test_endyear:int):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual','MAM','JJA','SON','DJF']
    Seasons = []
    test_CV_R2   = {}
    train_CV_R2  = {}
    geo_CV_R2    = {}
    RMSE         = {}
    NRMSE        = {}
    PWM_NRMSE    = {}
    slope        = {}
    PWAModel     = {}
    PWAMonitors  = {}
    
    ### 'AllPoints is not for AVD, but for all points Statistics
    test_CV_R2['AllPoints']   = {}
    train_CV_R2['AllPoints']   = {}
    geo_CV_R2['AllPoints']   = {}
    RMSE['AllPoints']   = {}
    NRMSE['AllPoints']   = {}
    PWM_NRMSE['AllPoints']   = {}
    slope['AllPoints']   = {}
    PWAModel['AllPoints']   = {}
    PWAMonitors['AllPoints']   = {}

    test_CV_R2['AllPoints']['AllPoints']    = -1.0
    train_CV_R2['AllPoints']['AllPoints']   = -1.0
    geo_CV_R2['AllPoints']['AllPoints']     = -1.0
    RMSE['AllPoints']['AllPoints']          = -1.0
    NRMSE['AllPoints']['AllPoints']         = -1.0
    PWM_NRMSE['AllPoints']['AllPoints']     = -1.0
    slope['AllPoints']['AllPoints']         = -1.0
    PWAModel['AllPoints']['AllPoints']      = -1.0
    PWAMonitors['AllPoints']['AllPoints']   = -1.0
    for imonth in MONTH:
        test_CV_R2['AllPoints'][imonth]    = -1.0
        train_CV_R2['AllPoints'][imonth]   = -1.0
        geo_CV_R2['AllPoints'][imonth]     = -1.0
        RMSE['AllPoints'][imonth]          = -1.0
        NRMSE['AllPoints'][imonth]         = -1.0
        PWM_NRMSE['AllPoints'][imonth]     = -1.0
        slope['AllPoints'][imonth]         = -1.0
        PWAModel['AllPoints'][imonth]      = -1.0
        PWAMonitors['AllPoints'][imonth]   = -1.0
    for iyear in range(test_endyear-test_beginyear+1):
            test_CV_R2[str(test_beginyear+iyear)]   = {}
            train_CV_R2[str(test_beginyear+iyear)]  = {}
            geo_CV_R2[str(test_beginyear+iyear)]    = {}
            RMSE[str(test_beginyear+iyear)]         = {}
            NRMSE[str(test_beginyear+iyear)]        = {}
            PWM_NRMSE[str(test_beginyear+iyear)]    = {}
            slope[str(test_beginyear+iyear)]        = {}
            PWAModel[str(test_beginyear+iyear)]     = {}
            PWAMonitors[str(test_beginyear+iyear)]  = {}
            
            for imonth in MONTH:
                test_CV_R2[str(test_beginyear+iyear)][imonth]   = -1.0
                train_CV_R2[str(test_beginyear+iyear)][imonth]  = -1.0
                geo_CV_R2[str(test_beginyear+iyear)][imonth]    = -1.0
                RMSE[str(test_beginyear+iyear)][imonth]         = -1.0
                NRMSE[str(test_beginyear+iyear)][imonth]        = -1.0
                PWM_NRMSE[str(test_beginyear+iyear)][imonth]    = -1.0
                slope[str(test_beginyear+iyear)][imonth]        = -1.0
                PWAModel[str(test_beginyear+iyear)][imonth]     = -1.0
                PWAMonitors[str(test_beginyear+iyear)][imonth]  = -1.0
            
    return test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE,slope, PWAModel, PWAMonitors


def initialize_AVD_CV_Alltime_dict():
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual','MAM','JJA','SON','DJF']
    test_CV_R2_Alltime   = {'Alltime':{}}
    train_CV_R2_Alltime  = {'Alltime':{}}
    geo_CV_R2_Alltime    = {'Alltime':{}}
    RMSE_Alltime         = {'Alltime':{}}
    NRMSE_Alltime        = {'Alltime':{}}
    PWM_NRMSE_Alltime    = {'Alltime':{}}
    slope_Alltime        = {'Alltime':{}}
    PWAModel_Alltime     = {'Alltime':{}}
    PWAMonitors_Alltime  = {'Alltime':{}}
    
    for imonth in MONTH:
            ## np.zeros((3),dtype=np.float64) - 0 - mean, 1 - min, 2 - max
            test_CV_R2_Alltime['Alltime'][imonth]  = np.zeros((3),dtype=np.float64)
            train_CV_R2_Alltime['Alltime'][imonth] = np.zeros((3),dtype=np.float64)
            geo_CV_R2_Alltime['Alltime'][imonth]   = np.zeros((3),dtype=np.float64)
            RMSE_Alltime['Alltime'][imonth]        = np.zeros((3),dtype=np.float64)
            NRMSE_Alltime['Alltime'][imonth]       = np.zeros((3),dtype=np.float64)
            PWM_NRMSE_Alltime['Alltime'][imonth]   = np.zeros((3),dtype=np.float64)
            slope_Alltime['Alltime'][imonth]       = np.zeros((3),dtype=np.float64)
            PWAModel_Alltime['Alltime'][imonth]    = np.zeros((3),dtype=np.float64)
            PWAMonitors_Alltime['Alltime'][imonth] = np.zeros((3),dtype=np.float64)
    return test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime, RMSE_Alltime, NRMSE_Alltime, PWM_NRMSE_Alltime,slope_Alltime, PWAModel_Alltime, PWAMonitors_Alltime

def initialize_Loss_Accuracy_Recordings(kfolds,n_models,epoch,batchsize):
    Training_losses_recording = np.zeros((kfolds,n_models,epoch*4000))
    Training_acc_recording    = np.zeros((kfolds,n_models,epoch*40))
    valid_losses_recording    = np.zeros((kfolds,n_models,epoch*4000))
    valid_acc_recording       = np.zeros((kfolds,n_models,epoch*40))
    print('Training_losses_recording.shape: '.format(Training_losses_recording.shape) + '----------------------')
    return Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording

def combine_kfolds_test_results(test_results_recording,kfold,sitesnumber):
    # Reshape the kfold_array to (kfold, sitesnumber)
    reshaped_array = test_results_recording.reshape(kfold, sitesnumber)

    # Create a mask for elements that are NaN in all kfold arrays
    nan_mask = np.all(np.isnan(reshaped_array), axis=0)
    
    # Replace NaN values with 0 for the purpose of addition
    reshaped_array_no_nan = np.nan_to_num(reshaped_array, nan=0.0)

    # Sum the arrays along the kfold axis
    result = np.sum(reshaped_array_no_nan, axis=0)
    
    # Restore NaN values where all kfold arrays had NaN
    result[nan_mask] = np.nan

    return result

def get_annual_longterm_array(beginyear, endyear, final_data_recording,obs_data_recording,sitesnumber,kfold):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    final_longterm_data = np.full((len(MONTH)*(endyear-beginyear+1),sitesnumber), np.nan, dtype=np.float64)
    obs_longterm_data   = np.full((len(MONTH)*(endyear-beginyear+1),sitesnumber), np.nan, dtype=np.float64)
    
    for iyear in range(endyear-beginyear+1):
        for imonth in range(len(MONTH)):
            final_longterm_data[len(MONTH)*iyear+imonth,:] = combine_kfolds_test_results(final_data_recording[str(beginyear+iyear)][MONTH[imonth]],kfold,sitesnumber)
            obs_longterm_data[len(MONTH)*iyear+imonth,:]   = combine_kfolds_test_results(obs_data_recording[str(beginyear+iyear)][MONTH[imonth]],kfold,sitesnumber)
            
    final_longterm_data = np.nanmean(final_longterm_data,axis=0)
    obs_longterm_data   = np.nanmean(obs_longterm_data,axis=0)
    return final_longterm_data, obs_longterm_data

def get_monthly_longterm_array(beginyear, imonth, endyear, final_data_recording,obs_data_recording,sitesnumber,kfold):   
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    final_longterm_data = np.full(((endyear-beginyear+1),sitesnumber), np.nan, dtype=np.float64)
    obs_longterm_data   = np.full(((endyear-beginyear+1),sitesnumber), np.nan, dtype=np.float64)
    
    for iyear in range(endyear-beginyear+1):
        final_longterm_data[iyear,:] = combine_kfolds_test_results(final_data_recording[str(beginyear+iyear)][MONTH[imonth]],kfold,sitesnumber)
        obs_longterm_data[iyear,:]   = combine_kfolds_test_results(obs_data_recording[str(beginyear+iyear)][MONTH[imonth]],kfold,sitesnumber)
        
    final_longterm_data = np.nanmean(final_longterm_data,axis=0)
    obs_longterm_data   = np.nanmean(obs_longterm_data,axis=0)
    return final_longterm_data, obs_longterm_data

def initialize_multimodels_CV_Dic(kfold:int, repeats:int, beginyears:list,):

    CV_R2 = {}
    CV_slope = {}
    CV_RMSE = {}

    annual_CV_R2 = {}
    annual_CV_slope = {}
    annual_CV_RMSE = {}

    month_CV_R2 = {}
    month_CV_slope = {}
    month_CV_RMSE = {}

    training_CV_R2 = {}
    training_annual_CV_R2 = {}
    training_month_CV_R2 = {}

    geophysical_CV_R2 = {}
    geophysical_annual_CV_R2 = {}
    geophysical_month_CV_R2 = {}


    CV_R2['Alltime']    = np.zeros((kfold * repeats + 1), dtype=np.float32)
    CV_slope['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    CV_RMSE['Alltime']  = np.zeros((kfold * repeats + 1), dtype=np.float32)

    annual_CV_R2['Alltime']    = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_slope['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    annual_CV_RMSE['Alltime']  = np.zeros((kfold * repeats + 1), dtype=np.float32)

    month_CV_R2['Alltime']    = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
    month_CV_slope['Alltime'] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
    month_CV_RMSE['Alltime']  = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

    training_CV_R2['Alltime']        = np.zeros((kfold * repeats + 1), dtype=np.float32)
    training_annual_CV_R2['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    training_month_CV_R2['Alltime']  = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

    geophysical_CV_R2['Alltime']        = np.zeros((kfold * repeats + 1), dtype=np.float32)
    geophysical_annual_CV_R2['Alltime'] = np.zeros((kfold * repeats + 1), dtype=np.float32)
    geophysical_month_CV_R2['Alltime']  = np.zeros((12, kfold * repeats + 1), dtype = np.float32)


    for imodel in range(len(beginyears)):

        CV_R2[str(beginyears[imodel])]    = np.zeros((kfold * repeats + 1), dtype=np.float32)
        CV_slope[str(beginyears[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        CV_RMSE[str(beginyears[imodel])]  = np.zeros((kfold * repeats + 1), dtype=np.float32)

        annual_CV_R2[str(beginyears[imodel])]    = np.zeros((kfold * repeats + 1), dtype=np.float32)
        annual_CV_slope[str(beginyears[imodel])] = np.zeros((kfold * repeats + 1), dtype=np.float32)
        annual_CV_RMSE[str(beginyears[imodel])]  = np.zeros((kfold * repeats + 1), dtype=np.float32)

        month_CV_R2[str(beginyears[imodel])]    = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
        month_CV_slope[str(beginyears[imodel])] = np.zeros((12, kfold * repeats + 1), dtype = np.float32)
        month_CV_RMSE[str(beginyears[imodel])]  = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

        training_CV_R2[str(beginyears[imodel])]          = np.zeros((kfold * repeats + 1), dtype=np.float32)
        training_annual_CV_R2[str(beginyears[imodel])]   = np.zeros((kfold * repeats + 1), dtype=np.float32)
        training_month_CV_R2[str(beginyears[imodel])]    = np.zeros((12, kfold * repeats + 1), dtype = np.float32)

        geophysical_CV_R2[str(beginyears[imodel])]          = np.zeros((kfold * repeats + 1), dtype=np.float32)
        geophysical_annual_CV_R2[str(beginyears[imodel])]   = np.zeros((kfold * repeats + 1), dtype=np.float32)
        geophysical_month_CV_R2[str(beginyears[imodel])]    = np.zeros((12, kfold * repeats + 1), dtype = np.float32)


    return training_CV_R2, training_month_CV_R2, training_annual_CV_R2, geophysical_CV_R2,geophysical_annual_CV_R2,geophysical_month_CV_R2,CV_R2, CV_slope, CV_RMSE, annual_CV_R2, annual_CV_slope, annual_CV_RMSE, month_CV_R2, month_CV_slope, month_CV_RMSE

def initialize_AnnualDataRecording_Dic(beginyears):
    annual_final_test = {}
    annual_obs_test   = {}
    for imodel in range(len(beginyears)):
        annual_final_test[str(beginyears[imodel])] = np.array([],dtype=np.float64)
        annual_obs_test[str(beginyears[imodel])] = np.array([],dtype=np.float64)
        annual_final_test['Alltime'] = np.array([],dtype=np.float64)
        annual_obs_test['Alltime'] = np.array([],dtype=np.float64)
    return annual_final_test, annual_obs_test

def initialize_MonthlyDataRecording_Dic(beginyears):
    monthly_final_test = {}
    monthly_obs_test   = {}
    MONTH = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    
    for imodel in range(len(beginyears)):
        monthly_final_test[str(beginyears[imodel])] = {}
        monthly_obs_test[str(beginyears[imodel])] = {}
    monthly_final_test['Alltime'] = {}
    monthly_obs_test['Alltime'] = {}
    for imonth in range(len(MONTH)):
        for imodel in range(len(beginyears)):
            monthly_final_test[str(beginyears[imodel])][MONTH[imonth]] = np.array([],dtype=np.float64)
            monthly_obs_test[str(beginyears[imodel])][MONTH[imonth]] = np.array([],dtype=np.float64)
        monthly_final_test['Alltime'][MONTH[imonth]] = np.array([],dtype=np.float64)
        monthly_obs_test['Alltime'][MONTH[imonth]] = np.array([],dtype=np.float64)
    return monthly_final_test, monthly_obs_test


def GetFixedNumber_TrainingIndex(test_index:np.array,train_index:np.array,fixed_test_number:int,fixed_train_number:int):
    """This function is used to find fixed number of test sites and fixed number of training sites in B-LOO CV

    Args:
        test_index (np.array): _description_
        train_index (np.array): _description_
        buffer (float): _description_
        fixed_test_number (int): _description_
        fixed_train_number (int): _description_
    """
    selected_test_index = np.random.choice(test_index,fixed_test_number,replace=False)
    #for isite in range(len(selected_test_index)):
     #   temp_index = selected_test_index[isite]
    #    train_index = find_sites_nearby(test_lat=sitelat[temp_index],test_lon=sitelon[temp_index],
       #                                 train_index=train_index,train_lat=sitelat,train_lon=sitelon,buffer_radius=buffer)
    #print('Radius: ',buffer,' - INIT Training index number: ',len(train_index))
    if len(train_index)<fixed_train_number:
        return selected_test_index,train_index
    else:
        selected_train_index = np.random.choice(train_index, fixed_train_number, replace=False)
        return selected_test_index,selected_train_index


############# BLOO CV toolkits ####################


def GetBufferTrainingIndex(test_index:np.array,train_index:np.array,buffer:float,sitelat:np.array, sitelon:np.array):
    """_summary_

    Args:
        test_index (np.array): _description_
        train_index (np.array): _description_
        buffer (float): _description_
    """
    time_start = time.time()
    for isite in range(len(test_index)):
        train_index = find_sites_nearby(test_lat=sitelat[test_index[isite]],test_lon=sitelon[test_index[isite]],train_index=train_index,
                                        train_lat=sitelat,train_lon=sitelon,buffer_radius=buffer)
    time_end = time.time()
    #print('Number of train index: ',len(train_index),'\nNumber of test index: ', len(test_index),'\nTime consume: ',str(np.round(time_end-time_start,4)),'s')
    return train_index


def find_sites_nearby(test_lat: np.float32, test_lon: np.float32,train_index:np.array,
                      train_lat: np.array, train_lon: np.array, buffer_radius: np.float32):
    """This function is used to get the sites index within the buffe area and exclue them from the training index. 

    Args:
        test_lat (np.float32): Test site latitude.
        test_lon (np.float32): Test site longitude.
        train_index (np.array): Training index(remain). This function should be in a loop,
        and all input training index already exclude other sites within the buffer zone near other testing site.
        train_lat (np.array): The initial sites lat array.
        train_lon (np.array): The initial sites lon array.
        buffer_radius (np.float32): The buffer radius.

    Returns:
        np.array : The train index exclude the sites within the input test sites surronding buffer zone.
    """
    lat_min = max(-69.95, (test_lat - 0.1 * buffer_radius))
    lat_max = min(69.95, (test_lat + 0.1 * buffer_radius))
    lon_min = max(-179.95, (test_lon - 0.1 * buffer_radius))
    lon_max = min(179.95, (test_lon + 0.1 * buffer_radius))
    # Find the sites within the square first
    lat_index = np.intersect1d(np.where(train_lat>lat_min),np.where(train_lat<lat_max))
    lon_index = np.intersect1d(np.where(train_lon>lon_min),np.where(train_lon<lon_max))
    sites_nearby_index = np.intersect1d(lat_index,lon_index)
           
    sites_lat_nearby = train_lat[sites_nearby_index]
    sites_lon_nearby = train_lon[sites_nearby_index]

    # Find the sites within the buffer zones
    sites_within_radius_index = np.array([],dtype=int)
    for isite in range(len(sites_nearby_index)):
        distance = calculate_distance(test_lat,test_lon,train_lat[sites_nearby_index[isite]],train_lon[sites_nearby_index[isite]])
        if distance < buffer_radius:
            sites_within_radius_index = np.append(sites_within_radius_index,sites_nearby_index[isite])
    sites_within_index,X_index,Y_index = np.intersect1d(train_index,sites_within_radius_index,return_indices=True)
    train_index = np.delete(train_index,X_index)
    return train_index


def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Radius of the Earth in kilometers (use 3956 for miles)
    r = 6371
    
    # Calculate the distance
    distance = r * c
    
    return distance

def calculate_distance_forArray(site_lat:np.float32,site_lon:np.float32,
                                SATLAT_MAP:np.array,SATLON_MAP:np.array,r=6371.01):
    if np.ndim(SATLAT_MAP) == 0:
        dist_map = calculate_distance(site_lat,site_lon,SATLAT_MAP,SATLON_MAP)
    elif np.ndim(SATLAT_MAP) == 1:
        dist_map = np.zeros(SATLAT_MAP.shape,dtype = np.float64)
        for ix in range(SATLAT_MAP.shape[0]):
            dist_map[ix] = calculate_distance(site_lat,site_lon,SATLAT_MAP[ix],SATLON_MAP[ix])
    elif np.ndim(SATLAT_MAP) == 2:
        dist_map = np.zeros(SATLAT_MAP.shape,dtype = np.float64)
        for ix in range(SATLAT_MAP.shape[0]):
            for iy in range(SATLAT_MAP.shape[1]):
                dist_map[ix,iy] = calculate_distance(site_lat,site_lon,SATLAT_MAP[ix,iy],SATLON_MAP[ix,iy])
   
   #other_sites_pos1_array = np.zeros(len(SATLAT_MAP),dtype=np.float64)
    #other_sites_pos2_array = np.zeros(len(SATLAT_MAP),dtype=np.float64)
    #for i in range(len(SATLAT_MAP)):
       # other_sites_pos1_array[i] = math.radians(SATLAT_MAP[i])
       # other_sites_pos2_array[i] = math.radians(SATLON_MAP[i])
    
    #site_pos1 = site_lat * np.pi / 180.0
    #site_pos2 = site_lon * np.pi / 180.0
    #other_sites_pos1_array = SATLAT_MAP * np.pi / 180.0
    #other_sites_pos2_array = SATLON_MAP * np.pi / 180.0
    #dist_map = r * np.arccos(np.sin(site_pos1)*np.sin(other_sites_pos1_array)+np.cos(site_pos1)*np.cos(other_sites_pos1_array)*np.cos(site_pos2-other_sites_pos2_array))
    
    return dist_map

def get_nearest_test_distance(area_test_index,area_train_index, site_lon, site_lat):
    """This function is used to calcaulate the nearest distance from one site in 
    testing datasets to the whole training datasets.

    Args:
        area_test_index (numpy): Testing index
        area_train_index (numpy): Training index
    return: nearest distances for testing datasets. len(area_test_index)
    """
    nearest_site_distance = np.full((len(area_test_index)),-999.99)
    for index in range(len(area_test_index)):
        temp_lat, temp_lon = site_lat[area_test_index[index]], site_lon[area_test_index[index]]
        other_sites_distances = calculate_distance_forArray(site_lat=temp_lat,site_lon=temp_lon,
                                                            SATLAT_MAP=site_lat[area_train_index],SATLON_MAP=site_lon[area_train_index])
        nearest_site_distance[index] = min(other_sites_distances[np.where(other_sites_distances>0.01)]) # We take 110 kilometers for one degree
    
    return nearest_site_distance

def get_coefficients(nearest_site_distance,cutoff_size,beginyear,endyear,months):
    """This function is used to calculate the coefficient of the combine with Geophysical PM2.5

    Args:
        nearest_site_distance (_type_): _description_
        beginyear (_type_): _description_
        endyear (_type_): _description_

    Returns:
        _type_: _description_
    """
    coefficient = (nearest_site_distance - cutoff_size)/(nearest_site_distance+0.0000001)
    coefficient[np.where(coefficient<0.0)]=0.0
    coefficient = np.square(coefficient)
    coefficients = np.zeros((len(months) * (endyear - beginyear + 1) * len(nearest_site_distance)), dtype=np.float64)  
    for i in range(len(months) * (endyear - beginyear + 1)):  
        coefficients[i * len(nearest_site_distance):(i + 1) * len(nearest_site_distance)] = coefficient
    
    return coefficients
    
