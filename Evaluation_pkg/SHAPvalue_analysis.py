import torch
import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
import random
import csv
import shap

from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.Model_Func import train, predict
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_start_end_YEAR
from Training_pkg.Statistic_Func import regress2, linear_regression, Cal_RMSE
from Training_pkg.Net_Construction import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Get_month_based_Index,Get_month_based_XY_indices,GetXIndex,GetYIndex,Get_XY_indices, Get_XY_arraies, Get_final_output, ForcedSlopeUnity_Func, CalculateAnnualR2, CalculateMonthR2, calculate_Statistics_results
from Evaluation_pkg.iostream import *
from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure, SHAPvalues_Analysis_figure


def Spatial_CV_SHAP_Analysis(width, height, sitesnumber,start_YYYY, TrainingDatasets, total_channel_names,main_stream_channel_names, side_stream_nchannel_names,):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    nchannel   = len(total_channel_names)
    seed       = 20230130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    if SHAP_Analysis_Calculation_Switch:
        site_index = np.array(range(sitesnumber))
        Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets,observation_data=SPECIES_OBS)
        rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)
        count = 0

        if LightGBM_setting:
            # For LightGBM: 2D arrays (samples, features)
            shap_values_values = np.zeros([0, nchannel], dtype=np.float32)
            shap_values_data = np.zeros([0, nchannel], dtype=np.float32)
        else:
            # For CNN: 4D arrays (samples, channels, height, width)
            shap_values_values = np.zeros([0, nchannel, width, height], dtype=np.float32)
            shap_values_data = np.zeros([0, nchannel, width, height], dtype=np.float32)

        shap_values_base = np.array([], dtype=np.float32)
        for imodel_year in range(len(beginyears)):
                #temp_TrainingDatasets = get_trainingdata_within_start_end_YEAR(initial_array=TrainingDatasets,training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                #Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=temp_TrainingDatasets,observation_data=SPECIES_OBS[(beginyears[imodel_year]-1998)*12*sitesnumber:(endyears[imodel_year]-1998+1)*12*sitesnumber])

                Normalized_TrainingData = get_trainingdata_within_start_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                for imodel_month in range(len(training_months)):
                    ## For each model, get the valid sites index and the initial array index.
                    ## Different sites to be splited into testing and training datasets.
                    valid_sites_index, temp_index_of_initial_array = Get_valid_index_for_temporal_periods(SPECIES_OBS=SPECIES_OBS,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_range=training_months[imodel_month],sitesnumber=sitesnumber)
                    imodel_siteindex = site_index[valid_sites_index] # This is equivalent to the temp_index_of_initial_array.
                    if LightGBM_setting:
                        LightGBM_model = load_month_based_lightgbm_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height)
                    else:
                        cnn_model = load_month_based_model(model_indir=model_outdir,typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],
                                                            month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, 
                                                        special_name=special_name, count=count, width=width, height=height)
                        cnn_model.eval()
                    for ifold, (train_index,test_index) in enumerate(rkf.split(imodel_siteindex)):
                        for iyear in range((endyears[imodel_year]-beginyears[imodel_year]+1)):
                            yearly_test_index   = Get_month_based_Index(index=test_index, model_beginyear=beginyears[imodel_year],beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                            yearly_train_index  = Get_month_based_Index(index=train_index, model_beginyear=beginyears[imodel_year],beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                            yearly_test_Yindex  = Get_month_based_Index(index=test_index,model_beginyear=beginyears[imodel_year],beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                            yearly_train_Yindex = Get_month_based_Index(index=train_index,model_beginyear=beginyears[imodel_year],beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                            
                            
                            
                            ## We are going to save the results each month in an array that has a length of
                            ## the number of total sites for each month. Therefore, we need to get the index
                            ## of the nonan values in the testing and training datasets in all sites.
                            nonan_yearly_test_index = np.where(~np.isnan(SPECIES_OBS[yearly_test_Yindex]))[0]
                            nonan_yearly_train_index = np.where(~np.isnan(SPECIES_OBS[yearly_train_Yindex]))[0]


                            ## Get the input predictors for the testing and training datasets.
                            yearly_test_input  = Normalized_TrainingData[nonan_yearly_test_index,:,:,:]
                            yearly_train_input = Normalized_TrainingData[nonan_yearly_train_index,:,:,:]
                            background_data_number = min(len(yearly_train_index),SHAP_Analysis_background_number)
                            data_to_explain_number = min(len(yearly_test_index), SHAP_Analysis_test_number)

                            if LightGBM_setting:
                                Back_Ground_Data_raw = yearly_train_input[np.sort(np.random.choice(yearly_train_input.shape[0], background_data_number, replace=False))]
                                Data_to_Explain_raw = yearly_test_input[np.sort(np.random.choice(yearly_test_input.shape[0], data_to_explain_number, replace=False))]
                                Back_Ground_Data = Back_Ground_Data_raw[:, :, height//2, width//2]  # Shape: (N, 27)
                                Data_to_Explain = Data_to_Explain_raw[:, :, height//2, width//2]    # Shape: (N, 27)
                                
                                print('Data_to_Explain.shape: {}, type: {}'.format(Data_to_Explain.shape, type(Data_to_Explain)))
    
                                LightGBM_Explainer = shap.TreeExplainer(model=LightGBM_model.model, data=Back_Ground_Data)
                                shap_values = LightGBM_Explainer.shap_values(Data_to_Explain,check_additivity=False)
                            else:
                                Back_Ground_Data = torch.Tensor(yearly_train_input[np.sort(np.random.choice(yearly_train_input.shape[0],background_data_number, replace=False))])
                                Data_to_Explain  = torch.Tensor(yearly_test_input[np.sort(np.random.choice(yearly_test_input.shape[0], data_to_explain_number, replace=False))])
                                print('Data_to_Explain.shape: {}, type: {}'.format(Data_to_Explain.shape, type(Data_to_Explain)))
                                Back_Ground_Data = Back_Ground_Data.to(device)
                                Data_to_Explain  = Data_to_Explain.to(device)
                                CNNModel_Explainer = shap.DeepExplainer(model=cnn_model,data=Back_Ground_Data)
                                shap_values = CNNModel_Explainer.shap_values(Data_to_Explain,check_additivity=False)
                                Data_to_Explain = Data_to_Explain.cpu().detach().numpy()                                
                                
                            shap_values = np.squeeze(shap_values)
                            print(shap_values.shape)
                            
                            shap_values_values = np.append(shap_values_values, shap_values, axis=0)
                            shap_values_data   = np.append(shap_values_data, Data_to_Explain, axis=0)
        
        save_SHAPValues_data_recording(shap_values_values=shap_values_values, shap_values_data=shap_values_data,
                                    species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,
                                    width=width,height=height)
    if SHAP_Analysis_visualization_Switch:
        shap_values_values, shap_values_data = load_SHAPValues_data_recording(species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,
                                                                    width=width,height=height)
        if SHAP_Analysis_plot_type == 'beeswarm':
            if len(shap_values_values.shape) == 4:
                print(f'Summing 4D SHAP values: {shap_values_values.shape}')
                shap_values_values = np.sum(shap_values_values, axis=(2,3))
                shap_values_data   = np.sum(shap_values_data, axis=(2,3))
            else:
                print(f'SHAP values already 2D: {shap_values_values.shape}')
                
            shap_values_data_min = np.min(shap_values_data,axis=0)
            shap_values_data_max = np.max(shap_values_data,axis=0)
            print('shap_values_data.shape: ', shap_values_data.shape)
            shap_values_data = (shap_values_data - shap_values_data_min) / (shap_values_data_max-shap_values_data_min)
            print(np.min(shap_values_data,axis=0),np.max(shap_values_data,axis=0))
            DISPLAY_NAMES = {'GeoNO2v513_GC': 'GeoNO2'}
            display_channel_names = [DISPLAY_NAMES.get(n, n) for n in total_channel_names]
            shap_values_with_feature_names = shap.Explanation(values=shap_values_values,data=shap_values_data,feature_names=display_channel_names)
        SHAPvalues_Analysis_figure(shap_values_with_feature_names=shap_values_with_feature_names,plot_type=SHAP_Analysis_plot_type,typeName=typeName,
                                    species=species,version=version,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,width=width,height=height,special_name=special_name)
    return

