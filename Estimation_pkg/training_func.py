import numpy as np
import gc
import os
import torch
import torch.nn as nn
from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.Model_Func import train, predict, train_lightgbm, predict_lightgbm
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_start_end_YEAR
from Training_pkg.Statistic_Func import regress2, linear_regression, Cal_RMSE
from Training_pkg.Net_Construction import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import *

from Estimation_pkg.data_func import Estimation_ForcedSlopeUnity_Func
from Estimation_pkg.iostream import save_trained_model_forEstimation,save_trained_month_based_model_forEstimation,save_ForcedSlope_forEstimation
from Estimation_pkg.utils import Estimation_ForcedSlopeUnity

def Train_Model_forEstimation(train_beginyears, train_endyears, training_months,width, height, sitesnumber,start_YYYY, TrainingDatasets,total_channel_names,main_stream_channel_names, side_stream_nchannel_names):
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets,observation_data=SPECIES_OBS)
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(total_channel_names)
    seed       = 20230130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    site_index = np.array(range(sitesnumber))
    for imodel_year in range(len(train_beginyears)):
        Normalized_TrainingData    = get_trainingdata_within_start_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=train_beginyears[imodel_year],training_end_YYYY=train_endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
        for imodel_month in range(len(training_months)):
            training_array_index       = Get_month_based_Index(index=site_index,model_beginyear=train_beginyears[imodel_year],beginyear=train_beginyears[imodel_year],endyear=train_endyears[imodel_year],month_index=training_months[imodel_month],sitenumber=sitesnumber)
            learning_objective_index   = Get_month_based_Index(index=site_index,model_beginyear=2005,beginyear=train_beginyears[imodel_year],endyear=train_endyears[imodel_year],month_index=training_months[imodel_month],sitenumber=sitesnumber)
            testing_array_index        = Get_month_based_Index(index=np.array(range(100)),model_beginyear=train_beginyears[imodel_year],beginyear=train_beginyears[imodel_year],endyear=train_endyears[imodel_year],month_index=training_months[imodel_month],sitenumber=sitesnumber) # These two testing arrrays are meaningless here
            teating_objective_index    = Get_month_based_Index(index=np.array(range(100)),model_beginyear=2005,beginyear=train_beginyears[imodel_year],endyear=train_endyears[imodel_year],month_index=training_months[imodel_month],sitenumber=sitesnumber) #

            
            X_train = Normalized_TrainingData[training_array_index,:,:,:]
            y_train = true_input[learning_objective_index]
            train_model_output = np.full(len(training_array_index),np.nan)
            train_mask = np.where(~np.isnan(y_train))[0]

            X_test  = Normalized_TrainingData[testing_array_index,:,:,:] 
            y_test  = true_input[teating_objective_index]
            test_mask  = np.where(~np.isnan(y_test))[0]
            
            if LightGBM_setting:
                from Training_pkg.Net_Construction import LightGBMModel
                model = LightGBMModel()                           
                train_loss, train_acc, valid_losses, test_acc = train_lightgbm(model=model,X_train=X_train[train_mask,:,:,:], y_train=y_train[train_mask], X_test=X_test[test_mask,:,:,:],y_test=y_test[test_mask], input_std=input_std,input_mean=input_mean,mean=mean, std=std, width=width,height=height,initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_nchannel_names)
            else: 
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                cnn_model = initial_network(width=width,main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_nchannel_names))
                cnn_model.to(device)
                torch.manual_seed(21)
                train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model, X_train=X_train[train_mask,:,:,:], y_train=y_train[train_mask], X_test=X_test[test_mask,:,:,:], y_test=y_test[test_mask], input_std=input_std,input_mean=input_mean,mean=mean, std=std, width=width,height=height,BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_nchannel_names,initial_channel_names=total_channel_names)
            if Estimation_ForcedSlopeUnity:
                if LightGBM_setting:
                    train_model_output[train_mask] = predict_lightgbm(inputarray=X_train[train_mask,:,:,:],  model=model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_nchannel_names)
                    save_trained_month_based_model_forEstimation(model=model, model_outdir=model_outdir, typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name,beginyear=train_beginyears[imodel_year], endyear=train_endyears[imodel_year], month_index=training_months[imodel_month], width=width, height=height)
                else:
                    train_model_output[train_mask] = predict(inputarray=X_train[train_mask,:,:,:],  model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_nchannel_names)
                    save_trained_month_based_model_forEstimation(model=cnn_model, model_outdir=model_outdir, typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name,beginyear=train_beginyears[imodel_year], endyear=train_endyears[imodel_year], month_index=training_months[imodel_month], width=width, height=height)
                train_final_data   = Get_final_output(train_model_output, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean, std,training_array_index)
                ForcedSlopeUnity_Dictionary_forEstimation = Estimation_ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=SPECIES_OBS[learning_objective_index],train_area_index=site_index,endyear=train_endyears[imodel_year],
                                                                                             beginyear=train_beginyears[imodel_year],month_index=training_months[imodel_month])
                save_ForcedSlope_forEstimation(ForcedSlopeUnity_Dictionary_forEstimation=ForcedSlopeUnity_Dictionary_forEstimation,model_outdir=model_outdir,typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name,beginyear=train_beginyears[imodel_year], endyear=train_endyears[imodel_year], month_index=training_months[imodel_month], width=width, height=height)
            
            del X_train, y_train
            gc.collect()
    del true_input, Initial_Normalized_TrainingData
    gc.collect()
    return