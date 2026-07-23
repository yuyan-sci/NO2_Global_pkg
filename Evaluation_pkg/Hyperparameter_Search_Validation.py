import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
import random
import csv
import shap
import wandb
import time

from wandb_LightGBM_config import wandb_sweep_parameters_return as wandb_LightGBM_sweep_parameters_return,wandb_initialize
from wandb_ResNet_config import wandb_sweep_parameters_return as wandb_ResNet_sweep_parameters_return ,wandb_initialize

from Evaluation_pkg.utils import *
from Evaluation_pkg.iostream import *
from Evaluation_pkg.data_func import *

from Training_pkg.Net_Construction import initial_network
from Training_pkg.utils import *
from Training_pkg.utils import epoch as config_epoch, batchsize as config_batchsize, lr0 as config_learning_rate0, ResNet_blocks_num as config_ResNet_blocks_num
from Training_pkg.iostream import *
from Training_pkg.data_func import * 
from Training_pkg.Statistic_Func import *
from Training_pkg.Model_Func import train, predict, train_lightgbm, predict_lightgbm

from Training_pkg.Net_Construction import LightGBMModel

####################################################################################
###                                Hyperparameters Search                         ###
####################################################################################
# This module is used to help search the optimized hyperparameters for the CNN model
# Here the cross-validation is not applied to accelerate the training and validation process
# This is only applied at the initial stage, and the final products should be validated by the cross-validation


def Hyperparameters_Search_Training_Testing_Validation(wandb_config,total_channel_names,main_stream_channel_names,side_stream_channel_names):
    
    epoch = config_epoch
    batchsize = config_batchsize
    lr0 = config_learning_rate0
    ResNet_blocks_num = config_ResNet_blocks_num

    if HSV_Spatial_splitting_Switch:
        Evaluation_type = 'Hyperparameters_Search_Validation_Spatial_Splitting'
    
    ### Get the hyperparameters from wandb sweep if wandb is applied
    if wandb.run is not None:
        print('Get wandb sweep parameters...')
        start_time = time.time()
        # Detect which type of sweep is running by checking for parameter names
        if hasattr(wandb_config, 'num_boost_round'):  # LightGBM sweep
            print('Detected LightGBM sweep')
            wandb_parameters = wandb_LightGBM_sweep_parameters_return(sweep_config=wandb_config)
            print(f"LightGBM parameters: {wandb_parameters}")
            
        elif hasattr(wandb_config, 'batch_size'):  # ResNet/CNN sweep
            print('Detected ResNet/CNN sweep')
            sweep_batchsize, sweep_learning_rate0, sweep_epoch, sweep_ResNet_blocks_num, sweep_channel_to_exclude = wandb_ResNet_sweep_parameters_return(sweep_config=wandb_config)
            batchsize = sweep_batchsize
            lr0 = sweep_learning_rate0
            epoch = sweep_epoch
            ResNet_blocks_num = sweep_ResNet_blocks_num
            print('Init_CNN_Datasets finished, time elapsed: ', time.time() - start_time)
            print('Epoch: ', epoch, ' Batch size: ', batchsize, ' Learning rate: ', lr0, 'ResNet_blocks_num: ', ResNet_blocks_num)
        else:
            print('WARNING: Could not detect sweep type!')

    print('@@@@@@@@@@@@@@@wandb name: ', wandb.run.name if wandb.run is not None else 'No wandb run')

    typeName = Get_typeName(bias=bias, normalize_bias=normalize_bias, normalize_species=normalize_species, absolute_species=absolute_species, log_species=False, species=species)    
    width, height, sitesnumber, start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=total_channel_names)    
    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, geolat, geolon = load_geophysical_species_data(species=species)
    population_data = load_coMonitor_Population()
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(total_channel_names)
    seed       = 20230130
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets,observation_data=SPECIES_OBS)
    
    site_index = np.array(range(sitesnumber))
    
    rkf = RepeatedKFold(n_splits=kfold, n_repeats=repeats, random_state=seed)

    final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording = initialize_AVD_DataRecording(beginyear=beginyears[0],endyear=endyears[-1])        
    Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording = initialize_Loss_Accuracy_Recordings(kfolds=kfold,n_models=len(HSV_Spatial_splitting_begindates),epoch=epoch,batchsize=batchsize)

    for imodel_year in range(len(beginyears)):
        Normalized_TrainingData = get_trainingdata_within_start_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)                    
        for imodel_month in range(len(training_months)):                
            valid_sites_index, temp_index_of_initial_array = Get_valid_index_for_temporal_periods(SPECIES_OBS=SPECIES_OBS,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_range=training_months[imodel_month],sitesnumber=sitesnumber)
            imodel_siteindex = site_index[valid_sites_index] # This is equivalent to the temp_index_of_initial_array.
            for ifold, (train_index,test_index) in enumerate(rkf.split(imodel_siteindex)):
                # train_index, test_index = train_test_split(imodel_siteindex,test_size=0.2,random_state=seed)
                train_index = imodel_siteindex[train_index]
                test_index  = imodel_siteindex[test_index]
                
                # test_index and train_index are the index of the array splited, not the value in the array.
                X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_month_based_XY_indices(train_index=train_index,test_index=test_index,model_beginyear=beginyears[imodel_year],beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_index=training_months[imodel_month], sitesnumber=sitesnumber)
                X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)
            
                ## Even though test_index and train_index are selected based on valid sites,
                ## there are still some missing points in some months.
                ## Therefore, we need to remove the missing points in the training and testing datasets.
                train_mask = np.where(~np.isnan(y_train))[0]
                test_mask  = np.where(~np.isnan(y_test))[0]

                # *------------------------------------------------------------------------------*#
                ## Training Process.
                # *------------------------------------------------------------------------------*#
                if LightGBM_setting:
                    # Initialize LightGBM model
                    model = LightGBMModel(params=wandb_parameters)
                    
                    print(f'Training LightGBM model for fold {ifold}, year {beginyears[imodel_year]}-{endyears[imodel_year]}, months {training_months[imodel_month]}')
                    
                    # Train model
                    train_loss, train_acc, valid_losses, test_acc = train_lightgbm(model=model,X_train=X_train[train_mask,:,:,:], y_train=y_train[train_mask], X_test=X_test[test_mask,:,:,:],y_test=y_test[test_mask], input_std=input_std,input_mean=input_mean,mean=mean, std=std, width=width,height=height,initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_channel_names)
                    # Record losses and accuracies
                    Training_losses_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(train_loss)] = train_loss
                    Training_acc_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(train_acc)] = train_acc
                    valid_losses_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(valid_losses)] = valid_losses
                    valid_acc_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(test_acc)] = test_acc
                        
                    # Save model
                    save_trained_month_based_lightgbm_model(lgb_model=model, model_outdir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species,nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height)
                    
                else:     
                    cnn_model = initial_network(width=width,main_stream_nchannel=len(main_stream_channel_names),
                                                side_stream_nchannel=len(side_stream_channel_names), blocks_num=ResNet_blocks_num)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    cnn_model.to(device)
                    torch.manual_seed(21)
                    train_loss, train_acc, valid_losses, test_acc = train(model=cnn_model, X_train=X_train[train_mask,:,:,:], y_train=y_train[train_mask], X_test=X_test[test_mask,:,:,:],
                                                                        y_test=y_test[test_mask], input_std=input_std,input_mean=input_mean,mean=mean, std=std, width=width,height=height,BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch,
                                                                initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_channel_names)
                    Training_losses_recording[ifold,imodel_year*len(training_months)+imodel_month,0:len(train_loss)] = train_loss
                    Training_acc_recording[ifold,imodel_year*len(training_months)+imodel_month,0:len(train_acc)]    = train_acc
                    valid_losses_recording[ifold,imodel_year*len(training_months)+imodel_month,0:len(valid_losses)]  = valid_losses
                    valid_acc_recording[ifold,imodel_year*len(training_months)+imodel_month,0:len(test_acc)]       = test_acc   
                    
                    save_trained_month_based_model(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height)
                            
                # *------------------------------------------------------------------------------*#
                ## Validation Process.
                # *------------------------------------------------------------------------------*#
                for iyear in range((endyears[imodel_year]-beginyears[imodel_year]+1)):
                    yearly_allsites_index  = Get_month_based_Index(index=site_index, model_beginyear=beginyears[imodel_year], beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                    yearly_allsites_Yindex = Get_month_based_Index(index=site_index,model_beginyear=2005,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                    yearly_test_index   = Get_month_based_Index(index=test_index, model_beginyear=beginyears[imodel_year],beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                    yearly_train_index  = Get_month_based_Index(index=train_index, model_beginyear=beginyears[imodel_year],beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                    yearly_test_Yindex  = Get_month_based_Index(index=test_index,model_beginyear=2005,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                    yearly_train_Yindex = Get_month_based_Index(index=train_index,model_beginyear=2005,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                    
                    ## Get the input predictors for the testing and training datasets.
                    yearly_test_input  = Normalized_TrainingData[yearly_test_index,:,:,:]
                    yearly_train_input = Normalized_TrainingData[yearly_train_index,:,:,:]
                    
                    ## We are going to save the results each month in an array that has a length of
                    ## the number of total sites for each month. Therefore, we need to get the index
                    ## of the nonan values in the testing and training datasets in all sites.
                    nonan_yearly_test_index = np.where(~np.isnan(SPECIES_OBS[yearly_test_Yindex]))[0]
                    nonan_yearly_train_index = np.where(~np.isnan(SPECIES_OBS[yearly_train_Yindex]))[0]

                    ## The yearly_test_index and yearly_test_Yindex are not the same indices, but they share the same length and 
                    ## have the same relative location. They only have different offets. 
                    nonan_yearly_test_predictor_datasets_within_time_periods_indices = np.where(np.in1d(yearly_allsites_index,yearly_test_index[nonan_yearly_test_index]))[0]
                    nonan_yearly_train_predictors_datasets_within_time_periods_indices = np.where(np.in1d(yearly_allsites_index,yearly_train_index[nonan_yearly_train_index]))[0] 
                    
                    Validation_Prediction = np.full(len(yearly_allsites_index),np.nan)
                    Training_Prediction   = np.full(len(yearly_allsites_index),np.nan)
                    Validation_obs_data   = np.full(len(yearly_allsites_index),np.nan)
                    Training_obs_data     = np.full(len(yearly_allsites_index),np.nan)
                    Geophysical_test_data = np.full(len(yearly_allsites_index),np.nan)
                    population_test_data  = np.full(len(yearly_allsites_index),np.nan)
                    
                    Validation_obs_data[nonan_yearly_test_predictor_datasets_within_time_periods_indices]   = SPECIES_OBS[yearly_test_Yindex[nonan_yearly_test_index]]
                    Training_obs_data[nonan_yearly_train_predictors_datasets_within_time_periods_indices]   = SPECIES_OBS[yearly_train_Yindex[nonan_yearly_train_index]]
                    Geophysical_test_data[nonan_yearly_test_predictor_datasets_within_time_periods_indices] = geophysical_species[yearly_test_Yindex[nonan_yearly_test_index]]
                    population_test_data[nonan_yearly_test_predictor_datasets_within_time_periods_indices]  = population_data[yearly_test_Yindex[nonan_yearly_test_index]]
                    if LightGBM_setting:
                        model = load_month_based_lightgbm_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height)
                        Validation_Prediction[nonan_yearly_test_predictor_datasets_within_time_periods_indices] = predict_lightgbm(inputarray=yearly_test_input[nonan_yearly_test_index,:,:,:],model=model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names, sidestream_channel_names=side_stream_channel_names)
                        Training_Prediction[nonan_yearly_train_predictors_datasets_within_time_periods_indices] = predict_lightgbm(inputarray=yearly_train_input[nonan_yearly_train_index,:,:,:], model=model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                    else:
                        cnn_model = load_month_based_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height)
                        Validation_Prediction[nonan_yearly_test_predictor_datasets_within_time_periods_indices]   = predict(inputarray=yearly_test_input[nonan_yearly_test_index,:,:,:], model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                        Training_Prediction[nonan_yearly_train_predictors_datasets_within_time_periods_indices]   = predict(inputarray=yearly_train_input[nonan_yearly_train_index,:,:,:],  model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                        
                    final_data = Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,yearly_allsites_Yindex)
                    train_final_data = Get_final_output(Training_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean, std,yearly_allsites_Yindex)
                
                    if combine_with_GeophysicalSpeceis_Switch:
                        nearest_distance = get_nearest_test_distance(area_test_index=test_index,area_train_index=train_index,site_lat=lat,site_lon=lon)
                        coeficient = get_coefficients(nearest_site_distance=nearest_distance,cutoff_size=cutoff_size,beginyear=beginyears[imodel_year],
                                            endyear = endyears[imodel_year],months=training_months[imodel_month])
                        final_data = (1.0-coeficient)*final_data + coeficient * geophysical_species[yearly_test_Yindex]
                    if ForcedSlopeUnity:
                        final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=Training_obs_data
                                                ,test_final_data=final_data,train_area_index=site_index,test_area_index=site_index,
                                                endyear=beginyears[imodel_year]+iyear,beginyear=beginyears[imodel_year]+iyear,month_index=training_months[imodel_month],EachMonth=EachMonthForcedSlopeUnity)
                    
                    # *------------------------------------------------------------------------------*#
                    ## Recording observation and prediction for this model this fold.
                    # *------------------------------------------------------------------------------*#
                    ### These recordings each year, each month has an array with the length of the 1 times the number of total sites.
                    ### Not interested points are filled with nan.
                    for imonth in range(len(training_months[imodel_month])):
                        final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]              = np.append(final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], final_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                        obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Validation_obs_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                        geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Geophysical_test_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                        training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]     = np.append(training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], train_final_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                        training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]       = np.append(training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Training_obs_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                        testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]] = np.append(testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], population_test_data[imonth*len(site_index):(imonth+1)*len(site_index)])

    save_month_based_data_recording(obs_data=obs_data_recording,final_data=final_data_recording,geo_data_recording=geo_data_recording,training_final_data_recording=training_final_data_recording,
                                    training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,lat_recording=lat,lon_recording=lon,
                                    species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height)

    save_loss_accuracy(model_outdir=model_outdir,loss=Training_losses_recording, accuracy=Training_acc_recording,valid_loss=valid_losses_recording, valid_accuracy=valid_acc_recording,typeName=typeName,
                version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height)       
            
    txtfile_outdir = txt_outdir + '{}/{}/Results/results-HSV/statistical_indicators/{}_{}_{}_{}Channel_{}x{}{}/'.format(species, version,typeName,species,version,nchannel,width,height,special_name)
    os.makedirs(txtfile_outdir, exist_ok=True)
    obs_data_recording, final_data_recording, geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_test_recording, lon_test_recording = load_month_based_data_recording(species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height)
    for iyear in range(len(test_beginyears)):
        test_beginyear = test_beginyears[iyear]
        test_endyear   = test_endyears[iyear]
        test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors, regional_number = calculate_Statistics_results(test_beginyear=test_beginyear, test_endyear=test_endyear,
                                                                                                            final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                            geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                            training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,
                                                                                                            masked_array_index=site_index,Area='Global',sitesnumber=sitesnumber,init_masked_index=site_index)
        
        print('1', 1, '\ntest_CV_R2', test_CV_R2, '\ntrain_CV_R2', train_CV_R2, '\ngeo_CV_R2', geo_CV_R2, '\nRMSE', RMSE, '\nNRMSE', NRMSE, '\nPWM_NRMSE', PWM_NRMSE, '\nslope', slope)

        print('test_CV_R2 AllPoint Annual', test_CV_R2['AllPoints']['Annual'])
        
        csvfile_outdir = txt_outdir + '{}/{}/Results/results-HSV/statistical_indicators/{}_{}_{}_{}_{}Channel_{}x{}{}/'.format(species,version,Evaluation_type,typeName,species,version,
                                                                                                                        len(main_stream_channel_names),width,height,special_name)   
        os.makedirs(csvfile_outdir, exist_ok=True)

        # If the recorded data is used to show the validation results, then load the data and calculate the statistics
        if wandb.run is not None:
            wandb.log({'test_R2': np.round(test_CV_R2['AllPoints']['Annual'],4),
                        'train_R2': np.round(train_CV_R2['AllPoints']['Annual'],4),
                        'geo_R2': np.round(geo_CV_R2['AllPoints']['Annual'],4),
                        'RMSE': np.round(RMSE['AllPoints']['Annual'],4),
                        'NRMSE': np.round(NRMSE['AllPoints']['Annual'],4),
                        'slope': np.round(slope['AllPoints']['Annual'],4),
                        })
            
            wandb.run.summary['test_R2'] = np.round(test_CV_R2['AllPoints']['Annual'], 4)
            wandb.run.summary['train_R2'] = np.round(train_CV_R2['AllPoints']['Annual'], 4)
            wandb.run.summary['geo_R2'] = np.round(geo_CV_R2['AllPoints']['Annual'], 4)
            wandb.run.summary['RMSE'] = np.round(RMSE['AllPoints']['Annual'], 4)
            wandb.run.summary['NRMSE'] = np.round(NRMSE['AllPoints']['Annual'], 4)
            wandb.run.summary['slope'] = np.round(slope['AllPoints']['Annual'], 4)
            
            print(f"Logged summary metrics - test_R2: {test_CV_R2['AllPoints']['Annual']:.4f}")
        
        if wandb.run is not None:
            api = wandb.Api()
            sweep = api.sweep(f"/{wandb.run.entity}/{wandb.run.project}/{wandb.run.sweep_id}")
            csvfile_outdir = csvfile_outdir + 'sweep-{}/'.format(sweep.name)
            os.makedirs(csvfile_outdir, exist_ok=True)
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}Channel_{}-{}_{}x{}_sweep-{}.csv'.format(species,version,
                                                                                                        Evaluation_type,typeName,
                                                                                                        len(main_stream_channel_names),HSV_Spatial_splitting_begindates[0],
                                                                                                        HSV_Spatial_splitting_enddates[-1],width,height,wandb.run.name)
        else:
            csvfile_outfile = csvfile_outdir + '{}_{}_{}_{}_{}Channel_{}-{}_{}x{}{}.csv'.format(species,version,
                                                                                                        Evaluation_type,typeName,
                                                                                                        len(main_stream_channel_names),HSV_Spatial_splitting_begindates[0],
                                                                                                        HSV_Spatial_splitting_enddates[-1],width,height,special_name)
        AVD_output_text(outfile=csvfile_outfile,status='w', Area='Global',test_beginyears=test_beginyear,test_endyears=test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                        slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)
        
    return