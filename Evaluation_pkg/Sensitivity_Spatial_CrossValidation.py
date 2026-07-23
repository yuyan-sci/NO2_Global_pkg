import torch
import numpy as np
import torch
import torch.nn as nn
import os
import gc
from sklearn.model_selection import RepeatedKFold
import random
import csv

from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.Model_Func import train, predict
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_start_end_YEAR
from Training_pkg.Statistic_Func import regress2, linear_regression, Cal_RMSE
from Training_pkg.Net_Construction import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import *
from Evaluation_pkg.iostream import *

def Sensitivity_Test_AVD_CrossValidation(width, height, sitesnumber,start_YYYY, TrainingDatasets,total_channel_names,main_stream_channel_names, side_stream_channel_names,sensitivity_test_channel_names, sensitivity_test_type):
    # *------------------------------------------------------------------------------*#
    ##   Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    beginyears = Sensitivity_Test_beginyears
    training_months = Sensitivity_Test_training_months
    sensitivity_test_names_suffix = ''
    for iname in sensitivity_test_channel_names:
        sensitivity_test_names_suffix += '-'+iname

    SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
    geophysical_species, lat, lon = load_geophysical_species_data(species=species)
    true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets,observation_data=SPECIES_OBS)
    population_data = load_coMonitor_Population()
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    nchannel   = len(total_channel_names)
    seed       = 20230130
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
    site_index = np.array(range(sitesnumber))
    
    rkf = RepeatedKFold(n_splits=Sensitivity_Test_kfold, n_repeats=repeats, random_state=seed)
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording = initialize_AVD_DataRecording(beginyear=beginyears[0],endyear=endyears[-1])
    Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording = initialize_Loss_Accuracy_Recordings(kfolds=kfold,n_models=len(beginyears)*len(training_months),epoch=epoch,batchsize=batchsize)
    

    if not Sensitivity_Test_Spatial_CV_test_only_Switch:
        for imodel_year in range(len(beginyears)):
                #temp_TrainingDatasets = get_trainingdata_within_start_end_YEAR(initial_array=TrainingDatasets,training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                #Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=temp_TrainingDatasets,observation_data=SPECIES_OBS[(beginyears[imodel_year]-2005)*12*sitesnumber:(endyears[imodel_year]-2005+1)*12*sitesnumber])
                Normalized_TrainingData = get_trainingdata_within_start_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                for imodel_month in range(len(training_months)):
                    ## For each model, get the valid sites index and the initial array index.
                    ## Different sites to be splited into testing and training datasets.
                    valid_sites_index, temp_index_of_initial_array = Get_valid_index_for_temporal_periods(SPECIES_OBS=SPECIES_OBS,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_range=training_months[imodel_month],sitesnumber=sitesnumber)
                    imodel_siteindex = site_index[valid_sites_index] # This is equivalent to the temp_index_of_initial_array.
                    for ifold, (train_index,test_index) in enumerate(rkf.split(imodel_siteindex)):
                        train_index = imodel_siteindex[train_index]
                        test_index  = imodel_siteindex[test_index]
                        X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_month_based_XY_indices(train_index=train_index,test_index=test_index,model_beginyear=beginyears[imodel_year],beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_index=training_months[imodel_month], sitesnumber=sitesnumber)
                        X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)
                        
                        ## Even though test_index and train_index are selected based on valid sites,
                        ## there are still some missing points in some months.
                        ## Therefore, we need to remove the missing points in the training and testing datasets.
                        train_mask = np.where(~np.isnan(y_train))[0]
                        test_mask  = np.where(~np.isnan(y_test))[0]

                        #print('X_train size: {}, X_test size: {}, y_train size: {}, y_test size: {} -------------------------------------------'.format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
                        # *------------------------------------------------------------------------------*#
                        ## Training Process.
                        # *------------------------------------------------------------------------------*#
                        if LightGBM_setting:
                            from Training_pkg.Net_Construction import LightGBMModel
                            lightgbm_model = LightGBMModel()
                            
                            print(f'Training LightGBM model for fold {ifold}, year {beginyears[imodel_year]}-{endyears[imodel_year]}, months {training_months[imodel_month]}')
                            
                            # Train model
                            train_loss, train_acc, valid_losses, test_acc = train_lightgbm(model=lightgbm_model,X_train=X_train[train_mask,:,:,:], y_train=y_train[train_mask], X_test=X_test[test_mask,:,:,:],y_test=y_test[test_mask], 
                                                                                           input_std=input_std,input_mean=input_mean,mean=mean, std=std, width=width,height=height,
                                                                                           initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_channel_names)
                            # Record losses and accuracies
                            Training_losses_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(train_loss)] = train_loss
                            Training_acc_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(train_acc)] = train_acc
                            valid_losses_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(valid_losses)] = valid_losses
                            valid_acc_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(test_acc)] = test_acc
                                
                            # Save model
                            save_sensitivity_test_trained_month_based_model(model=lightgbm_model, model_outdir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species,nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix)
                        else:
                            cnn_model = initial_network(width=width,main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_channel_names))

                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            cnn_model.to(device)
                            torch.manual_seed(21)
                            train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model, X_train=X_train[train_mask,:,:,:], y_train=y_train[train_mask], X_test=X_test[test_mask,:,:,:],
                                                                                    y_test=y_test[test_mask], input_std=input_std,input_mean=input_mean,mean=mean, std=std, width=width,height=height,BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch,
                                                                            initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_channel_names)
                            Training_losses_recording[ifold,imodel_year*len(training_months)+imodel_month,0:len(train_loss)] = train_loss
                            Training_acc_recording[ifold,imodel_year*len(training_months)+imodel_month,0:len(train_acc)]    = train_acc
                            valid_losses_recording[ifold,imodel_year*len(training_months)+imodel_month,0:len(valid_losses)]  = valid_losses
                            valid_acc_recording[ifold,imodel_year*len(training_months)+imodel_month,0:len(test_acc)]       = test_acc

                            save_sensitivity_test_trained_month_based_model(model=cnn_model, model_outdir=model_outdir, beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month],typeName=typeName, version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix)
                        # *------------------------------------------------------------------------------*#
                        ## Validation Process.
                        # *------------------------------------------------------------------------------*#
                        if LightGBM_setting:
                            lightgbm_model = load_sensitivity_test_trained_month_based_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix)
                        else:
                            cnn_model = load_sensitivity_test_trained_month_based_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix)
                        
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
                            ## have the sam erelative location. They only have different offets. 
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
                                Validation_Prediction[nonan_yearly_test_predictor_datasets_within_time_periods_indices] = predict_lightgbm(inputarray=yearly_test_input[nonan_yearly_test_index,:,:,:], model=lightgbm_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                                Training_Prediction[nonan_yearly_train_predictors_datasets_within_time_periods_indices] = predict_lightgbm(inputarray=yearly_train_input[nonan_yearly_train_index,:,:,:], model=lightgbm_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                            else:
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

                            
                            

                            for imonth in range(len(training_months[imodel_month])):
                                final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]              = np.append(final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], final_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Validation_obs_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Geophysical_test_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]     = np.append(training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], train_final_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]       = np.append(training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Training_obs_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]] = np.append(testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], population_test_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                    

            
        save_sensitivity_test_month_based_data_recording(obs_data=obs_data_recording,final_data=final_data_recording,geo_data_recording=geo_data_recording,training_final_data_recording=training_final_data_recording,
                                                         training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,lat_recording=lat,lon_recording=lon,
                                        species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix)
    obs_data_recording, final_data_recording, geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording,lat_test_recording, lon_test_recording = load_sensitivity_test_month_based_data_recording(species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix)
    txtfile_outdir = txt_outdir + '{}/{}/Results/results-Sensitivity_Tests/statistical_indicators/'.format(species, version)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    
    for iyear in range(len(Sensitivity_Test_test_beginyears)):
        Sensitivity_Test_test_beginyear = Sensitivity_Test_test_beginyears[iyear]
        Sensitivity_Test_test_endyear   = Sensitivity_Test_test_endyears[iyear]
        masked_array = Get_masked_array_index(masked_site_index=site_index,kfold=kfold,sitesnumber=sitesnumber)
        

        test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors, regional_number = calculate_Statistics_results(test_beginyear=Sensitivity_Test_test_beginyear, test_endyear=Sensitivity_Test_test_endyear,
                                                                                                            final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                            geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                            training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,
                                                                                                            masked_array_index=masked_array,Area='Global',sitesnumber=sitesnumber,init_masked_index=site_index)
        
        if sensitivity_test_type == 'exclusion':
            txt_outfile =  txtfile_outdir + 'Sensitivity_Tests_{}-{}_{}_{}_{}_{}Channel_{}x{}{}_Exclude{}.csv'.format(Sensitivity_Test_test_beginyear,Sensitivity_Test_test_endyear,typeName,species,version,nchannel,width,height,special_name,sensitivity_test_names_suffix)
        else:
            txt_outfile =  txtfile_outdir + 'Sensitivity_Tests_{}-{}_{}_{}_{}_{}Channel_{}x{}{}_Include{}.csv'.format(Sensitivity_Test_test_beginyear,Sensitivity_Test_test_endyear,typeName,species,version,nchannel,width,height,special_name,sensitivity_test_names_suffix)
       
        SensitivityTests_output_text(outfile=txt_outfile,status='w',Area='Global', test_beginyears=Sensitivity_Test_test_beginyear,test_endyears=Sensitivity_Test_test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                        slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names=sensitivity_test_names_suffix,regional_number=regional_number)
    
    if len(Sensitivity_Test_additional_test_regions) > 0:
        for iregion in Sensitivity_Test_additional_test_regions:
            mask_map, mask_lat, mask_lon = load_Global_Mask_data(region_name=iregion)
            # Check if mask loading failed
            if mask_lat is None or mask_lon is None:
                print(f"Warning: Could not load mask data for region '{iregion}'. Skipping statistics for this region.")
                continue # Skip to the next region
            masked_array_index = find_masked_latlon(mask_map=mask_map,mask_lat=mask_lat,mask_lon=mask_lon,test_lat=lat_test_recording,test_lon=lon_test_recording)
            for iyear in range(len(Sensitivity_Test_test_beginyears)):
                Sensitivity_Test_test_beginyear = Sensitivity_Test_test_beginyears[iyear]
                Sensitivity_Test_test_endyear   = Sensitivity_Test_test_endyears[iyear]
                masked_array   = Get_masked_array_index(masked_site_index=masked_array_index,kfold=kfold,sitesnumber=sitesnumber)
                test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors, regional_number = calculate_Statistics_results(test_beginyear=Sensitivity_Test_test_beginyear, test_endyear=Sensitivity_Test_test_endyear,
                                                                                                                    final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                    geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                    training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,
                                                                                                                    masked_array_index=masked_array,Area=iregion,sitesnumber=sitesnumber,init_masked_index=masked_array_index) 
            
                if sensitivity_test_type == 'exclusion':
                    txt_outfile =  txtfile_outdir + 'Sensitivity_Tests_{}-{}_{}_{}_{}_{}Channel_{}x{}{}_Exclude{}.csv'.format(Sensitivity_Test_test_beginyear,Sensitivity_Test_test_endyear,typeName,species,version,nchannel,width,height,special_name,sensitivity_test_names_suffix)
                else:
                    txt_outfile =  txtfile_outdir + 'Sensitivity_Tests_{}-{}_{}_{}_{}_{}Channel_{}x{}{}_Include{}.csv'.format(Sensitivity_Test_test_beginyear,Sensitivity_Test_test_endyear,typeName,species,version,nchannel,width,height,special_name,sensitivity_test_names_suffix)
                SensitivityTests_output_text(outfile=txt_outfile,status='a',Area=iregion, test_beginyears=Sensitivity_Test_test_beginyear,test_endyears=Sensitivity_Test_test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                            slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names=sensitivity_test_names_suffix,regional_number=regional_number)
        
    save_sensitivity_test_loss_accuracy(model_outdir=model_outdir,loss=Training_losses_recording, accuracy=Training_acc_recording,valid_loss=valid_losses_recording, valid_accuracy=valid_acc_recording,typeName=typeName,
                       version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix,)
    final_longterm_data, obs_longterm_data = get_annual_longterm_array(beginyear=Sensitivity_Test_test_beginyear, endyear=Sensitivity_Test_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording,sitesnumber=sitesnumber,kfold=Sensitivity_Test_kfold)
    save_sensitivity_test_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix,)
           
    for imonth in range(len(MONTH)):
        final_longterm_data, obs_longterm_data = get_monthly_longterm_array(beginyear=Sensitivity_Test_test_beginyear, imonth=imonth,endyear=Sensitivity_Test_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording,sitesnumber=sitesnumber,kfold=Sensitivity_Test_kfold)
        save_sensitivity_test_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height,sensitivity_test_type=sensitivity_test_type,sensitivity_variables_names_suffix=sensitivity_test_names_suffix)
      
    return

