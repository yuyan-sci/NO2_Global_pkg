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
from Training_pkg.Model_Func import train, predict, train_lightgbm, predict_lightgbm
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_start_end_YEAR
from Training_pkg.Statistic_Func import regress2, linear_regression, Cal_RMSE
from Training_pkg.Net_Construction import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import *
from Evaluation_pkg.iostream import *

from visualization_pkg.Assemble_Func import plot_save_loss_accuracy_figure
from visualization_pkg.Addtional_Plot_Func import plot_BLCO_test_train_buffers
from visualization_pkg.utils import *

def BLCO_AVD_forRawData_Spatial_CrossValidation(buffer_radius, BLCO_kfold, width, height, sitesnumber, start_YYYY, TrainingDatasets, total_channel_names, main_stream_channel_names, side_stream_channel_names):
    # *------------------------------------------------------------------------------*#
    ## Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    beginyears = BLCO_beginyears
    endyears   = BLCO_endyears
    training_months = BLCO_training_months
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
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording = initialize_AVD_DataRecording(beginyear=beginyears[0],endyear=endyears[-1])
    Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording = initialize_Loss_Accuracy_Recordings(kfolds=BLCO_kfold,n_models=len(beginyears)*len(training_months),epoch=epoch,batchsize=batchsize)
    test_sites_index_recording, train_sites_index_recording, excluded_sites_index_recording, testsites2trainsites_nearest_distances = initialize_BLCO_SitesFoldsRecording(beginyear=beginyears[0],endyear=endyears[-1])
    
    
    ## Initialize the index for BLCO
    ## If index == 1.0, then it is the test site.
    ## If index == -1.0, then it is the training site.
    ## If index == 0.0, then it is the excluded site.
    ## We have different sites used in different time range. 
    index_for_BLCO = np.zeros((BLCO_kfold,len(lat),len(beginyears),len(training_months)),dtype=np.int32)

    for imodel_year in range(len(beginyears)):
        for imodel_month in range(len(training_months)):
            total_months = (endyears[imodel_year]-beginyears[imodel_year]+1)*len(training_months[imodel_month])
            temp_observation_data = np.zeros((total_months,sitesnumber),dtype=np.int32)
            for iyear in range((endyears[imodel_year]-beginyears[imodel_year]+1)):
                for imonth in range(len(training_months[imodel_month])):
                    temp_observation_data[iyear*len(training_months)+imonth,:] = SPECIES_OBS[sitesnumber*((beginyears[imodel_year]+iyear-2005)*12+training_months[imodel_month][imonth]):
                                                                                                sitesnumber*((beginyears[imodel_year]+iyear-2005)*12+training_months[imodel_month][imonth]+1)]
                    ## Get the sites that have valid values during this time period.
                    valid_sites_index = ~np.all(np.isnan(temp_observation_data),axis=0)
                    temp_index_of_initial_array = np.where(valid_sites_index)[0]

            imodel_lat = lat[valid_sites_index]
            imodel_lon = lon[valid_sites_index]
            nearest_distances = np.array([],dtype=np.float32)
            for isite in range(len(imodel_lat)):
                ## get the nearest distance for each site to the rest of the sites.
                site_distances = calculate_distance_forArray(site_lat=imodel_lat[isite],site_lon=imodel_lon
                                                                [isite],SATLAT_MAP=imodel_lat,SATLON_MAP=imodel_lon)
                nearest_distances = np.append(nearest_distances,np.min(site_distances[np.where(site_distances>0.1)]))
            
            if utilize_self_isolated_sites:
                ## Get the index for Self-Isolated sites (have distances larger than buffer radius naturally) 
                ## and sites for BLeCO.
                Self_Isolated_sites_index = np.where(nearest_distances>=buffer_radius)[0]
                Sites_forBLeCO_index      = np.where(nearest_distances<buffer_radius)[0]
                
                self_isolated_fold_count = 0
                length_of_Self_Isolated_sites_index = len(Self_Isolated_sites_index)

                ## If there are Self-Isolated sites, then we need to split the Self-Isolated sites into different folds.
                if len(Self_Isolated_sites_index) > 0:

                    ## If the number of Self-Isolated sites is less than the number of BLCO_kfold, then we need to
                    ## split the Self-Isolated sites into different folds one by one.
                    if len(Self_Isolated_sites_index) < BLCO_kfold:
                        ## Here we append -999 to the Self_Isolated_sites_index to make the length of Self_Isolated_sites_index
                        ## equal to BLCO_kfold.
                        for i in range(BLCO_kfold - length_of_Self_Isolated_sites_index):
                            Self_Isolated_sites_index = np.append(Self_Isolated_sites_index,-999)

                        rkf = RepeatedKFold(n_splits=BLCO_kfold, n_repeats=repeats, random_state=seed)
                        for train_index, test_index in rkf.split(Self_Isolated_sites_index):
                            ## If the test_index is -999, then we set all self-Isolated sites for training index for this fold.
                            if Self_Isolated_sites_index[test_index] == -999:
                                temp_train_index = np.where(Self_Isolated_sites_index!=-999)
                                print(test_index,temp_train_index,Self_Isolated_sites_index[temp_train_index])
                                index_for_BLCO[self_isolated_fold_count,temp_index_of_initial_array[Self_Isolated_sites_index[temp_train_index]],imodel_year,imodel_month] = -1.0
                                self_isolated_fold_count += 1
                            
                            ## If this site is selected as the test site, then we set the index to 1.0. And we set the
                            ## rest of the sites (not filled sites, e.g., not equal to -999.0) as the training sites.
                            else:
                                temp_train_index = np.where(Self_Isolated_sites_index[train_index]!=-999)
                                print(test_index,train_index[temp_train_index],Self_Isolated_sites_index[train_index[temp_train_index]])
                                Self_Isolated_sites_index = Self_Isolated_sites_index.astype(int)
                                index_for_BLCO[self_isolated_fold_count,temp_index_of_initial_array[Self_Isolated_sites_index[test_index]],imodel_year,imodel_month]  = 1.0
                                index_for_BLCO[self_isolated_fold_count,temp_index_of_initial_array[Self_Isolated_sites_index[train_index[temp_train_index]]],imodel_year,imodel_month] = -1.0
                                self_isolated_fold_count += 1

                    ## If the number of Self-Isolated sites is larger than the number of BLCO_kfold, then we can split the
                    ## Self-Isolated sites into different folds directly, just like normal sppatial cross-validation.
                    else:
                        rkf = RepeatedKFold(n_splits=BLCO_kfold, n_repeats=repeats, random_state=seed)
                        for train_index, test_index in rkf.split(Self_Isolated_sites_index):
                            Self_Isolated_sites_index = Self_Isolated_sites_index.astype(int)
                            index_for_BLCO[self_isolated_fold_count,temp_index_of_initial_array[Self_Isolated_sites_index[test_index]],imodel_year,imodel_month]  = 1.0
                            index_for_BLCO[self_isolated_fold_count,temp_index_of_initial_array[Self_Isolated_sites_index[train_index]],imodel_year,imodel_month] = -1.0
                            self_isolated_fold_count += 1

                if len(Sites_forBLeCO_index) > 0:
                    ## If there are sites for BLeCO, then we need to split the sites for BLeCO into different folds using
                    ## the function derive_Test_Training_index_4Each_BLCO_fold.
                    Only_BLeCO_index = derive_Test_Training_index_4Each_BLCO_fold(kfolds=BLCO_kfold,number_of_SeedClusters=BLCO_seeds_number,site_lat=imodel_lat[Sites_forBLeCO_index],site_lon=imodel_lon[Sites_forBLeCO_index],
                                                                        BLCO_Buffer_Size=buffer_radius)
                    for ifold in range(BLCO_kfold):
                        
                        index_for_BLCO[ifold,temp_index_of_initial_array[Sites_forBLeCO_index[np.where(Only_BLeCO_index[ifold,:]==1.0)]],imodel_year,imodel_month] = 1.0
                        index_for_BLCO[ifold,temp_index_of_initial_array[Sites_forBLeCO_index[np.where(Only_BLeCO_index[ifold,:]==-1.0)]],imodel_year,imodel_month] = -1.0
                        
            else:
                Only_BLeCO_index = derive_Test_Training_index_4Each_BLCO_fold(kfolds=BLCO_kfold,number_of_SeedClusters=BLCO_seeds_number,site_lat=imodel_lat,site_lon=imodel_lon,
                                                                        BLCO_Buffer_Size=buffer_radius)
                
                for ifold in range(BLCO_kfold):
                        index_for_BLCO[ifold,temp_index_of_initial_array[np.where(Only_BLeCO_index[ifold,:]==1.0)],imodel_year,imodel_month] = 1.0
                        index_for_BLCO[ifold,temp_index_of_initial_array[np.where(Only_BLeCO_index[ifold,:]==-1.0)],imodel_year,imodel_month] = -1.0

    test_index_number = np.array([],dtype = int)
    train_index_number = np.array([],dtype=int)
      

    
    if not BLCO_Spatial_CV_test_only_Switch:
        for imodel_year in range(len(beginyears)):
                Normalized_TrainingData = get_trainingdata_within_start_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)
                for imodel_month in range(len(training_months)):
                    for ifold in range(BLCO_kfold):
                        test_index = np.where(index_for_BLCO[ifold,:,imodel_year,imodel_month] == 1.0)[0]
                        train_index = np.where(index_for_BLCO[ifold,:,imodel_year,imodel_month] == -1.0)[0]
                        excluded_index = np.where(index_for_BLCO[ifold,:,imodel_year,imodel_month] == 0.0)[0]
                        
                        
                        test_index_number = np.append(test_index_number,len(test_index))
                        train_index_number = np.append(train_index_number,len(train_index))
                        print('Buffer Size: {} km,No.{}-fold, test_index #: {}, train_index #: {}, total # of sites: {}'.format(buffer_radius,ifold+1,len(test_index),len(train_index),len(lat)))

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
                            from Training_pkg.Net_Construction import LightGBMModel
                            model = LightGBMModel()
                            
                            print(f'Training LightGBM model for fold {ifold}, buffer {buffer_radius}km, year {beginyears[imodel_year]}-{endyears[imodel_year]}, months {training_months[imodel_month]}')
                            
                            # Train model
                            train_loss, train_acc, valid_losses, test_acc = train_lightgbm(
                                model=model,
                                X_train=X_train[train_mask,:,:,:], 
                                y_train=y_train[train_mask], 
                                X_test=X_test[test_mask,:,:,:],
                                y_test=y_test[test_mask], 
                                input_std=input_std,
                                input_mean=input_mean,
                                mean=mean, 
                                std=std, 
                                width=width,
                                height=height,
                                initial_channel_names=total_channel_names,
                                main_stream_channels=main_stream_channel_names,
                                side_stream_channels=side_stream_channel_names
                            )
                            
                            # Record losses and accuracies
                            Training_losses_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(train_loss)] = train_loss
                            Training_acc_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(train_acc)] = train_acc
                            valid_losses_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(valid_losses)] = valid_losses
                            valid_acc_recording[ifold, imodel_year*len(training_months)+imodel_month, 0:len(test_acc)] = test_acc
                            
                            # Save model
                            save_trained_month_based_BLCO_lightgbm_model(lgb_model=model, model_outdir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height,buffer_radius=buffer_radius)
                        else:
                            cnn_model = initial_network(width=width,main_stream_nchannel=len(main_stream_channel_names),
                                                        side_stream_nchannel=len(side_stream_channel_names))
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
                            save_trained_month_based_BLCO_model(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height,buffer_radius=buffer_radius)
                        
                        if LightGBM_setting:
                            model = load_trained_month_based_BLCO_lightgbm_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height,buffer_radius=buffer_radius)
                        else:    
                            cnn_model = load_trained_month_based_BLCO_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=ifold, width=width, height=height,buffer_radius=buffer_radius)
                        
                        temp_testsites2trainsites_nearest_distances = np.full(len(site_index),np.nan,dtype=np.float32)
                        for isite in range(len(test_index)):
                            site_distances = calculate_distance_forArray(site_lat=lat[test_index[isite]],site_lon=lon[test_index[isite]],SATLAT_MAP=lat[train_index],SATLON_MAP=lon[train_index])
                            temp_testsites2trainsites_nearest_distances[test_index[isite]] = np.min(site_distances[np.where(site_distances>0.1)])
                        
        
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
                            temp_test_sites_index = np.full(len(site_index),np.nan)
                            temp_train_sites_index = np.full(len(site_index),np.nan)
                            temp_excluded_sites_index = np.full(len(site_index),np.nan)
                            temp_test_sites_index[test_index] = test_index
                            temp_train_sites_index[train_index] = train_index
                            temp_excluded_sites_index[excluded_index] = excluded_index
                            
                            Validation_obs_data[nonan_yearly_test_predictor_datasets_within_time_periods_indices]   = SPECIES_OBS[yearly_test_Yindex[nonan_yearly_test_index]]
                            Training_obs_data[nonan_yearly_train_predictors_datasets_within_time_periods_indices]   = SPECIES_OBS[yearly_train_Yindex[nonan_yearly_train_index]]
                            Geophysical_test_data[nonan_yearly_test_predictor_datasets_within_time_periods_indices] = geophysical_species[yearly_test_Yindex[nonan_yearly_test_index]]
                            population_test_data[nonan_yearly_test_predictor_datasets_within_time_periods_indices]  = population_data[yearly_test_Yindex[nonan_yearly_test_index]]

                            if LightGBM_setting:
                                Validation_Prediction[nonan_yearly_test_predictor_datasets_within_time_periods_indices] = predict_lightgbm(inputarray=yearly_test_input[nonan_yearly_test_index,:,:,:], model=model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                                Training_Prediction[nonan_yearly_train_predictors_datasets_within_time_periods_indices] = predict_lightgbm(inputarray=yearly_train_input[nonan_yearly_train_index,:,:,:], model=model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
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
                            ### These recordings each year, each month has an array with the length of the kfold times the number of total sites.
                            ### Not interested points are filled with nan.
                            for imonth in range(len(training_months[imodel_month])):
                                final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]              = np.append(final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], final_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Validation_obs_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Geophysical_test_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]     = np.append(training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], train_final_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]       = np.append(training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Training_obs_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]] = np.append(testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], population_test_data[imonth*len(site_index):(imonth+1)*len(site_index)])
                                test_sites_index_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]       = np.append(test_sites_index_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], temp_test_sites_index)
                                train_sites_index_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]      = np.append(train_sites_index_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], temp_train_sites_index)
                                excluded_sites_index_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]   = np.append(excluded_sites_index_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], temp_excluded_sites_index)
                                testsites2trainsites_nearest_distances[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]] = np.append(testsites2trainsites_nearest_distances[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], temp_testsites2trainsites_nearest_distances)
                        if Test_Train_Buffers_Distributions_plot:
                            if utilize_self_isolated_sites:
                                fig_outdir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-SelfIsolated_BLCO_Sites-Buffers-Distributions/Buffer-{}km/'.format(species, version,buffer_radius)
                            else:
                                fig_outdir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-BLCO_Sites-Buffers-Distributions/Buffer-{}km/'.format(species, version,buffer_radius)
                            if not os.path.isdir(fig_outdir):
                                os.makedirs(fig_outdir)
                            fig_outfile = fig_outdir + 'Buffer-{}km_Total-{}folds_Total-{}ClustersSeeds-No.{}-fold_BLCO_Sites-Buffers-Distributions.png'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,ifold)
                            plot_BLCO_test_train_buffers(train_index=train_index,test_index=test_index,excluded_index=excluded_index,sitelat=lat,sitelon=lon,
                                                        buffer_radius=buffer_radius,extent=BLCO_plot_extent,fig_outfile=fig_outfile)
        
        save_month_based_BLCO_data_recording(obs_data=obs_data_recording,final_data=final_data_recording,geo_data_recording=geo_data_recording,training_final_data_recording=training_final_data_recording,
                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,
                                                lat_recording=lat,lon_recording=lon,testsites2trainsites_nearest_distances=testsites2trainsites_nearest_distances,
                                                test_sites_index_recording=test_sites_index_recording,train_sites_index_recording=train_sites_index_recording,excluded_sites_index_recording=excluded_sites_index_recording,train_index_number=train_index_number,test_index_number=test_index_number,
                                            species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius,BLCO_kfold=BLCO_kfold,BLCO_seeds_number=BLCO_seeds_number)

                
                        
    obs_data_recording, final_data_recording, geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording,lat_test_recording, lon_test_recording,testsites2trainsites_nearest_distances,test_sites_index_recording,train_sites_index_recording,excluded_sites_index_recording,train_index_number, test_index_number = load_month_based_BLCO_data_recording(species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius,BLCO_kfold=BLCO_kfold,BLCO_seeds_number=BLCO_seeds_number)
    if utilize_self_isolated_sites:
        txtfile_outdir = txt_outdir + '{}/{}/Results/results-SelfIsolated_BLCOCV/statistical_indicators/{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}/'.format(species, version,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
    else:
        txtfile_outdir = txt_outdir + '{}/{}/Results/results-BLCOCV/statistical_indicators/{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}/'.format(species, version,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    for iyear in range(len(BLCO_test_beginyears)):
        BLCO_test_beginyear = BLCO_test_beginyears[iyear]
        BLCO_test_endyear   = BLCO_test_endyears[iyear]
        masked_array = Get_masked_array_index(masked_site_index=site_index,kfold=kfold,sitesnumber=sitesnumber)
        
        test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors,regional_number = calculate_Statistics_results(test_beginyear=BLCO_test_beginyear, test_endyear=BLCO_test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,masked_array_index=masked_array,Area='Global',init_masked_index=site_index,sitesnumber=sitesnumber)
        
        if utilize_self_isolated_sites:
            txt_outfile =  txtfile_outdir + 'SelfIsolated_BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(BLCO_test_beginyear,BLCO_test_endyear,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
        else:
            txt_outfile =  txtfile_outdir + 'BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(BLCO_test_beginyear,BLCO_test_endyear,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
        Output_Text_Sites_Number(outfile=txt_outfile, status='w', train_index_number=train_index_number, test_index_number=test_index_number, buffer=buffer_radius)
        AVD_output_text(outfile=txt_outfile,status='a',Area='Global',test_beginyears=BLCO_test_beginyear,test_endyears=BLCO_test_endyear, test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                        slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)
        
    for iregion in BLCO_additional_test_regions:
        mask_map, mask_lat, mask_lon = load_Global_Mask_data(region_name=iregion)
        masked_array_index = find_masked_latlon(mask_map=mask_map,mask_lat=mask_lat,mask_lon=mask_lon,test_lat=lat_test_recording,test_lon=lon_test_recording)
        for iyear in range(len(BLCO_test_beginyears)):
            BLCO_test_beginyear = BLCO_test_beginyears[iyear]
            BLCO_test_endyear   = BLCO_test_endyears[iyear]
            masked_array   = Get_masked_array_index(masked_site_index=masked_array_index,kfold=kfold,sitesnumber=sitesnumber)
                
            test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors, regional_number = calculate_Statistics_results(test_beginyear=BLCO_test_beginyear, test_endyear=BLCO_test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,masked_array_index=masked_array,Area=iregion,init_masked_index=masked_array_index,sitesnumber=sitesnumber)
            if utilize_self_isolated_sites:
                txt_outfile =  txtfile_outdir + 'SelfIsolated_BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(BLCO_test_beginyear,BLCO_test_endyear,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
            else:
                txt_outfile =  txtfile_outdir + 'BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(BLCO_test_beginyear,BLCO_test_endyear,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
            AVD_output_text(outfile=txt_outfile,status='a', Area=iregion,test_beginyears=BLCO_test_beginyear,test_endyears=BLCO_test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                                slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)

    save_BLCO_loss_accuracy(model_outdir=model_outdir,loss=Training_losses_recording, accuracy=Training_acc_recording,valid_loss=valid_losses_recording, valid_accuracy=valid_acc_recording,typeName=typeName,
                       version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height,buffer_radius=buffer_radius)
    final_longterm_data, obs_longterm_data = get_annual_longterm_array(beginyear=BLCO_test_beginyear, endyear=BLCO_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording,sitesnumber=sitesnumber,kfold=BLCO_kfold)
    save_BLCO_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius)
           
    for imonth in range(len(MONTH)):
        final_longterm_data, obs_longterm_data = get_monthly_longterm_array(beginyear=BLCO_test_beginyear, imonth=imonth,endyear=BLCO_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording,sitesnumber=sitesnumber,kfold=BLCO_kfold)
        save_BLCO_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius)
      
    return



def BLCO_AVD_Spatial_CrossValidation(buffer_radius, BLCO_kfold, width, height, sitesnumber, start_YYYY, TrainingDatasets, total_channel_names, main_stream_channel_names, side_stream_channel_names):
    # *------------------------------------------------------------------------------*#
    ## Initialize the array, variables and constants.
    # *------------------------------------------------------------------------------*#
    ### Get training data, label data, initial observation data and geophysical species
    beginyears = BLCO_beginyears
    endyears   = BLCO_endyears
    training_months = BLCO_training_months
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
    # *------------------------------------------------------------------------------*#
    ## Begining the Cross-Validation.
    ## Multiple Models will be trained in each fold.
    # *------------------------------------------------------------------------------*#
    final_data_recording, obs_data_recording, geo_data_recording, testing_population_data_recording, training_final_data_recording, training_obs_data_recording, training_dataForSlope_recording = initialize_AVD_DataRecording(beginyear=beginyears[0],endyear=endyears[-1])
    Training_losses_recording, Training_acc_recording, valid_losses_recording, valid_acc_recording = initialize_Loss_Accuracy_Recordings(kfolds=BLCO_kfold,n_models=len(beginyears)*len(training_months),epoch=epoch,batchsize=batchsize)
    lat_test_recording = np.array([],dtype=np.float32)
    lon_test_recording = np.array([],dtype=np.float32)
    test_sites_index_recording = {}
    train_sites_index_recording = {}
    excluded_sites_index_recording = {}

    if utilize_self_isolated_sites:
        nearest_distances = np.array([],dtype=np.float32)
        for isite in range(len(lat)):
            site_distances = calculate_distance_forArray(site_lat=lat[isite],site_lon=lon[isite],SATLAT_MAP=lat,SATLON_MAP=lon)
            nearest_distances = np.append(nearest_distances,np.min(site_distances[np.where(site_distances>0.1)]))
        site_index = np.array(range(len(lat)))
        index_for_BLCO = np.zeros((BLCO_kfold,len(lat)),dtype=np.int32)
        Self_Isolated_sites_index = np.where(nearest_distances>=buffer_radius)[0]
        Sites_forBLeCO_index      = np.where(nearest_distances<buffer_radius)[0]
        self_isolated_fold_count = 0
        length_of_Self_Isolated_sites_index = len(Self_Isolated_sites_index)
        if len(Self_Isolated_sites_index) > 0:
            if len(Self_Isolated_sites_index) < BLCO_kfold:
                for i in range(BLCO_kfold - length_of_Self_Isolated_sites_index):
                    Self_Isolated_sites_index = np.append(Self_Isolated_sites_index,-999)
                rkf = RepeatedKFold(n_splits=BLCO_kfold, n_repeats=repeats, random_state=seed)
                for train_index, test_index in rkf.split(Self_Isolated_sites_index):
                    if Self_Isolated_sites_index[test_index] == -999:
                        temp_train_index = np.where(Self_Isolated_sites_index!=-999)
                        print(test_index,temp_train_index,Self_Isolated_sites_index[temp_train_index])
                        index_for_BLCO[self_isolated_fold_count,Self_Isolated_sites_index[temp_train_index]] = -1.0
                        self_isolated_fold_count += 1
                    else:
                        temp_train_index = np.where(Self_Isolated_sites_index[train_index]!=-999)
                        print(test_index,train_index[temp_train_index],Self_Isolated_sites_index[train_index[temp_train_index]])
                        Self_Isolated_sites_index = Self_Isolated_sites_index.astype(int)
                        index_for_BLCO[self_isolated_fold_count,Self_Isolated_sites_index[test_index]]  = 1.0
                        index_for_BLCO[self_isolated_fold_count,Self_Isolated_sites_index[train_index[temp_train_index]]] = -1.0
                        self_isolated_fold_count += 1
            else:
                rkf = RepeatedKFold(n_splits=BLCO_kfold, n_repeats=repeats, random_state=seed)
                for train_index, test_index in rkf.split(Self_Isolated_sites_index):
                    Self_Isolated_sites_index = Self_Isolated_sites_index.astype(int)
                    index_for_BLCO[self_isolated_fold_count,Self_Isolated_sites_index[test_index]]  = 1.0
                    index_for_BLCO[self_isolated_fold_count,Self_Isolated_sites_index[train_index]] = -1.0
                    self_isolated_fold_count += 1
        if len(Sites_forBLeCO_index) > 0:
            Only_BLeCO_index = derive_Test_Training_index_4Each_BLCO_fold(kfolds=BLCO_kfold,number_of_SeedClusters=BLCO_seeds_number,site_lat=lat[Sites_forBLeCO_index],site_lon=lon[Sites_forBLeCO_index],
                                                                BLCO_Buffer_Size=buffer_radius)
            for ifold in range(BLCO_kfold):
                
                index_for_BLCO[ifold,Sites_forBLeCO_index[np.where(Only_BLeCO_index[ifold,:]==1.0)]] = 1.0
                index_for_BLCO[ifold,Sites_forBLeCO_index[np.where(Only_BLeCO_index[ifold,:]==-1.0)]] = -1.0
                
    else:
        index_for_BLCO = derive_Test_Training_index_4Each_BLCO_fold(kfolds=BLCO_kfold,number_of_SeedClusters=BLCO_seeds_number,site_lat=lat,site_lon=lon,
                                                                BLCO_Buffer_Size=buffer_radius)
    
    test_index_number = np.array([],dtype = int)
    train_index_number = np.array([],dtype=int)
    testsites2trainsites_nearest_distances =  np.array([],dtype=np.float32)
    if not BLCO_Spatial_CV_test_only_Switch:
        for ifold in range(BLCO_kfold):
            
            count = ifold
            test_index = np.where(index_for_BLCO[ifold,:] == 1.0)[0]
            train_index = np.where(index_for_BLCO[ifold,:] == -1.0)[0]
            excluded_index = np.where(index_for_BLCO[ifold,:] == 0.0)[0]
            test_index_number = np.append(test_index_number,len(test_index))
            train_index_number = np.append(train_index_number,len(train_index))
            lat_test_recording = np.append(lat_test_recording,lat[test_index])
            lon_test_recording = np.append(lon_test_recording,lon[test_index])
            test_sites_index_recording[str(ifold)] = test_index
            train_sites_index_recording[str(ifold)] = train_index
            excluded_sites_index_recording[str(ifold)] = excluded_index
            for isite in range(len(test_index)):
                site_distances = calculate_distance_forArray(site_lat=lat[test_index[isite]],site_lon=lon[test_index[isite]],SATLAT_MAP=lat[train_index],SATLON_MAP=lon[train_index])
                testsites2trainsites_nearest_distances = np.append(testsites2trainsites_nearest_distances,np.min(site_distances[np.where(site_distances>0.1)]))
            print('Buffer Size: {} km,No.{}-fold, test_index #: {}, train_index #: {}, total # of sites: {}'.format(buffer_radius,ifold+1,len(test_index),len(train_index),len(lat)))
            for imodel_year in range(len(beginyears)):
                Normalized_TrainingData = get_trainingdata_within_start_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)

                for imodel_month in range(len(training_months)):
                
                    X_Training_index, X_Testing_index, Y_Training_index, Y_Testing_index = Get_month_based_XY_indices(train_index=train_index,test_index=test_index,model_beginyear=beginyears[imodel_year],beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_index=training_months[imodel_month], sitesnumber=sitesnumber)
                    X_train, X_test, y_train, y_test = Get_XY_arraies(Normalized_TrainingData=Normalized_TrainingData,true_input=true_input,X_Training_index=X_Training_index,X_Testing_index=X_Testing_index,Y_Training_index=Y_Training_index,Y_Testing_index=Y_Testing_index)
                    #print('X_train size: {}, X_test size: {}, y_train size: {}, y_test size: {} -------------------------------------------'.format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
                    # *------------------------------------------------------------------------------*#
                    ## Training Process.
                    # *------------------------------------------------------------------------------*#
                    train_mask = np.where(~np.isnan(y_train))[0]
                    test_mask  = np.where(~np.isnan(y_test))[0]
                    cnn_model = initial_network(width=width,main_stream_nchannel=len(main_stream_channel_names),side_stream_nchannel=len(side_stream_channel_names))

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    cnn_model.to(device)
                    torch.manual_seed(21)
                    train_loss, train_acc, valid_losses, test_acc  = train(model=cnn_model, X_train=X_train[train_mask,:,:,:], y_train=y_train[train_mask], X_test=X_test[test_mask,:,:,:],
                                                                            y_test=y_test[test_mask], input_std=input_std,input_mean=input_mean,mean=mean, std=std, width=width,height=height,BATCH_SIZE=batchsize, learning_rate=lr0, TOTAL_EPOCHS=epoch,
                                                                    initial_channel_names=total_channel_names,main_stream_channels=main_stream_channel_names,side_stream_channels=side_stream_channel_names)
                    Training_losses_recording[count,imodel_year*len(training_months)+imodel_month,0:len(train_loss)] = train_loss
                    Training_acc_recording[count,imodel_year*len(training_months)+imodel_month,0:len(train_acc)]    = train_acc
                    valid_losses_recording[count,imodel_year*len(training_months)+imodel_month,0:len(valid_losses)]  = valid_losses
                    valid_acc_recording[count,imodel_year*len(training_months)+imodel_month,0:len(test_acc)]    = test_acc 

                    save_trained_month_based_BLCO_model(cnn_model=cnn_model, model_outdir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height,buffer_radius=buffer_radius)
                for iyear in range((endyears[imodel_year]-beginyears[imodel_year]+1)):
                    for imodel_month in range(len(training_months)):
                        yearly_test_index   = Get_month_based_XIndex(index=test_index, beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_train_index  = Get_month_based_XIndex(index=train_index, beginyear=(beginyears[imodel_year]+iyear),endyear=(beginyears[imodel_year]+iyear),month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_test_Yindex  = Get_month_based_YIndex(index=test_index,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_train_Yindex = Get_month_based_YIndex(index=train_index,beginyear=(beginyears[imodel_year]+iyear), endyear=(beginyears[imodel_year]+iyear), month_index=training_months[imodel_month],sitenumber=sitesnumber)
                        yearly_test_input  = Normalized_TrainingData[yearly_test_index,:,:,:]
                        yearly_train_input = Normalized_TrainingData[yearly_train_index,:,:,:]
                        Validation_Prediction = np.full(len(yearly_test_index),np.nan)
                        Training_Prediction   = np.full(len(yearly_train_index),np.nan)
                        Validation_obs_data   = SPECIES_OBS[yearly_test_Yindex]
                        Training_obs_data     = SPECIES_OBS[yearly_train_Yindex]
                        nonan_yearly_test_index  = np.where(~np.isnan(Validation_obs_data))[0]
                        nonan_yearly_train_index = np.where(~np.isnan(Training_obs_data))[0]

                        cnn_model = load_trained_month_based_BLCO_model(model_indir=model_outdir, typeName=typeName,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year], month_index=training_months[imodel_month], version=version, species=species, nchannel=nchannel, special_name=special_name, count=count, width=width, height=height,buffer_radius=buffer_radius)
                        Validation_Prediction[nonan_yearly_test_index] = predict(inputarray=yearly_test_input[nonan_yearly_test_index,:,:,:], model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                        Training_Prediction[nonan_yearly_train_index]   = predict(inputarray=yearly_train_input[nonan_yearly_train_index,:,:,:],  model=cnn_model, batchsize=3000, initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_stream_channel_names)
                        final_data = Get_final_output(Validation_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std,yearly_test_Yindex)
                        train_final_data = Get_final_output(Training_Prediction, geophysical_species,bias,normalize_bias,normalize_species,absolute_species,log_species,mean, std,yearly_train_Yindex)
                    
                        if combine_with_GeophysicalSpeceis_Switch:
                            nearest_distance = get_nearest_test_distance(area_test_index=test_index,area_train_index=train_index,site_lat=lat,site_lon=lon)
                            coeficient = get_coefficients(nearest_site_distance=nearest_distance,cutoff_size=cutoff_size,beginyear=beginyears[imodel_year],
                                                endyear = endyears[imodel_year],months=training_months[imodel_month])
                            final_data = (1.0-coeficient)*final_data + coeficient * geophysical_species[yearly_test_Yindex]
                        if ForcedSlopeUnity:
                            final_data = ForcedSlopeUnity_Func(train_final_data=train_final_data,train_obs_data=SPECIES_OBS[yearly_train_Yindex]
                                                    ,test_final_data=final_data,train_area_index=train_index,test_area_index=test_index,
                                                    endyear=beginyears[imodel_year]+iyear,beginyear=beginyears[imodel_year]+iyear,month_index=training_months[imodel_month],EachMonth=EachMonthForcedSlopeUnity)

                        # *------------------------------------------------------------------------------*#
                        ## Recording observation and prediction for this model this fold.
                        # *------------------------------------------------------------------------------*#

                        
                        Geophysical_test_data = geophysical_species[yearly_test_Yindex]
                        population_test_data  = population_data[yearly_test_Yindex]

                        for imonth in range(len(training_months[imodel_month])):
                            final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]              = np.append(final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], final_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Validation_obs_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]                = np.append(geo_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Geophysical_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                            training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]     = np.append(training_final_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], train_final_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                            training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]]       = np.append(training_obs_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], Training_obs_data[imonth*len(train_index):(imonth+1)*len(train_index)])
                            testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]] = np.append(testing_population_data_recording[str(beginyears[imodel_year]+iyear)][MONTH[training_months[imodel_month][imonth]]], population_test_data[imonth*len(test_index):(imonth+1)*len(test_index)])
                
               
            if Test_Train_Buffers_Distributions_plot:
                if utilize_self_isolated_sites:
                    fig_outdir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-SelfIsolated_BLCO_Sites-Buffers-Distributions/Buffer-{}km/'.format(species, version,buffer_radius)
                else:
                    fig_outdir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-BLCO_Sites-Buffers-Distributions/Buffer-{}km/'.format(species, version,buffer_radius)
                if not os.path.isdir(fig_outdir):
                    os.makedirs(fig_outdir)
                fig_outfile = fig_outdir + 'Buffer-{}km_Total-{}folds_Total-{}ClustersSeeds-No.{}-fold_BLCO_Sites-Buffers-Distributions.png'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,ifold)
                plot_BLCO_test_train_buffers(train_index=train_index,test_index=test_index,excluded_index=excluded_index,sitelat=lat,sitelon=lon,
                                            buffer_radius=buffer_radius,extent=BLCO_plot_extent,fig_outfile=fig_outfile)
        save_month_based_BLCO_data_recording(obs_data=obs_data_recording,final_data=final_data_recording,geo_data_recording=geo_data_recording,training_final_data_recording=training_final_data_recording,
                                             training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,
                                             lat_recording=lat_test_recording,lon_recording=lon_test_recording,testsites2trainsites_nearest_distances=testsites2trainsites_nearest_distances,
                                             test_sites_index_recording=test_sites_index_recording,train_sites_index_recording=train_sites_index_recording,excluded_sites_index_recording=excluded_sites_index_recording,train_index_number=train_index_number,test_index_number=test_index_number,
                                        species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius,BLCO_kfold=BLCO_kfold,BLCO_seeds_number=BLCO_seeds_number)

    obs_data_recording, final_data_recording, geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording,lat_test_recording, lon_test_recording,testsites2trainsites_nearest_distances,test_sites_index_recording,train_sites_index_recording,excluded_sites_index_recording,train_index_number, test_index_number = load_month_based_BLCO_data_recording(species=species,version=version,typeName=typeName,beginyear=beginyears[0],endyear=endyears[-1],nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius,BLCO_kfold=BLCO_kfold,BLCO_seeds_number=BLCO_seeds_number)
    if utilize_self_isolated_sites:
        txtfile_outdir = txt_outdir + '{}/{}/Results/results-SelfIsolated_BLCOCV/statistical_indicators/{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}/'.format(species, version,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
    else:
        txtfile_outdir = txt_outdir + '{}/{}/Results/results-BLCOCV/statistical_indicators/{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}/'.format(species, version,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
    if not os.path.isdir(txtfile_outdir):
        os.makedirs(txtfile_outdir)
    for iyear in range(len(BLCO_test_beginyears)):
        BLCO_test_beginyear = BLCO_test_beginyears[iyear]
        BLCO_test_endyear   = BLCO_test_endyears[iyear]
        
        test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors,regional_number = calculate_Statistics_results(test_beginyear=BLCO_test_beginyear, test_endyear=BLCO_test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,masked_array_index=np.array(range(len(lat_test_recording))),Area='North America')
        if utilize_self_isolated_sites:
            txt_outfile =  txtfile_outdir + 'SelfIsolated_BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(BLCO_test_beginyear,BLCO_test_endyear,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
        else:
            txt_outfile =  txtfile_outdir + 'BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(BLCO_test_beginyear,BLCO_test_endyear,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
        Output_Text_Sites_Number(outfile=txt_outfile, status='w', train_index_number=train_index_number, test_index_number=test_index_number, buffer=buffer_radius)
        AVD_output_text(outfile=txt_outfile,status='a',Area='North America',test_beginyears=BLCO_test_beginyear,test_endyears=BLCO_test_endyear, test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                        slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)
        
    for iregion in BLCO_additional_test_regions:
        mask_map, mask_lat, mask_lon = load_Global_Mask_data(region_name=iregion)
        masked_array_index = find_masked_latlon(mask_map=mask_map,mask_lat=mask_lat,mask_lon=mask_lon,test_lat=lat_test_recording,test_lon=lon_test_recording)
        for iyear in range(len(BLCO_test_beginyears)):
            BLCO_test_beginyear = BLCO_test_beginyears[iyear]
            BLCO_test_endyear   = BLCO_test_endyears[iyear]
            test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE, PWM_NRMSE, slope, PWAModel, PWAMonitors, regional_number = calculate_Statistics_results(test_beginyear=BLCO_test_beginyear, test_endyear=BLCO_test_endyear,
                                                                                                                final_data_recording=final_data_recording, obs_data_recording=obs_data_recording,
                                                                                                                geo_data_recording=geo_data_recording, training_final_data_recording=training_final_data_recording,
                                                                                                                training_obs_data_recording=training_obs_data_recording,testing_population_data_recording=testing_population_data_recording,masked_array_index=masked_array_index,Area=iregion)
            if utilize_self_isolated_sites:
                txt_outfile =  txtfile_outdir + 'SelfIsolated_BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(BLCO_test_beginyear,BLCO_test_endyear,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
            else:
                txt_outfile =  txtfile_outdir + 'BLCO-{}-{}_{}km-{}fold-{}ClusterSeeds-SpatialCV_{}_{}_{}_{}Channel_{}x{}{}.csv'.format(BLCO_test_beginyear,BLCO_test_endyear,buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName,species,version,nchannel,width,height,special_name)
            AVD_output_text(outfile=txt_outfile,status='a', Area=iregion,test_beginyears=BLCO_test_beginyear,test_endyears=BLCO_test_endyear,test_CV_R2=test_CV_R2, train_CV_R2=train_CV_R2, geo_CV_R2=geo_CV_R2, RMSE=RMSE, NRMSE=NRMSE,PMW_NRMSE=PWM_NRMSE,
                                slope=slope,PWM_Model=PWAModel,PWM_Monitors=PWAMonitors,regional_number=regional_number)

    save_BLCO_loss_accuracy(model_outdir=model_outdir,loss=Training_losses_recording, accuracy=Training_acc_recording,valid_loss=valid_losses_recording, valid_accuracy=valid_acc_recording,typeName=typeName,
                       version=version,species=species, nchannel=nchannel,special_name=special_name, width=width, height=height,buffer_radius=buffer_radius)
    final_longterm_data, obs_longterm_data = get_annual_longterm_array(beginyear=BLCO_test_beginyear, endyear=BLCO_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
    save_BLCO_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH='Annual',nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius)
           
    for imonth in range(len(MONTH)):
        final_longterm_data, obs_longterm_data = get_monthly_longterm_array(beginyear=BLCO_test_beginyear, imonth=imonth,endyear=BLCO_test_endyear, final_data_recording=final_data_recording,obs_data_recording=obs_data_recording)
        save_BLCO_data_recording(obs_data=obs_longterm_data,final_data=final_longterm_data,
                                species=species,version=version,typeName=typeName, beginyear='Alltime',MONTH=MONTH[imonth],nchannel=nchannel,special_name=special_name,width=width,height=height,buffer_radius=buffer_radius)
      
    return



def derive_Test_Training_index_4Each_BLCO_fold(kfolds, number_of_SeedClusters, site_lat, site_lon, BLCO_Buffer_Size):
    frac_testing  = 1.0/kfolds
    frac_training = 1.0 - frac_testing
    rkf = RepeatedKFold(n_splits=kfolds, n_repeats=1, random_state=20230130)
    number_of_test_sites = np.zeros((kfolds),dtype=np.int32)
    test_fold = 0
    for train_index, test_index in rkf.split(site_lat):
        number_of_test_sites[test_fold] = len(test_index)
        test_fold+=1
    # if # == -1   -> this site is for training for this fold, 
    # elif # == +1 -> this site is for testing for this fold.
    # elif # == 0  -> this site is exlcuded from training for this fold.
    index_for_BLCO = np.zeros((kfolds,len(site_lat)),dtype=np.int64) 

    # calculate local monitor density
    usite_density = np.zeros(len(site_lat),dtype=np.float64)

    for isite in range(len(site_lat)):
        temp_Distances = calculate_distance_forArray(site_lat=site_lat[isite],site_lon=site_lon[isite],SATLAT_MAP=site_lat,SATLON_MAP=site_lon)
        temp_Density   = len(np.where(temp_Distances < 200.0)[0])
        usite_density[isite] = temp_Density

    ispot = np.zeros((len(site_lat))) # record sites that are still available for selecting as test datasets.
    BLCO_criteria_radius = np.zeros((kfolds)) # this array is used to record the minimal criterial radius from sites to cluster seeds to select testing sites 
    # find stations that are still not withheld from selecting as the test sites.

    for ifold in range(kfolds):
        sites_unwithheld4testing = np.where(ispot == 0)[0].astype(int)
        sites_withheld4testing   = np.where(ispot > 0)[0].astype(int)

        # evenly divide stations by density, get the sites density limits by percentile
        density_percentile = np.percentile(usite_density[sites_unwithheld4testing], np.linspace(0,100,kfolds+1),interpolation='midpoint' )
        #randomly choose one stations within each density percentile range
        # cluster_seeds_index = np.full(number_of_SeedClusters,-1,dtype=np.int64)#np.zeros((len(density_percentile)-1),dtype=np.int64)
        #for icluster in range(len(cluster_seeds_index)):
            #sites_unwithheld4testing2 = np.intersect1d(np.where(usite_density[sites_unwithheld4testing]>=density_percentile[icluster]), np.where(usite_density[sites_unwithheld4testing]<=density_percentile[icluster+1]))
            
            #if len(sites_unwithheld4testing2)>0:
                
                #random_cluster_index      = np.random.randint(0,len(sites_unwithheld4testing2),1)
                #print(cluster_seeds_index,icluster,sites_unwithheld4testing2,sites_unwithheld4testing2[random_cluster_index])
                #while sites_unwithheld4testing2[random_cluster_index] in cluster_seeds_index:
                    #random_cluster_index      = np.random.randint(0,len(sites_unwithheld4testing2),1)
                #cluster_seeds_index[icluster] = sites_unwithheld4testing2[random_cluster_index].astype(int)
            #else:
               # None
        cluster_seeds_index = np.random.choice(range(len(sites_unwithheld4testing)), min(len(sites_unwithheld4testing),number_of_SeedClusters), replace=False)

        # --- print('sites_unwithheld4testing shape: {}, sites_unwithheld4testing 0:10 - {}, cluster_seeds_index[0:10]: {}'.format(sites_unwithheld4testing.shape,sites_unwithheld4testing[0:10],cluster_seeds_index[0:10]))
        
        # find distances between selected stations and other stations
        sites_unwithheld4testing_Distance = np.zeros((number_of_SeedClusters,len(sites_unwithheld4testing)))
        for icluster in range(len(cluster_seeds_index)):
            print('icluster: {}, \ncluster_seeds_index shape: {}, \nsites_unwithheld4testing shape:{}, \n site_lat shape:{}; site lon shape: {}, \nsites_unwithheld4testing_Distance shape:{}'.format(icluster,cluster_seeds_index.shape,sites_unwithheld4testing.shape,site_lat.shape,site_lon.shape,sites_unwithheld4testing_Distance.shape))
            temp_distance = calculate_distance_forArray(site_lat=site_lat[sites_unwithheld4testing[cluster_seeds_index[icluster]]],
                                                                                        site_lon=site_lon[sites_unwithheld4testing[cluster_seeds_index[icluster]]],
                                                                                        SATLAT_MAP=site_lat[sites_unwithheld4testing],SATLON_MAP=site_lon[sites_unwithheld4testing])
            sites_unwithheld4testing_Distance[icluster,:]= temp_distance
        # find the minimal distance of each sites to all seed clusters.

        Minimal_Distance2clusters = np.min(sites_unwithheld4testing_Distance,axis=0)
        Minimal_Distance2clusters_Sorted = np.sort(Minimal_Distance2clusters)
        
        ### calculate radius within which enough stations are located to fulfill this fold's quota.
        
        criterial_index = min(int(number_of_test_sites[ifold])-1,len(Minimal_Distance2clusters_Sorted)-1)
        BLCO_criteria_radius[ifold] = Minimal_Distance2clusters_Sorted[criterial_index]
        # store testing stations for this fold, find all sites with distances smaller than the criterial radius
        if criterial_index < number_of_SeedClusters:
            #print('ifold: ',ifold,cluster_seeds_index[0:criterial_index+1],criterial_index+1, len(Minimal_Distance2clusters_Sorted),)
            ispot[sites_unwithheld4testing[cluster_seeds_index[0:criterial_index+1]]]= ifold + 1
        else:    
            ispot[sites_unwithheld4testing[np.where(Minimal_Distance2clusters <= BLCO_criteria_radius[ifold] )]] = ifold + 1
        
        ifold_test_site_index       = np.where(ispot == (ifold+1))[0]
        ifold_init_train_site_index = np.where(ispot != (ifold+1))[0]

        ifold_train_site_index = GetBufferTrainingIndex(test_index=ifold_test_site_index,train_index=ifold_init_train_site_index,buffer=BLCO_Buffer_Size,sitelat=site_lat,sitelon=site_lon)
        index_for_BLCO[ifold,ifold_test_site_index]  = np.full((len(ifold_test_site_index)) , 1.0)
        index_for_BLCO[ifold,ifold_train_site_index] = np.full((len(ifold_train_site_index)),-1.0)

    return index_for_BLCO