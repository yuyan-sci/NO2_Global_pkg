import numpy as np
import time
import os
import gc
import netCDF4 as nc
from Estimation_pkg.utils import *
from Estimation_pkg.data_func import *
from Estimation_pkg.training_func import Train_Model_forEstimation
from Estimation_pkg.predict_func import map_predict,map_final_output, map_predict_LightGBM
from Estimation_pkg.iostream import save_ForcedSlopeUnity_final_map_data,load_ForcedSlope_forEstimation,load_map_data, load_trained_model_forEstimation,load_trained_month_based_model_forEstimation,save_final_map_data, load_estimation_map_data,save_combinedGeo_map_data

from Training_pkg.iostream import load_TrainingVariables
from Training_pkg.iostream import Learning_Object_Datasets, load_monthly_obs_data
from Training_pkg.data_func import normalize_Func
from Training_pkg.utils import *

from Evaluation_pkg.utils import *

def Estimation_Func(total_channel_names,mainstream_channel_names,side_channel_names):
    typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)

    if Train_model_Switch:
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=total_channel_names)
        Train_Model_forEstimation(train_beginyears=Training_beginyears,train_endyears=Training_endyears,training_months=Training_training_months,width=width,height=height,sitesnumber=sitesnumber,start_YYYY=start_YYYY,TrainingDatasets=TrainingDatasets,
                                  total_channel_names=total_channel_names,main_stream_channel_names=mainstream_channel_names,side_stream_nchannel_names=side_channel_names)
        
        del width, height, sitesnumber,start_YYYY, TrainingDatasets 
        gc.collect()
    
    if Map_estimation_Switch:
        SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
        width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=total_channel_names)
        Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets,observation_data=SPECIES_OBS)
        true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
    
        gc.collect()
        MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        MM = ['01','02','03','04','05','06','07','08','09','10','11','12']
        for imodel_year in range(len(Estiamtion_trained_beginyears)):
            for imodel_month in range(len(Estiamtion_trained_months)):
                model = load_trained_month_based_model_forEstimation(model_outdir=model_outdir,typeName=typeName,version=version,species=species, nchannel=len(total_channel_names),special_name=special_name,
                                                             beginyear=Estiamtion_trained_beginyears[imodel_year],endyear=Estiamtion_trained_endyears[imodel_year], month_index=Estiamtion_trained_months[imodel_month], width=width, height=height)
                if Estimation_ForcedSlopeUnity:
                    ForcedSlopeUnity_Dictionary_forEstimation = load_ForcedSlope_forEstimation(model_indir=model_outdir,typeName=typeName,version=version,species=species, nchannel=len(total_channel_names),special_name=special_name,
                                                             beginyear=Estiamtion_trained_beginyears[imodel_year],endyear=Estiamtion_trained_endyears[imodel_year], month_index=Estiamtion_trained_months[imodel_month], width=width, height=height)
                for YEAR in Estimation_years[imodel_year]:
                    for imonth in Estiamtion_months[imodel_month]:
                        print('YEAR: {}, MONTH: {}'.format(YEAR,MM[imonth]))
                        map_input = load_map_data(channel_names=total_channel_names,YYYY=YEAR,MM=MM[imonth])
                        if LightGBM_setting:
                            final_map_data = map_predict_LightGBM(inputmap=map_input,model=model,train_mean=input_mean,train_std=input_std,extent=Extent,width=width,nchannel=len(total_channel_names),YYYY=YEAR,MM=MM[imonth],total_channel_names=total_channel_names,main_stream_channel_names=mainstream_channel_names,side_channel_names=side_channel_names)
                        else:
                            final_map_data = map_predict(inputmap=map_input,model=model,train_mean=input_mean,train_std=input_std,extent=Extent,width=width,nchannel=len(total_channel_names),YYYY=YEAR,MM=MM[imonth],
                                                    total_channel_names=total_channel_names,main_stream_channel_names=mainstream_channel_names,side_channel_names=side_channel_names)
                        
                        
                        final_map_data = map_final_output(output=final_map_data,extent=Extent,YYYY=YEAR,MM=MM[imonth],SPECIES=species,bias=bias,
                                                        normalize_bias=normalize_bias,normalize_species=normalize_species,absolute_species=absolute_species,
                                                        log_species=log_species,mean=mean,std=std)
                        save_final_map_data(final_data=final_map_data,YYYY=YEAR,MM=MM[imonth],extent=Extent,SPECIES=species,version=version,special_name=special_name)
                        if Estimation_ForcedSlopeUnity:
                            temp_offset = ForcedSlopeUnity_Dictionary_forEstimation['offset'][str(YEAR)][MONTH[imonth]]
                            temp_slope  = ForcedSlopeUnity_Dictionary_forEstimation['slope'][str(YEAR)][MONTH[imonth]]
                            fill_mask = (final_map_data == -999.0)
                            final_map_data -= temp_offset
                            final_map_data /= temp_slope
                            final_map_data[fill_mask] = -999.0
                        save_ForcedSlopeUnity_final_map_data(final_data=final_map_data,YYYY=YEAR,MM=MM[imonth],extent=Extent,SPECIES=species,
                                                             version=version,special_name=special_name)
                        del map_input, final_map_data
                        gc.collect()

    if Derive_combinedGeo_MapData_Switch:
        coefficients = Get_coefficient_map()
        for imodel_year in range(len(Estiamtion_trained_beginyears)):
            for YEAR in Estimation_years[imodel_year]:
                for imodel_month in range(len(Estiamtion_months)):
                    for imonth in Estiamtion_months[imodel_month]:
                        MM = ['01','02','03','04','05','06','07','08','09','10','11','12']
                        CNN_Species = load_estimation_map_data(YYYY=YEAR,MM=MM[imonth],SPECIES=species,version=version,
                                                           special_name=special_name)
                        Combined_species = Combine_CNN_GeophysicalSpecies(CNN_Species=CNN_Species,coefficient=coefficients,YYYY=YEAR,MM=MM[imonth])
                        save_combinedGeo_map_data(final_data=Combined_species,YYYY=YEAR,MM=MM[imonth],extent=Extent,
                                              SPECIES=species,version=version,special_name=special_name)
                    
    return