import torch
import numpy as np
import netCDF4 as nc
from netCDF4 import Dataset
import os
import csv
from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import *
from Training_pkg.utils import *
from Training_pkg.Net_Construction import LightGBMModel
import lightgbm as lgb

def load_Global_Mask_data(region_name):
    mask_dir = '/path/to/supportData/Global_Masks/'
    file_path = os.path.join(mask_dir, f"{region_name}.nc")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No mask file found at {file_path!r}")

    with Dataset(file_path, 'r') as ds:
        try:
            mask_map = ds.variables['continent_mask'][:]
            lat      = ds.variables['lat'][:]
            lon      = ds.variables['lon'][:]
        except KeyError as e:
            raise KeyError(f"Variable {e.args[0]!r} not found in {file_path}") from e

    return mask_map, lat, lon

def save_sample_based_cv_lightgbm_model(lgb_model, model_outdir, typeName, version, 
                                         species, nchannel, special_name, fold, width, height):
    """Save LightGBM model for sample-based CV"""
    import os
    outdir = f'{model_outdir}/{species}/{version}/model/lightgbm_sampleCV_{typeName}_{species}_{version}_{nchannel}Channel_{width}x{height}{special_name}/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    filename = f'lightgbm_sampleCV_fold{fold}.txt'
    filepath = os.path.join(outdir, filename)
    
    lgb_model.model.save_model(filepath)
    print(f'Model saved: {filepath}')

def save_trained_sample_based_xgboost_model(xgboost_model, model_outdir, typeName, species, version, nchannel, special_name, fold, width, height):
    """Save XGBoost model for sample-based CV"""
    outdir = f'{model_outdir}/{species}/{version}/model/xgboost_sampleCV_{typeName}_{species}_{version}_{nchannel}Channel_{width}x{height}{special_name}/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    filename = f'xgboost_sampleCV_fold{fold}.txt'
    filepath = os.path.join(outdir, filename)

def save_trained_month_based_lightgbm_model(lgb_model, model_outdir, typeName, beginyear, endyear, 
                                             month_index, version, species, nchannel, special_name, 
                                             count, width, height):
    """Save LightGBM model"""
    outdir = f'{model_outdir}/{species}/{version}/model/lightgbm_{typeName}_{species}_{version}_{nchannel}Channel_{width}x{height}{special_name}/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    month_str = ''.join([str(m) for m in month_index])
    filename = f'lightgbm_{beginyear}_{endyear}_months{month_str}_fold{count}.txt'
    filepath = os.path.join(outdir, filename)
    
    lgb_model.model.save_model(filepath)
    print(f'LightGBM model saved: {filepath}')

def save_trained_month_based_xgboost_model(xgboost_model, model_outdir, typeName, beginyear, endyear, month_index, 
                                            version, species, nchannel, special_name, count, width, height):
    """Save XGBoost model"""
    outdir = f'{model_outdir}/{species}/{version}/model/xgboost_{typeName}_{species}_{version}_{nchannel}Channel_{width}x{height}{special_name}/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    month_str = ''.join([str(m) for m in month_index])
    filename = f'xgboost_{beginyear}_{endyear}_months{month_str}_fold{count}.txt'
    filepath = os.path.join(outdir, filename)
    
    xgboost_model.model.save_model(filepath)
    print(f'XGBoost model saved: {filepath}')

def load_month_based_xgboost_model(model_indir, typeName, beginyear, endyear, month_index, 
                                    version, species, nchannel, special_name, count, width, height):
    """Load XGBoost model"""
    indir = f'{model_indir}/{species}/{version}/model/xgboost_{typeName}_{species}_{version}_{nchannel}Channel_{width}x{height}{special_name}/'
    month_str = ''.join([str(m) for m in month_index])
    filename = f'xgboost_{beginyear}_{endyear}_months{month_str}_fold{count}.txt'
    filepath = os.path.join(indir, filename)
    
    model = XGBoostModel()
    model.model = xgb.Booster(model_file=filepath)
    model.best_iteration = model.model.best_iteration
    
    print(f'XGBoost model loaded: {filepath}')
    return model

def load_month_based_lightgbm_model(model_indir, typeName, beginyear, endyear, month_index, 
                                     version, species, nchannel, special_name, count, width, height):
    """Load LightGBM model"""    
    indir = f'{model_indir}/{species}/{version}/model/lightgbm_{typeName}_{species}_{version}_{nchannel}Channel_{width}x{height}{special_name}/'
    month_str = ''.join([str(m) for m in month_index])
    filename = f'lightgbm_{beginyear}_{endyear}_months{month_str}_fold{count}.txt'
    filepath = os.path.join(indir, filename)
    
    model = LightGBMModel()
    model.model = lgb.Booster(model_file=filepath)
    model.best_iteration = model.model.best_iteration
    
    print(f'LightGBM model loaded: {filepath}')
    return model

def save_trained_month_based_BLCO_lightgbm_model(lgb_model, model_outdir, typeName, beginyear, endyear, 
                                                  month_index, version, species, nchannel, special_name, 
                                                  count, width, height, buffer_radius):
    """Save LightGBM BLCO model"""
    import os
    outdir = f'{model_outdir}/{species}/{version}/model/lightgbm_{typeName}_{species}_{version}_{nchannel}Channel_{width}x{height}{special_name}_BLCO{buffer_radius}km/'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    month_str = ''.join([str(m) for m in month_index])
    filename = f'lightgbm_BLCO{buffer_radius}km_{beginyear}_{endyear}_months{month_str}_fold{count}.txt'
    filepath = os.path.join(outdir, filename)
    
    lgb_model.model.save_model(filepath)
    print(f'LightGBM BLCO model saved: {filepath}')


def load_trained_month_based_BLCO_lightgbm_model(model_indir, typeName, beginyear, endyear, month_index, 
                                                   version, species, nchannel, special_name, count, 
                                                   width, height, buffer_radius):
    """Load LightGBM BLCO model"""
    
    indir = f'{model_indir}/{species}/{version}/model/lightgbm_{typeName}_{species}_{version}_{nchannel}Channel_{width}x{height}{special_name}_BLCO{buffer_radius}km/'
    month_str = ''.join([str(m) for m in month_index])
    filename = f'lightgbm_BLCO{buffer_radius}km_{beginyear}_{endyear}_months{month_str}_fold{count}.txt'
    filepath = os.path.join(indir, filename)
    
    model = LightGBMModel()
    model.model = lgb.Booster(model_file=filepath)
    model.best_iteration = model.model.best_iteration
    
    print(f'LightGBM BLCO model loaded: {filepath}')
    return model

def save_trained_model(cnn_model, model_outdir, typeName, version, species, nchannel, special_name, count, width, height):
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    model_outfile = outdir +  'SpatialCV_{}_{}_{}x{}_{}Channel{}_No{}.pt'.format(typeName, species, width,height, nchannel,special_name, count)
    torch.save(cnn_model, model_outfile)

def save_trained_month_based_model(cnn_model, model_outdir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    model_outfile = outdir +  'SpatialCV_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}.pt'.format(typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count)
    torch.save(cnn_model, model_outfile)
    return

def save_trained_month_based_FixNumber_model(cnn_model, model_outdir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height,fixed_test_number,fixed_train_number):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    model_outfile = outdir +  'FixNumber_SpatialCV_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_{}fixed_test_number_{}fixed_train_number_No{}.pt'.format(typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name,fixed_test_number,fixed_train_number, count)
    torch.save(cnn_model, model_outfile)
    return

def save_trained_month_based_BLOO_model(cnn_model, model_outdir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height,buffer_radius):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    model_outfile = outdir +  'BLOO_SpatialCV_{}km_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}.pt'.format(buffer_radius,typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count)
    torch.save(cnn_model, model_outfile)
    return

def save_trained_month_based_BLCO_model(cnn_model, model_outdir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height,buffer_radius):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if utilize_self_isolated_sites:
        model_outfile = outdir +  'SelfIsolated_BLCO_SpatialCV_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}.pt'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count)
    
    else:
        model_outfile = outdir +  'BLCO_SpatialCV_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}.pt'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count)
    torch.save(cnn_model, model_outfile)
    return

def save_sensitivity_test_trained_month_based_model(model, model_outdir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height,sensitivity_variables_names_suffix,sensitivity_test_type):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if sensitivity_test_type == 'exclusion':
        model_outfile = outdir +  'SpatialCV_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}_exclude{}.pt'.format(typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count,sensitivity_variables_names_suffix)
    else:
         model_outfile = outdir +  'SpatialCV_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}_include{}.pt'.format(typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count,sensitivity_variables_names_suffix)
    torch.save(model, model_outfile)
    return


def save_sensitivity_test_trained_model(model, model_outdir, typeName, version, species, nchannel, special_name, count, width, height, sensitivity_variables_names_suffix,sensitivity_test_type):
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    if sensitivity_test_type == 'exclusion':
        model_outfile = outdir +  'SpatialCV_{}_{}_{}x{}_{}Channel{}_No{}_exclude{}.pt'.format(typeName, species, width,height, nchannel,special_name, count,sensitivity_variables_names_suffix)
    else:
        model_outfile = outdir +  'SpatialCV_{}_{}_{}x{}_{}Channel{}_No{}_include{}.pt'.format(typeName, species, width,height, nchannel,special_name, count,sensitivity_variables_names_suffix)
    
    torch.save(model, model_outfile)

def save_loss_accuracy(model_outdir, loss, accuracy, valid_loss, valid_accuracy, typeName, version, species, nchannel, special_name, width, height):

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    loss_outfile = outdir + 'SpatialCV_loss_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    accuracy_outfile = outdir + 'SpatialCV_accuracy_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    valid_loss_outfile = outdir + 'SpatialCV_valid_loss_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    valid_accuracy_outfile = outdir + 'SpatialCV_valid_accuracy_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    np.save(loss_outfile, loss)
    np.save(accuracy_outfile, accuracy)
    np.save(valid_loss_outfile, valid_loss)
    np.save(valid_accuracy_outfile, valid_accuracy)
    return


def save_sensitivity_test_loss_accuracy(model_outdir, loss, accuracy, valid_loss, valid_accuracy, typeName, version, species, nchannel, special_name, width, height, sensitivity_variables_names_suffix,sensitivity_test_type):

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    if sensitivity_test_type == 'exclusion':
        loss_outfile = outdir + 'SpatialCV_loss_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        accuracy_outfile = outdir + 'SpatialCV_accuracy_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        valid_loss_outfile = outdir + 'SpatialCV_valid_loss_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        valid_accuracy_outfile = outdir + 'SpatialCV_valid_accuracy_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    else:
        loss_outfile = outdir + 'SpatialCV_loss_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        accuracy_outfile = outdir + 'SpatialCV_accuracy_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        valid_loss_outfile = outdir + 'SpatialCV_valid_loss_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        valid_accuracy_outfile = outdir + 'SpatialCV_valid_accuracy_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        
    np.save(loss_outfile, loss)
    np.save(accuracy_outfile, accuracy)
    np.save(valid_loss_outfile, valid_loss)
    np.save(valid_accuracy_outfile, valid_accuracy)
    return

def save_month_based_data_recording(obs_data,final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording,lat_recording,lon_recording,species, version, typeName, beginyear, endyear, nchannel, special_name, width, height):
    outdir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    obs_data_outfile =  outdir + '{}-{}-Obs-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    final_data_outfile = outdir + '{}-{}-Final-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    geo_data_outfile = outdir + '{}-{}-Geo-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    training_final_data_outfile = outdir + '{}-{}-training_final_data-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    training_obs_data_outfile = outdir + '{}-{}-training_obs_data-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    testing_population_data_outfile = outdir + '{}-{}-testing_population_data-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)                                          
    lat_data_outfile = outdir + '{}-{}-lat-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    lon_data_outfile = outdir + '{}-{}-lon-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    np.save(geo_data_outfile, geo_data_recording)
    np.save(training_final_data_outfile, training_final_data_recording)
    np.save(training_obs_data_outfile, training_obs_data_recording)
    np.save(testing_population_data_outfile, testing_population_data_recording)
    np.save(lat_data_outfile, lat_recording)
    np.save(lon_data_outfile, lon_recording)
    return

def save_Fixnumber_month_based_data_recording(obs_data,final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording,lat_recording,lon_recording,species, version, typeName, beginyear, endyear, nchannel, special_name, width, height, test_number,train_number):
    outdir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    obs_data_outfile =  outdir + '{}-{}-Obs-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}_{}TestSites_{}TrainSites.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name,test_number,train_number)
    final_data_outfile = outdir + '{}-{}-Final-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}_{}TestSites_{}TrainSites.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name,test_number,train_number)
    geo_data_outfile = outdir + '{}-{}-Geo-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    training_final_data_outfile = outdir + '{}-{}-training_final_data-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    training_obs_data_outfile = outdir + '{}-{}-training_obs_data-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    testing_population_data_outfile = outdir + '{}-{}-testing_population_data-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    lat_data_outfile = outdir + '{}-{}-lat-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}_{}TestSites_{}TrainSites.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name,test_number,train_number)
    lon_data_outfile = outdir + '{}-{}-lon-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}_{}TestSites_{}TrainSites.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name,test_number,train_number)
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    np.save(geo_data_outfile, geo_data_recording)
    np.save(training_final_data_outfile, training_final_data_recording)
    np.save(training_obs_data_outfile, training_obs_data_recording)
    np.save(testing_population_data_outfile, testing_population_data_recording)
    np.save(lat_data_outfile, lat_recording)
    np.save(lon_data_outfile, lon_recording)
    return

def save_sensitivity_test_month_based_data_recording(obs_data,final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording,lat_recording,lon_recording,species, version, typeName, beginyear, endyear, nchannel, special_name, width, height,sensitivity_variables_names_suffix,sensitivity_test_type):
    outdir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    if sensitivity_test_type == 'exclusion':
        obs_data_outfile =  outdir + '{}-{}-Obs-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        final_data_outfile = outdir + '{}-{}-Final-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        geo_data_outfile = outdir + '{}-{}-Geo-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)
        training_final_data_outfile = outdir + '{}-{}-training_final_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)  
        training_obs_data_outfile = outdir + '{}-{}-training_obs_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name) 
        testing_population_data_outfile = outdir + '{}-{}-testing_population_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)  
        
        lat_data_outfile = outdir + '{}-{}-lat-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        lon_data_outfile = outdir + '{}-{}-lon-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, endyear
                                                                                              ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    else:
        obs_data_outfile =  outdir + '{}-{}-Obs-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        final_data_outfile = outdir + '{}-{}-Final-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        geo_data_outfile = outdir + '{}-{}-Geo-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)
        training_final_data_outfile = outdir + '{}-{}-training_final_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)  
        training_obs_data_outfile = outdir + '{}-{}-training_obs_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name) 
        testing_population_data_outfile = outdir + '{}-{}-testing_population_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)  
        
        lat_data_outfile = outdir + '{}-{}-lat-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        lon_data_outfile = outdir + '{}-{}-lon-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    np.save(geo_data_outfile, geo_data_recording)
    np.save(training_final_data_outfile, training_final_data_recording)
    np.save(training_obs_data_outfile, training_obs_data_recording)
    np.save(testing_population_data_outfile, testing_population_data_recording)
    np.save(lat_data_outfile, lat_recording)
    np.save(lon_data_outfile, lon_recording)
    return

def save_month_based_BLOO_data_recording(obs_data,final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording,lat_recording,lon_recording,species, version, typeName, beginyear, endyear, nchannel, special_name, width, height,buffer_radius):
    outdir = txt_outdir + '{}/{}/Results/results-BLOO_DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    obs_data_outfile =  outdir + '{}-{}-Obs-BLOODataRecording_{}km_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    final_data_outfile = outdir + '{}-{}-Final-BLOODataRecording_{}km_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    geo_data_outfile = outdir + '{}-{}-Geo-BLOODataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    training_final_data_outfile = outdir + '{}-{}-training_final_data-BLOODataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    training_obs_data_outfile = outdir + '{}-{}-training_obs_data-BLOODataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    testing_population_data_outfile = outdir + '{}-{}-testing_population_data-BLOODataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    lat_data_outfile = outdir + '{}-{}-lat-BLOODataRecording_{}km_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    lon_data_outfile = outdir + '{}-{}-lon-BLOODataRecording_{}km_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    np.save(geo_data_outfile, geo_data_recording)
    np.save(training_final_data_outfile, training_final_data_recording)
    np.save(training_obs_data_outfile, training_obs_data_recording)
    np.save(testing_population_data_outfile, testing_population_data_recording)
    np.save(lat_data_outfile, lat_recording)
    np.save(lon_data_outfile, lon_recording)
    return

def save_month_based_BLCO_data_recording(obs_data,final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording,lat_recording,lon_recording,testsites2trainsites_nearest_distances, test_sites_index_recording, train_sites_index_recording , excluded_sites_index_recording, train_index_number,test_index_number,
species, version, typeName, beginyear, endyear, nchannel, special_name, width, height,buffer_radius,BLCO_kfold,BLCO_seeds_number):
    if utilize_self_isolated_sites:
        outdir = txt_outdir + '{}/{}/Results/results-SelfIsolated_BLCO_DataRecording/'.format(species, version)
    else:
        outdir = txt_outdir + '{}/{}/Results/results-BLCO_DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    obs_data_outfile =  outdir + '{}-{}-Obs-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    final_data_outfile = outdir + '{}-{}-Final-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    geo_data_outfile = outdir + '{}-{}-Geo-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    training_final_data_outfile = outdir + '{}-{}-training_final_data-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    training_obs_data_outfile = outdir + '{}-{}-training_obs_data-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    testing_population_data_outfile = outdir + '{}-{}-testing_population_data-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    lat_data_outfile = outdir + '{}-{}-lat-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    lon_data_outfile = outdir + '{}-{}-lon-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    TestSites2TrainSites_nearest_distances_outfile =  outdir + '{}-{}-TestSites2TrainSites_nearest_distances-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    test_sites_index_recording_outfile =  outdir + '{}-{}-test_sites_index_recording-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    train_sites_index_recording_outfile =  outdir + '{}-{}-train_sites_index_recording-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    excluded_sites_index_recording_outfile =  outdir + '{}-{}-excluded_sites_index_recording-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    train_index_number_outfile = outdir + '{}-{}-train_index_number-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    test_index_number_outfile = outdir + '{}-{}-test_index_number-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    np.save(geo_data_outfile, geo_data_recording)
    np.save(training_final_data_outfile, training_final_data_recording)
    np.save(training_obs_data_outfile, training_obs_data_recording)
    np.save(testing_population_data_outfile, testing_population_data_recording)
    np.save(lat_data_outfile, lat_recording)
    np.save(lon_data_outfile, lon_recording)
    np.save(TestSites2TrainSites_nearest_distances_outfile, testsites2trainsites_nearest_distances)
    np.save(test_sites_index_recording_outfile,test_sites_index_recording)
    np.save(train_sites_index_recording_outfile,train_sites_index_recording)
    np.save(excluded_sites_index_recording_outfile,excluded_sites_index_recording)
    np.save(train_index_number_outfile, train_index_number)
    np.save(test_index_number_outfile,  test_index_number)
    return

def save_data_recording(obs_data, final_data,species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height):
    outdir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    obs_data_outfile   = outdir + '{}-{}-Obs-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    final_data_outfile = outdir + '{}-{}-Final-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    return

def save_sensitivity_test_data_recording(obs_data, final_data,species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height,sensitivity_variables_names_suffix,sensitivity_test_type):
    outdir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    if sensitivity_test_type == 'exclusion':
        obs_data_outfile   = outdir + '{}-{}-Obs-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        final_data_outfile = outdir + '{}-{}-Final-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    else:
        obs_data_outfile   = outdir + '{}-{}-Obs-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        final_data_outfile = outdir + '{}-{}-Final-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    return

def save_BLOO_data_recording(obs_data, final_data,species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height, buffer_radius):
    outdir = txt_outdir + '{}/{}/Results/results-BLOO_DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    obs_data_outfile   = outdir + '{}-{}-Obs-BLOODataRecording_longterm_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    final_data_outfile = outdir + '{}-{}-Final-BLOODataRecording_longterm_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    return

def save_BLCO_data_recording(obs_data, final_data,species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height, buffer_radius):
    if utilize_self_isolated_sites:
        outdir = txt_outdir + '{}/{}/Results/results-SelfIsolated_BLCO_DataRecording/'.format(species, version)
    else:
        outdir = txt_outdir + '{}/{}/Results/results-BLCO_DataRecording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir) 
    obs_data_outfile   = outdir + '{}-{}-Obs-BLCODataRecording_longterm_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number,beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    final_data_outfile = outdir + '{}-{}-Final-BLCODataRecording_longterm_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number,beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    np.save(obs_data_outfile, obs_data)
    np.save(final_data_outfile, final_data)
    return

def save_BLOO_loss_accuracy(model_outdir, loss, accuracy, valid_loss, valid_accuracy, typeName, version, species, nchannel, special_name, width, height, buffer_radius):

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    loss_outfile = outdir + 'BLOOCV_loss_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,typeName, species, width, height, nchannel,special_name)
    accuracy_outfile = outdir + 'BLOOCV_accuracy_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,typeName, species, width, height, nchannel,special_name)
    valid_loss_outfile = outdir + 'BLOOCV_valid_loss_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,typeName, species, width, height, nchannel,special_name)
    valid_accuracy_outfile = outdir + 'BLOOCV_valid_accuracy_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,typeName, species, width, height, nchannel,special_name)
    np.save(loss_outfile, loss)
    np.save(accuracy_outfile, accuracy)
    np.save(valid_loss_outfile, valid_loss)
    np.save(valid_accuracy_outfile, valid_accuracy)
    return


def save_BLCO_loss_accuracy(model_outdir, loss, accuracy, valid_loss, valid_accuracy, typeName, version, species, nchannel, special_name, width, height, buffer_radius):
    
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    if utilize_self_isolated_sites:
        loss_outfile = outdir + 'SelfIsolated_BLCOCV_loss_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        accuracy_outfile = outdir + 'SelfIsolated_BLCOCV_accuracy_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        valid_loss_outfile = outdir + 'SelfIsolated_BLCOCV_valid_loss_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        valid_accuracy_outfile = outdir + 'SelfIsolated_BLCOCV_valid_accuracy_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
    else:
        loss_outfile = outdir +'BLCOCV_loss_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        accuracy_outfile = outdir + 'BLCOCV_accuracy_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        valid_loss_outfile = outdir + 'BLCOCV_valid_loss_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius, BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        valid_accuracy_outfile = outdir + 'BLCOCV_valid_accuracy_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
    np.save(loss_outfile, loss)
    np.save(accuracy_outfile, accuracy)
    np.save(valid_loss_outfile, valid_loss)
    np.save(valid_accuracy_outfile, valid_accuracy)
    return

def save_SHAPValues_data_recording(shap_values_values:np.array, shap_values_data:np.array,species, version, typeName, beginyear, endyear, nchannel, special_name, width, height):
    outdir = txt_outdir + '{}/{}/Results/results-SHAPValues_data_recording/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    shap_values_values_outfile = outdir + '{}-{}-SHAP_values_values-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)

    shap_values_data_outfile   = outdir + '{}-{}-SHAP_values_data-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    np.save(shap_values_values_outfile, shap_values_values)

    np.save(shap_values_data_outfile,   shap_values_data)
    return


def load_SHAPValues_data_recording(species, version, typeName, beginyear, endyear, nchannel, special_name, width, height):
    indir = txt_outdir + '{}/{}/Results/results-SHAPValues_data_recording/'.format(species, version)
    shap_values_values_infile = indir + '{}-{}-SHAP_values_values-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    shap_values_data_infile   = indir + '{}-{}-SHAP_values_data-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    shap_values_values = np.load(shap_values_values_infile)#,allow_pickle=True).item()

    shap_values_data   = np.load(shap_values_data_infile)#,  allow_pickle=True).item()
    return shap_values_values, shap_values_data

def load_month_based_data_recording(species, version, typeName, beginyear, endyear, nchannel, special_name, width, height):
    indir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)

    obs_data_infile =  indir + '{}-{}-Obs-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    final_data_infile = indir + '{}-{}-Final-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    geo_data_infile = indir + '{}-{}-Geo-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    training_final_data_infile = indir + '{}-{}-training_final_data-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    training_obs_data_infile = indir + '{}-{}-training_obs_data-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    testing_population_data_infile = indir + '{}-{}-testing_population_data-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    lat_data_infile = indir + '{}-{}-lat-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    lon_data_infile = indir + '{}-{}-lon-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    obs_data = np.load(obs_data_infile,allow_pickle=True).item()
    final_data = np.load(final_data_infile,allow_pickle=True).item()
    geo_data_recording = np.load(geo_data_infile, allow_pickle=True).item()
    training_final_data_recording = np.load(training_final_data_infile, allow_pickle=True).item()
    training_obs_data_recording = np.load(training_obs_data_infile, allow_pickle=True).item()
    testing_population_data_recording = np.load(testing_population_data_infile, allow_pickle=True).item()                                                                                                                   
    lat_recording = np.load(lat_data_infile)
    lon_recording = np.load(lon_data_infile)
    return obs_data, final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_recording, lon_recording

def load_Fixnumber_month_based_data_recording(species, version, typeName, beginyear, endyear, nchannel, special_name, width, height, test_number,train_number):
    indir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)

    obs_data_infile =  indir + '{}-{}-Obs-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}_{}TestSites_{}TrainSites.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name,test_number,train_number)
    final_data_infile = indir + '{}-{}-Final-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}_{}TestSites_{}TrainSites.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name,test_number,train_number)
    geo_data_infile = indir + '{}-{}-Geo-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    training_final_data_infile = indir + '{}-{}-training_final_data-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    training_obs_data_infile = indir + '{}-{}-training_obs_data-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    testing_population_data_infile = indir + '{}-{}-testing_population_data-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    lat_data_infile = indir + '{}-{}-lat-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}_{}TestSites_{}TrainSites.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name,test_number,train_number)
    lon_data_infile = indir + '{}-{}-lon-FixnumberDataRecording_{}-{}_{}x{}_{}Channel{}_{}TestSites_{}TrainSites.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name,test_number,train_number)
    obs_data = np.load(obs_data_infile,allow_pickle=True).item()
    final_data = np.load(final_data_infile,allow_pickle=True).item()
    geo_data_recording = np.load(geo_data_infile, allow_pickle=True).item()
    training_final_data_recording = np.load(training_final_data_infile, allow_pickle=True).item()
    training_obs_data_recording = np.load(training_obs_data_infile, allow_pickle=True).item()
    testing_population_data_recording = np.load(testing_population_data_infile, allow_pickle=True).item()
    lat_recording = np.load(lat_data_infile)
    lon_recording = np.load(lon_data_infile)
    return obs_data, final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_recording, lon_recording


def load_sensitivity_test_month_based_data_recording(species, version, typeName, beginyear, endyear, nchannel, special_name, width, height,sensitivity_variables_names_suffix,sensitivity_test_type):
    if sensitivity_test_type == 'exclusion':
        indir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
        obs_data_infile =  indir + '{}-{}-Obs-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        final_data_infile = indir + '{}-{}-Final-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        geo_data_infile = indir + '{}-{}-Geo-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)
        training_final_data_infile = indir + '{}-{}-training_final_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)  
        training_obs_data_infile = indir + '{}-{}-training_obs_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name) 
        testing_population_data_infile = indir + '{}-{}-testing_population_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name) 
        lat_data_infile = indir + '{}-{}-lat-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        lon_data_infile = indir + '{}-{}-lon-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    else:
        indir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
        obs_data_infile =  indir + '{}-{}-Obs-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        final_data_infile = indir + '{}-{}-Final-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        geo_data_infile = indir + '{}-{}-Geo-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)
        training_final_data_infile = indir + '{}-{}-training_final_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name)  
        training_obs_data_infile = indir + '{}-{}-training_obs_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name) 
        testing_population_data_infile = indir + '{}-{}-testing_population_data-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name) 
        lat_data_infile = indir + '{}-{}-lat-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        lon_data_infile = indir + '{}-{}-lon-sensitivity_test-DataRecording_{}-{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, endyear
                                                                                                ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        
    obs_data = np.load(obs_data_infile,allow_pickle=True).item()
    final_data = np.load(final_data_infile,allow_pickle=True).item()
    geo_data_recording = np.load(geo_data_infile, allow_pickle=True).item()
    training_final_data_recording = np.load(training_final_data_infile, allow_pickle=True).item()
    training_obs_data_recording = np.load(training_obs_data_infile, allow_pickle=True).item()
    testing_population_data_recording = np.load(testing_population_data_infile, allow_pickle=True).item()
    lat_recording = np.load(lat_data_infile)
    lon_recording = np.load(lon_data_infile)
    return obs_data, final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_recording, lon_recording


def load_month_based_BLOO_data_recording(species, version, typeName, beginyear, endyear, nchannel, special_name, width, height,buffer_radius):
    indir = txt_outdir + '{}/{}/Results/results-BLOO_DataRecording/'.format(species, version)

    obs_data_infile =  indir + '{}-{}-Obs-BLOODataRecording_{}km_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    final_data_infile = indir + '{}-{}-Final-BLOODataRecording_{}km_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    geo_data_infile = indir + '{}-{}-Geo-BLOODataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    training_final_data_infile = indir + '{}-{}-training_final_data-BLOODataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    training_obs_data_infile = indir + '{}-{}-training_obs_data-BLOODataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    testing_population_data_infile = indir + '{}-{}-testing_population_data-BLOODataRecording_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name) 
    lat_data_infile = indir + '{}-{}-lat-BLOODataRecording_{}km_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    lon_data_infile = indir  + '{}-{}-lon-BLOODataRecording_{}km_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)
    obs_data = np.load(obs_data_infile,allow_pickle=True).item()
    final_data = np.load(final_data_infile,allow_pickle=True).item()
    geo_data_recording = np.load(geo_data_infile, allow_pickle=True).item()
    training_final_data_recording = np.load(training_final_data_infile, allow_pickle=True).item()
    training_obs_data_recording = np.load(training_obs_data_infile, allow_pickle=True).item()
    testing_population_data_recording = np.load(testing_population_data_infile, allow_pickle=True).item()
    lat_recording = np.load(lat_data_infile)
    lon_recording = np.load(lon_data_infile)
    return obs_data, final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_recording, lon_recording


def load_month_based_BLCO_data_recording(species, version, typeName, beginyear, endyear, nchannel, special_name, width, height,buffer_radius,BLCO_kfold,BLCO_seeds_number):
    if utilize_self_isolated_sites:
         indir = txt_outdir + '{}/{}/Results/results-SelfIsolated_BLCO_DataRecording/'.format(species, version)
    else:
        indir = txt_outdir + '{}/{}/Results/results-BLCO_DataRecording/'.format(species, version)

    obs_data_infile =  indir + '{}-{}-Obs-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    final_data_infile = indir + '{}-{}-Final-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    geo_data_infile = indir + '{}-{}-Geo-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    training_final_data_infile = indir + '{}-{}-training_final_data-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear
                                                                                            ,width, height, nchannel,special_name)  
    training_obs_data_infile = indir + '{}-{}-training_obs_data-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name) 
    testing_population_data_infile = indir + '{}-{}-testing_population_data-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)  
    lat_data_infile = indir + '{}-{}-lat-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species,buffer_radius, BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    lon_data_infile = indir + '{}-{}-lon-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    TestSites2TrainSites_nearest_distances_infile =  indir + '{}-{}-TestSites2TrainSites_nearest_distances-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear
     ,width, height, nchannel,special_name)
    test_sites_index_recording_infile =  indir + '{}-{}-test_sites_index_recording-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    train_sites_index_recording_infile =  indir + '{}-{}-train_sites_index_recording-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    excluded_sites_index_recording_infile =  indir + '{}-{}-excluded_sites_index_recording-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    train_index_number_infile = indir + '{}-{}-train_index_number-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    test_index_number_infile = indir + '{}-{}-test_index_number-BLCODataRecording_{}km_{}-folds_{}-ClusterSeeds_{}-{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, endyear,width, height, nchannel,special_name)
    obs_data = np.load(obs_data_infile,allow_pickle=True).item()
    final_data = np.load(final_data_infile,allow_pickle=True).item()
    geo_data_recording = np.load(geo_data_infile, allow_pickle=True).item()
    training_final_data_recording = np.load(training_final_data_infile, allow_pickle=True).item()
    training_obs_data_recording = np.load(training_obs_data_infile, allow_pickle=True).item()
    testing_population_data_recording = np.load(testing_population_data_infile, allow_pickle=True).item()
    lat_recording = np.load(lat_data_infile)
    lon_recording = np.load(lon_data_infile)
    testsites2trainsites_nearest_distances = np.load(TestSites2TrainSites_nearest_distances_infile, allow_pickle=True).item()
    test_sites_index_recording = np.load(test_sites_index_recording_infile, allow_pickle=True).item()
    train_sites_index_recording = np.load(train_sites_index_recording_infile, allow_pickle=True).item()
    excluded_sites_index_recording = np.load(excluded_sites_index_recording_infile, allow_pickle=True).item()
    train_index_number = np.load(train_index_number_infile)
    test_index_number  = np.load(test_index_number_infile)
    return obs_data, final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_recording, lon_recording,testsites2trainsites_nearest_distances,test_sites_index_recording,train_sites_index_recording,excluded_sites_index_recording, train_index_number, test_index_number


def load_coMonitor_Population():
    data = nc.Dataset(training_infile,'r')
    width = np.array(data.variables['width'][:])[0]
    height = np.array(data.variables['height'][:])[0]
    Population_Dataset = np.array(data.variables['Population'][:,int((width-1)/2),int((height-1)/2)])
    return Population_Dataset

def load_data_recording(species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height):
    indir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    obs_data_infile   = indir + '{}-{}-Obs-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    final_data_infile = indir + '{}-{}-Final-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    obs_data = np.load(obs_data_infile)
    final_data = np.load(final_data_infile)

    return obs_data, final_data

def load_month_based_model(model_indir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    indir = model_indir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    
    PATH = indir +  'SpatialCV_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}.pt'.format(typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.load(PATH, map_location=torch.device(device)).eval()
    model.to(device)
    return model
    
def load_trained_month_based_FixNumber_model(model_indir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height,fixed_test_number,fixed_train_number):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    indir = model_indir + '{}/{}/Results/results-Trained_Models/'.format(species, version)

    PATH = indir +  'FixNumber_SpatialCV_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_{}fixed_test_number_{}fixed_train_number_No{}.pt'.format(typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name,fixed_test_number,fixed_train_number, count)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.load(PATH, map_location=torch.device(device)).eval()
    model.to(device)
    return model

def load_trained_month_based_BLOO_model( model_indir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height,buffer_radius):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    indir = model_indir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    PATH = indir +  'BLOO_SpatialCV_{}km_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}.pt'.format(buffer_radius,typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.load(PATH, map_location=torch.device(device)).eval()
    model.to(device)
    return model

def load_trained_month_based_BLCO_model(model_indir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height,buffer_radius):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    if utilize_self_isolated_sites:
        indir = model_indir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
        PATH = indir +  'SelfIsolated_BLCO_SpatialCV_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}.pt'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count)
    else:
        indir = model_indir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
        PATH = indir +  'BLCO_SpatialCV_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}.pt'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.load(PATH, map_location=torch.device(device)).eval()
    model.to(device)
    return model


def load_sensitivity_test_trained_month_based_model( model_indir, typeName, beginyear,endyear,month_index, version, species, nchannel, special_name, count, width, height,sensitivity_variables_names_suffix,sensitivity_test_type):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    indir = model_indir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if sensitivity_test_type == 'exclusion':
        PATH  = indir +  'SpatialCV_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}_exclude{}.pt'.format(typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count,sensitivity_variables_names_suffix)
    else:
        PATH  = indir +  'SpatialCV_{}_{}_{}x{}_{}-{}_{}_{}Channel{}_No{}_include{}.pt'.format(typeName, species, width,height, beginyear,endyear,Selected_MONTHS_str,nchannel,special_name, count,sensitivity_variables_names_suffix)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = torch.load(PATH, map_location=torch.device(device)).eval()
    model.to(device)
    return model


def load_sensitivity_test_data_recording(species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height,sensitivity_variables_names_suffix,sensitivity_test_type):
    indir = txt_outdir + '{}/{}/Results/results-DataRecording/'.format(species, version)
    if sensitivity_test_type == 'exclusion':
        obs_data_infile   = indir + '{}-{}-Obs-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        final_data_infile = indir + '{}-{}-Final-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    else:
        obs_data_infile   = indir + '{}-{}-Obs-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        final_data_infile = indir + '{}-{}-Final-DataRecording_longterm_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name,sensitivity_variables_names_suffix)
         
    obs_data = np.load(obs_data_infile)
    final_data = np.load(final_data_infile)

    return obs_data, final_data


def load_loss_accuracy(model_outdir, typeName, version, species, nchannel, special_name, width, height):

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    loss_outfile = outdir +'SpatialCV_loss_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    accuracy_outfile = outdir + 'SpatialCV_accuracy_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    valid_loss_outfile = outdir + 'SpatialCV_valid_loss_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    valid_accuracy_outfile = outdir + 'SpatialCV_valid_accuracy_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, width, height, nchannel,special_name)
    loss = np.load(loss_outfile)
    accuracy = np.load(accuracy_outfile)
    valid_loss = np.load(valid_loss_outfile )
    valid_accuracy = np.load(valid_accuracy_outfile )
    return loss, accuracy, valid_loss, valid_accuracy

def load_sensitivity_test_loss_accuracy(model_outdir, typeName, version, species, nchannel, special_name, width, height,sensitivity_variables_names_suffix,sensitivity_test_type):

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    if sensitivity_test_type == 'exclusion':
        loss_outfile = outdir +'SpatialCV_loss_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        accuracy_outfile = outdir + 'SpatialCV_accuracy_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        valid_loss_outfile = outdir + 'SpatialCV_valid_loss_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        valid_accuracy_outfile = outdir + 'SpatialCV_valid_accuracy_{}_{}_{}x{}_{}Channel{}_exclude{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    else:
        loss_outfile = outdir +'SpatialCV_loss_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        accuracy_outfile = outdir + 'SpatialCV_accuracy_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        valid_loss_outfile = outdir + 'SpatialCV_valid_loss_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
        valid_accuracy_outfile = outdir + 'SpatialCV_valid_accuracy_{}_{}_{}x{}_{}Channel{}_include{}.npy'.format(typeName, species, width, height, nchannel,special_name,sensitivity_variables_names_suffix)
    loss = np.load(loss_outfile)
    accuracy = np.load(accuracy_outfile)
    valid_loss = np.load(valid_loss_outfile )
    valid_accuracy = np.load(valid_accuracy_outfile )
    return loss, accuracy, valid_loss, valid_accuracy

def load_BLOO_data_recording(species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height, buffer_radius):
    indir = txt_outdir + '{}/{}/Results/results-BLOO_DataRecording/'.format(species, version)
    obs_data_infile   = indir + '{}-{}-Obs-BLOODataRecording_longterm_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    final_data_infile = indir + '{}-{}-Final-BLOODataRecording_longterm_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    obs_data = np.load(obs_data_infile)
    final_data = np.load(final_data_infile)

    return obs_data, final_data

def load_BLOO_loss_accuracy(model_outdir, typeName, version, species, nchannel, special_name, width, height, buffer_radius):

    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    loss_outfile = outdir +'BLOOCV_loss_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius, typeName, species, width, height, nchannel,special_name)
    accuracy_outfile = outdir + 'BLOOCV_accuracy_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius, typeName, species, width, height, nchannel,special_name)
    valid_loss_outfile = outdir + 'BLOOCV_valid_loss_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius, typeName, species, width, height, nchannel,special_name)
    valid_accuracy_outfile = outdir + 'BLOO CV_valid_accuracy_{}km_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius, typeName, species, width, height, nchannel,special_name)
    loss = np.load(loss_outfile)
    accuracy = np.load(accuracy_outfile)
    valid_loss = np.load(valid_loss_outfile )
    valid_accuracy = np.load(valid_accuracy_outfile )
    return loss, accuracy, valid_loss, valid_accuracy

def load_BLCO_data_recording(species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height, buffer_radius):
    if utilize_self_isolated_sites:
        indir = txt_outdir + '{}/{}/Results/results-SelfIsolated_BLCO_DataRecording/'.format(species, version)
    else:
        indir = txt_outdir + '{}/{}/Results/results-BLCO_DataRecording/'.format(species, version)
    obs_data_infile   = indir + '{}-{}-Obs-BLOODataRecording_longterm_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number,beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    final_data_infile = indir + '{}-{}-Final-BLOODataRecording_longterm_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(typeName, species, buffer_radius,BLCO_kfold,BLCO_seeds_number, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    obs_data = np.load(obs_data_infile)
    final_data = np.load(final_data_infile)

    return obs_data, final_data

def load_BLCO_loss_accuracy(model_outdir, typeName, version, species, nchannel, special_name, width, height, buffer_radius):
    
    outdir = model_outdir + '{}/{}/Results/results-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    if utilize_self_isolated_sites:
        loss_outfile = outdir +'SelfIsolated_BLCOCV_loss_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        accuracy_outfile = outdir + 'SelfIsolated_BLCOCV_accuracy_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        valid_loss_outfile = outdir + 'SelfIsolated_BLCOCV_valid_loss_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius, BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        valid_accuracy_outfile = outdir + 'SelfIsolated_BLCOCV_valid_accuracy_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
    else:
        loss_outfile = outdir +'BLCOCV_loss_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        accuracy_outfile = outdir + 'BLCOCV_accuracy_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        valid_loss_outfile = outdir + 'BLCOCV_valid_loss_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius, BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
        valid_accuracy_outfile = outdir + 'BLCOCV_valid_accuracy_{}km_{}-folds_{}-ClusterSeeds_{}_{}_{}x{}_{}Channel{}.npy'.format(buffer_radius,BLCO_kfold,BLCO_seeds_number,typeName, species, width, height, nchannel,special_name)
    loss = np.load(loss_outfile)
    accuracy = np.load(accuracy_outfile)
    valid_loss = np.load(valid_loss_outfile )
    valid_accuracy = np.load(valid_accuracy_outfile )
    return loss, accuracy, valid_loss, valid_accuracy

def output_text(outfile:str,status:str,CV_R2,annual_CV_R2,month_CV_R2,training_annual_CV_R2,training_month_CV_R2,
                geo_annual_CV_R2, geo_month_CV_R2,
                CV_slope,annual_CV_slope,month_CV_slope,
                CV_RMSE,annual_CV_RMSE,month_CV_RMSE,

                beginyear:str,endyear:str,species:str,kfold:int,repeats:int):
    
    MONTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    CV_R2[-1] = np.mean(CV_R2[0:kfold * repeats])
    annual_CV_R2[-1] = np.mean(annual_CV_R2[0:kfold * repeats])
    CV_slope[-1] = np.mean(CV_slope[0:kfold * repeats])
    annual_CV_slope[-1] = np.mean(annual_CV_slope[0:kfold * repeats])
    CV_RMSE[-1] = np.mean(CV_RMSE[0:kfold * repeats])
    annual_CV_RMSE[-1] = np.mean(annual_CV_RMSE[0:kfold * repeats])
    training_annual_CV_R2[-1] = np.mean(training_annual_CV_R2[0:kfold * repeats])
    geo_annual_CV_R2[-1] = np.mean(geo_annual_CV_R2[0:kfold * repeats])

    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Species: {} ; Time Period: {} - {}'.format(species, beginyear, endyear)])
        writer.writerow(['R2 for monthly validation', '\nMax: ',str(np.round(np.max(CV_R2),4)),'Min: ',str(np.round(np.min(CV_R2),4)),
                         'Avg: ',str(np.round(CV_R2[-1],4)),'\nSlope for monthly validation Max: ',str(np.round(np.max(CV_slope),4)),'Min: ',str(np.round(np.min(CV_slope),4)),
                         'Avg: ',str(np.round(CV_slope[-1],4)),'\nRMSE for monthly validation Max: ',str(np.round(np.max(CV_RMSE),4)),'Min: ',str(np.round(np.min(CV_RMSE),4)),
                         'Avg: ',str(np.round(CV_RMSE[-1],4))])
        writer.writerow(['#####################   Annual average validation ####################', '\n R2 Max: ', str(np.round(np.max(annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_R2), 4)),
                         'Avg: ', str(np.round(annual_CV_R2[-1], 4)),'\nSlope for Annual average validation Max: ', str(np.round(np.max(annual_CV_slope), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_slope), 4)),
                         'Avg: ', str(np.round(annual_CV_slope[-1], 4)),'\nRMSE for Annual average validation Max: ', str(np.round(np.max(annual_CV_RMSE), 4)), 'Min: ',
                         str(np.round(np.min(annual_CV_RMSE), 4)),
                         'Avg: ', str(np.round(annual_CV_RMSE[-1], 4))])
        writer.writerow(['###################### Annual Training ####################', '\n Training R2 - Max: ', str(np.round(np.max(training_annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(training_annual_CV_R2), 4)),
                         'Avg: ', str(np.round(training_annual_CV_R2[-1], 4))])
        writer.writerow(['###################### Annual Geophysical ####################', '\n  R2 - Max: ', str(np.round(np.max(geo_annual_CV_R2), 4)), 'Min: ',
                         str(np.round(np.min(geo_annual_CV_R2), 4)),
                         'Avg: ', str(np.round(geo_annual_CV_R2[-1], 4))])
        for imonth in range(len(MONTH)):
            month_CV_R2[imonth,-1] = np.mean(month_CV_R2[imonth,0:kfold * repeats])
            month_CV_slope[imonth,-1] = np.mean(month_CV_slope[imonth,0:kfold * repeats])
            month_CV_RMSE[imonth,-1] = np.mean(month_CV_RMSE[imonth,0:kfold * repeats])
            training_month_CV_R2[imonth,-1] = np.mean(training_month_CV_R2[imonth,0:kfold * repeats])
            geo_month_CV_R2[imonth,-1] = np.mean(geo_month_CV_R2[imonth,0:kfold * repeats])
            writer.writerow([' -------------------------- {} ------------------------'.format(MONTH[imonth]), 
                             '\n R2 - Max: ', str(np.round(np.max(month_CV_R2[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_R2[imonth,:]), 4)), 'Avg: ',str(np.round(month_CV_R2[imonth,-1],4)),
                             '\nSlope - Max: ', str(np.round(np.max(month_CV_slope[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_slope[imonth,:]), 4)), 'Avg: ',str(np.round(month_CV_slope[imonth,-1],4)),
                             '\nRMSE -  Max: ', str(np.round(np.max(month_CV_RMSE[imonth,:]), 4)), 'Min: ',
                             str(np.round(np.min(month_CV_RMSE[imonth,:]), 4)), 'Avg: ',str(np.round(month_CV_RMSE[imonth,-1],4)),
                             '\nTraining R2 - Max: ',str(np.round(np.max(training_month_CV_R2[imonth,:]), 4)), 'Min: ',str(np.round(np.min(training_month_CV_R2[imonth,:]), 4)), 'Avg: ',
                             str(np.round(training_month_CV_R2[imonth,-1],4)),
                             '\nGeophysical R2 - Max: ',str(np.round(np.max(geo_month_CV_R2[imonth,:]), 4)), 'Min: ',str(np.round(np.min(geo_month_CV_R2[imonth,:]), 4)), 'Avg: ',
                             str(np.round(geo_month_CV_R2[imonth,-1],4))])

    return


def AVD_output_text(outfile:str,status:str,Area,test_beginyears,test_endyears,
                test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE,PMW_NRMSE,slope,PWM_Model, PWM_Monitors, regional_number):
    
    MONTH = ['Annual','MAM','JJA','SON','DJF','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime,RMSE_Alltime, NRMSE_Alltime, PWM_NRMSE_Alltime,slope_Alltime,PWAModel_Alltime,PWAMonitors_Alltime = calculate_Alltime_Statistics_results(test_beginyears,test_endyears,test_CV_R2, train_CV_R2, geo_CV_R2, RMSE,NRMSE,PMW_NRMSE, slope,PWM_Model,PWM_Monitors,Area)

    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Area: {} ; Time Period: {} - {}'.format(Area, test_beginyears, test_endyears), ' Total Site Number: {}'.format(regional_number)])
        
        for imonth in MONTH:
            if imonth == 'Annual':
                writer.writerow([' -------------------------- {} ------------------------'.format(imonth), 
                                '\n Test R2 - Avg: ', str(np.round(test_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(test_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(test_CV_R2_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(test_CV_R2_Alltime['Alltime'][imonth][3],4)), 'AllPoints Test R2 - Annual Average: ',
                                str(np.round(test_CV_R2['AllPoints']['Annual'],4)), 'AllPoints Test R2 - AllPoints: ',
                                str(np.round(test_CV_R2['AllPoints']['AllPoints'],4)),


                                '\n Slope - Avg: ', str(np.round(slope_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(slope_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(slope_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(slope_Alltime['Alltime'][imonth][3],4)), 'AllPoints Slope - Annual Average: ',
                                str(np.round(slope['AllPoints']['Annual'],4)), 'AllPoints Slope - AllPoints: ',
                                str(np.round(slope['AllPoints']['AllPoints'],4)),

                                '\n RMSE -  Avg: ', str(np.round(RMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(RMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(RMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(RMSE_Alltime['Alltime'][imonth][3],4)), 'AllPoints RMSE - Annual Average: ',
                                str(np.round(RMSE['AllPoints']['Annual'],4)), 'AllPoints RMSE - AllPoints: ',
                                str(np.round(RMSE['AllPoints']['AllPoints'],4)),

                                '\n NRMSE -  Avg: ', str(np.round(NRMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(NRMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(NRMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(NRMSE_Alltime['Alltime'][imonth][3],4)),'AllPoints NRMSE - Annual Average: ',
                                str(np.round(NRMSE['AllPoints']['Annual'],4)), 'AllPoints NRMSE - AllPoints: ',
                                str(np.round(NRMSE['AllPoints']['AllPoints'],4)),
                                
                                '\n PWM NRMSE -  Avg: ', str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][3],4)),'AllPoints PWM NRMSE - Annual Average: ',
                                str(np.round(PMW_NRMSE['AllPoints']['Annual'],4)), 'AllPoints PWM NRMSE - AllPoints: ',
                                str(np.round(PMW_NRMSE['AllPoints']['AllPoints'],4)),

                                '\n Training R2 - Avg: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(train_CV_R2_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][3],4)),'AllPoints Training R2 - Annual Average: ',
                                str(np.round(train_CV_R2['AllPoints']['Annual'],4)), 'AllPoints Training R2 - AllPoints: ',
                                str(np.round(train_CV_R2['AllPoints']['AllPoints'],4)),


                                '\n Geophysical R2 - Avg: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][3],4)),'AllPoints Geophysical R2 - Annual Average: ',
                                str(np.round(geo_CV_R2['AllPoints']['Annual'],4)), 'AllPoints Geophysical R2 - AllPoints: ',
                                str(np.round(geo_CV_R2['AllPoints']['AllPoints'],4)),
                                
                                '\n PWA Model - Avg: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(PWAModel_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][3],4)),'AllPoints PWA Model - Annual Average: ',
                                str(np.round(PWM_Model['AllPoints']['Annual'],4)), 'AllPoints PWA Model - AllPoints: ',
                                str(np.round(PWM_Model['AllPoints']['AllPoints'],4)),

                                '\n PWA Monitors - Avg: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(PWAMonitors_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][3],4)),'AllPoints PWA Monitors - Annual Average: ',
                                str(np.round(PWM_Monitors['AllPoints']['Annual'],4)), 'AllPoints PWA Monitors - AllPoints: ',
                                str(np.round(PWM_Monitors['AllPoints']['AllPoints'],4)),
                                ])
            else:
                writer.writerow([' -------------------------- {} ------------------------'.format(imonth), 
                                '\n Test R2 - Avg: ', str(np.round(test_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(test_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(test_CV_R2_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(test_CV_R2_Alltime['Alltime'][imonth][3],4)), 'AllPoints Test R2 - AllPoints: ',
                                str(np.round(test_CV_R2['AllPoints'][imonth],4)),


                                '\n Slope - Avg: ', str(np.round(slope_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(slope_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(slope_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(slope_Alltime['Alltime'][imonth][3],4)), 'AllPoints Slope - AllPoints: ',
                                str(np.round(slope['AllPoints'][imonth],4)),

                                '\n RMSE -  Avg: ', str(np.round(RMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(RMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(RMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(RMSE_Alltime['Alltime'][imonth][3],4)), 'AllPoints RMSE - AllPoints: ',
                                str(np.round(RMSE['AllPoints'][imonth],4)),

                                '\n NRMSE -  Avg: ', str(np.round(NRMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(NRMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(NRMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(NRMSE_Alltime['Alltime'][imonth][3],4)), 'AllPoints NRMSE - AllPoints: ',
                                str(np.round(NRMSE['AllPoints'][imonth],4)),
                                
                                '\n PWM NRMSE -  Avg: ', str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][3],4)), 'AllPoints PWM NRMSE - AllPoints: ',
                                str(np.round(PMW_NRMSE['AllPoints'][imonth],4)),

                                '\n Training R2 - Avg: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(train_CV_R2_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][3],4)),'AllPoints Training R2 - AllPoints: ',
                                str(np.round(train_CV_R2['AllPoints'][imonth],4)),


                                '\n Geophysical R2 - Avg: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][3],4)), 'AllPoints Geophysical R2 - AllPoints: ',
                                str(np.round(geo_CV_R2['AllPoints'][imonth],4)),
                                
                                '\n PWA Model - Avg: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(PWAModel_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][3],4)), 'AllPoints PWA Model - AllPoints: ',
                                str(np.round(PWM_Model['AllPoints'][imonth],4)),

                                '\n PWA Monitors - Avg: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(PWAMonitors_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][3],4)), 'AllPoints PWA Monitors - AllPoints: ',
                                str(np.round(PWM_Monitors['AllPoints'][imonth],4)),
                                ])
                

    return 


def SensitivityTests_output_text(outfile:str,status:str,Area,test_beginyears,test_endyears,
                test_CV_R2, train_CV_R2, geo_CV_R2, RMSE, NRMSE,PMW_NRMSE,slope,PWM_Model, PWM_Monitors,regional_number,sensitivity_variables_names,sensitivity_test_type):
    
    MONTH = ['Annual','MAM','JJA','SON','DJF','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    test_CV_R2_Alltime, train_CV_R2_Alltime, geo_CV_R2_Alltime,RMSE_Alltime, NRMSE_Alltime, PWM_NRMSE_Alltime,slope_Alltime,PWAModel_Alltime,PWAMonitors_Alltime = calculate_Alltime_Statistics_results(test_beginyears,test_endyears,test_CV_R2, train_CV_R2, geo_CV_R2, RMSE,NRMSE,PMW_NRMSE, slope,PWM_Model,PWM_Monitors,Area)
    Exclude_Variables = ''
    for iname in sensitivity_variables_names:
         Exclude_Variables += ' '+iname

    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        if sensitivity_test_type == 'exclusion':
            writer.writerow(['Area: {} ; Time Period: {} - {}; Exclude Variables: {}'.format(Area, test_beginyears, test_endyears, Exclude_Variables), ' Total Site Number: {}'.format(regional_number)])
        else:
            writer.writerow(['Area: {} ; Time Period: {} - {}; Include Variables: {}'.format(Area, test_beginyears, test_endyears, Exclude_Variables), ' Total Site Number: {}'.format(regional_number)])
        
        for imonth in MONTH:
            if imonth == 'Annual':
                writer.writerow([' -------------------------- {} ------------------------'.format(imonth), 
                                '\n Test R2 - Avg: ', str(np.round(test_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(test_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(test_CV_R2_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(test_CV_R2_Alltime['Alltime'][imonth][3],4)), 'AllPoints Test R2 - Annual Average: ',
                                str(np.round(test_CV_R2['AllPoints']['Annual'],4)), 'AllPoints Test R2 - AllPoints: ',
                                str(np.round(test_CV_R2['AllPoints']['AllPoints'],4)),


                                '\n Slope - Avg: ', str(np.round(slope_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(slope_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(slope_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(slope_Alltime['Alltime'][imonth][3],4)), 'AllPoints Slope - Annual Average: ',
                                str(np.round(slope['AllPoints']['Annual'],4)), 'AllPoints Slope - AllPoints: ',
                                str(np.round(slope['AllPoints']['AllPoints'],4)),

                                '\n RMSE -  Avg: ', str(np.round(RMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(RMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(RMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(RMSE_Alltime['Alltime'][imonth][3],4)), 'AllPoints RMSE - Annual Average: ',
                                str(np.round(RMSE['AllPoints']['Annual'],4)), 'AllPoints RMSE - AllPoints: ',
                                str(np.round(RMSE['AllPoints']['AllPoints'],4)),

                                '\n NRMSE -  Avg: ', str(np.round(NRMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(NRMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(NRMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(NRMSE_Alltime['Alltime'][imonth][3],4)),'AllPoints NRMSE - Annual Average: ',
                                str(np.round(NRMSE['AllPoints']['Annual'],4)), 'AllPoints NRMSE - AllPoints: ',
                                str(np.round(NRMSE['AllPoints']['AllPoints'],4)),
                                
                                '\n PWM NRMSE -  Avg: ', str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][3],4)),'AllPoints PWM NRMSE - Annual Average: ',
                                str(np.round(PMW_NRMSE['AllPoints']['Annual'],4)), 'AllPoints PWM NRMSE - AllPoints: ',
                                str(np.round(PMW_NRMSE['AllPoints']['AllPoints'],4)),

                                '\n Training R2 - Avg: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(train_CV_R2_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][3],4)),'AllPoints Training R2 - Annual Average: ',
                                str(np.round(train_CV_R2['AllPoints']['Annual'],4)), 'AllPoints Training R2 - AllPoints: ',
                                str(np.round(train_CV_R2['AllPoints']['AllPoints'],4)),


                                '\n Geophysical R2 - Avg: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][3],4)),'AllPoints Geophysical R2 - Annual Average: ',
                                str(np.round(geo_CV_R2['AllPoints']['Annual'],4)), 'AllPoints Geophysical R2 - AllPoints: ',
                                str(np.round(geo_CV_R2['AllPoints']['AllPoints'],4)),
                                
                                '\n PWA Model - Avg: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(PWAModel_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][3],4)),'AllPoints PWA Model - Annual Average: ',
                                str(np.round(PWM_Model['AllPoints']['Annual'],4)), 'AllPoints PWA Model - AllPoints: ',
                                str(np.round(PWM_Model['AllPoints']['AllPoints'],4)),

                                '\n PWA Monitors - Avg: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(PWAMonitors_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][3],4)),'AllPoints PWA Monitors - Annual Average: ',
                                str(np.round(PWM_Monitors['AllPoints']['Annual'],4)), 'AllPoints PWA Monitors - AllPoints: ',
                                str(np.round(PWM_Monitors['AllPoints']['AllPoints'],4)),
                                ])
            else:
                writer.writerow([' -------------------------- {} ------------------------'.format(imonth), 
                                '\n Test R2 - Avg: ', str(np.round(test_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(test_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(test_CV_R2_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(test_CV_R2_Alltime['Alltime'][imonth][3],4)), 'AllPoints Test R2 - AllPoints: ',
                                str(np.round(test_CV_R2['AllPoints'][imonth],4)),


                                '\n Slope - Avg: ', str(np.round(slope_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(slope_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(slope_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(slope_Alltime['Alltime'][imonth][3],4)), 'AllPoints Slope - AllPoints: ',
                                str(np.round(slope['AllPoints'][imonth],4)),

                                '\n RMSE -  Avg: ', str(np.round(RMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(RMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(RMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(RMSE_Alltime['Alltime'][imonth][3],4)), 'AllPoints RMSE - AllPoints: ',
                                str(np.round(RMSE['AllPoints'][imonth],4)),

                                '\n NRMSE -  Avg: ', str(np.round(NRMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(NRMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(NRMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(NRMSE_Alltime['Alltime'][imonth][3],4)), 'AllPoints NRMSE - AllPoints: ',
                                str(np.round(NRMSE['AllPoints'][imonth],4)),
                                
                                '\n PWM NRMSE -  Avg: ', str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][0], 4)), 'Min: ',
                                str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][1], 4)), 'Max: ',str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(PWM_NRMSE_Alltime['Alltime'][imonth][3],4)), 'AllPoints PWM NRMSE - AllPoints: ',
                                str(np.round(PMW_NRMSE['AllPoints'][imonth],4)),

                                '\n Training R2 - Avg: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(train_CV_R2_Alltime['Alltime'][imonth][2],4)),
                                'STD: ',str(np.round(train_CV_R2_Alltime['Alltime'][imonth][3],4)),'AllPoints Training R2 - AllPoints: ',
                                str(np.round(train_CV_R2['AllPoints'][imonth],4)),


                                '\n Geophysical R2 - Avg: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(geo_CV_R2_Alltime['Alltime'][imonth][3],4)), 'AllPoints Geophysical R2 - AllPoints: ',
                                str(np.round(geo_CV_R2['AllPoints'][imonth],4)),
                                
                                '\n PWA Model - Avg: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(PWAModel_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(PWAModel_Alltime['Alltime'][imonth][3],4)), 'AllPoints PWA Model - AllPoints: ',
                                str(np.round(PWM_Model['AllPoints'][imonth],4)),

                                '\n PWA Monitors - Avg: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][0], 4)), 'Min: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][1], 4)), 'Max: ',
                                str(np.round(PWAMonitors_Alltime['Alltime'][imonth][2],4)), 
                                'STD: ',str(np.round(PWAMonitors_Alltime['Alltime'][imonth][3],4)), 'AllPoints PWA Monitors - AllPoints: ',
                                str(np.round(PWM_Monitors['AllPoints'][imonth],4)),
                                ])
                

    return 

def Output_Text_Sites_Number(outfile:str,status:str,train_index_number:np.array,test_index_number:np.array, buffer:float):
    with open(outfile,status) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Buffer size - {} km'.format(buffer),
                        '\nTraining index number - Max: ',str(np.max(train_index_number)),' Min: ',str(np.min(train_index_number)),
                         'Average: ',str(np.mean(train_index_number)),
                         '\n Testing index number - Max: ',str(np.max(test_index_number)),' Min: ',str(np.min(test_index_number)),
                         'Average: ',str(np.mean(test_index_number)),
                         ' \n---------------------------------------------------------'])
    return