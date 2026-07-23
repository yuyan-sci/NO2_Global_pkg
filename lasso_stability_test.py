import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import toml
import os
import pickle

from Training_pkg.iostream import load_TrainingVariables, load_geophysical_biases_data, load_geophysical_species_data, load_monthly_obs_data, Learning_Object_Datasets
from Training_pkg.utils import *
from Training_pkg.data_func import normalize_Func, get_trainingdata_within_start_end_YEAR
from Training_pkg.Net_Construction import *
from Training_pkg.Statistic_Func import linear_regression

from Evaluation_pkg.utils import *
from Evaluation_pkg.data_func import Get_valid_index_for_temporal_periods,Get_month_based_Index,Get_month_based_XY_indices,GetXIndex,GetYIndex,Get_XY_indices, Get_XY_arraies, Get_final_output, ForcedSlopeUnity_Func, CalculateAnnualR2, CalculateMonthR2, calculate_Statistics_results
from Evaluation_pkg.iostream import *

from lassonet import LassoNetRegressor, LassoNetRegressorCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from functools import partial
import torch

from visualization_pkg.LassoNet_plot import plot_Stability_Selection

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.toml')
cfg = toml.load(config_path)
cfg = toml.load('/path/to/NO2_DL_global_2019/NO2_global_pkg/config.toml')
# *------------------------------------------------------------------------------*#
##   Initialize the array, variables and constants.
# *------------------------------------------------------------------------------*#
### Get training data, label data, initial observation data and geophysical species
width, height, sitesnumber,start_YYYY, TrainingDatasets = load_TrainingVariables(nametags=channel_names)
SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
geophysical_species, geolat, geolon = load_geophysical_species_data(species=species)
true_input, mean, std = Learning_Object_Datasets(bias=bias,Normalized_bias=normalize_bias,Normlized_Speices=normalize_species,Absolute_Species=absolute_species,Log_PM25=log_species,species=species)
Initial_Normalized_TrainingData, input_mean, input_std = normalize_Func(inputarray=TrainingDatasets,observation_data=SPECIES_OBS)
population_data = load_coMonitor_Population()
MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
total_channel_names, main_stream_channel_names, side_channel_names = Get_channel_names(channels_to_exclude=[])
nchannel   = len(total_channel_names)
seed       = 20190130
typeName   = Get_typeName(bias=bias, normalize_bias=normalize_bias,normalize_species=normalize_species, absolute_species=absolute_species, log_species=log_species, species=species)
site_index = np.array(range(sitesnumber))

imodel_year = 0
Normalized_TrainingData = get_trainingdata_within_start_end_YEAR(initial_array=Initial_Normalized_TrainingData, training_start_YYYY=beginyears[imodel_year],training_end_YYYY=endyears[imodel_year],start_YYYY=start_YYYY,sitesnumber=sitesnumber)

print('shape of Normalized_TrainingData: ', Normalized_TrainingData.shape)
valid_sites_index, temp_index_of_initial_array = Get_valid_index_for_temporal_periods(SPECIES_OBS=SPECIES_OBS,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_range=list(range(0, 12)),sitesnumber=sitesnumber)
imodel_siteindex = site_index[valid_sites_index]

train_index = imodel_siteindex
test_index  = imodel_siteindex
X_train, X_test, y_train, y_test = train_test_split(Normalized_TrainingData, true_input)

#Flatten
train_mask = np.where(~np.isnan(y_train))[0]
test_mask  = np.where(~np.isnan(y_test))[0]

X_train_masked = X_train[train_mask,:,2,2]
y_train_masked = y_train[train_mask]
X_test_masked = X_test[test_mask,:,2,2]
y_test_masked = y_test[test_mask]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CV_model_default = LassoNetRegressorCV(device=device)
model_default = LassoNetRegressor(device=device)

# default_model_name_list = ['LassoNet_default', 'LassoNetCV_default']

# for model, model_name in zip([model_default, CV_model_default], default_model_name_list): 
#     print(f"\n--- Stability selection for {model_name} ---")
#     oracle, order, wrong, paths, prob = model.stability_selection(X_train_masked, y_train_masked)
#     # results_dir = txt_outdir + '{}/{}/Results/results-lassonet-stability-selection/{}_{}_{}channel/'.format(species, 'v0.0.1', '_ResNet_MSE_5x5', model_name, 84)
#     results_dir = txt_outdir + '{}/{}/Results/results-lassonet-stability-selection/{}_{}_{}channel/'.format(species, version,special_name, model_name, nchannel)
#     os.makedirs(results_dir, exist_ok=True)
#     np.save(os.path.join(results_dir, f'{version}_{model_name}_oracle.npy'), oracle)
#     np.save(os.path.join(results_dir, f'{version}_{model_name}_order.npy'), order)
#     np.save(os.path.join(results_dir, f'{version}_{model_name}_wrong.npy'), wrong)
#     with open(os.path.join(results_dir, f'{version}_{model_name}_paths.pkl'), 'wb') as f:
#         pickle.dump(paths, f)
#     np.save(os.path.join(results_dir, f'{version}_{model_name}_prob.npy'), prob)
#     # lassonet_plot_dir = txt_outdir + '{}/{}/Figures/figures-LassoNet-stability-selection/{}_{}_{}channel/'.format(species, 'v0.0.1', '_ResNet_MSE_5x5', model_name, nchannel)
#     lassonet_plot_dir = txt_outdir + '{}/{}/Figures/figures-LassoNet-stability-selection/{}_{}_{}channel/'.format(species, version, special_name, model_name, nchannel)
#     os.makedirs(lassonet_plot_dir, exist_ok=True)

#     plot_Stability_Selection(results_dir, lassonet_plot_dir, model_name, nchannel, total_channel_names, Normalized_TrainingData, version)


def slug_hidden(hidden):
    # e.g., (96, 48) -> "h96_48" ; (64,) -> "h64"
    return "h" + "_".join(str(x) for x in hidden)
candidates = [(256, 128, 64, 32), (512, 256, 128, 64)] #[(32,), (48,), (64,), (64,32), (96,48), (128,64)]
for hidden in candidates:
    hslug = slug_hidden(hidden)

    # model  = LassoNetRegressor(device=device,   hidden_dims=hidden, batch_size=512,
    #                            lambda_start=1e-3, optim=partial(torch.optim.Adam, lr=lr0, betas=(Adam_beta0, Adam_beta1), eps=Adam_eps),
    #                            verbose=True)
    model_cv = LassoNetRegressorCV(device=device, hidden_dims=hidden, batch_size=512,
                                   lambda_start=1e-3, optim=partial(torch.optim.Adam, lr=lr0, betas=(Adam_beta0, Adam_beta1), eps=Adam_eps),
                                   verbose=True)

    for mdl, base_name in [(model_cv, "LassoNetCV_custom")]:
        tag = f"{base_name}_{hslug}"
        print(f"\n--- Stability selection for {tag} ---")
        oracle, order, wrong, paths, prob = mdl.stability_selection(X_train_masked, y_train_masked)

        results_dir = os.path.join(
            txt_outdir, f"{species}/{version}/Results/results-lassonet-stability-selection/{special_name}_{tag}_{nchannel}channel"
        )
        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, f"{version}_{tag}_oracle.npy"), oracle)
        np.save(os.path.join(results_dir, f"{version}_{tag}_order.npy"),  order)
        np.save(os.path.join(results_dir, f"{version}_{tag}_wrong.npy"),  wrong)
        with open(os.path.join(results_dir, f"{version}_{tag}_paths.pkl"), "wb") as f:
            pickle.dump(paths, f)
        np.save(os.path.join(results_dir, f"{version}_{tag}_prob.npy"),   prob)

        plot_dir = os.path.join(
            txt_outdir, f"{species}/{version}/Figures/figures-LassoNet-stability-selection/{special_name}_{tag}_{nchannel}channel"
        )
        os.makedirs(plot_dir, exist_ok=True)
        plot_Stability_Selection(results_dir, plot_dir, tag, nchannel, total_channel_names, Normalized_TrainingData, version)