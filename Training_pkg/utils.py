import torch
import torch.nn as nn

import toml
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.environ.get('NO2_CONFIG_PATH', os.path.join(os.path.dirname(current_dir), 'config.toml'))
cfg = toml.load(config_path)

from Evaluation_pkg.utils import LassoNet_Stability_Selection_Switch, Hyperparameters_Search_Validation_Switch, LassoNet_channel_names

#######################################################################################
# Observation Path
obs_dir = cfg['Pathway']['observations-dir']

geophysical_species_data_dir    = obs_dir['geophysical_species_data_dir']
geophysical_biases_data_dir     = obs_dir['geophysical_biases_data_dir']
ground_observation_data_dir     = obs_dir['ground_observation_data_dir']
geophysical_species_data_infile = obs_dir['geophysical_species_data_infile']
geophysical_biases_data_infile  = obs_dir['geophysical_biases_data_infile']
ground_observation_data_infile  = obs_dir['ground_observation_data_infile']
#######################################################################################
# Training file Path
Training_dir = cfg['Pathway']['TrainingModule-dir']

training_infile = Training_dir['training_infile']
model_outdir = Training_dir['model_outdir']

#######################################################################################
Config_outdir = cfg['Pathway']['Config-outdir']['Config_outdir']
#######################################################################################
# identity settings
identity = cfg['Training-Settings']['identity']

special_name = identity['special_name']
version = identity['version']

#######################################################################################
# Hyperparameters settings
HyperParameters = cfg['Training-Settings']['hyper-parameters']

channel_names = HyperParameters['channel_names']
epoch = HyperParameters['epoch']
batchsize = HyperParameters['batchsize']
normalization_type = HyperParameters['normalization_type']
Hyperparameters_Search_Validation_Switch = cfg['Hyperparameters_Search_Validation-Settings']['Hyperparameters_Search_Validation_Switch']

#######################################################################################
tree_structure_settings         = cfg['Training-Settings']['tree_structure_settings']
LightGBM_setting                = tree_structure_settings['LightGBM']['Settings']
LightGBM_objective              = tree_structure_settings['LightGBM']['objective']
LightGBM_metric                 = tree_structure_settings['LightGBM']['metric']
LightGBM_boosting_type          = tree_structure_settings['LightGBM']['boosting_type']
LightGBM_data_sample_strategy   = tree_structure_settings['LightGBM']['data_sample_strategy']
LightGBM_device                 = tree_structure_settings['LightGBM']['device']
LightGBM_num_leaves             = tree_structure_settings['LightGBM']['num_leaves']
LightGBM_learning_rate          = tree_structure_settings['LightGBM']['learning_rate']
LightGBM_feature_fraction       = tree_structure_settings['LightGBM']['feature_fraction']
LightGBM_bagging_fraction       = tree_structure_settings['LightGBM']['bagging_fraction']
LightGBM_bagging_freq           = tree_structure_settings['LightGBM']['bagging_freq']
LightGBM_verbose                = tree_structure_settings['LightGBM']['verbose']
LightGBM_num_boost_round        = tree_structure_settings['LightGBM']['num_boost_round']
LightGBM_early_stopping_rounds  = tree_structure_settings['LightGBM']['early_stopping_rounds']

LightGBM_top_rate               = tree_structure_settings['LightGBM']['top_rate']
LightGBM_other_rate             = tree_structure_settings['LightGBM']['other_rate']
LightGBM_max_bin                = tree_structure_settings['LightGBM']['max_bin']
LightGBM_lambda_l1              = tree_structure_settings['LightGBM']['lambda_l1']
LightGBM_lambda_l2              = tree_structure_settings['LightGBM']['lambda_l2']
LightGBM_min_data_in_leaf       = tree_structure_settings['LightGBM']['min_data_in_leaf']
LightGBM_min_gain_to_split      = tree_structure_settings['LightGBM']['min_gain_to_split']
LightGBM_max_depth              = tree_structure_settings['LightGBM']['max_depth']
LightGBM_enable_bundle          = tree_structure_settings['LightGBM']['enable_bundle']
LightGBM_num_threads            = tree_structure_settings['LightGBM']['num_threads']
LightGBM_deterministic          = tree_structure_settings['LightGBM']['deterministic']  

# XGBoost Settings
XGBoost_setting                = tree_structure_settings['XGBoost']['Settings']
XGBoost_objective              = tree_structure_settings['XGBoost']['objective']
XGBoost_eval_metric            = tree_structure_settings['XGBoost']['eval_metric']
XGBoost_booster                = tree_structure_settings['XGBoost']['booster']
XGBoost_sample_type            = tree_structure_settings['XGBoost']['sample_type']
XGBoost_normalize_type         = tree_structure_settings['XGBoost']['normalize_type']
XGBoost_rate_drop              = tree_structure_settings['XGBoost']['rate_drop']
XGBoost_skip_drop              = tree_structure_settings['XGBoost']['skip_drop']
XGBoost_tree_method            = tree_structure_settings['XGBoost']['tree_method']
XGBoost_device                 = tree_structure_settings['XGBoost']['device']
XGBoost_max_leaves             = tree_structure_settings['XGBoost']['max_leaves']
XGBoost_learning_rate          = tree_structure_settings['XGBoost']['learning_rate']
XGBoost_colsample_bytree       = tree_structure_settings['XGBoost']['colsample_bytree']
XGBoost_subsample              = tree_structure_settings['XGBoost']['subsample']
XGBoost_verbosity              = tree_structure_settings['XGBoost']['verbosity']
XGBoost_num_boost_round        = tree_structure_settings['XGBoost']['num_boost_round']
XGBoost_early_stopping_rounds  = tree_structure_settings['XGBoost']['early_stopping_rounds']
XGBoost_max_bin                = tree_structure_settings['XGBoost']['max_bin']
XGBoost_reg_alpha              = tree_structure_settings['XGBoost']['reg_alpha']
XGBoost_reg_lambda             = tree_structure_settings['XGBoost']['reg_lambda']
XGBoost_min_child_weight       = tree_structure_settings['XGBoost']['min_child_weight']
XGBoost_gamma                  = tree_structure_settings['XGBoost']['gamma']
XGBoost_max_depth              = tree_structure_settings['XGBoost']['max_depth']
XGBoost_n_jobs                 = tree_structure_settings['XGBoost']['n_jobs']

######################################################################################
# Ensemble Settings
ensemble_settings                = cfg['Training-Settings']['ensemble_settings']
Ensemble_setting                 = ensemble_settings['Ensemble']['Settings']
Ensemble_base_models             = ensemble_settings['Ensemble']['base_models']
Ensemble_strategy                = ensemble_settings['Ensemble']['strategy']
Ensemble_custom_weights          = ensemble_settings['Ensemble']['custom_weights']
Ensemble_evaluate_individual     = ensemble_settings['Ensemble']['evaluate_individual_models']
Ensemble_save_predictions        = ensemble_settings['Ensemble']['save_ensemble_predictions']

######################################################################################
# Net Structure Settings

net_structure_settings = cfg['Training-Settings']['net_structure_settings']

TwoCombineModels_Settings = net_structure_settings['TwoCombineModels']['Settings']
TwoCombineModels_Variable = net_structure_settings['TwoCombineModels']['Variable']
TwoCombineModels_threshold = net_structure_settings['TwoCombineModels']['threshold']

ResNet_setting      = net_structure_settings['ResNet']['Settings']
ResNet_Blocks       = net_structure_settings['ResNet']['Blocks']
ResNet_blocks_num   = net_structure_settings['ResNet']['blocks_num']



ResNet_MLP_setting      = net_structure_settings['ResNet_MLP']['Settings']
ResNet_MLP_Blocks       = net_structure_settings['ResNet_MLP']['Blocks']
ResNet_MLP_blocks_num   = net_structure_settings['ResNet_MLP']['blocks_num']

ResNet_Classification_Settings                        = net_structure_settings['ResNet_Classification']['Settings']
ResNet_Classification_Blocks                          = net_structure_settings['ResNet_Classification']['Blocks']
ResNet_Classification_blocks_num                      = net_structure_settings['ResNet_Classification']['blocks_num']
ResNet_Classification_left_bin                        = net_structure_settings['ResNet_Classification']['left_bin']
ResNet_Classification_right_bin                       = net_structure_settings['ResNet_Classification']['right_bin']
ResNet_Classification_bins_number                     = net_structure_settings['ResNet_Classification']['bins_number']

ResNet_MultiHeadNet_Settings                          = net_structure_settings['ResNet_MultiHeadNet']['Settings']
ResNet_MultiHeadNet_Blocks                            = net_structure_settings['ResNet_MultiHeadNet']['Blocks']
ResNet_MultiHeadNet_blocks_num                        = net_structure_settings['ResNet_MultiHeadNet']['blocks_num']
ResNet_MultiHeadNet_left_bin                          = net_structure_settings['ResNet_MultiHeadNet']['left_bin']
ResNet_MultiHeadNet_right_bin                         = net_structure_settings['ResNet_MultiHeadNet']['right_bin']
ResNet_MultiHeadNet_bins_number                       = net_structure_settings['ResNet_MultiHeadNet']['bins_number']
ResNet_MultiHeadNet_regression_portion                = net_structure_settings['ResNet_MultiHeadNet']['regression_portion']
ResNet_MultiHeadNet_classifcation_portion             = net_structure_settings['ResNet_MultiHeadNet']['classifcation_portion']

LateFusion_setting      = net_structure_settings['LateFusion']['Settings']
LateFusion_Blocks       = net_structure_settings['LateFusion']['Blocks']
LateFusion_blocks_num   = net_structure_settings['LateFusion']['blocks_num']
LateFusion_initial_channels     = net_structure_settings['LateFusion']['initial_channels']
LateFusion_latefusion_channels  = net_structure_settings['LateFusion']['LateFusion_channels']

MultiHeadLateFusion_settings               = net_structure_settings['MultiHeadLateFusion']['Settings']
MultiHeadLateFusion_Blocks                 = net_structure_settings['MultiHeadLateFusion']['Blocks']
MultiHeadLateFusion_blocks_num             = net_structure_settings['MultiHeadLateFusion']['blocks_num']
MultiHeadLateFusion_initial_channels       = net_structure_settings['MultiHeadLateFusion']['initial_channels']
MultiHeadLateFusion_LateFusion_channels    = net_structure_settings['MultiHeadLateFusion']['LateFusion_channels']
MultiHeadLateFusion_left_bin               = net_structure_settings['MultiHeadLateFusion']['left_bin']
MultiHeadLateFusion_right_bin              = net_structure_settings['MultiHeadLateFusion']['right_bin']
MultiHeadLateFusion_bins_number            = net_structure_settings['MultiHeadLateFusion']['bins_number']
MultiHeadLateFusion_regression_portion     = net_structure_settings['MultiHeadLateFusion']['regression_portion']
MultiHeadLateFusion_classifcation_portion  = net_structure_settings['MultiHeadLateFusion']['classifcation_portion']

UNet_setting = net_structure_settings['UNet']['Settings']
UNet_depth = net_structure_settings['UNet']['depth']
UNet_use_residual_blocks = net_structure_settings['UNet']['use_residual_blocks']
UNet_blocks_per_level = net_structure_settings['UNet']['blocks_per_level']
UNet_use_attention = net_structure_settings['UNet']['use_attention']
UNet_bottleneck_type = net_structure_settings['UNet']['bottleneck_type']
UNet_remove_pooling = net_structure_settings['UNet']['remove_pooling']
UNet_decoder_channels = net_structure_settings['UNet']['decoder_channels']
#######################################################################################
# Optimizer settings

Optimizer_settings = cfg['Training-Settings']['optimizer']

Adam_settings      = Optimizer_settings['Adam']['Settings']
Adam_beta0         = Optimizer_settings['Adam']['beta0']
Adam_beta1         = Optimizer_settings['Adam']['beta1']
Adam_eps           = Optimizer_settings['Adam']['eps']
Adam_L2_regularization   = Optimizer_settings['L2_regularization']
Adam_weight_decay  = Optimizer_settings['Adam']['weight_decay']
#######################################################################################
# learning rate settings
lr_settings = cfg['Training-Settings']['learning_rate']
lr0 = lr_settings['learning_rate0']


### Strategy
ExponentialLR = lr_settings['ExponentialLR']['Settings']
ExponentialLR_gamma = lr_settings['ExponentialLR']['gamma']

CosineAnnealingLR = lr_settings['CosineAnnealingLR']['Settings']
CosineAnnealingLR_T_max = lr_settings['CosineAnnealingLR']['T_max']
CosineAnnealingLR_eta_min = lr_settings['CosineAnnealingLR']['eta_min']

CosineAnnealingRestartsLR = lr_settings['CosineAnnealingRestartsLR']['Settings']
CosineAnnealingRestartsLR_T_0 = lr_settings['CosineAnnealingRestartsLR']['T_0']
CosineAnnealingRestartsLR_T_mult = lr_settings['CosineAnnealingRestartsLR']['T_mult']
CosineAnnealingRestartsLR_eta_min = lr_settings['CosineAnnealingRestartsLR']['eta_min']

#######################################################################################
# dropout settings
dropout_settings = cfg['Training-Settings']['dropout']
standard_dropout = dropout_settings['standard_dropout']
dropout_rate = dropout_settings['dropout_rate']
fc_dropout_rate = dropout_settings['fc_dropout_rate']
DropBlock = dropout_settings['DropBlock']
dropblock_rate = dropout_settings['dropblock_rate']
dropblock_size = dropout_settings['dropblock_size']
#######################################################################################
He_initialization_settings = cfg['Training-Settings']['He_initialization']
He_initialization = He_initialization_settings['He_initialization']
#######################################################################################
Data_Augmentation_Settings = cfg['Training-Settings']['Data_Augmentation']
Data_Augmentation_Switch = Data_Augmentation_Settings['Data_Augmentation_Switch']
#######################################################################################
# activation func settings
activation_func_settings = cfg['Training-Settings']['activation_func']
activation_func_name = activation_func_settings['activation_func_name']
ReLU_ACF = activation_func_settings['ReLU']['Settings']
Tanh_ACF = activation_func_settings['Tanh']['Settings']
GeLU_ACF = activation_func_settings['GeLU']['Settings']
Sigmoid_ACF = activation_func_settings['Sigmoid']['Settings']

#######################################################################################
# Learning Objectives Settings
learning_objective = cfg['Training-Settings']['learning-objective']

species = learning_objective['species']

bias = learning_objective['bias']
normalize_bias = learning_objective['normalize_bias']
normalize_species = learning_objective['normalize_species']
absolute_species = learning_objective['absolute_species']
log_species = learning_objective['log_species']

#######################################################################################
# Loss Function Settings
Loss_Func = cfg['Training-Settings']['Loss-Functions']

Loss_type = Loss_Func['Loss_type']
Classification_loss_type = Loss_Func['Classification_loss_type']
Adj_MSE_alpha = Loss_Func['Adj_MSE_alpha']
Adj_MSE_beta = Loss_Func['Adj_MSE_beta']
GeoMSE_Lamba1_Penalty1 = Loss_Func['GeoMSE_Lamba1_Penalty1']
GeoMSE_Lamba1_Penalty2 = Loss_Func['GeoMSE_Lamba1_Penalty2']
GeoMSE_Gamma = Loss_Func['GeoMSE_Gamma']
ResNet_MultiHeadNet_regression_loss_coefficient = Loss_Func['ResNet_MultiHeadNet_regression_loss_coefficient']
ResNet_MultiHeadNet_classfication_loss_coefficient = Loss_Func['ResNet_MultiHeadNet_classfication_loss_coefficient']
#######################################################################################
# CombineWithGeophysical Settings
CombineWithGeophysical = cfg['Training-Settings']['CombineWithGeophysical']

combine_with_GeophysicalSpeceis_Switch                = CombineWithGeophysical['combine_with_GeophysicalSpeceis_Switch']
cutoff_size                                           = CombineWithGeophysical['cutoff_size']



training_infile = training_infile.format(species,species)
geophysical_biases_data_infile  = geophysical_biases_data_infile.format(species)
geophysical_species_data_infile = geophysical_species_data_infile.format(species)
ground_observation_data_infile  = ground_observation_data_infile.format(species)

def activation_function_table():
    if ReLU_ACF == True:
        return 'relu' #nn.ReLU()
    elif Tanh_ACF == True:
        return 'tanh' #nn.Tanh()
    elif GeLU_ACF == True:
        return 'gelu' #nn.GELU()
    elif Sigmoid_ACF == True:
        return 'sigmoid' #nn.Sigmoid()
    

def lr_strategy_lookup_table(optimizer):
    if ExponentialLR:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=ExponentialLR_gamma)
    elif CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=CosineAnnealingLR_T_max,eta_min=CosineAnnealingLR_eta_min)
    elif CosineAnnealingRestartsLR:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=CosineAnnealingRestartsLR_T_0,T_mult=CosineAnnealingRestartsLR_T_mult,eta_min=CosineAnnealingRestartsLR_eta_min)



def find_latfusion_index(total_channel_names,initial_channels,late_fusion_channels):
    initial_channel_index = []
    for i in range(len(initial_channels)):
        initial_channel_index.append(total_channel_names.index(initial_channels[i]))
    
    latefusion_channel_index = []
    for i in range(len(late_fusion_channels)):
        latefusion_channel_index.append(total_channel_names.index(late_fusion_channels[i]))
    
    return initial_channel_index, latefusion_channel_index
    

def optimizer_lookup(model_parameters,learning_rate):
    if Adam_settings:
        if Adam_L2_regularization:
            return torch.optim.Adam(params=model_parameters,betas=(Adam_beta0, Adam_beta1),eps=Adam_eps, lr=learning_rate, weight_decay=Adam_weight_decay)
        else:
            return torch.optim.Adam(params=model_parameters,betas=(Adam_beta0, Adam_beta1),eps=Adam_eps, lr=learning_rate)
            
def Get_channel_names(channels_to_exclude:list):
    if ResNet_setting or LightGBM_setting or Hyperparameters_Search_Validation_Switch or ResNet_MLP_setting or ResNet_Classification_Settings or ResNet_MultiHeadNet_Settings or UNet_setting:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = channel_names.copy()
            side_channel_names = []
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = channel_names.copy()
            side_channel_names = []
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
    elif LassoNet_Stability_Selection_Switch:
        if len(channels_to_exclude) == 0:
            total_channel_names = LassoNet_channel_names.copy()
            main_stream_channel_names = LassoNet_channel_names.copy()
            side_channel_names = []
        else:
            total_channel_names = LassoNet_channel_names.copy()
            main_stream_channel_names = LassoNet_channel_names.copy()
            side_channel_names = []
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
    elif LateFusion_setting:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = LateFusion_initial_channels.copy()
            side_channel_names = LateFusion_latefusion_channels.copy()
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = LateFusion_initial_channels.copy()
            side_channel_names = LateFusion_latefusion_channels.copy()
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in side_channel_names:
                    side_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the side channel list.'.format(channels_to_exclude[ichannel]))
    elif MultiHeadLateFusion_settings:
        if len(channels_to_exclude) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = MultiHeadLateFusion_initial_channels.copy()
            side_channel_names = MultiHeadLateFusion_LateFusion_channels.copy()
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = MultiHeadLateFusion_initial_channels.copy()
            side_channel_names = MultiHeadLateFusion_LateFusion_channels.copy()
            for ichannel in range(len(channels_to_exclude)):
                if channels_to_exclude[ichannel] in total_channel_names:
                    total_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the total channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in main_stream_channel_names:
                    main_stream_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the main channel list.'.format(channels_to_exclude[ichannel]))
                if channels_to_exclude[ichannel] in side_channel_names:
                    side_channel_names.remove(channels_to_exclude[ichannel])
                else:
                    print('{} is not in the side channel list.'.format(channels_to_exclude[ichannel]))

    return total_channel_names, main_stream_channel_names, side_channel_names

def Add_channel_names(channels_to_add:list):
    if ResNet_setting or ResNet_MLP_setting or ResNet_Classification_Settings or ResNet_MultiHeadNet_Settings or UNet_setting:
        if len(channels_to_add) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = channel_names.copy()
            side_channel_names = []
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = channel_names.copy()
            side_channel_names = []
            for ichannel in range(len(channels_to_add)):
                if channels_to_add[ichannel] in total_channel_names:
                    print('{} is in the initial channel list.'.format(channels_to_add[ichannel]))
                else:
                    total_channel_names.append(channels_to_add[ichannel])
                if channels_to_add[ichannel] in main_stream_channel_names:
                    print('{} is in the main channel list.'.format(channels_to_add[ichannel]))
                else:
                    main_stream_channel_names.append(channels_to_add[ichannel])
    elif LateFusion_setting:
        if len(channels_to_add) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = LateFusion_initial_channels.copy()
            side_channel_names = LateFusion_latefusion_channels.copy()
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = LateFusion_initial_channels.copy()
            side_channel_names = LateFusion_latefusion_channels.copy()
            for ichannel in range(len(channels_to_add)):
                if channels_to_add[ichannel] in total_channel_names:
                    print('{} is in the total channel list.'.format(channels_to_add[ichannel]))
                    
                else:
                    total_channel_names.append(channels_to_add[ichannel])
                if channels_to_add[ichannel] in main_stream_channel_names:
                    print('{} is in the main channel list.'.format(channels_to_add[ichannel]))
                else:
                    main_stream_channel_names.append(channels_to_add[ichannel])
    elif MultiHeadLateFusion_settings:
        if len(channels_to_add) == 0:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = MultiHeadLateFusion_initial_channels.copy()
            side_channel_names = MultiHeadLateFusion_LateFusion_channels.copy()
        else:
            total_channel_names = channel_names.copy()
            main_stream_channel_names = MultiHeadLateFusion_initial_channels.copy()
            side_channel_names = MultiHeadLateFusion_LateFusion_channels.copy()
            for ichannel in range(len(channels_to_add)):
                if channels_to_add[ichannel] in total_channel_names:
                    print('{} is in the total channel list.'.format(channels_to_add[ichannel]))
                else:
                    total_channel_names.append(channels_to_add[ichannel])
                    
                if channels_to_add[ichannel] in main_stream_channel_names:
                    print('{} is in the main channel list.'.format(channels_to_add[ichannel]))
                else:
                    main_stream_channel_names.remove(channels_to_add[ichannel])
                    

    return total_channel_names, main_stream_channel_names, side_channel_names