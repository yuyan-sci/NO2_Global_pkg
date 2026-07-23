import wandb
from Training_pkg.utils import *

def wandb_initialize():
    # Initialize a new wandb run
    wandb.init(
        # Set the wandb entity where your project will be logged
        entity="yany1-washington-uiversity-in-st-louis",
        # Set the wandb project where this run will be logged
        project="BenchMark",
        # Track hyperparameters and run metadata
        config={},  # This config will be ignored if wandb_sweep_config is used
    )
    return

def wandb_sweep_config():
    # Define the sweep configuration for LightGBM (native API)
    sweep_configuration = {
        'name': 'LightGBM_GOSS_Sweep',  # Name of the sweep
        'entity': 'yany1-washington-uiversity-in-st-louis',
        'project': 'ACAG-Global_NO2-LightGBM-20260102',
        'method': "grid",  # 'grid', 'random', 'bayes' - bayes is recommended for efficiency
        'metric': {
            'name': 'test_R2',
            'goal': 'maximize'
        },
        'parameters': {
            'objective': {
                'values': ["regression"] #"huber", "regression", "regression_l2"
            },
            # Learning parameters
            'learning_rate': {
                'values': [0.001] #0.01, 0.005
            },
            'num_boost_round': {
                'values': [500, 2000, 5000, 10000]
            },
            
            # Tree structure (main complexity control)
            'num_leaves': {
                'values': [511, 1023, 2047]  # 2^n - 1
            },
            'max_depth': {
                'values': [-1]  # -1 means no limit
            },
            'min_data_in_leaf': {
                'values': [20, 50, 100, 200]
            },
            # Histogram bins (affects speed/accuracy trade-off)
            'max_bin': {
                'values': [511]
            },            
            # Feature sampling 0.5, 0.7, 0.9, 1.0
            'feature_fraction': {
                'values': [1.0]
            },
            
            # GOSS parameters (if using boosting_type='goss')
            'top_rate': {
                'values': [0.2]  # keep top X% largest gradients
            },
            'other_rate': {
                'values': [0.1]  # sample X% of small gradients
            },
            
            # Regularization 0.0, 0.5, 1.0, 5.0
            'lambda_l1': {
                'values': [0.0]
            },
            'lambda_l2': {
                'values': [0.0]
            },
            'min_gain_to_split': {
                'values': [0.0]
            },
            
            # Early stopping
            'early_stopping_rounds': {
                'values': [200]
            },
            
            # Feature selection parameter
            'channel_to_exclude': {
                'values': [
                    []
                ]
            }
        }
    }
    return sweep_configuration

def wandb_sweep_parameters_return(sweep_config):
    """Extract LightGBM parameters from wandb sweep config (native API format)"""
    print('wandb_sweep_parameters_return: ', sweep_config)

    fixed_params = {
        'metric': LightGBM_metric,
        'boosting_type': LightGBM_boosting_type,
        'device_type': LightGBM_device,
        'verbose': int(LightGBM_verbose),
        'bagging_fraction': float(LightGBM_bagging_fraction),
        'bagging_freq': int(LightGBM_bagging_freq),
        'num_threads': int(LightGBM_num_threads),
        'enable_bundle': bool(LightGBM_enable_bundle),
        'deterministic': bool(LightGBM_deterministic),
        'force_col_wise': True,
    }
    
    # Swept parameters (varying across runs)
    sweep_params = {
        'objective': str(sweep_config.objective),
        'learning_rate': float(sweep_config.learning_rate),
        'num_boost_round': int(sweep_config.num_boost_round),
        'num_leaves': int(sweep_config.num_leaves),
        'max_depth': int(sweep_config.max_depth),
        'min_data_in_leaf': int(sweep_config.min_data_in_leaf),
        'feature_fraction': float(sweep_config.feature_fraction),
        'top_rate': float(sweep_config.top_rate),
        'other_rate': float(sweep_config.other_rate),
        'lambda_l1': float(sweep_config.lambda_l1),
        'lambda_l2': float(sweep_config.lambda_l2),
        'min_gain_to_split': float(sweep_config.min_gain_to_split),
        'max_bin': int(sweep_config.max_bin),
        'channel_to_exclude': getattr(sweep_config, 'channel_to_exclude', None),
    }
    
    # Combine parameters
    params = {**fixed_params, **sweep_params}

    return params