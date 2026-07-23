from numba import none
import wandb
from Training_pkg.utils import version


def wandb_initialize():
    # Initialize a new wandb run
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="yany1-washington-uiversity-in-st-louis",
        # Set the wandb project where this run will be logged.
        project="BenchMark",
        # Track hyperparameters and run metadata.
        
        config={

        },## This config will be ignored if wandb_sweep_config is used
    )
    # Return the wandb run object
    return

def wandb_sweep_config():
    # Define the sweep configuration
    sweep_configuration = {
        'name':'HSV_Sweep',  # Name of the sweep
        'entity': 'yany1-washington-uiversity-in-st-louis',  # Your wandb project name
        'project': 'ACAG-Global_NO2-2foldCV-ReLu',  # version
        'method': "grid",  # 'grid', 'random', 'bayes'
        'metric': {
            'name': 'test_R2',
            'goal': 'maximize'  # 'minimize' or 'maximize'
        },
        'parameters': {
            'learning_rate0': {
                'values': [0.001, 0.0001]
            },
            'batch_size': {
                'values': [64, 32, 16, 128]
            },
            'epoch':{
                'values': [51,71,31]
            },
            'ResNet_blocks_num':{
                'values': [[1,1,0,1],[1,1,1,1],[1,0,1,1],[1,0,0,1]]
            },
            'channel_to_exclude': {
                'values': [[],
                           ['GeoNO2'],
                            ['GCHP_NO2'], ['GCHP_OH'],['GCHP_O3'], 
                            ['NDVI'], ['ISA'],
                            ['NO_emi'], ['Total_DM'],
                            ['V10M'], ['U10M'], ['TP'], ['TSW'], ['PS'], ['T2M'], ['RH'], ['PBLH'],['USTAR'],
                            ['elevation'], ['Population'],
                            ['major_roads'], ['log_minor_roads'],
                            ['log_major_roads'], ['log_minor_roads_new'],
                            ['forests_density'], ['shrublands_distance'], ['croplands_distance'],
                            ['urban_builtup_lands_buffer-6500'], ['water_bodies_distance']
                            ],
                        }
                    }
    }
    return sweep_configuration

def wandb_sweep_parameters_return(sweep_config):
    print('wandb_sweep_parameters_return: ', sweep_config)
    batchsize_value = sweep_config.batch_size
    learning_rate0_value = sweep_config.learning_rate0
    epoch_value = sweep_config.epoch
    ResNet_blocks_num_value = sweep_config.ResNet_blocks_num
    channel_to_exclude_value = getattr(sweep_config, 'channel_to_exclude', None)
    return batchsize_value, learning_rate0_value, epoch_value, ResNet_blocks_num_value, channel_to_exclude_value