import numpy as np
from Training_pkg.utils import normalization_type

def normalize_Func(inputarray:np.array,observation_data:np.array):
    nonan_index = np.where(~np.isnan(observation_data))[0]
    normalized_inputarray = inputarray.copy()
    array_for_mean = inputarray[nonan_index,:,:].copy()
    array_for_std = inputarray[nonan_index,:,:].copy()
    if normalization_type == 'Gaussian':
        input_mean = np.mean(array_for_mean,axis=0)
        input_std  = np.std(array_for_std,axis=0)
        input_std[np.where(input_std==0)] = 1.0
    elif normalization_type == 'MinMax':
        input_mean = np.min(array_for_mean,axis=0)
        input_std  = np.max(array_for_std,axis=0) - input_mean
    elif normalization_type == 'None':
        input_mean = np.mean(array_for_mean,axis=0)
        input_std = np.std(array_for_std,axis=0)
        
    normalized_inputarray -= input_mean
    normalized_inputarray /= input_std

    print('normalized_inputarray shape:',normalized_inputarray.shape)
    return normalized_inputarray,input_mean,input_std
    


def get_trainingdata_within_start_end_YEAR(initial_array,training_start_YYYY, training_end_YYYY, start_YYYY, sitesnumber):
    final_array = initial_array[(training_start_YYYY-start_YYYY)*12*sitesnumber:(training_end_YYYY-start_YYYY+1)*12*sitesnumber,:,:,:]
    return final_array