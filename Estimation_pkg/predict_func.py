import numpy as np
import torch
import time
from Estimation_pkg.data_func import get_extent_index, get_landtype
from Estimation_pkg.utils import inputfiles_table
from Training_pkg.Model_Func import predict
from Training_pkg.Model_Func import predict_lightgbm

def map_predict_LightGBM(inputmap: np.array, model, train_mean: np.array, train_std: np.array, 
                         extent: list, width: int, nchannel: int, YYYY: str, MM: str, 
                         total_channel_names, main_stream_channel_names, side_channel_names):
    """
    Map prediction function specifically for LightGBM models.
    Extracts center pixel and predicts on 2D data.
    """
    lat_index_global, lon_index_global = get_extent_index(extent)
    landtype = get_landtype(YYYY, extent)
    output = np.full((len(lat_index_global), len(lon_index_global)), -999.0, dtype=np.float32)
    
    print(f'{YYYY}-{MM} Prediction is beginning!')
    Total_start_time = time.time()
    batchsize = 5000
    half_width = (width - 1) // 2
    
    # Get map dimensions
    map_height, map_width = inputmap.shape[1], inputmap.shape[2]
    
    for ix in range(len(lat_index_global)):
        land_index = np.where(landtype[ix, :] != 0)
        
        if ix % 100 == 0:
            print(f'Processing: {100 * ix / len(lat_index_global):.2f}%')
        
        if len(land_index[0]) == 0:
            continue
        
        temp_input = np.zeros((len(land_index[0]), nchannel, width, width), dtype=np.float32)
        
        for iy in range(len(land_index[0])):
            # Use GLOBAL indices directly (no offset needed)
            global_lat_idx = lat_index_global[ix]
            global_lon_idx = lon_index_global[land_index[0][iy]]
            
            # Extract window from full global map
            lat_start = global_lat_idx - half_width
            lat_end = global_lat_idx + half_width + 1
            lon_start = global_lon_idx - half_width
            lon_end = global_lon_idx + half_width + 1
            
            # Skip pixels too close to boundaries
            if (lat_start < 0 or lat_end > map_height or 
                lon_start < 0 or lon_end > map_width):
                # Skip boundary pixels or fill with zeros
                temp_input[iy, :, :, :] = 0
                continue
            
            extracted = inputmap[:, lat_start:lat_end, lon_start:lon_end]
            
            if extracted.shape != (nchannel, width, width):
                temp_input[iy, :, :, :] = 0
                continue
            
            temp_input[iy, :, :, :] = extracted
        
        # Normalize
        temp_input -= train_mean
        temp_input /= train_std
        
        # Predict
        temp_output = predict_lightgbm(
            inputarray=temp_input, 
            model=model, 
            batchsize=batchsize,
            initial_channel_names=total_channel_names,
            mainstream_channel_names=main_stream_channel_names,
            sidestream_channel_names=side_channel_names)
        
        output[ix, land_index[0]] = temp_output
    
    Total_end_time = time.time()
    print(f'{YYYY}-{MM} Prediction completed in {Total_end_time - Total_start_time:.2f}s')
    
    return output

def map_predict(inputmap:np.array, model, train_mean:np.array,train_std:np.array, extent:list,width:int, nchannel:int,
 YYYY:str, MM:str, total_channel_names, main_stream_channel_names,side_channel_names):

    lat_index, lon_index = get_extent_index(extent)
    landtype = get_landtype(YYYY,extent)
    output = np.full((len(lat_index),len(lon_index)),-999.0,dtype=np.float32)
    print(YYYY,MM,' Prediction is beginning!')
    Total_start_time = time.time()
    batchsize = 5000
    for ix in range(len(lat_index)):
        land_index = np.where(landtype[ix,:] != 0)
        
        print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
        if len(land_index[0]) == 0:
            None
        else:
            temp_input = np.zeros((len(land_index[0]), nchannel, width, width), dtype=np.float32)
            GET_INPUT_TIME_START = time.time()
            for iy in range(len(land_index[0])):
                # Calculate indices
                half_width = (width - 1) // 2
                lat_start = int(lat_index[ix] - half_width)
                lat_end = int(lat_index[ix] + half_width + 1)
                lon_start = int(lon_index[land_index[0][iy]] - half_width)
                lon_end = int(lon_index[land_index[0][iy]] + half_width + 1)
                
                # Check boundaries and skip or handle edge pixels
                map_height, map_width = inputmap.shape[1], inputmap.shape[2]
                if (lat_start < 0 or lat_end > map_height or 
                    lon_start < 0 or lon_end > map_width):
                    temp_input[iy, :, :, :] = 0  # Or skip this pixel
                    continue
                
                # Extract the window
                temp_input[iy, :, :, :] = inputmap[:, lat_start:lat_end, lon_start:lon_end]
                
            temp_input -= train_mean
            temp_input /= train_std

            GET_INPUT_TIME_END =time.time()
            GET_INPUT_TIME = GET_INPUT_TIME_END - GET_INPUT_TIME_START
            print('Get Input Time is ', GET_INPUT_TIME, 's, the number of datasets is ', len(land_index[0]))

            GET_PREDICT_TIME_START = time.time()
            temp_output = predict(inputarray=temp_input, model=model, batchsize=batchsize,initial_channel_names=total_channel_names,mainstream_channel_names=main_stream_channel_names,sidestream_channel_names=side_channel_names)
            GET_PREDICT_TIME_END = time.time()
            GET_PREDICT_TIME = GET_PREDICT_TIME_END - GET_PREDICT_TIME_START
            print('Predict time is ', GET_PREDICT_TIME, 's, the number of datasets is ', len(land_index[0]),'batchsize: ', batchsize)

            output[ix,land_index[0]] = temp_output
    Total_end_time = time.time()
    Total_map_predict_time = Total_end_time - Total_start_time
    print(YYYY, MM, 'Prediction Ended! Time is ', Total_map_predict_time, 's', '\nShape of Map:', output.shape)

    return output

def map_final_output(output,extent,YYYY,MM, SPECIES, bias,normalize_bias,normalize_species,absolute_species,log_species,mean,std):
    lat_index, lon_index = get_extent_index(extent)
    infiles = inputfiles_table(YYYY=YYYY,MM=MM)
    GeoSpecies = np.load(infiles['GeoNO2'])
    
    if bias == True:
        final_data = output + GeoSpecies[lat_index[0]:lat_index[-1]+1,lon_index[0]:lon_index[-1]+1]
    elif normalize_bias == True:
        final_data = output * std + mean + GeoSpecies[lat_index[0]:lat_index[-1]+1,lon_index[0]:lon_index[-1]+1]
    elif normalize_species == True:
        final_data = output * std + mean
    elif absolute_species == True:
        final_data = output
    elif log_species == True:
        final_data = np.exp(output) - 1
    return final_data