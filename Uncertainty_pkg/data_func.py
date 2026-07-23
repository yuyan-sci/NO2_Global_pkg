import numpy as np 
import os
from scipy.interpolate import NearestNDInterpolator
from sklearn.neighbors import BallTree
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess
import time

from Training_pkg.iostream import load_monthly_obs_data
from Training_pkg.utils import *

from Estimation_pkg.data_func import get_landtype,get_extent_index
from Estimation_pkg.utils import *

from Evaluation_pkg.utils import *
from Evaluation_pkg.iostream import *

from Uncertainty_pkg.iostream import save_data_for_LOWESS_calculation,load_data_for_LOWESS_calculation,load_GeoLatLon,load_GeoLatLon_Map,save_nearest_site_distances_forEachPixel,save_nearby_site_distances_forEachPixel
from Uncertainty_pkg.utils import *

def Get_datasets_for_LOWESS_Calculation(total_channel_names,width,height,total_sites):
    nchannel = len(total_channel_names)
    SPECIES_OBS, sitelat, sitelon  = load_monthly_obs_data(species)
    total_obs_data = {}
    total_final_data = {}
    total_nearest_distances_data = {}
    total_nearbysites_distances_data = {}

    init_bins = np.linspace(0,Max_distances_for_Bins,Number_of_Bins)
    output_bins = np.array(range(len(init_bins)-1))*round(Max_distances_for_Bins/Number_of_Bins)
    
    Keys = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual','MAM','JJA','SON','DJF']
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    SEASONS = ['MAM','JJA','SON','DJF']
    MONTHS_inSEASONS = [['Mar', 'Apr', 'May'],[ 'Jun', 'Jul', 'Aug'],['Sep', 'Oct', 'Nov'],['Dec','Jan', 'Feb']]

    YEARS = [str(iyear) for iyear in range(Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear+1)]
    typeName = Get_typeName(bias=bias,normalize_bias=normalize_bias,normalize_species=normalize_species,absolute_species=absolute_species,log_species=log_species,species=species)

    for ikey in Keys:
        print('Processing the month/season: {}'.format(ikey))
        total_obs_data[ikey] = np.array([],dtype=np.float64)
        total_final_data[ikey] = np.array([],dtype=np.float64)
        total_nearest_distances_data[ikey] = np.array([],dtype=np.float64)
        total_nearbysites_distances_data[ikey] = np.array([],dtype=np.float64)

    for radius in Uncertainty_Buffer_radii_forUncertainty:
        print('Processing the radius: {}'.format(radius))
        
        obs_data, final_data,geo_data_recording,training_final_data_recording,training_obs_data_recording,testing_population_data_recording, lat_recording, lon_recording,testsites2trainsites_nearest_distances,test_sites_index_recording,train_sites_index_recording,excluded_sites_index_recording, train_index_number, test_index_number=load_month_based_BLCO_data_recording(species, version, typeName, Uncertainty_BLISCO_beginyear, Uncertainty_BLISCO_endyear, nchannel, special_name, width, height,radius,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers)
        
        for iyear in YEARS:
            print('Processing the year: {}'.format(iyear))
            obs_data[iyear]['Annual'] = np.full((12,total_sites),np.nan)
            final_data[iyear]['Annual'] = np.full((12,total_sites),np.nan)
            testsites2trainsites_nearest_distances[iyear]['Annual'] = np.full((12,total_sites),np.nan)
            annual_nearby_distances = np.full((12,total_sites),np.nan)
            for im, imonth in enumerate(MONTHS):
                print('Processing the month: {}'.format(imonth))
                temp_month_distances_recording = np.array([],dtype=np.float64)
                for ifold in range(Uncertainty_BLISCO_kfolds):
                    print('Processing the kfold: {}/{}'.format(ifold+1,Uncertainty_BLISCO_kfolds))
                    temp_ifold_distances_recording = np.full(total_sites,np.nan,dtype=np.float64)
                    ifold_test_sites_index = test_sites_index_recording[iyear][imonth][ifold*total_sites:(ifold+1)*total_sites]
                    ifold_train_sites_index = train_sites_index_recording[iyear][imonth][ifold*total_sites:(ifold+1)*total_sites]
                    nonan_ifold_test_index = np.where(~np.isnan(ifold_test_sites_index))[0]
                    nonan_ifold_train_index = np.where(~np.isnan(ifold_train_sites_index))[0]
                    for isite in range(len(nonan_ifold_test_index)):
                        temp_nearbysites_distance =  Get_NearbySites_Distances_Info(sitelat[nonan_ifold_test_index[isite]],sitelon[nonan_ifold_test_index[isite]],sitelat[nonan_ifold_train_index],sitelon[nonan_ifold_train_index],number_of_nearby_sites_forAverage,nearby_sites_distances_mode)
                        temp_ifold_distances_recording[nonan_ifold_test_index[isite]] = temp_nearbysites_distance
                    temp_month_distances_recording = np.append(temp_month_distances_recording,temp_ifold_distances_recording)
                temp_obs_data = combine_kfolds_test_results(obs_data[iyear][imonth],Uncertainty_BLISCO_kfolds,total_sites)
                temp_final_data = combine_kfolds_test_results(final_data[iyear][imonth],Uncertainty_BLISCO_kfolds,total_sites)
                temp_testsites2trainsites_nearest_distances = combine_kfolds_test_results(testsites2trainsites_nearest_distances[iyear][imonth],Uncertainty_BLISCO_kfolds,total_sites)
                temp_nearby_distances = combine_kfolds_test_results(temp_month_distances_recording,Uncertainty_BLISCO_kfolds,total_sites)

                obs_data[iyear]['Annual'][im,:] = temp_obs_data
                final_data[iyear]['Annual'][im,:] = temp_final_data
                testsites2trainsites_nearest_distances[iyear]['Annual'][im,:] = temp_testsites2trainsites_nearest_distances
                annual_nearby_distances[im,:] = temp_nearby_distances

                total_obs_data[imonth] = np.append(total_obs_data[imonth],temp_obs_data)
                total_final_data[imonth] = np.append(total_final_data[imonth],temp_final_data)
                total_nearbysites_distances_data[imonth] = np.append(total_nearbysites_distances_data[imonth],temp_nearby_distances)
                total_nearest_distances_data[imonth] = np.append(total_nearest_distances_data[imonth],temp_testsites2trainsites_nearest_distances)
            obs_data[iyear]['Annual'] = np.nanmean(obs_data[iyear]['Annual'],axis=0)
            final_data[iyear]['Annual'] = np.nanmean(final_data[iyear]['Annual'],axis=0)
            testsites2trainsites_nearest_distances[iyear]['Annual'] = np.nanmean(testsites2trainsites_nearest_distances[iyear]['Annual'],axis=0)
            annual_nearby_distances = np.nanmean(annual_nearby_distances,axis=0)
            

            for iseason in range(len(SEASONS)):
                season_nearby_distances = np.full((3,total_sites),np.nan)
                obs_data[iyear][SEASONS[iseason]] = np.full((3,total_sites),np.nan)
                final_data[iyear][SEASONS[iseason]] = np.full((3,total_sites),np.nan)
                testsites2trainsites_nearest_distances[iyear][SEASONS[iseason]] = np.full((3,total_sites),np.nan)

                for imonth in range(len(MONTHS_inSEASONS[iseason])):
                    temp_month_distances_recording = np.array([],dtype=np.float64)
                    for ifold in range(Uncertainty_BLISCO_kfolds):
                        temp_ifold_distances_recording = np.full(total_sites,np.nan,dtype=np.float64)
                        ifold_test_sites_index = test_sites_index_recording[iyear][MONTHS_inSEASONS[iseason][imonth]][ifold*total_sites:(ifold+1)*total_sites]
                        ifold_train_sites_index = train_sites_index_recording[iyear][MONTHS_inSEASONS[iseason][imonth]][ifold*total_sites:(ifold+1)*total_sites]
                        nonan_ifold_test_index = np.where(~np.isnan(ifold_test_sites_index))[0]
                        nonan_ifold_train_index = np.where(~np.isnan(ifold_train_sites_index))[0]
                        for isite in range(len(nonan_ifold_test_index)):
                            temp_nearbysites_distance =  Get_NearbySites_Distances_Info(sitelat[nonan_ifold_test_index[isite]],sitelon[nonan_ifold_test_index[isite]],sitelat[nonan_ifold_train_index],sitelon[nonan_ifold_train_index],number_of_nearby_sites_forAverage,nearby_sites_distances_mode)
                            temp_ifold_distances_recording[nonan_ifold_test_index[isite]] = temp_nearbysites_distance
                        temp_month_distances_recording = np.append(temp_month_distances_recording,temp_ifold_distances_recording)
                    temp_obs_data = combine_kfolds_test_results(obs_data[iyear][MONTHS_inSEASONS[iseason][imonth]],Uncertainty_BLISCO_kfolds,total_sites)
                    temp_final_data = combine_kfolds_test_results(final_data[iyear][MONTHS_inSEASONS[iseason][imonth]],Uncertainty_BLISCO_kfolds,total_sites)
                    temp_testsites2trainsites_nearest_distances = combine_kfolds_test_results(testsites2trainsites_nearest_distances[iyear][MONTHS_inSEASONS[iseason][imonth]],Uncertainty_BLISCO_kfolds,total_sites)
                    temp_nearby_distances = combine_kfolds_test_results(temp_month_distances_recording,Uncertainty_BLISCO_kfolds,total_sites)

                    obs_data[iyear][SEASONS[iseason]][imonth,:] = temp_obs_data
                    final_data[iyear][SEASONS[iseason]][imonth,:] = temp_final_data
                    testsites2trainsites_nearest_distances[iyear][SEASONS[iseason]][imonth,:] = temp_testsites2trainsites_nearest_distances
                    season_nearby_distances[imonth,:] = temp_nearby_distances

                    total_obs_data[SEASONS[iseason]] = np.append(total_obs_data[SEASONS[iseason]],temp_obs_data)
                    total_final_data[SEASONS[iseason]] = np.append(total_final_data[SEASONS[iseason]],temp_final_data)
                    total_nearbysites_distances_data[SEASONS[iseason]] = np.append(total_nearbysites_distances_data[SEASONS[iseason]],temp_nearby_distances)
                    total_nearest_distances_data[SEASONS[iseason]] = np.append(total_nearest_distances_data[SEASONS[iseason]],temp_testsites2trainsites_nearest_distances)
                obs_data[iyear][SEASONS[iseason]] = np.nanmean(obs_data[iyear][SEASONS[iseason]],axis=0)
                final_data[iyear][SEASONS[iseason]] = np.nanmean(final_data[iyear][SEASONS[iseason]],axis=0)
                testsites2trainsites_nearest_distances[iyear][SEASONS[iseason]] = np.nanmean(testsites2trainsites_nearest_distances[iyear][SEASONS[iseason]],axis=0)
                season_nearby_distances = np.nanmean(season_nearby_distances,axis=0)
                
                total_obs_data[SEASONS[iseason]] = np.append(total_obs_data[SEASONS[iseason]],obs_data[iyear][SEASONS[iseason]])
                total_final_data[SEASONS[iseason]] = np.append(total_final_data[SEASONS[iseason]],final_data[iyear][SEASONS[iseason]])
                total_nearbysites_distances_data[SEASONS[iseason]] = np.append(total_nearbysites_distances_data[SEASONS[iseason]],season_nearby_distances)
                total_nearest_distances_data[SEASONS[iseason]] = np.append(total_nearest_distances_data[SEASONS[iseason]],testsites2trainsites_nearest_distances)
        
        
            total_obs_data['Annual'] = np.append(total_obs_data['Annual'],obs_data[iyear]['Annual'])
            total_final_data['Annual'] = np.append(total_final_data['Annual'],final_data[iyear]['Annual'])
            total_nearbysites_distances_data['Annual'] = np.append(total_nearbysites_distances_data['Annual'],annual_nearby_distances)
            total_nearest_distances_data['Annual'] = np.append(total_nearest_distances_data['Annual'],testsites2trainsites_nearest_distances[iyear]['Annual'])
        save_data_for_LOWESS_calculation(total_obs_data,total_final_data,total_nearbysites_distances_data,total_nearest_distances_data,radius,nchannel,width,height)
    return

def Get_LOWESS_values_for_Uncertainty(total_channel_names,width,height,total_sites):
    Keys = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual','MAM','JJA','SON','DJF']
    nchannel = len(total_channel_names)
    SPECIES_OBS, sitelat, sitelon  = load_monthly_obs_data(species)
    init_bins = np.linspace(0,Max_distances_for_Bins,Number_of_Bins)
    output_bins = np.array(range(len(init_bins)-1))*round(Max_distances_for_Bins/Number_of_Bins)
    LOWESS_values = {}
    rRMSE         = {}

    for ikey in Keys:
        print('Processing the month/season: {}'.format(ikey))
        LOWESS_values[ikey] = np.array([],dtype=np.float64)
        rRMSE[ikey] = np.array([],dtype=np.float64)
    for index, radius in enumerate(Uncertainty_Buffer_radii_forUncertainty):
        if index == 0:
            print('Get into the first radius: {}'.format(radius))
            total_obs_data,total_final_data,total_nearbysites_distances_data,total_nearest_distances_data = load_data_for_LOWESS_calculation(radius,nchannel,width,height)
        else:
            print('Get into the next radius: {}'.format(radius))
            temp_obs_data,temp_final_data,temp_nearbysites_distances_data,temp_nearest_distances_data = load_data_for_LOWESS_calculation(radius,nchannel,width,height)
            for ikey in Keys:
                total_obs_data[ikey] = np.append(total_obs_data[ikey],temp_obs_data[ikey])
                total_final_data[ikey] = np.append(total_final_data[ikey],temp_final_data[ikey])
                total_nearbysites_distances_data[ikey] = np.append(total_nearbysites_distances_data[ikey],temp_nearbysites_distances_data[ikey])
                total_nearest_distances_data[ikey] = np.append(total_nearest_distances_data[ikey],temp_nearest_distances_data[ikey])
    for imonth in Keys:
        distances = total_nearbysites_distances_data[imonth].copy()
        temp_obs  = total_obs_data[imonth].copy()
        temp_final= total_final_data[imonth].copy()
        number_each_bin = np.array([],dtype=np.float64)
        for i in range(len(init_bins)-1):
            index = np.where(distances<init_bins[i+1])[0]
            temp_rRMSE = Cal_NRMSE_forUncertainty_Bins(temp_final[index],temp_obs[index],Low_percentile_remove,High_percentile_remove)
            rRMSE[imonth] = np.append(rRMSE[imonth],temp_rRMSE)
            distances = np.delete(distances,index)
            temp_final= np.delete(temp_final,index)
            temp_obs  = np.delete(temp_obs,index)
            number_each_bin = np.append(number_each_bin,len(index))
        temp_lowess_result = lowess(rRMSE[imonth], output_bins, frac=LOWESS_frac)
        smoothed_x = temp_lowess_result[:, 0]
        smoothed_y = temp_lowess_result[:, 1]
        LOWESS_values[imonth] = smoothed_y
        
    return LOWESS_values,rRMSE,output_bins

def Get_NearbySites_Distances_Info(test_lat,test_lon,train_lat_array,train_lon_array,number_of_nearby_sites_forAverage,nearby_sites_distances_mode='mean'):
    dist_map = calculate_distance_forArray(test_lat,test_lon,train_lat_array,train_lon_array)
    dist_map.sort()
    if  nearby_sites_distances_mode == 'mean':
        distance = np.mean(dist_map[0:number_of_nearby_sites_forAverage])
    if nearby_sites_distances_mode == 'median':
        if number_of_nearby_sites_forAverage%2 == 1:
            distance = dist_map[int((number_of_nearby_sites_forAverage-1)/2)]
        else:
            distance = np.mean(dist_map[int(number_of_nearby_sites_forAverage)/2-1:int(number_of_nearby_sites_forAverage/2)+1])
    return distance
    
def Cal_NRMSE_forUncertainty_Bins(init_final_data,init_obs_data,low_percentile,high_percentile):
    nonnan_index = np.where(~np.isnan(init_obs_data))
    final_data = init_final_data[nonnan_index].copy()
    obs_data = init_obs_data[nonnan_index].copy()
    ratio = final_data/obs_data
    percentage_array = np.array(range(21))*5
    low_percentile_index = int(low_percentile/5.0)
    high_percentile_index = int(high_percentile/5.0)
    threshold_array = np.percentile(ratio,percentage_array)
    ratio_forCalculation_index = np.where((ratio>=threshold_array[low_percentile_index])&(ratio<=threshold_array[high_percentile_index]))
    #RMSE = np.sqrt(mean_squared_error(ratio[ratio_forCalculation_index], ratio[ratio_forCalculation_index]/ratio[ratio_forCalculation_index]))
    #RMSE = np.sqrt(mean_squared_error(final_data[ratio_forCalculation_index]/obs_data[ratio_forCalculation_index], obs_data[ratio_forCalculation_index]/obs_data[ratio_forCalculation_index]))
    RMSE = np.sqrt(mean_squared_error(final_data[ratio_forCalculation_index],obs_data[ratio_forCalculation_index]))
    #RMSE = np.sqrt(mean_squared_error(final_data,obs_data))
    
    RMSE = round(RMSE, 2)
    
    #NRMSE = RMSE
    NRMSE = RMSE/np.mean(obs_data[ratio_forCalculation_index])
    #NRMSE = RMSE/np.mean(obs_data)
    return NRMSE

def convert_distance_to_rRMSE_uncertainty(distances_bins_array, BLCO_rRMSE_LOWESS_values, map_distances,):
    print('Get into the convert_distance_to_rRMSE_uncertainty!!!!!')
    map_uncertainty = np.zeros(map_distances.shape,dtype=np.float64)
    pixels_index = np.where(map_distances < distances_bins_array[0])
    map_uncertainty[pixels_index] = BLCO_rRMSE_LOWESS_values[0]
    for iradius in range(len(distances_bins_array)-1):
        d_left  = distances_bins_array[iradius]
        d_right = distances_bins_array[iradius+1]
        rRMSE_left  = BLCO_rRMSE_LOWESS_values[iradius]
        rRMSE_right = BLCO_rRMSE_LOWESS_values[iradius+1]
        pixels_index = np.where((map_distances >= d_left) & (map_distances < d_right))
        print('d_left: {}, d_right: {}, rRMSE_left: {}, rRMSE_right: {}'.format(d_left,d_right,rRMSE_left,rRMSE_right))
        map_uncertainty[pixels_index] = (map_distances[pixels_index]-d_left)/(d_right-d_left) * (rRMSE_right - rRMSE_left) +rRMSE_left
        #print('Distance: ', map_distances[pixels_index], 'Uncertainty: ',map_uncertainty[pixels_index])
    
    d_left  = distances_bins_array[0]
    d_right = distances_bins_array[-1]
    rRMSE_left  = BLCO_rRMSE_LOWESS_values[0]
    rRMSE_right = BLCO_rRMSE_LOWESS_values[-1] 
    outrange_pixels_index = np.where(map_distances >= distances_bins_array[-1])
    if BLCO_rRMSE_LOWESS_values[-1] >= BLCO_rRMSE_LOWESS_values[-2]:
        slope = abs(BLCO_rRMSE_LOWESS_values[-1]-BLCO_rRMSE_LOWESS_values[-2])/(distances_bins_array[-1]-distances_bins_array[-2])
        map_uncertainty[outrange_pixels_index] = slope*(map_distances[outrange_pixels_index]-distances_bins_array[-1])+BLCO_rRMSE_LOWESS_values[-1]
    else:
        slope,intercept = m, b = np.polyfit(distances_bins_array,BLCO_rRMSE_LOWESS_values,1)#abs(BLCO_rRMSE_LOWESS_values[-1]-BLCO_rRMSE_LOWESS_values[0])/(distances_bins_array[-1]-distances_bins_array[0])
        map_uncertainty[outrange_pixels_index] = slope*(map_distances[outrange_pixels_index]-distances_bins_array[-1])+BLCO_rRMSE_LOWESS_values[-1]
        #map_uncertainty[outrange_pixels_index] = rRMSE_right #(map_distances[outrange_pixels_index]-d_left)/(d_right-d_left) * (rRMSE_right - rRMSE_left) +rRMSE_left
    
    return map_uncertainty

def get_nearest_site_distance_for_each_pixel():
    SATLAT,SATLON = load_GeoLatLon()
    lat_index, lon_index = get_extent_index(Extent)
    extent_lat_map,extent_lon_map = get_extent_lat_lon_map(lat_index=lat_index,lon_index=lon_index,SATLAT=SATLAT,SATLON=SATLON)
    SPECIES, sites_lat,sites_lon = load_monthly_obs_data(species=species)
    landtype = get_landtype(YYYY=2015,extent=Extent)
    interp_start = time.time()
    #interp = NearestNDInterpolator(list(zip(sites_lat,sites_lon)),sites_index)
    #nearest_index_map = interp(tSATLAT_map,tSATLON_map)
    interp_lat = NearestNDInterpolator(list(zip(sites_lat,sites_lon)),sites_lat)
    interp_lon = NearestNDInterpolator(list(zip(sites_lat,sites_lon)),sites_lon)
    nearest_lat_map = interp_lat(extent_lat_map,extent_lon_map)
    nearest_lon_map = interp_lon(extent_lat_map,extent_lon_map)
    
    interp_end   = time.time()

    interp_total = interp_end - interp_start
    print('Finish the nearest interpolation! Time costs:',interp_total,' seconds')
    nearest_distance_map = np.full(nearest_lat_map.shape,1000.0)
    for ix in range(len(lat_index)):
        land_index = np.where(landtype[ix,:] != 0)
        print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
        if len(land_index[0]) == 0:
            print('No lands.')
            None
        else:
            start_time = time.time()
            nearest_distance_map[ix,land_index[0]] = calculate_distance_forArray(nearest_lat_map[ix,land_index[0]],
                                                                        nearest_lon_map[ix,land_index[0]],
                                                                        extent_lat_map[ix,land_index[0]],extent_lon_map[ix,land_index[0]])
            print(nearest_distance_map[ix,land_index[0]])
            end_time = time.time()
            Get_distance_forOneLatitude_time = end_time - start_time
            print('Time for getting distance for one latitude', Get_distance_forOneLatitude_time, 's, the number of pixels is ', len(land_index[0]))

    save_nearest_site_distances_forEachPixel(nearest_distance_map=nearest_distance_map,extent_lat=SATLAT[lat_index],extent_lon=SATLON[lon_index])
    return 

def get_nearby_sites_distances_for_each_pixel():
    SATLAT,SATLON = load_GeoLatLon()
    lat_index, lon_index = get_extent_index(Extent)
    extent_lat_map,extent_lon_map = get_extent_lat_lon_map(lat_index=lat_index,lon_index=lon_index,SATLAT=SATLAT,SATLON=SATLON)
    SPECIES, sites_lat,sites_lon = load_monthly_obs_data(species=species)
    sitesnumber = len(sites_lat)
    site_index = np.array(range(sitesnumber))
    MM = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    SEASONS = ['MAM','JJA','SON','DJF']
    MONTHS_inSEASONS = [[2, 3, 4],[ 5, 6, 7],[8, 9, 10],[11,0,1]]
    for imodel_year in range(len(Initial_Estimation_Map_trained_beginyears)):
        Yearly_nearby_distance_map = np.full((12,extent_lat_map.shape[0],extent_lat_map.shape[1]),np.nan,dtype=np.float64)
        for imodel_month in range(len(Initial_Estimation_Map_trained_months)):
            valid_sites_index, temp_index_of_initial_array = Get_valid_index_for_temporal_periods(SPECIES_OBS=SPECIES,beginyear=beginyears[imodel_year],endyear=endyears[imodel_year],month_range=training_months[imodel_month],sitesnumber=sitesnumber)
            imodel_siteindex = site_index[valid_sites_index] # This is equivalent to the temp_index_of_initial_array.

            sites_loc = np.array([sites_lat[imodel_siteindex],sites_lon[imodel_siteindex]]).T
            sites_loc = np.radians(sites_loc)
            tree = BallTree(sites_loc, metric='haversine',leaf_size=2) # build ball tree
            landtype = get_landtype(YYYY=2015,extent=Extent)
            nearby_distance_map = np.full(extent_lat_map.shape,np.nan,dtype=np.float64)
            for ix in range(len(lat_index)):
                land_index = np.where(landtype[ix,:] != 0)
                print('It is procceding ' + str(np.round(100*(ix/len(lat_index)),2))+'%.' )
                if len(land_index[0]) == 0:
                    print('No lands.')
                    None
                else:
                    start_time = time.time()
                    temp_pixels_lat_lon = np.radians(np.array([extent_lat_map[ix,land_index[0]], extent_lon_map[ix,land_index[0]]]).T)
                    dist, ind = tree.query(temp_pixels_lat_lon,k=number_of_nearby_sites_forAverage)  # find sites nearby
                    dist = dist * 6371 # convert to km
                    if  nearby_sites_distances_mode == 'mean':
                        distances = np.mean(dist,axis=1)
                    if nearby_sites_distances_mode == 'median':
                        if number_of_nearby_sites_forAverage%2 == 1:
                            distances = dist[:,int((number_of_nearby_sites_forAverage-1)/2)]
                        else:
                            distances = np.mean(dist[:,int(number_of_nearby_sites_forAverage)/2-1:int(number_of_nearby_sites_forAverage/2)+1],axis=1)
                    nearby_distance_map[ix,land_index[0]] = distances
                    end_time = time.time()
                    Get_distance_forOneLatitude_time = end_time - start_time
                    print('Time for getting distance for one latitude', Get_distance_forOneLatitude_time, 's, the number of pixels is ', len(land_index[0]))
            for imonth in range(len(Initial_Estimation_Map_trained_months[imodel_month])):
                Yearly_nearby_distance_map[Initial_Estimation_Map_trained_months[imodel_month][imonth],:] = nearby_distance_map

        for iyear in range((Initial_Estimation_Map_trained_beginyears[imodel_year]-Initial_Estimation_Map_trained_endyears[imodel_year]+1)):
            annual_average_nearby_distance = np.nanmean(Yearly_nearby_distance_map,axis=0)
            save_nearby_site_distances_forEachPixel(nearby_distance_map=annual_average_nearby_distance,extent_lat=SATLAT[lat_index],extent_lon=SATLON[lon_index],YYYY=Initial_Estimation_Map_trained_beginyears[imodel_year]+iyear,MM='Annual')
            for iseason in range(len(SEASONS)):
                seasonal_nearby_distance = np.full((3,extent_lat_map.shape[0],extent_lat_map.shape[1]),np.nan,dtype=np.float64)
                for imonth in range(len(MONTHS_inSEASONS[iseason])):
                    seasonal_nearby_distance[imonth,:] = np.nanmean(Yearly_nearby_distance_map[MONTHS_inSEASONS[iseason][imonth],:],axis=0)
                seasonal_average_nearby_distance = np.nanmean(seasonal_nearby_distance,axis=0)
                save_nearby_site_distances_forEachPixel(nearby_distance_map=seasonal_average_nearby_distance,extent_lat=SATLAT[lat_index],extent_lon=SATLON[lon_index],YYYY=Initial_Estimation_Map_trained_beginyears[imodel_year]+iyear,MM=SEASONS[iseason])
            for imonth in range(len(Initial_Estimation_Map_trained_months[imodel_month])):
                save_nearby_site_distances_forEachPixel(nearby_distance_map=Yearly_nearby_distance_map[Initial_Estimation_Map_trained_months[imodel_month][imonth],:],extent_lat=SATLAT[lat_index],extent_lon=SATLON[lon_index],YYYY=Initial_Estimation_Map_trained_beginyears[imodel_year]+iyear,MM=MM[Initial_Estimation_Map_trained_months[imodel_month][imonth]])
    return

def calculate_distances_for_ArraysAndArrays(lat_array_1,lon_array_1,lat_array_2,lon_array_2):
    """_summary_

    Args:
        lat_array_1 (_type_): 1D
        lon_array_1 (_type_): 1D
        lat_array_2 (_type_): 1D
        lon_array_2 (_type_): 1D
    """
    new_lat_array1 = lat_array_1[:,np.newaxis]
    d_lat = new_lat_array1 - lat_array_2
    dist_map = np.full(d_lat.shape,1000.0)
    for i in range(len(lat_array_1)):
        dist_map[i,:] = calculate_distance_forArray(site_lat=lat_array_1[i],site_lon=lon_array_1[i],SATLAT_MAP=lat_array_2,SATLON_MAP=lon_array_2)
    dist_map = np.sort(dist_map,axis=1)
    if  nearby_sites_distances_mode == 'mean':
        distance = np.mean(dist_map[:,0:number_of_nearby_sites_forAverage],axis=1)
    if nearby_sites_distances_mode == 'median':
        if number_of_nearby_sites_forAverage%2 == 1:
            distance = dist_map[:,int((number_of_nearby_sites_forAverage-1)/2)]
        else:
            distance = np.mean(dist_map[:,int(number_of_nearby_sites_forAverage)/2-1:int(number_of_nearby_sites_forAverage/2)+1],axis=1)
    return distance