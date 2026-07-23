import numpy as np
import netCDF4 as nc
import os 

from Uncertainty_pkg.data_func import *
from Uncertainty_pkg.iostream import *
from Uncertainty_pkg.utils import *

from Training_pkg.iostream import load_TrainingVariables
from Training_pkg.utils import *

from Estimation_pkg.iostream import load_estimation_map_data,load_Annual_estimation_map_data

from visualization_pkg.Assemble_Func import plot_save_uncertainty_map_figure,plot_save_uncertainty_LOWESS_bins_relationship_figure

def Derive_Estimation_Uncertainty(total_channel_names,width,height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','MAM','JJA','SON','DJF', 'Annual']
    Uncertainty_Estimation_months_to_Season_dic = {'12':'DJF', '01':'DJF', '02':'DJF',
                                                   '03':'MAM', '04':'MAM', '05':'MAM', 
                                                   '06':'JJA', '07':'JJA', '08':'JJA', 
                                                   '09':'SON', '10':'SON', '11':'SON',  
                                                   'MAM':'MAM', 'JJA':'JJA', 'SON':'SON', 'DJF':'DJF', 'Annual':'Annual'}
    #Uncertainty_Estimation_months_to_Month_dic = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr', '05':'May', '06':'Jun', '07':'Jul', 
    #                                              '08':'Aug', '09':'Sep', '10':'Oct', '11':'Nov', '12':'Dec', 'MAM':'MAM','JJA':'JJA',
    #                                              'SON':'SON','DJF':'DJF', 'Annual':'Annual'}
    if Derive_distances_map_Switch:
        get_nearby_sites_distances_for_each_pixel()
    if Derive_datasets_for_LOWESS_Calculation_Switch:
        SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
        total_sitesnumber = len(lat)
        Get_datasets_for_LOWESS_Calculation(total_channel_names,width,height,total_sitesnumber)
        
    if Derive_BLISCO_LOWESS_Uncertainty_Switch:
        SPECIES_OBS, lat, lon = load_monthly_obs_data(species=species)
        total_sitesnumber = len(lat)
        LOWESS_values,bins_rRMSE,output_bins = Get_LOWESS_values_for_Uncertainty(total_channel_names,width,height,total_sitesnumber)
        save_LOWESS_values_bins(LOWESS_values_dic=LOWESS_values,rRMSE_dic=bins_rRMSE,bins=output_bins,nchannels=len(total_channel_names),width=width,height=height)
    
    
    if Derive_rRMSE_map_Switch: 
        MM = ['01','02','03','04','05','06','07','08','09','10','11','12']
        LOWESS_values,bins_rRMSE,output_bins = load_LOWESS_values_bins(nchannels=len(total_channel_names),width=width,height=height)
        for iyear in range(len(Uncertainty_Estimation_years)):
            for imonth in range(len(MONTH)):
                print('Derive rRMSE map - YEAR:{}, MONTH:{}'.format(Uncertainty_Estimation_years[iyear],MONTH[imonth]))
                distances_map = load_pixels_nearby_sites_distances_map(YYYY=Uncertainty_Estimation_years[iyear],MM=MONTH[imonth])
                rRMSE_Map = convert_distance_to_rRMSE_uncertainty(distances_bins_array=output_bins,BLCO_rRMSE_LOWESS_values=LOWESS_values[MONTH[imonth]],map_distances=distances_map)
                save_rRMSE_uncertainty_Map(Map_rRMSE=rRMSE_Map,YYYY=Uncertainty_Estimation_years[iyear],MM=MONTH[imonth])
            
    if Derive_absolute_Uncertainty_map_Switch:
        for iyear in range(len(Uncertainty_Estimation_years)):
            for imonth in range(len(Uncertainty_Estimation_months)):
                print('Derive Absolute Uncertainty - YEAR:{}, MONTH:{}'.format(Uncertainty_Estimation_years[iyear],Uncertainty_Estimation_months[imonth]))
                if Uncertainty_Estimation_months[imonth] == 'Annual':
                    Estimation_Map, lat, lon = load_Annual_estimation_map_data(YYYY=Uncertainty_Estimation_years[iyear],
                                                          SPECIES=species,version=version,special_name=special_name)
                else:
                    Estimation_Map,lat, lon = load_estimation_map_data(YYYY=Uncertainty_Estimation_years[iyear],MM=Uncertainty_Estimation_months[imonth],
                                                          SPECIES=species,version=version,special_name=special_name)
                rRMSE_Map,lat, lon      = load_rRMSE_map_data(Uncertainty_Estimation_years[iyear], MM=Uncertainty_Estimation_months_to_Season_dic[Uncertainty_Estimation_months[imonth]],version=version,special_name=special_name)
                print('rRMSE type:{}, Estimation type:{}'.format(type(rRMSE_Map),type(Estimation_Map)))
                Absolute_Uncertainty_Map = rRMSE_Map * Estimation_Map
                save_absolute_uncertainty_data(final_data=Absolute_Uncertainty_Map,YYYY=Uncertainty_Estimation_years[iyear],
                                               MM=Uncertainty_Estimation_months[imonth])
    if Uncertainty_visualization_Switch:
        if Uncertainty_Map_plot:
            
            typeNAME = Get_typeName(bias=bias,normalize_bias=normalize_bias,normalize_species=normalize_species,
                                    absolute_species=absolute_species,log_species=log_species,species=species)
            plot_save_uncertainty_map_figure(typeName=typeNAME,width=width,height=height,species=species,version=version,Area=Uncertainty_Plot_Area,PLOT_YEARS=Uncertainty_Estimation_years,PLOT_MONTHS=Uncertainty_Estimation_months)
        if Uncertainty_BLISCO_LOWESS_relationship_plot:
            LOWESS_values,bins_rRMSE,output_bins = load_LOWESS_values_bins(nchannels=len(total_channel_names),width=width,height=height)
            plot_save_uncertainty_LOWESS_bins_relationship_figure(LOWESS_dic=LOWESS_values,rRMSE_dic=bins_rRMSE,output_bins=output_bins,nchannel=len(total_channel_names),
                                                                  width=width,height=height,species=species,version=version)
            

    return