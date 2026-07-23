import shap
from Estimation_pkg.iostream import load_estimation_map_data,load_ForcedSlopeUnity_estimation_map_data
from Uncertainty_pkg.iostream import load_absolute_uncertainty_map_data
from visualization_pkg.iostream import (
    save_BLISCO_LOWESS_distances_relationship_figure,
    load_monthly_obs_data_forEstimationMap,
    save_ForcedSlopeUnity_estimation_map_figure,
    save_shap_analysis_figures,
    save_loss_accuracy_figure,
    save_estimation_map_figure,
    load_Population_MapData,
    save_uncertainty_map_figure
)
from visualization_pkg.Training_plot import plot_loss_accuracy_with_epoch, plot_valid_training_loss_accuracy_with_epoch_together
from visualization_pkg.Estimation_plot import Plot_Species_Map_Figures
from visualization_pkg.Uncertainty_plot import Plot_Species_Uncertainty_Map_Figures,Plot_LOWESS_values_bins_Figure
from visualization_pkg.Evaluation_plot import shap_value_plot
from Estimation_pkg.utils import Map_Plot_Extent
from Uncertainty_pkg.utils import *
from visualization_pkg.utils import *
from Training_pkg.utils import *

def SHAPvalues_Analysis_figure(shap_values_with_feature_names:shap._explanation.Explanation,plot_type,typeName, species,version,beginyear,endyear,nchannel,width,height,special_name ):
    outfile = save_shap_analysis_figures(plot_type=plot_type,typeName=typeName,species=species,version=version,
                                        beginyear=beginyear,endyear=endyear,nchannel=nchannel,width=width,height=height,special_name=special_name)
    shap_value_plot(shap_values_with_feature_names=shap_values_with_feature_names,plot_type=plot_type,outfile=outfile)
    return

def plot_save_loss_accuracy_figure(loss, accuracy, valid_loss, valid_accuracy, typeName, species, version, nchannel, width, height, special_name):
    training_fig_outfile, valid_fig_outfile, Combine_fig_outfile = save_loss_accuracy_figure(typeName=typeName,species=species,version=version,nchannel=nchannel,width=width,height=height,special_name=special_name)
    
    plot_loss_accuracy_with_epoch(loss_recording=loss,accuracy_recording=accuracy,outfile=training_fig_outfile)
    plot_loss_accuracy_with_epoch(loss_recording=valid_loss, accuracy_recording=valid_accuracy, outfile=valid_fig_outfile)
    plot_valid_training_loss_accuracy_with_epoch_together(training_loss_recording=loss,training_accuracy_recording=accuracy,valid_accuracy_recording=valid_accuracy,
                                                        valid_loss_recording=valid_loss,outfile=Combine_fig_outfile)
    return

def plot_save_estimation_map_figure(Estimation_Map_Plot:bool,ForcedSlopeUnity_Map_Plot_Switch:bool,typeName:str,width:int,height:int,species:str,version:str,Area:str,PLOT_YEARS:list,PLOT_MONTHS:list):
    SPECIES_OBS, site_lat, site_lon  = load_monthly_obs_data_forEstimationMap(species=species)
    MONTHS = ['01','02','03','04','05','06','07','08','09','10','11','12']
    if Estimation_Map_Plot:
        for YEAR in PLOT_YEARS:
            for MM in PLOT_MONTHS:
                print('YEAR: {}, MONTH: {}'.format(YEAR, MM))

                Estimation_Map_Figure_outfile = save_estimation_map_figure(typeName=typeName,species=species,
                                                                   version=version,Area=Area,nchannel=len(channel_names),width=width,height=height,
                                                                   special_name=special_name,YYYY=YEAR,MM=MM)
                SPECIES_Map, lat, lon = load_estimation_map_data(YYYY=YEAR,MM=MM,SPECIES=species,version=version,special_name=special_name)
                Population_Map, Pop_lat, Pop_lon = load_Population_MapData(YYYY=YEAR,MM=MM)
                month_index = MONTHS.index(MM)
                temp_Species_Obs = SPECIES_OBS[:,(YEAR-2005)*12+month_index]
                Plot_Species_Map_Figures(PM25_Map=SPECIES_Map,PM25_LAT=lat,PM25_LON=lon,PM25_Sites=temp_Species_Obs,PM25_Sites_LAT=site_lat,PM25_Sites_LON=site_lon,Population_Map=Population_Map,population_Lat=Pop_lat,population_Lon=Pop_lon,extent=Map_Plot_Extent,outfile=Estimation_Map_Figure_outfile
                                         ,YYYY=YEAR,MM=MM)
    if ForcedSlopeUnity_Map_Plot_Switch:
        for YEAR in PLOT_YEARS:
            for MM in PLOT_MONTHS:
                print('ForcedSlopeUnity_Map_Plot YEAR: {}, MONTH: {}'.format(YEAR, MM))
                Estimation_Map_ForcedSlopeUnity_Figure_outfile = save_ForcedSlopeUnity_estimation_map_figure(typeName=typeName,species=species,
                                                                   version=version,Area=Area,nchannel=len(channel_names),width=width,height=height,
                                                                   special_name=special_name,YYYY=YEAR,MM=MM)
                month_index = MONTHS.index(MM)
                SPECIES_Map, lat, lon = load_ForcedSlopeUnity_estimation_map_data(YYYY=YEAR,MM=MM,SPECIES=species,version=version,special_name=special_name)
                Population_Map, Pop_lat, Pop_lon = load_Population_MapData(YYYY=YEAR,MM=MM)
                month_index = MONTHS.index(MM)
                temp_Species_Obs = SPECIES_OBS[:,(YEAR-2005)*12+month_index]
                Plot_Species_Map_Figures(PM25_Map=SPECIES_Map,PM25_LAT=lat,PM25_LON=lon,PM25_Sites=temp_Species_Obs,PM25_Sites_LAT=site_lat,PM25_Sites_LON=site_lon,Population_Map=Population_Map,population_Lat=Pop_lat,population_Lon=Pop_lon,extent=Map_Plot_Extent,outfile=Estimation_Map_ForcedSlopeUnity_Figure_outfile
                                         ,YYYY=YEAR,MM=MM)

    return

def plot_save_uncertainty_map_figure(typeName:str,width:int,height:int,species:str,version:str,Area:str,PLOT_YEARS:list,PLOT_MONTHS:list):
    MONTH = ['Annual','01','02','03','04','05','06','07','08','09','10','11','12']
    for YEAR in PLOT_YEARS:
        for MM in PLOT_MONTHS:
            print('YEAR: {}, MONTH: {}'.format(YEAR, MM))
            Uncertainty_Map_Figure_outfile = save_uncertainty_map_figure(typeName=typeName,species=species,YYYY=YEAR,MM=MM,
                                                                   version=version,Area=Area,nchannel=len(channel_names),width=width,height=height,
                                                                   special_name=special_name)
            SPECIES_Uncertainty_Map, lat, lon = load_absolute_uncertainty_map_data(YYYY=YEAR,MM=MM,version=version,special_name=special_name)
            Population_Map, Pop_lat, Pop_lon  = load_Population_MapData(YYYY=YEAR,MM=MM)
            Plot_Species_Uncertainty_Map_Figures(Uncertainty_Map=SPECIES_Uncertainty_Map,PM25_LAT=lat,PM25_LON=lon,Population_Map=Population_Map,population_Lat=Pop_lat,population_Lon=Pop_lon,extent=Uncertainty_Plot_Extent,outfile=Uncertainty_Map_Figure_outfile
                                         ,YYYY=YEAR,MM=MM)
    return

def plot_save_uncertainty_LOWESS_bins_relationship_figure(LOWESS_dic,rRMSE_dic,output_bins,nchannel,width:int,height:int,species:str,version:str,):
    LOWESS_bins_relationship_figure_outfile = save_BLISCO_LOWESS_distances_relationship_figure(nchannel=nchannel,species=species,special_name=special_name,version=version,width=width,height=height)
    Plot_LOWESS_values_bins_Figure(LOWESS_vallues_dic=LOWESS_dic,rRMSE_dic=rRMSE_dic,output_bins=output_bins,outfile=LOWESS_bins_relationship_figure_outfile)
    return