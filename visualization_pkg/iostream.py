import os
import netCDF4 as nc
from Uncertainty_pkg.utils import *
from visualization_pkg.utils import *
from Estimation_pkg.utils import *
from Training_pkg.utils import *

import pandas as pd
import numpy as np

def save_shap_analysis_figures(plot_type,typeName, species,version,beginyear,endyear,nchannel,width,height,special_name ):
    fig_outdir = SHAP_Analysis_outdir + '{}/{}/Figures/figures-SHAP_Analysis/'.format(species, version)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    fig_outfile = fig_outdir + '{}-plots_{}_{}_{}_{}Channel_{}-{}_{}x{}{}.png'.format(plot_type,typeName,species,version,nchannel,beginyear,endyear,width,height,special_name)        
    return fig_outfile

def save_loss_accuracy_figure(typeName, species,version,nchannel,width,height,special_name ):
    fig_outdir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-Loss_Accuracy/'.format(species, version)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    training_fig_outfile =  fig_outdir + 'SpatialCV_Training_{}_{}_{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,nchannel,width,height,special_name)
    valid_fig_outfile =  fig_outdir + 'SpatialCV_Valid_{}_{}_{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,nchannel,width,height,special_name)
    Combine_fig_outfile = fig_outdir + 'SpatialCV_Combine_Training_Valid_{}_{}_{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,nchannel,width,height,special_name)
    return training_fig_outfile, valid_fig_outfile, Combine_fig_outfile

def save_estimation_map_figure(typeName, species,version,Area,YYYY,MM,nchannel,width,height,special_name ):
    fig_outdir = Estimation_Map_outdir + '{}/{}/Figures/figures-Estimation_Map/{}/'.format(species, version,YYYY)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    estimation_map_fig_outfile =  fig_outdir + 'EstimationMap_{}_{}_{}_{}_{}{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,Area,YYYY,MM,nchannel,width,height,special_name)
    return estimation_map_fig_outfile

def save_ForcedSlopeUnity_estimation_map_figure(typeName, species,version,Area,YYYY,MM,nchannel,width,height,special_name ):
    fig_outdir = Estimation_Map_outdir + '{}/{}/Figures/figures-ForcedSlopeUnity_Estimation_Map/{}/'.format(species, version,YYYY)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    estimation_map_fig_outfile =  fig_outdir + 'EstimationMap_ForcedSlopeUnity_{}_{}_{}_{}_{}{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,Area,YYYY,MM,nchannel,width,height,special_name)
    return estimation_map_fig_outfile

def save_uncertainty_map_figure(typeName, species,version,Area,YYYY,MM,nchannel,width,height,special_name ):
    fig_outdir = Uncertainty_Map_outdir + '{}/{}/Figures/figures-Uncertainty_Map/{}/'.format(species, version,YYYY)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    uncertainty_map_fig_outfile =  fig_outdir + 'UncertaintyMap_{}_{}_{}_{}_{}{}_{}Channel_{}x{}{}.png'.format(typeName,species,version,Area,YYYY,MM,nchannel,width,height,special_name)
    return uncertainty_map_fig_outfile

def save_BLISCO_LOWESS_distances_relationship_figure(nchannel, species,version,width,height,special_name ):
    fig_outdir = Uncertainty_Map_outdir + '{}/{}/Figures/figures-BLISCO_LOWESS_distances_relationship/'.format(species, version)
    if not os.path.isdir(fig_outdir):
        os.makedirs(fig_outdir)
    uncertainty_map_fig_outfile =  fig_outdir + 'BLISCO_LOWESS_distances_relationship_{}_{}_{}-folds_{}-SeedsNumbers_0-{}km-{}bins_{}-Mode_{}-NearbySites_{}-{}_{}channels_{}x{}{}.png'.format(version,species,Uncertainty_BLISCO_kfolds,Uncertainty_BLISCO_seeds_numbers,Max_distances_for_Bins,Number_of_Bins,nearby_sites_distances_mode,number_of_nearby_sites_forAverage,Uncertainty_BLISCO_beginyear,Uncertainty_BLISCO_endyear,nchannel,width,height,special_name)
    return uncertainty_map_fig_outfile

def load_Population_MapData(YYYY,MM):
    inputfiles = inputfiles_table(YYYY=YYYY,MM=MM)
    infile = inputfiles['Population']
    tempdata = np.load(infile)
    output = tempdata
    lat = np.linspace(10.005,69.995,6000)
    lon = np.linspace(-169.995,-40.005,13000)
    return output,lat,lon
    
def load_monthly_obs_data_forEstimationMap(species:str):
    infile = ground_observation_data_dir + ground_observation_data_infile
    data = nc.Dataset(infile)
    SPECIES_OBS = data.variables[species][:]
    SPECIES_OBS = np.array(SPECIES_OBS)

    lat = data.variables["latitude"][:]
    lon = data.variables["longitude"][:]
    lat = np.array(lat)
    lon = np.array(lon)

    return SPECIES_OBS, lat, lon 