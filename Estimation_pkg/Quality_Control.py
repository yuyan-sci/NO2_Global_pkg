import numpy as np
import time
import os
import gc
import netCDF4 as nc
from Estimation_pkg.utils import *
from Estimation_pkg.data_func import *
from Estimation_pkg.training_func import Train_Model_forEstimation
from Estimation_pkg.predict_func import map_predict,map_final_output
from Estimation_pkg.iostream import Monthly_PWM_no2_output_text,Monthly_PWM_no2_output_text,load_Annual_estimation_map_data, save_annual_final_map_data, save_final_map_data, load_estimation_map_data,save_combinedGeo_map_data

from Training_pkg.iostream import load_TrainingVariables,Learning_Object_Datasets
from Training_pkg.data_func import normalize_Func
from Training_pkg.Statistic_Func import Calculate_PWA_PM25
from Training_pkg.utils import *

from Evaluation_pkg.utils import *

from Mask_func_pkg.utils import *
from Mask_func_pkg.iostream import load_cropped_mask_map

from visualization_pkg.iostream import load_Population_MapData

def Calculate_Regional_PWM_PM_Components():
    MM = ['01','02','03','04','05','06','07','08','09','10','11','12']
    outdir =  txt_outdir + '{}/{}/Results/results-QC_PWM-PM/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        Global_Masks_Dic = {}
        Global_Monthly_PWM_Dic = {}
        Global_Annual_PWM_Dic = {}
        for iregion in additional_test_regions:
            print('{}'.format(iregion))
            Global_Masks_Dic[iregion], Lat, Lon = load_cropped_mask_map(Continent_Name=iregion,region_type_name='continent_mask')
            if Monthly_Analysis_Switch:
                Global_Monthly_PWM_Dic[iregion] = np.zeros((len(Analysis_YEARS)*len(Analysis_MONTH)),dtype=np.float32)
            if Annual_Analysis_Switch:
                Global_Annual_PWM_Dic[iregion] = np.zeros((len(Analysis_YEARS)),dtype=np.float32)


    if Monthly_Analysis_Switch:
        for iyear in range(len(Analysis_YEARS)):
                for imonth in range(len(Analysis_MONTH)):
                    print('YEAR: {}, MM: {}'.format(Analysis_YEARS[iyear],MM[Analysis_MONTH[imonth]]))
                    SPECIES_Map = np.zeros((13000,36000),dtype=np.float32)
                    init_SPECIES_Map, lat, lon = load_estimation_map_data(YYYY=Analysis_YEARS[iyear],MM=MM[Analysis_MONTH[imonth]],SPECIES=species,version=version,special_name=special_name)
                    SPECIES_Map[5:12995,5:35995] = init_SPECIES_Map
                    Population_Map, Pop_lat, Pop_lon = load_Population_MapData(YYYY=Analysis_YEARS[iyear],MM=MM[Analysis_MONTH[imonth]],)

                    for iregion in additional_test_regions:
                        Masked_SPECIES = Global_Masks_Dic[iregion]*SPECIES_Map
                        Global_Monthly_PWM_Dic[iregion][iyear*12+imonth] = Calculate_PWA_PM25(Population_array=Population_Map,PM25_array=Masked_SPECIES)

        outfile = outdir + 'Monthly_Global_Analysis_{}-{}{}.csv'.format(species,version,special_name)
        Monthly_PWM_no2_output_text(PWM_PM_dic=Global_Monthly_PWM_Dic,species=species,YYYY=Analysis_YEARS,MM=[MM[i] for i in Analysis_MONTH],outfile=outfile,areas_list=additional_test_regions)


    if Annual_Analysis_Switch:
        
        for iyear in range(len(Analysis_YEARS)):
            print('YEAR: {}'.format(Analysis_YEARS[iyear]))
            indir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(species,version,Analysis_YEARS[iyear])
            infile =  indir + 'Annual_{}_{}_{}{}.nc'.format(species,version,Analysis_YEARS[iyear],special_name)
            if os.path.exists(infile):
                SPECIES_Map = np.zeros((13000,36000),dtype=np.float32)
                SPECIES_Map[5:5995,5:12995], lat, lon = load_Annual_estimation_map_data(YYYY=Analysis_YEARS[iyear],SPECIES=species,version=version,special_name=special_name)
                Population_Map, Pop_lat, Pop_lon = load_Population_MapData(YYYY=Analysis_YEARS[iyear],MM='01')

                for iregion in additional_test_regions:
                    Masked_SPECIES = Global_Masks_Dic[iregion]*SPECIES_Map
                    Global_Annual_PWM_Dic[iregion][iyear] = Calculate_PWA_PM25(Population_array=Population_Map,PM25_array=Masked_SPECIES)

            else:
                temp_annual_map = np.zeros((13000,36000),dtype=np.float32)
                for imonth in range(12):
                    SPECIES_Map = np.zeros((13000,36000),dtype=np.float32)
                    init_SPECIES_Map, lat, lon = load_estimation_map_data(YYYY=Analysis_YEARS[iyear],MM=MM[imonth],SPECIES=species,version=version,special_name=special_name)
                    SPECIES_Map[5:5995,5:12995] = init_SPECIES_Map
                    temp_annual_map += SPECIES_Map
                temp_annual_map = temp_annual_map/12.0
                save_annual_final_map_data(final_data=temp_annual_map[5:5995,5:12995],YYYY=Analysis_YEARS[iyear],extent=Extent,SPECIES=species,version=version,special_name=special_name)
                Population_Map, Pop_lat, Pop_lon = load_Population_MapData(YYYY=Analysis_YEARS[iyear],MM='01')

                for iregion in additional_test_regions:
                    Masked_SPECIES = Global_Masks_Dic[iregion]*SPECIES_Map
                    Global_Annual_PWM_Dic[iregion][iyear] = Calculate_PWA_PM25(Population_array=Population_Map,PM25_array=Masked_SPECIES)

        outfile = outdir + 'Annual_Global_Analysis_{}-{}{}.csv'.format(species,version,special_name)
        Monthly_PWM_no2_output_text(PWM_PM_dic=Global_Annual_PWM_Dic,species=species,YYYY=Analysis_YEARS,outfile=outfile,areas_list=additional_test_regions)

    return
