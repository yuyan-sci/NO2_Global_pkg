Obs_version       = 'v7_filtered'
training_version  = ''
special_name      = ''
version               = 'Global_NO2_v7'
TrainingData_outdir = f'/path/to/NO2_DL_global/TrainingDatasets/{version}{special_name}/'

#######################################################################################################################
#######################################################################################################################
###################################################### INPUT VARIABLES ################################################
#######################################################################################################################
#######################################################################################################################
Observation_indir                   = f'/path/to/NO2_DL_global/TrainingDatasets/{version}{special_name}/'

BASE_INPUT_PATH                     = '/path/to/NO2_DL_global/input_variables/'

ML_NO2_input_indir                  = BASE_INPUT_PATH + 'ML_input/'

GeoNO2_input_indir                  = BASE_INPUT_PATH + 'GeoNO2_input/'
GeoNO2v513_input_indir              = BASE_INPUT_PATH + 'GeoNO2-v5.13_input/'
GeoNO2v513_geoupscaled_input_indir  = BASE_INPUT_PATH + 'GeoNO2-v5.13_geoupscaled_input/'
GeoNO2v513_downscaled_input_indir   = BASE_INPUT_PATH + 'GeoNO2-v5.13_downscaled/'
GCHP_input_dir                      = BASE_INPUT_PATH + 'GCHP_input/'
Meteorology_input_indir             = BASE_INPUT_PATH + 'Meteorology_input/'
CEDS_Emissions_input_indir          = BASE_INPUT_PATH + 'CEDS_Anthro_Emissions_01_input/'
LandCover_input_indir               = BASE_INPUT_PATH + 'LandCover_input/'
Geographical_Variables_input_indir  = BASE_INPUT_PATH + 'Geographical_Variables_input/'
GFED4_input_indir                   = BASE_INPUT_PATH + 'GFED4_Emissions_input/'
Population_input_indir              = BASE_INPUT_PATH + 'Population_input/'
OpenStreetMap_nearest_dist_indir    = BASE_INPUT_PATH + 'OpenStreetMap_RoadDensity_NearestDistances_forEachPixels_input/'
OpenStreetMap_road_density_indir    = BASE_INPUT_PATH + 'OpenStreetMap_RoadDensity_input/'
OpenStreetMap_buffer_indir          = BASE_INPUT_PATH + 'OpenStreetMap_RoadDensity_Buffer_forEachPixels_input/'
NDVI_input_indir                    = BASE_INPUT_PATH + 'NDVI_input/'
ISA_input_indir                     = BASE_INPUT_PATH + 'ISA_input/'
LandCover_Density_input_indir       = BASE_INPUT_PATH + 'LandCover_input/'
LandCover_Distance_input_indir      = BASE_INPUT_PATH + 'LandCover_NearestDistances_forEachPixels_input/'
LandCover_Buffer_input_indir        = BASE_INPUT_PATH + 'LandCover_buffer_forEachPixels_input/'

def inputfiles_table(YYYY, MM):
    # 'GeoNO2-v2' channel: DOWNSCALED GeoNO2 (R_fine * boxcar10, OMI large-scale x
    # TROPOMI fine structure from ref years 2019/2022/2023) for the OMI era
    # 2005-01..2018-05; native 1 km v5.13 GeoNO2 for the TROPOMI era 2018-06..2023.
    # Same cutover as build_downscaled_geono2.py (is_omi = year<2018 or
    # (year==2018 and month<=5)); both dirs share the same GeoNO2_trop_GC_...npy
    # filename, only the parent directory differs.
    _geono2v2_is_omi   = (int(YYYY) < 2018) or (int(YYYY) == 2018 and int(MM) <= 5)
    _GeoNO2v2_indir    = GeoNO2v513_downscaled_input_indir if _geono2v2_is_omi else GeoNO2v513_input_indir

    inputfiles_dic = {
        #####################[Variables from ML_NO2] ###################
        # v7 anchors live in {year}_v7/ (regenerated from the v7 estimation maps);
        # the v5 originals in 2019/ are preserved untouched.
        'ML_NO2_2019'             : ML_NO2_input_indir + '2019/ML_NO2_001x001_Global_map_2019{}.npy'.format(MM),
        'ML_NO2_2019_FSU'         : ML_NO2_input_indir + '2019_v7/ML_NO2_FSU_001x001_Global_map_2019{}.npy'.format(MM),
        'ML_NO2_2019_Map'         : ML_NO2_input_indir + '2019_v7/ML_NO2_Map_001x001_Global_map_2019{}.npy'.format(MM),
        'ML_NO2_2022_FSU'         : ML_NO2_input_indir + '2022_v7/ML_NO2_FSU_001x001_Global_map_2022{}.npy'.format(MM),
        'ML_NO2_2022_Map'         : ML_NO2_input_indir + '2022_v7/ML_NO2_Map_001x001_Global_map_2022{}.npy'.format(MM),
        #####################[Variables from Geophysical Species] ###################
        'GeoNO2'                  : GeoNO2v513_input_indir + '{}/GeoNO2_trop_GC_001x001_Global_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'GeoNO2-v2'               : _GeoNO2v2_indir + '{}/GeoNO2_trop_GC_001x001_Global_map_{}{}.npy'.format(YYYY,YYYY,MM),
        ##################### [Variables from GEOS-Chem] ###################
        'GCHP_NO2'                : GCHP_input_dir + '{}/gchp_NO2_001x001_Global_map_{}{}.npy'.format(YYYY,YYYY,MM),
        ##################### [Variables from Satellite Observations] ###################   
        'SatColNO2_TM'            : GeoNO2v513_input_indir + '{}/SatColNO2_TM5_001x001_Global_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'SatColNO2_GC'            : GeoNO2v513_input_indir + '{}/SatColNO2_GC_001x001_Global_map_{}{}.npy'.format(YYYY,YYYY,MM),
        ##################### [Variables from CEDS Emissions] ###################
        'NO_emi'             : CEDS_Emissions_input_indir + '{}/NO-em-anthro_CMIP_v2025-04_CEDS_Total_001x001_Global_{}{}.npy'.format(YYYY,YYYY,MM),

        ##################### [Variables from GFED4 Dry Matter Emissions] ###################
        'Total_DM'           : GFED4_input_indir + '{}/GFED4-DM_TOTL_001x001_{}{}.npy'.format(YYYY,YYYY,MM),

        ##################### [Variables from Meteorology] ###################
        'GRN'                : Meteorology_input_indir + '{}/GRN_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'LAI'                : Meteorology_input_indir + '{}/LAI_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'PS'                 : Meteorology_input_indir + '{}/PS_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'PBLH'               : Meteorology_input_indir + '{}/PBLH_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'TP'                 : Meteorology_input_indir + '{}/PRECTOT_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'RH'                 : Meteorology_input_indir + '{}/RH_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'T2M'                  : Meteorology_input_indir + '{}/T2M_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'U10M'               : Meteorology_input_indir + '{}/U10M_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'V10M'               : Meteorology_input_indir + '{}/V10M_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'USTAR'              : Meteorology_input_indir + '{}/USTAR_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
        'TSW'                : Meteorology_input_indir + '{}/GWETTOP_001x001_GL_map_{}{}.npy'.format(YYYY,YYYY,MM),
    
        ##################### [Geographical Information] ###################
        'S1'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_1.npy',
        'S2'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_2.npy',
        'S3'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates/Spherical_Coordinates_3.npy',
        'x'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates_2/Spherical_Coordinates_1.npy',
        'y'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates_2/Spherical_Coordinates_2.npy',
        'z'                 : Geographical_Variables_input_indir + 'Spherical_Coordinates_2/Spherical_Coordinates_3.npy',
        'lat'                : '/path/to/NO2_DL_global/input_variables/tSATLAT_global_MAP.npy',
        'lon'                : '/path/to/NO2_DL_global/input_variables/tSATLON_global_MAP.npy',
        'elevation'          : Geographical_Variables_input_indir + 'elevation/elevation_001x001_Global.npy',

        ###################### [Population Information] ####################
        'Population'         : Population_input_indir + 'WorldPopGrid-{}-0.01.npy'.format(YYYY),
    
        ##################### [Variables from NDVI] ##################
        'NDVI'               : NDVI_input_indir + '{}/NDVI-MOD13A3_001x001_Global_{}{}.npy'.format(YYYY,YYYY,MM),

        ##################### [Variables from ISA] ##################
        'ISA'                : ISA_input_indir + '2010/ISA_NOAA_001x001_Global_2010.npy',
        
        ##################### [Variables from Land Cover: Density] ###################
        'forests_density'            : LandCover_Density_input_indir + 'forests/forests-MCD12C1_LandCover_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_density'         : LandCover_Density_input_indir + 'shrublands/shrublands-MCD12C1_LandCover_001x001_Global_{}.npy'.format(YYYY),
        'croplands_density'          : LandCover_Density_input_indir + 'croplands/croplands-MCD12C1_LandCover_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_density': LandCover_Density_input_indir + 'Urban-Builtup-Lands/Urban-Builtup-Lands-MCD12C1_LandCover_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_density'       : LandCover_Density_input_indir + 'Water-Bodies/Water-Bodies-MCD12C1_LandCover_001x001_Global_{}.npy'.format(YYYY),
        
        ##################### [Variables from Land Cover: Distance] ###################
        'forests_distance'            : LandCover_Distance_input_indir + 'forests/forests-MCD12C1_forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_distance'         : LandCover_Distance_input_indir + 'shrublands/shrublands-MCD12C1_forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_distance'          : LandCover_Distance_input_indir + 'croplands/croplands-MCD12C1_forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_distance': LandCover_Distance_input_indir + 'Urban-Builtup-Lands_NearestDistances/Urban-Builtup-Lands_NearestDistances-MCD12C1_forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_distance'       : LandCover_Distance_input_indir + 'Water-Bodies_NearestDistances/Water-Bodies_NearestDistances-MCD12C1_forEachPixel_001x001_Global_{}.npy'.format(YYYY),      
        
        ##################### [Variables from Land Cover: Buffer] ###################
        'forests_buffer-500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-0.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-0.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-0.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-0.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-0.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),   
        'forests_buffer-1000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-1-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-1000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-1-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-1000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-1-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-1000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-1-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-1000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-1-forEachPixel_001x001_Global_{}.npy'.format(YYYY),   
        'forests_buffer-1500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-1.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-1500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-1.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-1500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-1.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-1500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-1.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-1500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-1.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),   
        'forests_buffer-2000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-2-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-2000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-2-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-2000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-2-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-2000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-2-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-2000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-2-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-2500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-2.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-2500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-2.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-2500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-2.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-2500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-2.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-2500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-2.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-3000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-3-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-3000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-3-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-3000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-3-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-3000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-3-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-3000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-3-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-3500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-3.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-3500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-3.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-3500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-3.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-3500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-3.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-3500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-3.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-4000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-4-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-4000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-4-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-4000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-4-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-4000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-4-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-4000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-4-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-4500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-4.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-4500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-4.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-4500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-4.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-4500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-4.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-4500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-4.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-5000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-5000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-5000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-5000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-5000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-5500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-5.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-5500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-5.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-5500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-5.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-5500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-5.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-5500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-5.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-6000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-6-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-6000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-6-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-6000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-6-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-6000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-6-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-6000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-6-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-6500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-6.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-6500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-6.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-6500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-6.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-6500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-6.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-6500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-6.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-7000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-7-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-7000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-7-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-7000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-7-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-7000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-7-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-7000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-7-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-7500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-7.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-7500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-7.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-7500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-7.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-7500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-7.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-7500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-7.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-8000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-8-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-8000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-8-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-8000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-8-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-8000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-8-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-8000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-8-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-8500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-8.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-8500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-8.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-8500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-8.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-8500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-8.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-8500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-8.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-9000'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-9-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-9000'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-9-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-9000'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-9-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-9000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-9-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-9000'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-9-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-9500'            : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-9.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-9500'         : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-9.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-9500'          : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-9.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-9500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-9.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-9500'       : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-9.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-10000'           : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-10-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-10000'        : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-10-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-10000'         : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-10-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-10000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-10-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-10000'      : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-10-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-10500'           : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-10.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-10500'        : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-10.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-10500'         : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-10.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-10500': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-10.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-10500'      : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-10.5-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        'forests_buffer-11000'           : LandCover_Buffer_input_indir + 'forests/forests-MCD12C1_Buffer-11-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'shrublands_buffer-11000'        : LandCover_Buffer_input_indir + 'shrublands/shrublands-MCD12C1_Buffer-11-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'croplands_buffer-11000'         : LandCover_Buffer_input_indir + 'croplands/croplands-MCD12C1_Buffer-11-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'urban_builtup_lands_buffer-11000': LandCover_Buffer_input_indir + 'Urban_Builtup_Lands/Urban_Builtup_Lands-MCD12C1_Buffer-11-forEachPixel_001x001_Global_{}.npy'.format(YYYY),
        'water_bodies_buffer-11000'      : LandCover_Buffer_input_indir + 'Water_Bodies/Water_Bodies-MCD12C1_Buffer-11-forEachPixel_001x001_Global_{}.npy'.format(YYYY),       
        
        ##################### [Open Street Map Road Density nearest distances] ###################
        'major_roads'        : OpenStreetMap_road_density_indir + '2025/OpenStreetMap-Global-major_roads-RoadDensityMap_2025.npy',
        'minor_roads'        : OpenStreetMap_road_density_indir + '2025/OpenStreetMap-Global-minor_roads-RoadDensityMap_2025.npy',
        'major_roads_new'    : OpenStreetMap_road_density_indir + '2025/OpenStreetMap-Global-major_roads_new-RoadDensityMap_2025.npy',
        'minor_roads_new'    : OpenStreetMap_road_density_indir + '2025/OpenStreetMap-Global-minor_roads_new-RoadDensityMap_2025.npy',
        
        'log_major_roads'    : OpenStreetMap_road_density_indir + '2025/OpenStreetMap-Global-major_roads_log-RoadDensityMap_2025.npy',
        'log_minor_roads'    : OpenStreetMap_road_density_indir + '2025/OpenStreetMap-Global-minor_roads_log-RoadDensityMap_2025.npy',
        'log_major_roads_new': OpenStreetMap_road_density_indir + '2025/OpenStreetMap-Global-major_roads_new_log-RoadDensityMap_2025.npy',
        'log_minor_roads_new': OpenStreetMap_road_density_indir + '2025/OpenStreetMap-Global-minor_roads_new_log-RoadDensityMap_2025.npy',
        
        'major_roads_dist'   : OpenStreetMap_nearest_dist_indir + 'major_roads/OpenStreetMap-major_roads-NearestDistanceforEachPixel_001x001_Global_2025.npy',
        'minor_roads_dist'   : OpenStreetMap_nearest_dist_indir + 'minor_roads/OpenStreetMap-minor_roads-NearestDistanceforEachPixel_001x001_Global_2025.npy',
        'major_roads_new_dist': OpenStreetMap_nearest_dist_indir + 'major_roads_new/OpenStreetMap-major_roads_new-NearestDistanceforEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_dist': OpenStreetMap_nearest_dist_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-NearestDistanceforEachPixel_001x001_Global_2025.npy',
        
        'major_roads_buffer-500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-0.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-0.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-0.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-0.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-1000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-1-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-1000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-1-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-1000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-1-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-1000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-1-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-1500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-1.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-1500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-1.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-1500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-1.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-1500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-1.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-2000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-2-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-2000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-2-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-2000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-2-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-2000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-2-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-2500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-2.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-2500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-2.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-2500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-2.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-2500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-2.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-3000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-3-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-3000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-3-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-3000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-3-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-3000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-3-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-3500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-3.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-3500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-3.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-3500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-3.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-3500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-3.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-4000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-4-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-4000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-4-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-4000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-4-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-4000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-4-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-4500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-4.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-4500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-4.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-4500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-4.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-4500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-4.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-5000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-5000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-5000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-5000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-5500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-5.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-5500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-5.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-5500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-5.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-5500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-5.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-6000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-6-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-6000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-6-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-6000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-6-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-6000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-6-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-6500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-6.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-6500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-6.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-6500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-6.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-6500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-6.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-7000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-7-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-7000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-7-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-7000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-7-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-7000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-7-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-7500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-7.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-7500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-7.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-7500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-7.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-7500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-7.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-8000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-8-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-8000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-8-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-8000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-8-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-8000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-8-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-8500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-8.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-8500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-8.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-8500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-8.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-8500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-8.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-9000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-9-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-9000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-9-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-9000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-9-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-9000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-9-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-9500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-9.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-9500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-9.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-9500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-9.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-9500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-9.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-10000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-10-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-10000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-10-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-10000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-10-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-10000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-10-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-10500' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-10.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-10500': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-10.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-10500' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-10.5-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-10500': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-10.5-forEachPixel_001x001_Global_2025.npy',
        'major_roads_buffer-11000' : OpenStreetMap_buffer_indir + 'major_roads/OpenStreetMap-major_roads-Buffer-11-forEachPixel_001x001_Global_2025.npy',
        'major_roads_new_buffer-11000': OpenStreetMap_buffer_indir + 'major_roads_new/OpenStreetMap-major_roads_new-Buffer-11-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_buffer-11000' : OpenStreetMap_buffer_indir + 'minor_roads/OpenStreetMap-minor_roads-Buffer-11-forEachPixel_001x001_Global_2025.npy',
        'minor_roads_new_buffer-11000': OpenStreetMap_buffer_indir + 'minor_roads_new/OpenStreetMap-minor_roads_new-Buffer-11-forEachPixel_001x001_Global_2025.npy',
        
    }
    return inputfiles_dic