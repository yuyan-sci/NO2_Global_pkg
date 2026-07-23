from Population_input_pkg.data_func import *
from Population_input_pkg.iostream import *

def load_crop_interpolate_save_PopulationData(Left_Point_YEARS, Right_Point_YEARS, Aimed_YEAR, Extent):
    for istage in range(len(Left_Point_YEARS)):
        Left_Point_Map,  lat, lon = load_init_Global_Population_density(YEAR=Left_Point_YEARS[istage])
        Right_Point_Map, lat, lon = load_init_Global_Population_density(YEAR=Right_Point_YEARS[istage])
        Cropped_Left_Point_Map  = crop_Mapdata(Init_MapData=Left_Point_Map,  lat=lat,lon=lon,Extent=Extent)
        Cropped_Right_Point_Map = crop_Mapdata(Init_MapData=Right_Point_Map, lat=lat,lon=lon,Extent=Extent)
        print(Aimed_YEAR)
        TempMap = temporal_linear_interporlate_MapData(YEAR=Aimed_YEAR,Left_Point_YEAR=Left_Point_YEARS[istage],
                                                        Right_Point_YEAR=Right_Point_YEARS[istage], Left_Point_Map=Cropped_Left_Point_Map,
                                                        Right_Point_Map=Cropped_Right_Point_Map)
        save_cropped_interpolated_PopulationMap(Population_MapData=TempMap, YEAR=Aimed_YEAR)
    return

def copy_paste_Population_data(Copy_YEARS, Paste_YEARS):
    for iyear in range(len(Copy_YEARS)):
        TempMap = load_cropped_interpolated_PopulationMap(YEAR=Copy_YEARS[iyear])
        save_cropped_interpolated_PopulationMap(Population_MapData=TempMap, YEAR=Paste_YEARS[iyear])
    return