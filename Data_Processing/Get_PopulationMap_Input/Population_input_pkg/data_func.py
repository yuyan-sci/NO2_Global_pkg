import numpy as np


def crop_Mapdata(Init_MapData,lat,lon,Extent):
    bottom_lat = Extent[0]
    top_lat    = Extent[1]
    left_lon   = Extent[2]
    right_lon  = Extent[3]
    lat_start_index = round((bottom_lat - lat[0])* 100 )
    lon_start_index = round((left_lon - lon[0]) * 100 )
    lat_end_index = round((top_lat - lat[0]) * 100 )
    lon_end_index = round((right_lon - lon[0])*100)
    cropped_mapdata = Init_MapData[lat_start_index:lat_end_index+1,lon_start_index:lon_end_index+1]
    return cropped_mapdata

def temporal_linear_interporlate_MapData(YEAR, Left_Point_YEAR, Right_Point_YEAR, Left_Point_Map, Right_Point_Map):
    coefficient = (Right_Point_YEAR - YEAR)/np.abs(Right_Point_YEAR - Left_Point_YEAR)
    TempMap = coefficient * Left_Point_Map + (1-coefficient) * Right_Point_Map
    print('TempMap.shape: ',TempMap.shape)
    return TempMap