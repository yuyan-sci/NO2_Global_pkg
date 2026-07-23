import numpy as np

def get_regional_mask_map(regional_mask_array,LANDigIND_0p01,tSATLAT,tSATLON):
    Mask_map = np.zeros((len(tSATLAT),len(tSATLON)))
    index = np.where(regional_mask_array == 1)
    LAT_index = np.mod(LANDigIND_0p01[index]-1,13000)
    LON_index = np.floor(LANDigIND_0p01[index]/13000).astype(int)
    Mask_map[LAT_index,LON_index] = 1.0
    return Mask_map

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