from Mask_func_pkg.data_func import *
from Mask_func_pkg.iostream import *
from Mask_func_pkg.utils import *

def Convert2Cropped_MaskMap(region_type_name:str, Continent_Name_list:list):
    LANDigIND_0p01, LANDigLAT_0p01, LANDigLON_0p01 = load_mask_index_files()
    GL_GeoLAT, GL_GeoLON = load_GL_GeoLatLon()
    US_GeoLAT, US_GeoLON = load_US_GeoLatLon()
    for Continent_Name in Continent_Name_list:
        print(Continent_Name)
        Mask_Array = load_initial_mask(Continent_Name=Continent_Name, region_type_name=region_type_name)
        Mask_Array = np.squeeze(Mask_Array)
        Mask_map   = get_regional_mask_map(regional_mask_array=Mask_Array,LANDigIND_0p01=LANDigIND_0p01,
                                           tSATLAT=GL_GeoLAT,tSATLON=GL_GeoLON)
        Cropped_Mask_Map = crop_Mapdata(Init_MapData=Mask_map,lat=GL_GeoLAT,lon=GL_GeoLON, Extent=[US_GeoLAT[0],US_GeoLAT[-1],US_GeoLON[0], US_GeoLON[-1]])
        save_cropped_mask_map(Cropped_Map_Data=Cropped_Mask_Map,Geo_lat=US_GeoLAT, Geo_lon=US_GeoLON,Continent_Name=Continent_Name, region_type_name=region_type_name)
    return