import netCDF4 as nc
import numpy as np
from GFED_input_pkg.data_func import get_GFED_indices, interpolate_GFED_map
from GFED_input_pkg.iostream import *
from GFED_input_pkg.utils import *

def interpolate_save_GFED_mapdata(GetIndices, YEARS, MONTHS, Nametags):
    if GetIndices:
        get_GFED_indices()
        print('Indices saved!!!!!1')
    GC_LAT, GC_LON = load_Coarse_LatLon()
    GEOLAT, GEOLON = load_global_GeoLatLon()
    lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy = load_GFED4_interpolate_indices(indir=Offline_GFED_outdir)        
    for YEAR in YEARS:
        for MONTH in MONTHS:
            temp_dataset =  load_OFFLINE_GFED4_dataset(indir=indir,YEAR=YEAR,MONTH=MONTH)
            for keyword in Nametags:
                print('YEAR: {}, MONTH: {}, keyword: {}'.format(YEAR, MONTH, keyword))
                temp_mapdata = get_certain_emissions(data=temp_dataset,keyword=keyword)
                interpolate_GFED_map(init_monthly_mapdata=temp_mapdata,GeoLAT=GEOLAT,GeoLON=GEOLON,lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,
                                        lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,dx=dx,dy=dy,YEAR=YEAR,MONTH=MONTH,nametag=keyword)

    return

def copy_paste_Population_data(Copy_YEARS, Paste_YEARS, MONTHS):
    for iyear in range(len(Copy_YEARS)):
        for imonth in range(len(MONTHS)):
            TempMap = load_GFED_interpolated_mapdata(nametag='DM_TOTL',YEAR=Copy_YEARS[iyear],MONTH=MONTHS[imonth])
            save_GFED_interpolated_mapdata(mapdata=TempMap,nametag='DM_TOTL',YEAR=Paste_YEARS[iyear],MONTH=MONTHS[imonth])
    return