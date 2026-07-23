from Meteo_input_pkg.data_func import *
from Meteo_input_pkg.utils import *
from Meteo_input_pkg.iostream import *
import numpy as np

def calculate_save_interpolate_indices():
    get_meteorology_interpolate_indices()
    return

def average_interpolate_meteorology_map(YYYY,MM,filetag,nametag,Area,lev):
    DD = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
        '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
        '25', '26', '27', '28', '29', '30', '31']
    
    GC_LAT, GC_LON = load_global_GC_LatLon()
    GEOLAT, GEOLON = load_global_GeoLatLon()
    lat_nearest_array,lat_ceil_array,lat_floor_array,lon_nearest_array,lon_ceil_array,lon_floor_array,dx,dy = load_meteo_interpolate_indices(indir=meteorology_mapdata_outdir)

    lookup_table = File_lookup_table()
    for iyear in range(len(YYYY)):
        Days = [31,29,31,30,31,30,31,31,30,31,30,31] if YYYY[iyear] in ['2008', '2012', '2016', '2020', '2024'] else [31,28,31,30,31,30,31,31,30,31,30,31] 
        for imonth in range(len(MM)):
            for itag in range(len(filetag)):
                print('YEAR: {}, MONTH: {}, Variable: {}'.format(YYYY[iyear],MM[imonth],nametag[itag]))
                temp_days_average = np.zeros((len(GC_LAT),len(GC_LON), Days[imonth]),dtype=np.float64)
                for iday in range(Days[imonth]):
                    meteo_data = load_meteorology_file(filetag=filetag[itag],YYYY=YYYY[iyear],MM=MM[imonth],DD=DD[iday],Area=Area)
                    temp_days_average[:,:,iday] = np.mean(lookup_table[filetag[itag]](meteo_data,nametag[itag],lev),axis=0)
                temp_monthly_average = np.mean(temp_days_average, axis = 2)
                interpolate_meteorology_map(init_mapdata=temp_monthly_average,GeoLAT=GEOLAT,GeoLON=GEOLON,
                                                                                lat_ceil_array=lat_ceil_array,lat_floor_array=lat_floor_array,
                                                                                lon_ceil_array=lon_ceil_array,lon_floor_array=lon_floor_array,
                                                                                dx=dx,dy=dy,tagname=nametag[itag],YEAR=YYYY[iyear],MONTH=MM[imonth],Area=Area)

    return