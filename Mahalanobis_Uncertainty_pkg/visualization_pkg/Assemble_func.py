
from map_uncertainty_func.iostream import load_absolute_uncertainty_map,load_estimation_map_data,load_mahalanobis_distance_map,load_rRMSE_map
import numpy as np
from visualization_pkg.iostream import get_absolute_uncertainty_filepath,get_map_estimation_filepath,get_mahalanobis_distance_filepath, get_rRMSE_uncertainty_filepath,get_rRMSE_uncertainty_filepath
from visualization_pkg.map_plot_func import Plot_Map_estimation_Map_Figures,Plot_absolute_Uncertainty_Map_Figures,Plot_Mahalanobis_distance_Map_Figures, Plot_rRMSE_Uncertainty_Map_Figures

from visualization_pkg.utils import extent


def plot_map_estimation_data(species,map_estimation_version,YYYY,MM,map_estimation_special_name):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']
    print(f"Plotting map estimation data for species: {species}, version: {map_estimation_version}, date: {YYYY}-{MONTH[MM]}")
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')
    plot_map = np.zeros((len(SATLAT),len(SATLON)),dtype=np.float64)
    if MM != 12:
        map_data, lat, lon = load_estimation_map_data(YYYY=YYYY, MM=MONTH[MM], SPECIES=species, version=map_estimation_version, special_name=map_estimation_special_name)
        h, w = map_data.shape
        plot_map[5:5+h, 5:5+w] = map_data
    else:
        for m in range(12):
            monthly_map, lat, lon = load_estimation_map_data(YYYY=YYYY, MM=MONTH[m], SPECIES=species, version=map_estimation_version, special_name=map_estimation_special_name)
            h, w = monthly_map.shape
            plot_map[5:5+h, 5:5+w] += monthly_map
        plot_map = plot_map / 12.0
    map_estiamtion_filepath = get_map_estimation_filepath(species=species,version=map_estimation_version,YYYY=YYYY,MM=MONTH[MM],special_name=map_estimation_special_name)

    vmax_dic = {
        'NO2'  : 15,
    }
    Plot_Map_estimation_Map_Figures(species=species,map_estimation_map=plot_map,extent=extent,
                                        Mahalanobis_LAT=SATLAT, Mahalanobis_LON=SATLON,
                                        YYYY=YYYY,MM=MONTH[MM],
                                        outfile=map_estiamtion_filepath,
                                        vmin=0,vmax=vmax_dic[species])
    return

def plot_longterm_average_map_estimation_data(species,map_estimation_version,YYYY_list:np.int32,MM:np.int32,map_estimation_special_name):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','AllMonths']
    if MM != 12:
        print(f"Plotting longterm average map estimation data for species: {species}, version: {map_estimation_version}, month: {MONTH[MM]}")
    else:
        print(f"Plotting longterm average map estimation data for species: {species}, version: {map_estimation_version}, all months")
    longterm_average_map = None
    count = 0
    for YYYY in YYYY_list:
        if MM != 12:
            temp_map = np.zeros((13000,36000),dtype=np.float64)
            monthly_map, lat, lon = load_estimation_map_data(YYYY=YYYY, MM=MONTH[MM], SPECIES=species, version=map_estimation_version, special_name=map_estimation_special_name)
            h, w = monthly_map.shape
            temp_map[5:5+h, 5:5+w] = monthly_map
        else:
            temp_map = np.zeros((13000,36000),dtype=np.float64)
            for m in range(12):
                monthly_map, lat, lon = load_estimation_map_data(YYYY=YYYY, MM=MONTH[m], SPECIES=species, version=map_estimation_version, special_name=map_estimation_special_name)
                h, w = monthly_map.shape
                temp_map[5:5+h, 5:5+w] += monthly_map
            temp_map = temp_map / 12.0
        if longterm_average_map is None:
            longterm_average_map = np.zeros_like(temp_map)
        longterm_average_map += temp_map
        count += 1
    longterm_average_map = longterm_average_map / count
    longterm_average_map_estimation_filepath = get_map_estimation_filepath(species=species,version=map_estimation_version,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],special_name=map_estimation_special_name)
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')
    vmax_dic = {
        'NO2'  : 15,
    }
    Plot_Map_estimation_Map_Figures(species=species,map_estimation_map=longterm_average_map,extent=extent,
                                        Mahalanobis_LAT=SATLAT, Mahalanobis_LON=SATLON,
                                        YYYY='{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                        outfile=longterm_average_map_estimation_filepath,
                                        vmin=0,vmax=vmax_dic[species])
    
def plot_absolute_uncertainty_map(species,map_estimation_version,map_estimation_special_name,YYYY,MM,obs_version,nearby_sites_number,vmin,vmax):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']
    print(f"Plotting absolute uncertainty map for species: {species}, version: {map_estimation_version}, date: {YYYY}-{MONTH[MM]}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    absolute_uncertainty_filepath = get_absolute_uncertainty_filepath(species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY=YYYY,MM=MONTH[MM],
                                                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    absolute_uncertainty_map = load_absolute_uncertainty_map(species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY=YYYY,MM=MONTH[MM],
                                                           obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')
    Plot_absolute_Uncertainty_Map_Figures(species=species,absolute_uncertainty_map=absolute_uncertainty_map,extent=extent,
                                       Mahalanobis_LAT=SATLAT, Mahalanobis_LON=SATLON,
                                       YYYY=YYYY,MM=MONTH[MM],
                                       outfile=absolute_uncertainty_filepath,
                                       vmin=vmin,vmax=vmax)
    return

def plot_longterm_average_absolute_uncertainty_map(species,map_estimation_version,map_estimation_special_name,YYYY_list:np.int32,MM:np.int32,obs_version,nearby_sites_number,vmin,vmax):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','AllMonths']
    if MM != 12:
        print(f"Plotting longterm average absolute uncertainty map for species: {species}, version: {map_estimation_version}, month: {MONTH[MM]}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    else:
        print(f"Plotting longterm average absolute uncertainty map for species: {species}, version: {map_estimation_version}, all months, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    absolute_uncertainty_map = load_absolute_uncertainty_map(species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')
    Plot_absolute_Uncertainty_Map_Figures(species=species,absolute_uncertainty_map=absolute_uncertainty_map,extent=extent,
                                       Mahalanobis_LAT=SATLAT, Mahalanobis_LON=SATLON,
                                       YYYY='{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                       outfile=get_absolute_uncertainty_filepath(species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number),
                                       vmin=vmin,vmax=vmax)
    return
        
            
def plot_rRMSE_uncertainty_map(species,version,YYYY,MM,obs_version,nearby_sites_number,vmin,vmax,special_name=''):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']
    print(f"Plotting rRMSE uncertainty map for species: {species}, version: {version}, date: {YYYY}-{MONTH[MM]}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    rRMSE_uncertainty_map = load_rRMSE_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                           obs_version=obs_version,nearby_sites_number=nearby_sites_number,
                                           special_name=special_name)

    rRMSE_uncertainty_filepath = get_rRMSE_uncertainty_filepath(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                               obs_version=obs_version,nearby_sites_number=nearby_sites_number,
                                                               special_name=special_name)
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')

    Plot_rRMSE_Uncertainty_Map_Figures(species=species,rRMSE_uncertainty_map=rRMSE_uncertainty_map,extent=extent,
                                       Mahalanobis_LAT=SATLAT, Mahalanobis_LON=SATLON,
                                       YYYY=YYYY,MM=MONTH[MM],
                                       outfile=rRMSE_uncertainty_filepath,
                                       vmin=vmin,vmax=vmax)

    return 



def plot_mahalanobis_distance_map(species,version,YYYY,MM,obs_version,nearby_sites_number):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']
    print(f"Plotting Mahalanobis distance map for species: {species}, version: {version}, date: {YYYY}-{MONTH[MM]}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                             obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    
    mahalanobis_distance_filepath = get_mahalanobis_distance_filepath(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                                     obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')

    Plot_Mahalanobis_distance_Map_Figures(species=species,mahalanobis_distance_map=mahalanobis_distance_map,extent=extent,
                                         Mahalanobis_LAT=SATLAT, Mahalanobis_LON=SATLON,
                                         YYYY=YYYY,MM=MONTH[MM],
                                         outfile=mahalanobis_distance_filepath)

    return 

def plot_longterm_average_mahalanobis_distance_map(species,version,YYYY_list:np.int32,MM:np.int32,obs_version,nearby_sites_number):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','AllMonths']
    if MM != 12:
        print(f"Plotting longterm average Mahalanobis distance map for species: {species}, version: {version}, month: {MONTH[MM]}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    else:
        print(f"Plotting longterm average Mahalanobis distance map for species: {species}, version: {version}, all months, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                                             obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    SATLAT = np.load('/path/to/NO2_DL_global/input_variables/tSATLAT_global.npy')
    SATLON = np.load('/path/to/NO2_DL_global/input_variables/tSATLON_global.npy')
    Plot_Mahalanobis_distance_Map_Figures(species=species,mahalanobis_distance_map=mahalanobis_distance_map,extent=extent,
                                         Mahalanobis_LAT=SATLAT, Mahalanobis_LON=SATLON,
                                         YYYY='{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                         outfile=get_mahalanobis_distance_filepath(species=species,version=version,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                                                     obs_version=obs_version,nearby_sites_number=nearby_sites_number))

    return