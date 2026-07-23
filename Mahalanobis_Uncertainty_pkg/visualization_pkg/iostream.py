import os
from data_func.utils import Figure_outdir

def get_map_estimation_filepath(species,version,YYYY,MM,special_name):
    """
    Constructs the file path for map estimation data based on the base directory and date string.

    Parameters:
    base_dir (str): The base directory where the data is stored.
    date_str (str): The date string in 'YYYYMMDD' format.

    Returns:
    str: The constructed file path.
    """
    outdir = Figure_outdir + '{}/{}/Mahalanobis_Uncertainty/figures/Map_Estimation/{}/'.format(species,version,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filepath = outdir + '{}_{}_{}{}{}.png'.format(species,version,YYYY,MM,special_name)
    return filepath

def get_absolute_uncertainty_filepath(species,version,special_name,YYYY,MM,obs_version,nearby_sites_number):
    """
    Constructs the file path for absolute uncertainty data based on the base directory and date string.

    Parameters:
    base_dir (str): The base directory where the data is stored.
    date_str (str): The date string in 'YYYYMMDD' format.

    Returns:
    str: The constructed file path.
    """
    outdir = Figure_outdir + '{}/{}/Mahalanobis_Uncertainty/figures/Absolute_Uncertainty_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filepath = outdir + 'RawObs-{}_pixels_nearby_{}_sites_absolute_uncertainty_map_{}{}{}.png'.format(obs_version,nearby_sites_number,YYYY,MM,special_name)
    return filepath

def get_mahalanobis_distance_filepath(species,version,YYYY,MM,obs_version,nearby_sites_number):
    """
    Constructs the file path for Mahalanobis distance data based on the base directory and date string.

    Parameters:
    base_dir (str): The base directory where the data is stored.
    date_str (str): The date string in 'YYYYMMDD' format.

    Returns:
    str: The constructed file path.
    """
    outdir = Figure_outdir + '{}/{}/Mahalanobis_Uncertainty/figures/Mahalanobis_distance_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filepath = outdir + 'RawObs-{}_pixels_nearby_{}_sites_mahalanobis_distance_map_{}{}.png'.format(obs_version,nearby_sites_number,YYYY,MM)
    return filepath

def get_rRMSE_uncertainty_filepath(species,version,YYYY,MM,obs_version,nearby_sites_number,special_name=''):
    """
    Constructs the file path for rRMSE uncertainty data based on the base directory and date string.

    Parameters:
    base_dir (str): The base directory where the data is stored.
    date_str (str): The date string in 'YYYYMMDD' format.

    Returns:
    str: The constructed file path.
    """
    outdir = Figure_outdir + '{}/{}/Mahalanobis_Uncertainty/figures/rRMSE_Map/sitesnumber-{}/{}/'.format(species,version,nearby_sites_number,YYYY)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    filepath = outdir + 'RawObs-{}_pixels_nearby_{}_sites_rRMSE_uncertainty_map_{}{}{}.png'.format(obs_version,nearby_sites_number,YYYY,MM,special_name)
    return filepath