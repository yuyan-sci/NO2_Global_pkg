import numpy as np

def get_CNN_training_site_data(initial_array, Height, Width,lat_index, lon_index,nsite):
    """Vectorized version for faster site data extraction"""
    CNN_training = np.zeros((nsite, Height, Width), dtype=np.float64)
    
    # Vectorize the extraction using advanced indexing
    half_height = (Height - 1) // 2
    half_width = (Width - 1) // 2
    
    # Pre-compute row and column indices
    lat_start = (lat_index - half_height).astype(int)
    lat_end = (lat_index + half_height + 1).astype(int)
    lon_start = (lon_index - half_width).astype(int)
    lon_end = (lon_index + half_width + 1).astype(int)
    
    # Extract windows for all sites at once
    for j in range(nsite):
        CNN_training[j,:,:] = initial_array[lat_start[j]:lat_end[j], lon_start[j]:lon_end[j]]

    return CNN_training

def get_nearest_point_index(sitelon, sitelat, lon_grid, lat_grid):
    '''
    func: get the index of stations on the grids map
    inputs:
        sitelon, sitelat: stations location, eg:[42.353,110.137] 0th dim:lat 1st dim:lat
        lon_grid: grids longitude
        lat_grid: grids latitude
    return:
        index: [index_lat,index_lon]
    '''
    # step1: get the spatial resolution; Default: the latitude and longitude have the same resolution
    det = 0.01
    # step2:
    lon_min = np.min(lon_grid)
    lat_min = np.min(lat_grid)
    index_lon = np.round((sitelon - lon_min) / det)
    index_lat = np.round((sitelat - lat_min) / det)
    index_lon = index_lon.astype(int)
    index_lat = index_lat.astype(int)
    
    return index_lon,index_lat