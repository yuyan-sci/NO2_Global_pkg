import numpy as np
import time
from scipy.spatial import cKDTree
from LandCover_nearest_distances_pkg.iostream import load_Global_GeoLatLon, load_Global_GeoLatLon_Map, save_landcover_nearest_pixels

# Earth radius in kilometers
r = 6371.0

def calculate_distance(site_lat, site_lon, other_sites_lat, other_sites_lon):
    site_pos1 = site_lat * np.pi / 180.0
    site_pos2 = site_lon * np.pi / 180.0
    other_sites_pos1 = other_sites_lat * np.pi / 180.0
    other_sites_pos2 = other_sites_lon * np.pi / 180.0
    dist_map = r * np.arccos(np.sin(site_pos1)*np.sin(other_sites_pos1)+np.cos(site_pos1)*np.cos(other_sites_pos1)*np.cos(site_pos2-other_sites_pos2))
    return dist_map

def calculate_distance_forArray(nearest_lat_array, nearest_lon_array, site_lat_array, site_lon_array):
    # Assuming this function calculates Haversine distance between corresponding points in arrays
    # Convert degrees to radians
    site_pos1 = site_lat_array * np.pi / 180.0
    site_pos2 = site_lon_array * np.pi / 180.0
    other_sites_pos1_array = nearest_lat_array * np.pi / 180.0
    other_sites_pos2_array = nearest_lon_array * np.pi / 180.0
    
    # Haversine formula components
    dlon = other_sites_pos2_array - site_pos2
    dlat = other_sites_pos1_array - site_pos1
    a = np.sin(dlat / 2)**2 + np.cos(site_pos1) * np.cos(other_sites_pos1_array) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    dist_map = r * c
    return dist_map

# Renamed input parameters slightly for clarity
def get_nearest_landcoverpixels_distance_for_each_pixel(landcover_lats, landcover_lons, init_nametag, YYYY, Area):
    # Load the target grid (2D)
    Global_GeoLAT_MAP, Global_GeoLON_MAP = load_Global_GeoLatLon_Map()
    
    print("Preparing KDTree...")
    start_time = time.time()
    
    # Check if there are any land cover pixels found
    if len(landcover_lats) == 0:
        print(f"Warning: No land cover pixels found for {init_nametag}. Skipping distance calculation.")
        # Create an empty or default distance map (e.g., filled with a large value or NaN)
        nearest_distance_map = np.full(Global_GeoLAT_MAP.shape, np.nan) 
    else:
        # Combine land cover coordinates into shape (n_points, 2)
        landcover_points = np.vstack((landcover_lats, landcover_lons)).T
        
        # Build the KDTree
        tree = cKDTree(landcover_points)
        
        # Prepare query points (flatten the target grid and stack lat/lon)
        target_lats_flat = Global_GeoLAT_MAP.flatten()
        target_lons_flat = Global_GeoLON_MAP.flatten()
        target_points = np.vstack((target_lats_flat, target_lons_flat)).T
        
        print(f"Built KDTree with {len(landcover_points)} points. Querying for {len(target_points)} target points...")
        
        # Query the tree for the index (idx) of the nearest neighbor (k=1)
        # We ignore the returned distance (`dist`) because it's Euclidean, not great-circle.
        dist, idx = tree.query(target_points, k=1)
        
        # Get the coordinates of the nearest land cover points using the indices
        nearest_landcover_lats = landcover_lats[idx]
        nearest_landcover_lons = landcover_lons[idx]
        
        print("Calculating great-circle distances...")
        # Calculate the actual great-circle distance
        nearest_distance_flat = calculate_distance_forArray(
            nearest_landcover_lats, nearest_landcover_lons,
            target_lats_flat, target_lons_flat
        )
        
        # Reshape the distances back to the original map shape
        nearest_distance_map = nearest_distance_flat.reshape(Global_GeoLAT_MAP.shape)

    end_time = time.time()
    print(f'Finished nearest distance calculation for {init_nametag}. Time costs: {end_time - start_time:.2f} seconds')
    
    # Save the resulting distance map
    nametag_out = init_nametag + '_NearestDistances'
    # Use the correct iostream function if it differs
    save_landcover_nearest_pixels(mapdata=nearest_distance_map, nametag=nametag_out, YEAR=YYYY, Area=Area) 
    
    return
