import numpy as np
import geopandas as gpd
from rasterio.transform import from_origin
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from rasterio.transform import xy
from affine import Affine

def get_transform(lat_min, lon_min, resolution):
    # Create affine transform (for converting grid indices to lat/lon)
    transform = Affine.translation(lon_min, lat_min) * Affine.scale(resolution, resolution) 
    return transform


def calculate_lengths_in_grid(gdf, resolution, n_lat, n_lon, transform):
    print(f"Starting calculate_lengths_in_grid: n_lat={n_lat}, n_lon={n_lon}")
    # Initialize grid
    grid_lengths = np.zeros((n_lat, n_lon), dtype=np.float32)
    
    
    
    # Build spatial index for faster lookups
    sindex = gdf.sindex
    

    for i in range(n_lat):
        for j in range(n_lon):
            #print('i: {}/{}, j: {}/{}'.format(i,n_lat,j,n_lon))

            # Compute grid cell bounds
            cell_bounds = rasterio.transform.xy(transform, i, j, offset='center')
            cell_box = box(cell_bounds[0] - resolution / 2, cell_bounds[1] - resolution / 2,
                           cell_bounds[0] + resolution / 2, cell_bounds[1] + resolution / 2)
            
            # Find candidate geometries using spatial index
            possible_matches_idx = list(sindex.intersection(cell_box.bounds))
            if len(possible_matches_idx) != 0:
                print('i: {}/{}, j: {}/{}'.format(i,n_lat,j,n_lon), '  possible_matches_idx:', possible_matches_idx)
            clipped_gdf = gdf.iloc[possible_matches_idx].copy()
            clipped_gdf["clipped_geom"] = clipped_gdf.intersection(cell_box).to_crs(epsg=32633)
            
            # gdf.intersects(cell_box): Filters the GeoDataFrame to only include geometries that intersect the grid cell. 
            # It doesn’t change the shape of the geometries, it just selects the relevant ones.
            # gdf.intersection(cell_box): Actually modifies the geometries so that only the part of each geometry 
            # that is inside the grid cell remains. 
            # It produces new geometries that fit within the grid cell boundaries.

            # Calculate the total length of the LineStrings in this cell
            total_length_in_cell = clipped_gdf['clipped_geom'].length.sum()
            grid_lengths[i, j] = total_length_in_cell
    
    return grid_lengths

def original_calculate_lengths_in_grid(gdf, resolution, n_lat, n_lon, transform):
    # Function to calculate length of LineStrings in each grid cell
    grid_lengths = np.zeros((n_lat, n_lon), dtype=np.float32)

    for i in range(n_lat):
        for j in range(n_lon):
            print('i: {}/{}, j: {}/{}'.format(i,n_lat,j,n_lon))
            # Define the bounds of the current grid cell
            cell_bounds = rasterio.transform.xy(transform, i, j, offset='center')
            cell_box = box(cell_bounds[0] - resolution / 2, cell_bounds[1] - resolution / 2,
                           cell_bounds[0] + resolution / 2, cell_bounds[1] + resolution / 2)
            
            # Clip the LineStrings to the grid cell
            clipped_gdf = gdf[gdf.intersects(cell_box)].copy()
            clipped_gdf['clipped_geom'] = clipped_gdf.intersection(cell_box)
            # gdf.intersects(cell_box): Filters the GeoDataFrame to only include geometries that intersect the grid cell. 
            # It doesn’t change the shape of the geometries, it just selects the relevant ones.
            # gdf.intersection(cell_box): Actually modifies the geometries so that only the part of each geometry 
            # that is inside the grid cell remains. 
            # It produces new geometries that fit within the grid cell boundaries.
            
            clipped_gdf['clipped_geom'] = clipped_gdf['clipped_geom'].to_crs(epsg=32633)
            # Calculate the total length of the LineStrings in this cell
            total_length_in_cell = clipped_gdf['clipped_geom'].length.sum()
            grid_lengths[i, j] = total_length_in_cell

    print("Completed calculate_lengths_in_grid")            
    return grid_lengths