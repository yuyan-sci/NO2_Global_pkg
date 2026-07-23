#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import xarray as xr
import warnings
import gc
from collections import defaultdict
warnings.filterwarnings('ignore')

# Configuration
EARTH_RADIUS_KM = 6371.0

# File paths
Obs_version = 'v7'
obs_file = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/combined/combined_global_no2_2005-2023_v7_filtered.csv'
lat_grid_file = '/path/to/NO2_DL_global/input_variables/tSATLAT_global_MAP.npy'
lon_grid_file = '/path/to/NO2_DL_global/input_variables/tSATLON_global_MAP.npy'
gchp_dir = '/path/to/GeoNO2-v5.13/'
output_dir = f'/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_{Obs_version}/'

os.makedirs(output_dir, exist_ok=True)

def get_nearest_point_index(sitelon, sitelat, lon_grid_shape, lat_grid_shape, grid_min_lon, grid_min_lat):
    """
    Optimized version that doesn't need full grid arrays in memory
    """
    det = 0.01
    index_lon = np.round((sitelon - grid_min_lon) / det).astype(np.int32)
    index_lat = np.round((sitelat - grid_min_lat) / det).astype(np.int32)
    
    # Clip to bounds
    index_lat = np.clip(index_lat, 0, lat_grid_shape[0] - 1)
    index_lon = np.clip(index_lon, 0, lon_grid_shape[1] - 1)
    
    return index_lon, index_lat

def apply_corrections_vectorized(base, obs, alkylnitrates, hno3, pan, alpha=0.15, beta=0.95):
    """Apply correction factors to observations."""
    denom = base + alkylnitrates + alpha * hno3 + beta * pan
    denom_safe = np.where(denom == 0, 1, denom)
    cf = obs * (base / denom_safe)
    return cf

def load_observations_optimized(years):
    """Load and preprocess observations with memory optimization."""
    print("Loading observations...")
    
    # Load only required columns to save memory
    required_cols = ['lat', 'lon', 'year', 'mon', 'no2']
    obs_df = pd.read_csv(obs_file, usecols=required_cols, low_memory=False)
    
    # Filter years immediately
    obs_df = obs_df[obs_df['year'].isin(years)]
    
    # Remove invalid observations
    valid_mask = ~(pd.isna(obs_df['lat']) | pd.isna(obs_df['lon']) | pd.isna(obs_df['no2']))
    obs_clean = obs_df[valid_mask].copy()
    del obs_df  # Free memory
    gc.collect()
    
    print(f"Valid observations: {len(obs_clean):,}")
    
    # Use more memory-efficient grouping
    site_coords = obs_clean[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    site_coords['site_id'] = range(len(site_coords))
    
    # Merge back efficiently
    obs_clean = obs_clean.merge(site_coords, on=['lat', 'lon'])
    
    print(f"Unique observation sites: {len(site_coords)}")
    print(f"Total observations: {len(obs_clean):,}")
    
    return obs_clean, site_coords

def get_gchp_coordinates_only(years):
    """Get GCHP grid coordinates without loading data."""
    print("Getting GCHP grid coordinates...")
    
    for year in years:
        year_dir = os.path.join(gchp_dir, str(year))
        for month in range(1, 13):
            month_str = f"{month:02d}"
            gchp_file = os.path.join(year_dir, f'1x1km.GeoNO2.{year}{month_str}.MonMean.nc')
            
            if os.path.exists(gchp_file):
                try:
                    # Open with minimal memory usage
                    with xr.open_dataset(gchp_file, engine='netcdf4') as ds:
                        lat_1d = ds.lat.values.astype(np.float32)
                        lon_1d = ds.lon.values.astype(np.float32)
                        
                        # Just get coordinate info, not full grid
                        grid_shape = (len(lat_1d), len(lon_1d))
                        min_lat, max_lat = float(lat_1d.min()), float(lat_1d.max())
                        min_lon, max_lon = float(lon_1d.min()), float(lon_1d.max())
                        
                        return {
                            'lat_1d': lat_1d,
                            'lon_1d': lon_1d,
                            'grid_shape': grid_shape,
                            'lat_range': (min_lat, max_lat),
                            'lon_range': (min_lon, max_lon),
                            'min_lat': min_lat,
                            'min_lon': min_lon
                        }
                        
                except Exception as e:
                    print(f"Error reading coordinates from {gchp_file}: {e}")
                    continue
    
    raise ValueError("No valid GCHP files found!")

def match_sites_to_gchp_optimized(site_coords, gchp_coords):
    """Match sites to GCHP grid with minimal memory usage."""
    print("Matching observation sites to GCHP grid...")
    
    sitelat = site_coords['lat'].values
    sitelon = site_coords['lon'].values
    
    # Use optimized index calculation
    index_lon, index_lat = get_nearest_point_index(
        sitelon=sitelon, 
        sitelat=sitelat,
        lon_grid_shape=gchp_coords['grid_shape'], 
        lat_grid_shape=gchp_coords['grid_shape'],
        grid_min_lon=gchp_coords['min_lon'],
        grid_min_lat=gchp_coords['min_lat']
    )
    
    # Calculate GCHP coordinates for matched sites
    gchp_lat_matched = gchp_coords['lat_1d'][index_lat]
    gchp_lon_matched = gchp_coords['lon_1d'][index_lon]
    
    # Add matching info
    site_coords_matched = site_coords.copy()
    site_coords_matched['gchp_lat_index'] = index_lat.astype(np.int32)
    site_coords_matched['gchp_lon_index'] = index_lon.astype(np.int32)
    site_coords_matched['gchp_lat'] = gchp_lat_matched
    site_coords_matched['gchp_lon'] = gchp_lon_matched
    
    print(f"Grid index ranges:")
    print(f"  Lat indices: {index_lat.min()} to {index_lat.max()}")
    print(f"  Lon indices: {index_lon.min()} to {index_lon.max()}")
    
    # Group by grid indices efficiently
    print("Grouping sites by GCHP grid cell...")
    
    grouped = site_coords_matched.groupby(['gchp_lat_index', 'gchp_lon_index']).agg({
        'site_id': lambda x: list(x),
        'lat': 'mean',
        'lon': 'mean',
        'gchp_lat': 'first',
        'gchp_lon': 'first'
    }).reset_index()
    
    grouped['grouped_site_id'] = range(len(grouped))
    grouped['num_original_sites'] = grouped['site_id'].apply(len)
    
    print(f"Original sites: {len(site_coords_matched)}")
    print(f"Grouped sites: {len(grouped)}")
    
    return grouped, site_coords_matched

def load_single_gchp_file(filepath):
    """Load single GCHP file with minimal memory footprint."""
    try:
        with xr.open_dataset(filepath, engine='netcdf4') as ds:
            return {
                'gchp_NO2': ds['gchp_NO2'].values.astype(np.float32),
                'geophysical_NO2': ds['filled_GeoNO2_trop'].values.astype(np.float32),
                'gchp_alkylnitrates': ds['gchp_alkylnitrates'].values.astype(np.float32),
                'gchp_HNO3': ds['gchp_HNO3'].values.astype(np.float32),
                'gchp_PAN': ds['gchp_PAN'].values.astype(np.float32)
            }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_datasets_incremental(obs_df, grouped_sites, years):
    """Create datasets using vectorized processing."""
    print("Creating datasets with vectorized processing...")

    # Create time coordinates and index mapping
    time_coords = []
    time_to_idx = {}
    for year in years:
        for month in range(1, 13):
            idx = len(time_coords)
            time_coords.append(pd.Timestamp(year, month, 1))
            time_to_idx[(year, month)] = idx

    n_groups = len(grouped_sites)
    n_times = len(time_coords)

    print(f"Initializing arrays: {n_groups} sites x {n_times} times")

    obs_corrected = np.full((n_groups, n_times), np.nan, dtype=np.float32)
    obs_uncorrected = np.full((n_groups, n_times), np.nan, dtype=np.float32)
    geo_data = np.full((n_groups, n_times), np.nan, dtype=np.float32)

    # Build site_id -> group_idx mapping
    site_to_group = {}
    for group_idx, (_, row) in enumerate(grouped_sites.iterrows()):
        for sid in row['site_id']:
            site_to_group[sid] = group_idx
    obs_df['group_idx'] = obs_df['site_id'].map(site_to_group)

    # Pre-aggregate observations by (year, month, group_idx)
    print("Pre-aggregating observations...")
    obs_agg = obs_df.groupby(['year', 'mon', 'group_idx'])['no2'].mean()

    # Vectorized extraction using fancy indexing
    lat_indices = grouped_sites['gchp_lat_index'].values
    lon_indices = grouped_sites['gchp_lon_index'].values

    print("Processing year-months (ONE GeoNO2 file at a time to bound memory)...")
    for year in years:
        for month in range(1, 13):
            # Load a single 1km global GeoNO2 file (5 grids ~= 18GB) per iteration.
            # Reassigning gchp_data each month frees the previous one, so peak
            # memory is ~1 month. (Pre-loading all 12 at once peaked ~220GB -> OOM.)
            time_idx = time_to_idx[(year, month)]
            gchp_file = os.path.join(gchp_dir, str(year), f'1x1km.GeoNO2.{year}{month:02d}.MonMean.nc')
            if not os.path.exists(gchp_file):
                continue
            gchp_data = load_single_gchp_file(gchp_file)
            if gchp_data is None:
                continue
            print(f"  {year}-{month:02d}: loaded GeoNO2", flush=True)

            # Extract geophysical NO2 for ALL sites at once
            geo_data[:, time_idx] = gchp_data['geophysical_NO2'][lat_indices, lon_indices]

            # Get pre-aggregated observations for this (year, month)
            try:
                month_obs = obs_agg.loc[(year, month)]
            except KeyError:
                continue

            grp_idx = month_obs.index.values.astype(np.intp)
            obs_vals = month_obs.values.astype(np.float32)

            obs_uncorrected[grp_idx, time_idx] = obs_vals

            # Vectorized correction for all sites with observations
            base = gchp_data['gchp_NO2'][lat_indices[grp_idx], lon_indices[grp_idx]]
            alkyl = gchp_data['gchp_alkylnitrates'][lat_indices[grp_idx], lon_indices[grp_idx]]
            hno3 = gchp_data['gchp_HNO3'][lat_indices[grp_idx], lon_indices[grp_idx]]
            pan = gchp_data['gchp_PAN'][lat_indices[grp_idx], lon_indices[grp_idx]]

            obs_corrected[grp_idx, time_idx] = apply_corrections_vectorized(
                base, obs_vals, alkyl, hno3, pan
            )
    
    print("Creating xarray datasets...")
    
    # Create corrected observation dataset
    ds_obs_corrected = xr.Dataset(
        data_vars={
            'NO2': (
                ('sites', 'time'),
                obs_corrected,
                {
                    'units': 'ppb',
                    'long_name': 'Corrected NO2 observations (grouped by grid cell)',
                    '_FillValue': np.nan}),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 
                 'long_name': 'Average observation site latitude within grid cell'}),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 
                 'long_name': 'Average observation site longitude within grid cell'})
        },
        coords={
            'sites': ('sites', grouped_sites['grouped_site_id'].values),
            'time': ('time', time_coords),
        },
        attrs={
            'title': 'Monthly NO2 Corrected Observations (Grouped by GCHP Grid Cell)',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Create uncorrected observation dataset
    ds_obs_uncorrected = xr.Dataset(
        data_vars={
            'NO2': (
                ('sites', 'time'),
                obs_uncorrected,
                {
                    'units': 'ppb',
                    'long_name': 'Uncorrected NO2 observations (grouped by grid cell)',
                    '_FillValue': np.nan}),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 
                 'long_name': 'Average observation site latitude within grid cell'}),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 
                 'long_name': 'Average observation site longitude within grid cell'})
        },
        coords=ds_obs_corrected.coords,
        attrs={
            'title': 'Monthly NO2 Uncorrected Observations (Grouped by GCHP Grid Cell)',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Create geophysical dataset
    ds_geo = xr.Dataset(
        data_vars={
            'NO2': (
                ('sites', 'time'),
                geo_data,
                {
                    'units': 'ppb',
                    'long_name': 'Geophysical NO2 (grouped by grid cell)',
                    '_FillValue': np.nan
                }
            ),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 
                 'long_name': 'Average observation site latitude within grid cell'}),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 
                 'long_name': 'Average observation site longitude within grid cell'})
        },
        coords=ds_obs_corrected.coords,
        attrs={
            'title': 'Monthly Geophysical NO2 (Grouped by GCHP Grid Cell)',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Create bias dataset (corrected)
    bias_data_corrected = obs_corrected - geo_data
    ds_bias_corrected = xr.Dataset(
        data_vars={
            'NO2': (
                ('sites', 'time'),
                bias_data_corrected,
                {
                    'units': 'ppb',
                    'long_name': 'NO2 bias (grouped corrected observations - geophysical)',
                    '_FillValue': np.nan
                }
            ),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 
                 'long_name': 'Average observation site latitude within grid cell'}),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 
                 'long_name': 'Average observation site longitude within grid cell'})
        },
        coords=ds_obs_corrected.coords,
        attrs={
            'title': 'Monthly NO2 Bias (Grouped Corrected Observations - Geophysical)',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    # Create bias dataset (uncorrected)
    bias_data_uncorrected = obs_uncorrected - geo_data
    ds_bias_uncorrected = xr.Dataset(
        data_vars={
            'NO2': (
                ('sites', 'time'),
                bias_data_uncorrected,
                {
                    'units': 'ppb',
                    'long_name': 'NO2 bias (grouped uncorrected observations - geophysical)',
                    '_FillValue': np.nan
                }
            ),
            'latitude': (
                'sites',
                grouped_sites['lat'].values,
                {'units': 'degrees_north', 
                 'long_name': 'Average observation site latitude within grid cell'}),
            'longitude': (
                'sites',
                grouped_sites['lon'].values,
                {'units': 'degrees_east', 
                 'long_name': 'Average observation site longitude within grid cell'})
        },
        coords=ds_obs_corrected.coords,
        attrs={
            'title': 'Monthly NO2 Bias (Grouped Uncorrected Observations - Geophysical)',
            'created_on': pd.Timestamp.now().isoformat()
        }
    )
    
    return ds_obs_corrected, ds_obs_uncorrected, ds_geo, ds_bias_corrected, ds_bias_uncorrected

def save_datasets_efficiently(ds_obs_corrected, ds_obs_uncorrected, ds_geo, ds_bias_corrected, ds_bias_uncorrected):
    """Save datasets with compression and chunking for efficiency."""
    print("Saving datasets with optimized compression...")
    
    # Optimized encoding for better compression and faster I/O
    encoding = {
        'NO2': {
            'zlib': True, 
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float64',
            'chunksizes': (min(1000, ds_obs_corrected.dims['sites']), min(12, ds_obs_corrected.dims['time']))
        }
    }
    
    encoding_geo = {
        'NO2': {
            'zlib': True, 
            'complevel': 4,
            'shuffle': True,
            'dtype': 'float64',
            'chunksizes': (min(1000, ds_geo.dims['sites']), min(12, ds_geo.dims['time']))
        }
    }

    
    # Save corrected observation file
    obs_corrected_file_out = os.path.join(output_dir, f'NO2_observation_corrected_{Obs_version}_filtered_v5.13.nc')
    ds_obs_corrected.to_netcdf(obs_corrected_file_out, encoding=encoding)
    print(f"Saved: {obs_corrected_file_out}")
    
    # Save uncorrected observation file
    obs_uncorrected_file_out = os.path.join(output_dir, f'NO2_observation_uncorrected_{Obs_version}_filtered_v5.13.nc')
    ds_obs_uncorrected.to_netcdf(obs_uncorrected_file_out, encoding=encoding)
    print(f"Saved: {obs_uncorrected_file_out}")
    
    # Save geophysical file
    geo_file_out = os.path.join(output_dir, f'NO2_geophysical_{Obs_version}_filtered_v5.13.nc')
    ds_geo.to_netcdf(geo_file_out, encoding=encoding_geo)
    print(f"Saved: {geo_file_out}")
    
    # Save corrected bias file
    bias_corrected_file_out = os.path.join(output_dir, f'NO2_bias_corrected_{Obs_version}_filtered_v5.13.nc')
    ds_bias_corrected.to_netcdf(bias_corrected_file_out, encoding=encoding)
    print(f"Saved: {bias_corrected_file_out}")
    
    # Save uncorrected bias file
    bias_uncorrected_file_out = os.path.join(output_dir, f'NO2_bias_uncorrected_{Obs_version}_filtered_v5.13.nc')
    ds_bias_uncorrected.to_netcdf(bias_uncorrected_file_out, encoding=encoding)
    print(f"Saved: {bias_uncorrected_file_out}")

def print_memory_usage():
    """Print current memory usage."""
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")

def main():
    """Optimized main execution function."""
    print("Starting optimized NO2 analysis...")
    print("="*60)
    
    print_memory_usage()
    
    # Step 1: Load observations (optimized)
    years=list(range(2005, 2024))
    obs_df, site_coords = load_observations_optimized(years)
    print_memory_usage()
    
    # Step 2: Get GCHP coordinates only (not full data)
    gchp_coords = get_gchp_coordinates_only(years)
    print_memory_usage()
    
    # Step 3: Match sites to GCHP grid (optimized)
    grouped_sites, _ = match_sites_to_gchp_optimized(site_coords, gchp_coords)
    del site_coords  # Free memory
    gc.collect()
    print_memory_usage()
    
    # Step 4: Create datasets with incremental processing
    ds_obs_corrected, ds_obs_uncorrected, ds_geo, ds_bias_corrected, ds_bias_uncorrected = create_datasets_incremental(obs_df, grouped_sites, years)
    del obs_df  # Free memory
    gc.collect()
    print_memory_usage()
    
    # Step 5: Save with optimized compression
    save_datasets_efficiently(ds_obs_corrected, ds_obs_uncorrected, ds_geo, ds_bias_corrected, ds_bias_uncorrected)
    
    # Print summary statistics
    obs_corrected_valid = ~np.isnan(ds_obs_corrected['NO2'].values)
    obs_uncorrected_valid = ~np.isnan(ds_obs_uncorrected['NO2'].values)
    print(f"\nSummary:")
    print(f"Grouped sites: {len(grouped_sites)}")
    print(f"Valid corrected observations: {np.sum(obs_corrected_valid):,}")
    print(f"Valid uncorrected observations: {np.sum(obs_uncorrected_valid):,}")
    print(f"Corrected data coverage: {100*np.mean(obs_corrected_valid):.1f}%")
    print(f"Uncorrected data coverage: {100*np.mean(obs_uncorrected_valid):.1f}%")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    
    print_memory_usage()

if __name__ == "__main__":
    main()