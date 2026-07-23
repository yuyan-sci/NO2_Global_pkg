import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import numpy as np
import matplotlib.ticker as tick
import matplotlib.colors as colors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import os
import sys
import cartopy
cartopy.config['data_dir'] = '/path/to/supportData/shapefile'

# Add parent directory to path to import from Estimation_pkg
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Estimation_pkg.utils import inputfiles_table


def load_input_variable(variable_name, year, month):
    try:
        YYYY = str(year)
        MM = month
        
        # Get input files dictionary
        input_files = inputfiles_table(YYYY, MM)
        
        if variable_name not in input_files:
            print(f"Warning: Variable '{variable_name}' not found in input files table")
            return None
            
        filepath = input_files[variable_name]
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            return None
            
        # Load the data
        data = np.load(filepath)
        
        return data
        
    except Exception as e:
        print(f"Error loading variable '{variable_name}': {str(e)}")
        return None


def Plot_InputVariable_Map(variable_name, variable_data, LAT, LON, extent, outfile, year, month, 
                           vmin=None, vmax=None, cmap='viridis', title=None):
    MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Handle NaN values
    variable_data = np.nan_to_num(variable_data, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Set extent [lon_min, lon_max, lat_min, lat_max] for cartopy
    map_extent = [extent[2], extent[3], extent[0], extent[1]]
    
    ax.set_aspect(1.25)
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', 
                                              edgecolor='none', facecolor='white'))
    ax.add_feature(cfeat.COASTLINE, linewidth=0.15)
    ax.add_feature(cfeat.LAKES, linewidth=0.05, facecolor='white')
    ax.add_feature(cfeat.BORDERS, linewidth=0.1)
    
    # Determine colorbar range if not provided
    if vmin is None or vmax is None:
        valid_data = variable_data[~np.isnan(variable_data)]
        if len(valid_data) > 0:
            if vmin is None:
                vmin = np.percentile(valid_data, 2)
            if vmax is None:
                vmax = np.percentile(valid_data, 10)
    
    # Plot the data
    pcm = plt.pcolormesh(LON, LAT, variable_data, 
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, 
                         norm=colors.Normalize(vmin=vmin, vmax=vmax))
    
    # Add month and year text
    month_idx = int(month) - 1
    month_name = MONTH_NAMES[month_idx]
    ax.text(map_extent[0] + 0.02 * abs(map_extent[1] - map_extent[0]),
            map_extent[2] + 0.05 * abs(map_extent[3] - map_extent[2]),
            f'{year} {month_name}', 
            style='italic', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add variable statistics
    valid_data = variable_data[~np.isnan(variable_data)]
    if len(valid_data) > 0:
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        ax.text(map_extent[0] + 0.02 * abs(map_extent[1] - map_extent[0]),
                map_extent[2] + 0.12 * abs(map_extent[3] - map_extent[2]),
                f'Mean: {mean_val:.2f} ± {std_val:.2f}', 
                style='italic', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar = plt.colorbar(pcm, location='right', fraction=0.046, 
                        shrink=0.7, aspect=40.0, 
                        orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=10)
    
    # Set colorbar label
    if title:
        cbar.set_label(title, fontsize=12)
    else:
        cbar.set_label(variable_name, fontsize=12)
    
    # Add title
    plot_title = title if title else f'{variable_name}'
    plt.title(plot_title, fontsize=14, pad=10)
    
    # Save figure
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, format='png', dpi=300, 
                bbox_inches='tight', transparent=False)
    plt.close()
    
    print(f"Saved: {outfile}")


def Plot_Multiple_InputVariables(variables_list, year, month, extent, outdir, 
                                 lat_array=None, lon_array=None):
    
    print(f"\n{'='*60}")
    print(f"Plotting Input Variables for {year}-{month}")
    print(f"{'='*60}\n")
    
    # Load latitude and longitude if not provided
    if lat_array is None or lon_array is None:
        print("Loading latitude and longitude arrays...")
        try:
            input_files = inputfiles_table(str(year), month)
            lat_array = np.load(input_files['lat'])
            lon_array = np.load(input_files['lon'])
            print(f"  Lat shape: {lat_array.shape}, Lon shape: {lon_array.shape}")
        except Exception as e:
            print(f"Error loading lat/lon: {e}")
            return
    
    # Variable-specific plotting parameters
    variable_configs = {
        'lat': {'cmap': 'RdBu_r', 'vmin': -60, 'vmax': 70, 'title': 'Latitude (°)'},
        'lon': {'cmap': 'RdBu_r', 'vmin': -180, 'vmax': 180, 'title': 'Longitude (°)'},
        'GeoNO2': {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 15, 'title': 'GeoNO₂ (ppb)'},
        'GCHP_NO2': {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 15, 'title': 'GCHP NO₂ (ppb)'},
        'ML_NO2_2019': {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 20, 'title': 'ML NO₂ 2019 (ppb)'},
        'ML_NO2_2023': {'cmap': 'YlOrRd', 'vmin': 0, 'vmax': 20, 'title': 'ML NO₂ 2023 (ppb)'},
        'Population': {'cmap': 'YlGnBu', 'vmin': 0, 'vmax': 1000, 'title': 'Population'},
        'NO_emi': {'cmap': 'Reds', 'vmin': 0, 'vmax': 1e-9, 'title': 'NO Emissions'},
        'Total_DM': {'cmap': 'hot_r', 'vmin': 0, 'vmax': 1e-8, 'title': 'Total Dry Matter'},
        'NDVI': {'cmap': 'RdYlGn', 'vmin': -1, 'vmax': 1, 'title': 'NDVI'},
        'ISA': {'cmap': 'gray_r', 'vmin': 0, 'vmax': 100, 'title': 'ISA (%)'},
        'T2M': {'cmap': 'RdBu_r', 'vmin': None, 'vmax': None, 'title': 'Temperature 2m (K)'},
        'RH': {'cmap': 'Blues', 'vmin': 0, 'vmax': 1, 'title': 'Relative Humidity (%)'},
        'PBLH': {'cmap': 'viridis', 'vmin': 0, 'vmax': None, 'title': 'PBLH (m)'},
        'TP': {'cmap': 'Blues', 'vmin': 0, 'vmax': None, 'title': 'Total Precipitation'},
        'PS': {'cmap': 'plasma', 'vmin': None, 'vmax': None, 'title': 'Surface Pressure'},
        'U10M': {'cmap': 'RdBu_r', 'vmin': -10, 'vmax': 10, 'title': 'U Wind 10m (m/s)'},
        'V10M': {'cmap': 'RdBu_r', 'vmin': -10, 'vmax': 10, 'title': 'V Wind 10m (m/s)'},
        'USTAR': {'cmap': 'viridis', 'vmin': 0, 'vmax': None, 'title': 'Friction Velocity'},
        'elevation': {'cmap': 'terrain', 'vmin': -7500, 'vmax': 7500, 'title': 'Elevation (m)'},
        # Land Cover Density
        'forests_density': {'cmap': 'Greens', 'vmin': 0, 'vmax': 1, 'title': 'Forests Density'},
        'shrublands_density': {'cmap': 'YlGn', 'vmin': 0, 'vmax': 1, 'title': 'Shrublands Density'},
        'croplands_density': {'cmap': 'YlOrBr', 'vmin': 0, 'vmax': 1, 'title': 'Croplands Density'},
        'urban_builtup_lands_density': {'cmap': 'Greys', 'vmin': 0, 'vmax': 1, 'title': 'Urban Built-up Lands Density'},
        'water_bodies_density': {'cmap': 'Blues', 'vmin': 0, 'vmax': 1, 'title': 'Water Bodies Density'},
        # Land Cover Distance
        'forests_distance': {'cmap': 'Greens_r', 'vmin': 0, 'vmax': 70, 'title': 'Distance to Forests (km)'},
        'shrublands_distance': {'cmap': 'YlGn_r', 'vmin': 0, 'vmax': 70, 'title': 'Distance to Shrublands (km)'},
        'croplands_distance': {'cmap': 'YlOrBr_r', 'vmin': 0, 'vmax': 70, 'title': 'Distance to Croplands (km)'},
        'urban_builtup_lands_distance': {'cmap': 'Greys_r', 'vmin': 0, 'vmax': 70, 'title': 'Distance to Urban Lands (km)'},
        'water_bodies_distance': {'cmap': 'Blues_r', 'vmin': 0, 'vmax': 70, 'title': 'Distance to Water Bodies (km)'},
        # Road Density
        'major_roads': {'cmap': 'Reds', 'vmin': 0, 'vmax': None, 'title': 'Major Roads Density'},
        'minor_roads': {'cmap': 'Oranges', 'vmin': 0, 'vmax': None, 'title': 'Minor Roads Density'},
        'major_roads_new': {'cmap': 'Reds', 'vmin': 0, 'vmax': None, 'title': 'Major Roads (New) Density'},
        'minor_roads_new': {'cmap': 'Oranges', 'vmin': 0, 'vmax': None, 'title': 'Minor Roads (New) Density'},
        'log_major_roads': {'cmap': 'Reds', 'vmin': None, 'vmax': None, 'title': 'Log Major Roads Density'},
        'log_minor_roads': {'cmap': 'Oranges', 'vmin': None, 'vmax': None, 'title': 'Log Minor Roads Density'},
        'log_major_roads_new': {'cmap': 'Reds', 'vmin': None, 'vmax': None, 'title': 'Log Major Roads (New) Density'},
        'log_minor_roads_new': {'cmap': 'Oranges', 'vmin': None, 'vmax': None, 'title': 'Log Minor Roads (New) Density'},
        # Road Distance
        'major_roads_dist': {'cmap': 'Reds_r', 'vmin': 0, 'vmax': None, 'title': 'Distance to Major Roads (km)'},
        'minor_roads_dist': {'cmap': 'Oranges_r', 'vmin': 0, 'vmax': None, 'title': 'Distance to Minor Roads (km)'},
        'major_roads_new_dist': {'cmap': 'Reds_r', 'vmin': 0, 'vmax': None, 'title': 'Distance to Major Roads (New) (km)'},
        'minor_roads_new_dist': {'cmap': 'Oranges_r', 'vmin': 0, 'vmax': None, 'title': 'Distance to Minor Roads (New) (km)'},
    }
    
    # Add buffer variables (binary 0/1 values) dynamically
    # All buffer variables use binary colormap: white (0) to blue (1)
    # Land cover buffer variables
    for land_type in ['forests', 'shrublands', 'croplands', 'urban_builtup_lands', 'water_bodies']:
        for buffer_size in ['500', '1000', '1500', '2000', '2500', '3000', '3500', '4000', 
                           '4500', '5000', '5500', '6000', '6500', '7000', '7500', '8000',
                           '8500', '9000', '9500', '10000', '10500', '11000']:
            var_name = f'{land_type}_buffer-{buffer_size}'
            buffer_km = float(buffer_size) / 1000
            land_display = land_type.replace('_', ' ').title()
            variable_configs[var_name] = {
                'cmap': 'Blues',
                'vmin': 0,
                'vmax': 1,
                'title': f'{land_display} Buffer {buffer_km}km (Binary)'
            }
    
    # Road buffer variables
    for road_type in ['major_roads', 'minor_roads', 'major_roads_new', 'minor_roads_new']:
        for buffer_size in ['500', '1000', '1500', '2000', '2500', '3000', '3500', '4000',
                           '4500', '5000', '5500', '6000', '6500', '7000', '7500', '8000',
                           '8500', '9000', '9500', '10000', '10500', '11000']:
            var_name = f'{road_type}_buffer-{buffer_size}'
            buffer_km = float(buffer_size) / 1000
            road_display = road_type.replace('_', ' ').title()
            variable_configs[var_name] = {
                'cmap': 'Blues',
                'vmin': 0,
                'vmax': 1,
                'title': f'{road_display} Buffer {buffer_km}km (Binary)'
            }
    
    # Plot each variable
    success_count = 0
    failed_vars = []
    
    for var_name in variables_list:
        print(f"\nProcessing: {var_name}")
        print("-" * 40)
        
        try:
            # Load variable data
            var_data = load_input_variable(var_name, year, month)
            
            if var_data is None:
                print(f"  Skipping {var_name} - data not available")
                failed_vars.append(var_name)
                continue
            
            print(f"  Data shape: {var_data.shape}")
            print(f"  Data range: [{np.nanmin(var_data):.4f}, {np.nanmax(var_data):.4f}]")
            
            # Get variable-specific config or use defaults
            config = variable_configs.get(var_name, {
                'cmap': 'viridis', 
                'vmin': None, 
                'vmax': None, 
                'title': var_name
            })
            
            # Create output directory structure
            year_month_dir = os.path.join(outdir, f'Input_Variables_Plots/{var_name}', 
                                        f'{year}', f'{year}{month}')
            os.makedirs(year_month_dir, exist_ok=True)
            
            # Create output filename
            outfile = os.path.join(year_month_dir, 
                                   f'{var_name}_{year}{month}_Global.png')
            
            # Plot the variable
            Plot_InputVariable_Map(
                variable_name=var_name,
                variable_data=var_data,
                LAT=lat_array,
                LON=lon_array,
                extent=extent,
                outfile=outfile,
                year=year,
                month=month,
                vmin=config.get('vmin'),
                vmax=config.get('vmax'),
                cmap=config.get('cmap', 'viridis'),
                title=config.get('title', var_name)
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"  Error plotting {var_name}: {str(e)}")
            failed_vars.append(var_name)
    
    # Summary
    print(f"  Successfully plotted: {success_count}/{len(variables_list)} variables")
    if failed_vars:
        print(f"  Failed variables: {', '.join(failed_vars)}")
    print(f"  Output directory: {outdir}")