import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
cmap = plt.cm.get_cmap('RdYlBu_r')
from matplotlib.colors import ListedColormap, BoundaryNorm

from OSM_pkg.iostream import load_global_buffer_map, load_RoadDensity_nearest_pixels, load_global_road_map, load_global_GeoLatLon_Map

# entry_list = ['primary','secondary', 'motorway', 'trunk', 'tertiary', 'residential','unclassified']

entry_list = [
    'major_roads', 'major_roads_log', 'major_roads_new', 'major_roads_new_log',
    'minor_roads', 'minor_roads_new', 'minor_roads_new_log', 'secondary'
]
YEAR = 2025

# Load global lat/lon grid once
y, x = load_global_GeoLatLon_Map()

cmap = plt.cm.RdYlBu_r

#-----1. Plot roaddensity map
fig1, axes1 = plt.subplots(
    nrows=4, ncols=2,
    figsize=(12, 16),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

for ax, entry in zip(axes1.flat, entry_list):
    data = load_global_road_map(YEAR=YEAR, entry=entry)
    mesh = ax.pcolormesh(
        x, y, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0, vmax=5000
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    ax.set_title(entry.replace('_', ' ').title(), pad=8)

# Remove any unused axes if fewer than 8 entries
for i in range(len(entry_list), 8):
    fig1.delaxes(axes1.flat[i])

# Add a single shared horizontal colorbar
cbar = fig1.colorbar(
    mesh, ax=axes1.ravel().tolist(),
    orientation='horizontal', pad=0.02
)
cbar.set_label('Road Density')

# Add a main title and adjust layout
fig1.suptitle(f'Global Road Density Categories – {YEAR}', fontsize=20, y=0.97)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
out_dir = (
    '/path/to/NO2_DL_global_2019/NO2_global_pkg/'
    'input_variables/OpenStreetMap_RoadDensity_input/'
    f'{YEAR}/'
)
fig1.savefig(f'{out_dir}Global_RoadDensity_{YEAR}.png', dpi=300)
plt.close(fig1)

#-------------2. plot distance map
# Create a 4×2 grid of PlateCarree subplots
fig2, axes2 = plt.subplots(
    nrows=4, ncols=2,
    figsize=(12, 16),
    subplot_kw={'projection': ccrs.PlateCarree()}
)
for ax, entry in zip(axes2.flat, entry_list):
    data = load_RoadDensity_nearest_pixels(YEAR=YEAR, nametag=entry, Area='Global')
    mesh = ax.pcolormesh(
        x, y, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0, vmax=2000
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    ax.set_title(entry.replace('_', ' ').title(), pad=8)

# Remove any unused axes if fewer than 8 entries
for i in range(len(entry_list), 8):
    fig2.delaxes(axes2.flat[i])

# Add a single shared horizontal colorbar
cbar = fig2.colorbar(
    mesh, ax=axes2.ravel().tolist(),
    orientation='horizontal', pad=0.02
)
cbar.set_label('Distance to Nearest Roads')

# Add a main title and adjust layout
fig2.suptitle(f'Global Distance to Nearest Roads– {YEAR}', fontsize=20, y=0.97)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
out_dir = (
    '/path/to/NO2_DL_global_2019/NO2_global_pkg/'
    'input_variables/OpenStreetMap_RoadDensity_NearestDistances_forEachPixels_input/'
)
fig2.savefig(f'{out_dir}Global Distance to Nearest Roads_{YEAR}.png', dpi=300)
plt.close(fig2)

#-------3. plot buffer map
# Create a 4×2 grid of PlateCarree subplots
fig3, axes3 = plt.subplots(
    nrows=4, ncols=2,
    figsize=(12, 16),
    subplot_kw={'projection': ccrs.PlateCarree()}
)
for ax, entry in zip(axes3.flat, entry_list):
    data = load_global_buffer_map(YEAR=YEAR, nametag=entry, Area='Global', buffer=1)
    mesh = ax.pcolormesh(
        x, y, data,
        transform=ccrs.PlateCarree(),
        cmap=ListedColormap(['gray', 'green']),
        vmin=0, vmax=1
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    ax.set_title(entry.replace('_', ' ').title(), pad=8)

# Remove any unused axes if fewer than 8 entries
for i in range(len(entry_list), 8):
    fig3.delaxes(axes3.flat[i])

# Add a single shared horizontal colorbar
cbar = fig3.colorbar(
    mesh, ax=axes3.ravel().tolist(),
    orientation='horizontal', pad=0.02
)
cbar.set_label('Buffer Distance (1km) to Nearest Roads')

# Add a main title and adjust layout
fig3.suptitle(f'Global Buffer Distance (1km) to Nearest Roads– {YEAR}', fontsize=20, y=0.97)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
out_dir = (
    '/path/to/NO2_DL_global_2019/NO2_global_pkg/'
    'input_variables/OpenStreetMap_RoadDensity_Buffer_forEachPixels_input/'
)
fig3.savefig(f'{out_dir}Global Buffer Distance to Nearest Roads_{YEAR}.png', dpi=300)
plt.close(fig3)