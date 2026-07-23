import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from CEDSAnthroEmissions_pkg.iostream import load_global_GeoLatLon_Map, load_Anthro_Emi_interpolated_mapdata
from CEDSAnthroEmissions_pkg.utils import Anthropogenic_Emission_outdir


def verify_anthro_emi_mapdata(YYYY, nametag, Area, MONTH):
    """
    Plot a single nametag's emission data for a specific month and year
    """
    cmap = plt.cm.get_cmap('RdYlBu_r')
    mapdata = load_Anthro_Emi_interpolated_mapdata(nametag=nametag, YEAR=YYYY, MONTH=MONTH, Area=Area)
    y, x = load_global_GeoLatLon_Map()
    
    fig, ax = plt.subplots(figsize=(6, 4), 
                        subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-180, 180, -70, 70], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(x, y, mapdata, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        vmin=np.nanmin(mapdata),
                        vmax=np.nanmax(mapdata))

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02)
    cbar.set_label(f'{nametag} Anthropogenic Emission')

    plt.title(f'Global CEDS {nametag} Anthropogenic Emission {YYYY} {MONTH}', pad=20)

    plt.tight_layout()
    fig.savefig(f'Global_{nametag}_{YYYY}_{MONTH}.png')
    plt.close()

def plot_multiple_emissions(YYYY, nametags, Area, MONTH):
    """
    Plot multiple emission types (nametags) for a specific month and year in a single figure
    
    Parameters:
    -----------
    YYYY : str
        Year to plot
    nametags : list
        List of emission types to plot (up to 6)
    Area : str
        Area to plot (e.g., 'Global')
    MONTH : str
        Month to plot (e.g., '01' for January)
    """
    if len(nametags) > 9:
        print("Warning: Only the first 9 nametags will be plotted")
        nametags = nametags[:9]
    
    # Calculate grid layout based on number of nametags
    if len(nametags) <= 3:
        nrows, ncols = 1, len(nametags)
    else:
        nrows, ncols = 3, 3
    
    fig = plt.figure(figsize=(ncols*4, nrows*3))
    cmap = plt.cm.get_cmap('RdYlBu_r')
    y, x = load_global_GeoLatLon_Map()
    
    for i, nametag in enumerate(nametags):
        ax = fig.add_subplot(nrows, ncols, i+1, projection=ccrs.PlateCarree())
        
        # Load data for this nametag
        mapdata = load_Anthro_Emi_interpolated_mapdata(nametag=nametag, YEAR=YYYY, MONTH=MONTH, Area=Area)
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        # ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_extent([-180, 180, -70, 70], crs=ccrs.PlateCarree())
        
        # Plot data
        mesh = ax.pcolormesh(x, y, mapdata, 
                            transform=ccrs.PlateCarree(),
                            cmap=cmap,
                            vmin=0,
                            vmax=np.nanmax(mapdata)*0.5)
        
        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02, fraction=0.05)
        cbar.set_label(f'{nametag}')
        
        # Add title
        ax.set_title(f'{nametag}', fontsize=10)
    
    # Add overall title
    plt.suptitle(f'Global CEDS Anthropogenic Emissions {YYYY} {MONTH}', fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt_dir = Anthropogenic_Emission_outdir + '{}/'.format(YYYY)
    fig.savefig(f'{plt_dir}Global_emissions_{YYYY}_{MONTH}.png', dpi=300, bbox_inches='tight')
    plt.close()