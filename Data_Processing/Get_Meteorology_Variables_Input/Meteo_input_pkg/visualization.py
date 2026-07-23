import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from Meteo_input_pkg.iostream import load_global_GeoLatLon_Map, load_meteo_mapdata
from Meteo_input_pkg.utils import meteorology_mapdata_outdir

def plot_multiple_METEO(tagname, Area, YYYY, MONTH):
    if len(tagname) > 9:
        print("Warning: Only the first 9 tagname will be plotted")
        tagname = tagname[:9]
    
    # Calculate grid layout based on number of tagname
    if len(tagname) <= 3:
        nrows, ncols = 1, len(tagname)
    else:
        nrows, ncols = 3, 3 
    
    fig = plt.figure(figsize=(ncols*4, nrows*3))
    cmap = plt.cm.get_cmap('RdYlBu_r')
    y, x = load_global_GeoLatLon_Map()
    
    for i, nametag in enumerate(tagname):
        ax = fig.add_subplot(nrows, ncols, i+1, projection=ccrs.PlateCarree())
        
        # Load data for this nametag
        mapdata = load_meteo_mapdata(tagname=nametag, YYYY=YYYY, MM=MONTH, Area=Area)
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        # ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
        
        # Plot data
        mesh = ax.pcolormesh(x, y, mapdata, 
                            transform=ccrs.PlateCarree(),
                            cmap=cmap,
                            vmin=np.nanmin(mapdata),
                            vmax=np.nanmax(mapdata))
        
        # Add colorbar
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02, fraction=0.05)
        cbar.set_label(f'{nametag}')
        
        # Add title
        ax.set_title(f'{nametag}', fontsize=10)
    
    # Add overall title
    plt.suptitle(f'GEOS-IT {YYYY} {MONTH}', fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    plt_dir = meteorology_mapdata_outdir + '{}/'.format(YYYY)
    fig.savefig(plt_dir + f'GEOS-IT_{YYYY}_{MONTH}.png', dpi=300, bbox_inches='tight')
    plt.close()