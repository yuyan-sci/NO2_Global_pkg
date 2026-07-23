import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

from ISA_pkg.iostream import load_global_GeoLatLon_Map, load_ISA_interpolated_mapdata
from ISA_pkg.utils import ISA_input_outdir

cmap = plt.cm.get_cmap('RdYlBu_r')

def verify_ISA_mapdata(YYYY):
    
    mapdata = load_ISA_interpolated_mapdata(YYYY)
    y,x = load_global_GeoLatLon_Map()
    
    fig, ax = plt.subplots(figsize=(6, 4), 
                        subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(x, y, mapdata, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        vmin=np.nanmin(mapdata),
                        vmax=np.nanmax(mapdata))

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02)
    cbar.set_label(f'ISA %')

    plt.title(f'Global ISA {YYYY}', pad=20)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(f'{ISA_input_outdir}{YYYY}/'), exist_ok=True)
    fig.savefig(f'{ISA_input_outdir}{YYYY}/Global_ISA_{YYYY}.png', dpi=300)
    plt.close()