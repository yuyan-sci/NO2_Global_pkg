import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

from NDVI_pkg.iostream import load_global_GeoLatLon_Map,load_NDVI_interpolated_mapdata
from NDVI_pkg.utils import NDVI_outdir

def verify_NDVI_mapdata(YYYY,MM):
    cmap = plt.cm.get_cmap('RdYlBu_r')
    mapdata = load_NDVI_interpolated_mapdata(YYYY=YYYY,MM=MM)
    y,x = load_global_GeoLatLon_Map()
    
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
    cbar.set_label(f'NDVI')

    plt.title(f'Global NDVI {YYYY}-{MM}', pad=20)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(f'{NDVI_outdir}{YYYY}/'), exist_ok=True)
    fig.savefig(f'{NDVI_outdir}{YYYY}/Global_NDVI_{YYYY}_{MM}.png', dpi=300)
    plt.close()