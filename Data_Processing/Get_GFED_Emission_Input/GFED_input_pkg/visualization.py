import numpy as np
import gc
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from GFED_input_pkg.iostream import load_global_GeoLatLon_Map, load_GFED_interpolated_mapdata
from GFED_input_pkg.utils import Offline_GFED_outdir

def verify_anthro_emi_mapdata(YYYY, MONTH):
    cmap = plt.cm.get_cmap('RdYlBu_r')
    mapdata = load_GFED_interpolated_mapdata(nametag='DM_TOTL', YEAR=YYYY, MONTH=MONTH)
    y, x = load_global_GeoLatLon_Map()
    
    fig, ax = plt.subplots(figsize=(6, 4), 
                        subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())
    
    valid_data = mapdata[~np.isnan(mapdata)]
    if len(valid_data) > 0:
        # Use percentiles for better color scaling
        vmin = np.nanpercentile(mapdata, 1)  # 1st percentile
        vmax = np.nanpercentile(mapdata, 99)  # 99th percentile
        
        print(f"Data range: min={np.nanmin(mapdata)}, max={np.nanmax(mapdata)}")
        print(f"Using color range: vmin={vmin}, vmax={vmax} (1-99 percentiles)")
    else:
        vmin = 0
        vmax = 1
        print("Warning: No valid data found")
    mesh = ax.pcolormesh(x, y, mapdata, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax)

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02)
    cbar.set_label(f'DM_TOTL')

    plt.title(f'Global GFED DM_TOTL Emission {YYYY} {MONTH}', pad=20)

    plt.tight_layout()
    fig.savefig(f'{Offline_GFED_outdir}/{YYYY}/Global_GFED_{YYYY}_{MONTH}.png', dpi=300, bbox_inches='tight')
    plt.close()
    gc.collect()