import numpy as np
import gc
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from Population_input_pkg.iostream import load_cropped_interpolated_PopulationMap

def verify_Population_mapdata(YEAR):
    cmap = plt.cm.get_cmap('RdYlBu_r')
    mapdata = load_cropped_interpolated_PopulationMap(YEAR=YEAR)
    
    print(f'{YEAR} shape', mapdata.shape)
    print(f'{YEAR} max', np.nanmax(mapdata))
    print(f'{YEAR} min', np.nanmin(mapdata))
    print(f'{YEAR} mean', np.nanmean(mapdata))
    
    x = np.load('/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLON_global_MAP.npy')
    y = np.load('/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/tSATLAT_global_MAP.npy')
    
    fig, ax = plt.subplots(figsize=(6, 4), 
                        subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5)  
    ax.set_extent([-180, 180, -60, 70], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(x, y, mapdata, 
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        vmin=0,
                        vmax=1000)

    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.02)
    cbar.set_label(f'Population')

    plt.title(f'Global Population {YEAR}', pad=20)
    plt.tight_layout()
    
    plt_dir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/Population_input/'
    
    fig.savefig(f'{plt_dir}Global_Population_{YEAR}.png', dpi=300)
    plt.close()
    gc.collect()