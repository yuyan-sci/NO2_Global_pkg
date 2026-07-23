from click import style
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy as crt
import numpy as np
#from .Statistic_Func import Calculate_PWA_PM25, linear_regression,linear_slope
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import xarray as xr
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.ticker as tick
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Training_pkg.Statistic_Func import Calculate_PWA_PM25
from Evaluation_pkg.utils import calculate_distance_forArray
from visualization_pkg.utils import crop_map_data

def plot_BLCO_test_train_buffers(train_index, test_index, excluded_index, sitelat, sitelon, buffer_radius,extent,fig_outfile):
    ax = plt.axes(projection=ccrs.PlateCarree())
    bottom_lat = extent[0]
    left_lon     = extent[2]
    up_lat       = extent[1]
    right_lon   = extent[3]
    extent = [left_lon,right_lon,bottom_lat,up_lat]
    ax.set_extent(extent)
    
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'land', '50m', edgecolor='none', facecolor=cfeat.COLORS['land']))
    ax.add_feature(cfeat.BORDERS, linewidth=0.1)
    ax.add_feature(cfeat.BORDERS, linewidth=0.1)
    ax.add_feature(cfeat.LAKES, linewidth = 0.05)
    nearest_distances = np.array([],dtype=np.float32)
    for isite in range(len(test_index)):
        site_distances = calculate_distance_forArray(site_lat=sitelat[test_index[isite]],site_lon=sitelon[test_index[isite]],SATLAT_MAP=sitelat[train_index],SATLON_MAP=sitelon[train_index])
        nearest_distances = np.append(nearest_distances,np.min(site_distances[np.where(site_distances>0.1)]))
        ax.add_patch(mpatches.Circle(xy=[sitelon[test_index[isite]], sitelat[test_index[isite]]], radius=buffer_radius*0.01, color='white', alpha=0.8, transform=ccrs.PlateCarree(), zorder=6))
        average_neaerest_distance = round(np.average(nearest_distances),1)
            
    plt.scatter(sitelon[test_index], sitelat[test_index], s=10,
            linewidths=0.1, marker='*', edgecolors='red',c='red',
            alpha=0.8,label='Test Sites - {}\n Average Distance {}'.format(len(test_index),average_neaerest_distance),zorder=10)
    plt.scatter(sitelon[train_index], sitelat[train_index], s=3,  
            linewidths=0.1, marker='o', edgecolors='black',c='black',
            alpha=0.8,label='Training Sites - {}'.format(len(train_index)),zorder=8)
    plt.scatter(sitelon[excluded_index], sitelat[excluded_index], s=3,
            linewidths=0.1, marker='X',c='blue',
            alpha=0.5,label='Excluded Sites - {}'.format(len(excluded_index)),zorder=8)
    plt.legend(fontsize=7,markerscale = 1.2,loc=3,handlelength=1.0,handletextpad=0.3,borderpad=0.3,labelspacing=0.2)
    plt.savefig(fig_outfile, format='png', dpi=2000, transparent=True,bbox_inches='tight')
    plt.close()
    return
