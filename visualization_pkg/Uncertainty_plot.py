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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Training_pkg.Statistic_Func import Calculate_PWA_PM25
from Training_pkg.utils import species
from visualization_pkg.utils import crop_map_data
from Uncertainty_pkg.utils import *


def Plot_LOWESS_values_bins_Figure(LOWESS_vallues_dic,rRMSE_dic,output_bins,outfile):
      Plot_Months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Annual']
      color = ['#0047AB','#FFB3BA','#FFDFBA','#BAFFC9','#E6194B','#F58231','#8B4513','#00CCBC','#BAE1FF',
            '#6B8E23','#1C1C1C','#FFE119','#D4AF37']

      symbols = ['o','^','v','<','>','s','p','h','H','+','X','D','d']
      count = 0
      fig1 = plt.figure(figsize=[15,10])
      ax1 = fig1.add_axes([0.1,0.1,0.8,0.8])
      X_label_values = np.array(range(round(Max_distances_for_Bins/50)+1))*50
      xlabels = [str(radius) for radius in X_label_values]
      total_ymax_lim = 0.6
      for Plot_Month in Plot_Months:
            temp_max_values = np.max(LOWESS_vallues_dic[Plot_Month])
            total_ymax_lim = max(temp_max_values,total_ymax_lim)
            plt.title('Uncertainty - {} BLISCO rRMSE Results'.format(species),fontsize=36)
            plt.scatter(output_bins,rRMSE_dic[Plot_Month],marker = symbols[count],c=color[count],linewidth=3,label = '{}'.format(Plot_Month))
            plt.plot(output_bins,LOWESS_vallues_dic[Plot_Month],c=color[count],linewidth=3)#,label = 'Linear Regression {} rRMSE'.format(Plot_Month))
            count += 1
      plt.yticks(size=24)
      plt.ylim(0.05, total_ymax_lim+0.1)
      ax1.set_xticks(X_label_values,xlabels,fontsize=24,rotation=-60)
      plt.ylabel('rRMSE', fontdict={ 'size'   : 36})
      plt.xlabel('Distances (km)', fontdict={'size'   : 36})
      plt.legend(loc=2,fontsize=24,frameon=True, reverse=False,bbox_to_anchor=(1.01,1.01))
      plt.savefig(outfile, format = 'png', dpi= 2000, transparent = True,bbox_inches='tight')
      plt.close()
      return
def Plot_Species_Uncertainty_Map_Figures(Uncertainty_Map:np.array,PM25_LAT:np.array,PM25_LON:np.array,Population_Map:np.array,
                             population_Lat:np.array, population_Lon:np.array, extent:np.array, outfile:str,YYYY,MM):
    MONTH = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    Uncertainty_Map[np.where(Uncertainty_Map < 0)] = 0
    Uncertainty_Map = np.nan_to_num(Uncertainty_Map, nan=5.0, posinf=3.0, neginf=2.0)

    Cropped_Population_Map = crop_map_data(Population_Map,population_Lat,population_Lon,extent)
    Croppeed_PM25_Map      = crop_map_data(Uncertainty_Map, PM25_LAT, PM25_LON,extent)
    PWA_PM25 = Calculate_PWA_PM25(Population_array=Cropped_Population_Map, PM25_array=Croppeed_PM25_Map)
    ax = plt.axes(projection=ccrs.PlateCarree())
    m1 = 0
    m2 = 10.0 #np.mean(Uncertainty_Map) * 1.5 #min(int(abs(PWA_PM25) * 2.5),120)
    extent = [extent[2],extent[3],extent[0],extent[1]]
    
    #print('PM25_Lat Dim:',PM25_LAT.shape, 'PM25_Lon Dim:', PM25_LON.shape, 'PM25_Map Dim:', PM25_Map.shape,'PWA: ',PWA_PM25)
    
    print('extent:', extent)
    ax.set_aspect(1.25)
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    #ax.add_feature(cfeat.COASTLINE,linewidth = 0.15) 
    ax.add_feature(cfeat.LAKES, linewidth = 0.05)
    ax.add_feature(cfeat.BORDERS, linewidth=0.1)
    pcm = plt.pcolormesh(PM25_LON, PM25_LAT,Uncertainty_Map,transform=ccrs.PlateCarree(),
          cmap = 'plasma',norm=colors.Normalize(vmin = m1, vmax = m2))
    #ax.add_feature(cfeat.OCEAN) 

    #RMSE = round(np.sqrt(mean_squared_error(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],
    #                                      pre_pm25_site[area_index])),2)
    #R2 = round(linear_regression(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],pre_pm25_site[area_index]),2)    
    #ax.text(extent[2], extent[1]-0.1*abs(extent[1]), '$R^2 = $' + str(R2), style='italic', fontsize=12)
    #ax.text(extent[2], extent[1], '$RMSE = $' + str(RMSE), style='italic', fontsize=12)
    
    ax.text(extent[0]+0.01*abs(extent[1]-extent[0]),extent[2]+0.05*abs(extent[3]-extent[2]),'PWM ' + r'$\rm{NO_{2} = }$' + str(round(PWA_PM25,1)) +r' $\rm{(ppb)}$', style='italic',fontsize = 6)
    ax.text(extent[0]+0.01*abs(extent[1]-extent[0]),extent[2]+0.10*abs(extent[3]-extent[2]),'{} {}'.format(YYYY,MM), style='italic',fontsize = 6)
    
   # plt.scatter(PM25_Sites_LON, PM25_Sites_LAT, c=PM25_Sites, s=0.1,
    #                linewidths=0.1, marker='o', edgecolors='black', vmin=0, vmax=m2,
     #               cmap='YlOrRd',
     #              alpha=0.4)
    
    ## Global colorbar parameters fraction=0.35, pad=-1.63, shrink=0.5, aspect=50.0
    cbar = plt.colorbar(pcm, location = 'right',fraction=0.15, shrink=0.5,aspect=40.0, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('NO$_{2}$' + '' + r'$\rm{(ppb)}$')
    cbar.ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='k', alpha=0.7, linestyle='--')
    ##gl.top_labels = False  ##关闭上侧坐标显示
    #gl.right_labels = False  ##关闭右侧坐标显示
    #gl.xformatter = LONGITUDE_FORMATTER  ##坐标刻度转换为经纬度样式
    #gl.yformatter = LATITUDE_FORMATTER
    #gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1], 10))
    #gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3], 10))
    #gl.xlabel_style = {'size': 3.5}
    #gl.ylabel_style = {'size': 3.5}
    plt.savefig(outfile, format = 'png', dpi= 2000, transparent = True,bbox_inches='tight')
    plt.close()

    return

def Plot_Species_Uncertainty_rRMSE_Map_Figures(rRMSE_Uncertainty_Map:np.array,PM25_LAT:np.array,PM25_LON:np.array,Population_Map:np.array,
                             population_Lat:np.array, population_Lon:np.array, extent:np.array, outfile:str,YYYY,MM):
    MONTH = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    rRMSE_Uncertainty_Map[np.where(rRMSE_Uncertainty_Map < 0)] = 0
    Cropped_Population_Map = crop_map_data(Population_Map,population_Lat,population_Lon,extent)
    Croppeed_PM25_Map      = crop_map_data(rRMSE_Uncertainty_Map, PM25_LAT, PM25_LON,extent)
    PWA_PM25 = Calculate_PWA_PM25(Population_array=Cropped_Population_Map, PM25_array=Croppeed_PM25_Map)
    ax = plt.axes(projection=ccrs.PlateCarree())
    m1 = 0
    m2 = 40.0 #np.mean(Uncertainty_Map) * 1.5 #min(int(abs(PWA_PM25) * 2.5),120)
    extent = [extent[2],extent[3],extent[0],extent[1]]
    
    #print('PM25_Lat Dim:',PM25_LAT.shape, 'PM25_Lon Dim:', PM25_LON.shape, 'PM25_Map Dim:', PM25_Map.shape,'PWA: ',PWA_PM25)
    
    print('extent:', extent)
    ax.set_aspect(1.25)
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    #ax.add_feature(cfeat.COASTLINE,linewidth = 0.15) 
    ax.add_feature(cfeat.LAKES, linewidth = 0.05)
    ax.add_feature(cfeat.BORDERS, linewidth=0.1)
    pcm = plt.pcolormesh(PM25_LON, PM25_LAT,rRMSE_Uncertainty_Map,transform=ccrs.PlateCarree(),
          cmap = 'plasma',norm=colors.Normalize(vmin = m1, vmax = m2))
    #ax.add_feature(cfeat.OCEAN) 

    #RMSE = round(np.sqrt(mean_squared_error(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],
    #                                      pre_pm25_site[area_index])),2)
    #R2 = round(linear_regression(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],pre_pm25_site[area_index]),2)    
    #ax.text(extent[2], extent[1]-0.1*abs(extent[1]), '$R^2 = $' + str(R2), style='italic', fontsize=12)
    #ax.text(extent[2], extent[1], '$RMSE = $' + str(RMSE), style='italic', fontsize=12)
    
    ax.text(extent[0]+0.01*abs(extent[1]-extent[0]),extent[2]+0.10*abs(extent[3]-extent[2]),'{} {}'.format(YYYY,MM), style='italic',fontsize = 6)
    
   # plt.scatter(PM25_Sites_LON, PM25_Sites_LAT, c=PM25_Sites, s=0.1,
    #                linewidths=0.1, marker='o', edgecolors='black', vmin=0, vmax=m2,
     #               cmap='YlOrRd',
     #              alpha=0.4)
    
    ## Global colorbar parameters fraction=0.35, pad=-1.63, shrink=0.5, aspect=50.0
    cbar = plt.colorbar(pcm, location = 'right',fraction=0.15, shrink=0.5,aspect=40.0, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('rRMSE')
    cbar.ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='k', alpha=0.7, linestyle='--')
    ##gl.top_labels = False  ##关闭上侧坐标显示
    #gl.right_labels = False  ##关闭右侧坐标显示
    #gl.xformatter = LONGITUDE_FORMATTER  ##坐标刻度转换为经纬度样式
    #gl.yformatter = LATITUDE_FORMATTER
    #gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1], 10))
    #gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3], 10))
    #gl.xlabel_style = {'size': 3.5}
    #gl.ylabel_style = {'size': 3.5}
    plt.savefig(outfile, format = 'png', dpi= 2000, transparent = True,bbox_inches='tight')
    plt.close()

    return
