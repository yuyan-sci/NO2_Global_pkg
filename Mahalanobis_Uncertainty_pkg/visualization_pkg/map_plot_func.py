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

def Plot_Map_estimation_Map_Figures(species,map_estimation_map:np.array,Mahalanobis_LAT:np.array,Mahalanobis_LON:np.array, extent:np.array, outfile:str,YYYY,MM,vmin=0,vmax=15):
    title_name = {
        'NO2' : r'NO$_{2}$',
    }
    map_estimation_map[np.where(map_estimation_map < 0)] = 0
    map_estimation_map = np.nan_to_num(map_estimation_map, nan=5.0, posinf=3.0, neginf=2.0)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ## make the color scale with (0,1,3,5,8,10,15,20,25,50,80,100)
    larger_zero_pixels = np.where(map_estimation_map > 0)
    
    extent = [extent[2],extent[3],extent[0],extent[1]]

    print('extent:', extent)
    ax.set_aspect(1.25)
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeat.COASTLINE,linewidth = 0.15) 
    ax.add_feature(cfeat.LAKES, linewidth   = 0.1,facecolor='white')
    ax.add_feature(cfeat.BORDERS, linewidth = 0.1)
    pcm = plt.pcolormesh(Mahalanobis_LON, Mahalanobis_LAT,map_estimation_map,transform=ccrs.PlateCarree(),
          cmap = 'plasma',norm=colors.Normalize(vmin=vmin, vmax=vmax))
    #ax.add_feature(cfeat.OCEAN) 

    #RMSE = round(np.sqrt(mean_squared_error(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],
    #                                      pre_pm25_site[area_index])),2)
    #R2 = round(linear_regression(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],pre_pm25_site[area_index]),2)    
    #ax.text(extent[2], extent[1]-0.1*abs(extent[1]), '$R^2 = $' + str(R2), style='italic', fontsize=12)
    #ax.text(extent[2], extent[1], '$RMSE = $' + str(RMSE), style='italic', fontsize=12)
    SPEC_NAME = title_name[species]
    ax.text(extent[1]+0.01*abs(extent[1]-extent[0]),extent[2]+0.05*abs(extent[3]-extent[2]),'{} {} {}'.format(SPEC_NAME,YYYY,MM), style='italic',fontsize = 10)

    # Global colorbar parameters fraction=0.35, pad=-1.63, shrink=0.5, aspect=50.0
    #cbar = plt.colorbar(pcm, location = 'right', fraction=0.15, shrink=0.35, aspect=40.0, anchor=(-5.7,0.5), orientation='vertical', extend='both')
    #ticks = np.linspace(vmin, vmax, 5)
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    #cbar.ax.tick_params(labelsize=12)
    #cbar.set_label('{}'.format(title_name[species]) + '' + r'$\rm{(\mu g/m^3)}$')
    #cbar.ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='k', alpha=0.7, linestyle='--')
    ##gl.top_labels = False  ##关闭上侧坐标显示
    #gl.right_labels = False  ##关闭右侧坐标显示
    #gl.xformatter = LONGITUDE_FORMATTER  ##坐标刻度转换为经纬度样式
    #gl.yformatter = LATITUDE_FORMATTER
    #gl.xlocator = mticker.FixedLocator(np.arange(extent[0], extent[1], 10))
    #gl.ylocator = mticker.FixedLocator(np.arange(extent[2], extent[3], 10))
    #gl.xlabel_style = {'size': 3.5}
    #gl.ylabel_style = {'size': 3.5}

    cbar = plt.colorbar(pcm, location = 'right', fraction=0.15, shrink=0.30, aspect=40.0, anchor=(-5.7,0.5), orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('{}'.format(title_name[species]) + '' + r'$\rm{(ppb)}$',size=8)
    #cbar.set_label('NO$_{2}$' + '' + r'$\rm{(ppb)}$')
    cbar.ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    
    plt.savefig(outfile, format = 'png', dpi= 500, transparent = True,bbox_inches='tight')
    plt.close()

    return

def Plot_absolute_Uncertainty_Map_Figures(species,absolute_uncertainty_map:np.array,Mahalanobis_LAT:np.array,Mahalanobis_LON:np.array, extent:np.array, outfile:str,YYYY,MM,vmin=0,vmax=20):

    title_name = {
        'NO2' : r'NO$_{2}$',
    }
    absolute_uncertainty_map[np.where(absolute_uncertainty_map < 0)] = 0
    absolute_uncertainty_map = np.nan_to_num(absolute_uncertainty_map, nan=5.0, posinf=3.0, neginf=2.0)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ## make the color scale with (0,1,3,5,8,10,15,20,25,50,80,100)
    larger_zero_pixels = np.where(absolute_uncertainty_map > 0)
    
    extent = [extent[2],extent[3],extent[0],extent[1]]

    print('extent:', extent)
    ax.set_aspect(1.25)
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeat.COASTLINE,linewidth = 0.15) 
    ax.add_feature(cfeat.LAKES, linewidth   = 0.05,facecolor='white')
    ax.add_feature(cfeat.BORDERS, linewidth = 0.1)
    pcm = plt.pcolormesh(Mahalanobis_LON, Mahalanobis_LAT,absolute_uncertainty_map,transform=ccrs.PlateCarree(),
          cmap = 'plasma',norm=colors.Normalize(vmin=vmin, vmax=vmax))
    #ax.add_feature(cfeat.OCEAN) 

    #RMSE = round(np.sqrt(mean_squared_error(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],
    #                                      pre_pm25_site[area_index])),2)
    #R2 = round(linear_regression(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],pre_pm25_site[area_index]),2)    
    #ax.text(extent[2], extent[1]-0.1*abs(extent[1]), '$R^2 = $' + str(R2), style='italic', fontsize=12)
    #ax.text(extent[2], extent[1], '$RMSE = $' + str(RMSE), style='italic', fontsize=12)
    SPEC_NAME = title_name[species]
    ax.text(extent[1]+0.01*abs(extent[1]-extent[0]),extent[2]+0.05*abs(extent[3]-extent[2]),'{} {} {}'.format(SPEC_NAME,YYYY,MM), style='italic',fontsize = 10,fontweight='bold')
    cbar = plt.colorbar(pcm, location = 'right', fraction=0.15, shrink=0.30, aspect=40.0, anchor=(-5.7,0.5), orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Uncertainties' + '' + r'$\rm{(ppb)}$',size=8)
    #cbar.set_label('PM$_{2.5}$' + '' + r'$\rm{(\mu g/m^3)}$')
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
    plt.savefig(outfile, format = 'png', dpi= 500, transparent = True,bbox_inches='tight')
    plt.close()

    return


def Plot_rRMSE_Uncertainty_Map_Figures(species,rRMSE_uncertainty_map:np.array,Mahalanobis_LAT:np.array,Mahalanobis_LON:np.array, extent:np.array, outfile:str,YYYY,MM,vmin=0.2,vmax=0.6):

    title_name = {
        'NO2' : r'NO$_{2}$',
    }
    rRMSE_uncertainty_map[np.where(rRMSE_uncertainty_map < 0)] = 0
    rRMSE_uncertainty_map = np.nan_to_num(rRMSE_uncertainty_map, nan=5.0, posinf=3.0, neginf=2.0)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ## make the color scale with (0,1,3,5,8,10,15,20,25,50,80,100)
    
    
    extent = [extent[2],extent[3],extent[0],extent[1]]

    print('extent:', extent)
    ax.set_aspect(1.25)
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeat.COASTLINE,linewidth = 0.15) 
    ax.add_feature(cfeat.LAKES, linewidth   = 0.05,facecolor='white')
    ax.add_feature(cfeat.BORDERS, linewidth = 0.1)
    pcm = plt.pcolormesh(Mahalanobis_LON, Mahalanobis_LAT,rRMSE_uncertainty_map,transform=ccrs.PlateCarree(),
          cmap = 'plasma',norm=colors.Normalize(vmin=vmin, vmax=vmax))
    #ax.add_feature(cfeat.OCEAN) 

    #RMSE = round(np.sqrt(mean_squared_error(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],
    #                                      pre_pm25_site[area_index])),2)
    #R2 = round(linear_regression(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],pre_pm25_site[area_index]),2)    
    #ax.text(extent[2], extent[1]-0.1*abs(extent[1]), '$R^2 = $' + str(R2), style='italic', fontsize=12)
    #ax.text(extent[2], extent[1], '$RMSE = $' + str(RMSE), style='italic', fontsize=12)
    SPEC_NAME = title_name[species]
    ax.text(extent[1]+0.01*abs(extent[1]-extent[0]),extent[2]+0.05*abs(extent[3]-extent[2]),'{} {} {}'.format(SPEC_NAME,YYYY,MM), style='italic',fontsize = 10,fontweight='bold')
    cbar = plt.colorbar(pcm, location = 'right', fraction=0.15, shrink=0.30, aspect=40.0, anchor=(-5.7,0.5), orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Uncertainties' + '' + r'$\rm{(unitless)}$',size=8)
    #cbar.set_label('PM$_{2.5}$' + '' + r'$\rm{(\mu g/m^3)}$')
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
    plt.savefig(outfile, format = 'png', dpi= 500, transparent = True,bbox_inches='tight')
    plt.close()

    return



def Plot_Mahalanobis_distance_Map_Figures(species,mahalanobis_distance_map:np.array,Mahalanobis_LAT:np.array,Mahalanobis_LON:np.array, extent:np.array, outfile:str,YYYY,MM):
    MONTH = ['Jan', 'Feb', 'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    title_name= {
        'NO2' : r'NO$_{2}$',
    }
    mahalanobis_distance_map[np.where(mahalanobis_distance_map < 0)] = 0
    mahalanobis_distance_map = np.nan_to_num(mahalanobis_distance_map, nan=5.0, posinf=3.0, neginf=2.0)
    mahalanobis_distance_map = np.log1p(mahalanobis_distance_map)
    ax = plt.axes(projection=ccrs.PlateCarree())

    cbar_levels = [0.5, 1.0, 1.5, 2.0, 2.5]
    extent = [extent[2],extent[3],extent[0],extent[1]]

    print('extent:', extent)
    ax.set_aspect(1.25)
    ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.add_feature(cfeat.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='none', facecolor='white'))
    ax.add_feature(cfeat.COASTLINE,linewidth = 0.15) 
    ax.add_feature(cfeat.LAKES, linewidth   = 0.05,facecolor='white')
    ax.add_feature(cfeat.BORDERS, linewidth = 0.1)
    pcm = plt.pcolormesh(Mahalanobis_LON, Mahalanobis_LAT,mahalanobis_distance_map,transform=ccrs.PlateCarree(),
          cmap = 'plasma',norm=colors.Normalize(vmin=cbar_levels[0], vmax=cbar_levels[-1]))
    #ax.add_feature(cfeat.OCEAN) 

    #RMSE = round(np.sqrt(mean_squared_error(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],
    #                                      pre_pm25_site[area_index])),2)
    #R2 = round(linear_regression(sitePM25[area_index, (yyyy[iyear]-1998)*12+mm[imonth]],pre_pm25_site[area_index]),2)    
    #ax.text(extent[2], extent[1]-0.1*abs(extent[1]), '$R^2 = $' + str(R2), style='italic', fontsize=12)
    #ax.text(extent[2], extent[1], '$RMSE = $' + str(RMSE), style='italic', fontsize=12)
    
    SPEC_NAME = title_name[species]
    # Global colorbar parameters fraction=0.35, pad=-1.63, shrink=0.5, aspect=50.0
    ax.text(extent[1]+0.01*abs(extent[1]-extent[0]),extent[2]+0.05*abs(extent[3]-extent[2]),'{} {} {}'.format(SPEC_NAME,YYYY,MM), style='italic',fontsize = 10,fontweight='bold')
    cbar = plt.colorbar(pcm, location = 'right', fraction=0.15, shrink=0.30, aspect=40.0, anchor=(-5.7,0.5), orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Mahalanobis Dist',size=8)
    cbar.ax.xaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))
    
    plt.savefig(outfile, format = 'png', dpi= 500, transparent = True,bbox_inches='tight')
    plt.close()

    return
