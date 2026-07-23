import os
import shap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
from sklearn.metrics import mean_squared_error,r2_score
from Training_pkg.Statistic_Func import regress2, linear_regression
from Training_pkg.utils import species
from visualization_pkg.utils import Loss_Accuracy_outdir,species_plot_tag_Name

nrows = 2
ncols = 2
proj = ccrs.PlateCarree()
aspect = (179)/(60+70)
height = 5.0
width = aspect * height
vpad = 0.03 * height
hpad = 0.02 * width
hlabel = 0.12 * height*2
vlabel = 0.1 * height*2
hmargin = 0.03 * width
vmargin = 0.03 * height*2
cbar_height = 0.48 * height
cbar_width = 0.015 * width
cbar_height_2 = 0.9 * (height*2 - vlabel)
cbar_width_2 = 0.08 * (width + height*2)

figwidth = width + height + hmargin*2 + cbar_width_2
figheight = height*2 + vmargin*2


def shap_value_plot(shap_values_with_feature_names:shap._explanation.Explanation,plot_type:str,outfile:str):
    if plot_type == 'beeswarm':
        tag_name = species_plot_tag_Name()
        shap.plots.beeswarm(shap_values_with_feature_names, show=False, max_display=21)

        # Get the current figure and axes for customization
        fig = plt.gcf()

        #cbar = fig.axes[-1]  # The colorbar is usually the last axis in the figure
        #cbar.set_ylim(-1,1)
        

        cbar = fig.get_axes()[-1]  # Retrieve the colorbar axis
        cbar.set_yticks([ 0, 1.0])
        cbar.set_yticklabels(["0","1"])
        plt.xlabel('Impact on {} (ppb)'.format(tag_name))
        cbar.set_ylabel('Predictor variables values')
        plt.savefig(outfile,format='png',dpi=1000, bbox_inches='tight')
        plt.close()
    return

def every_point_regression_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    species, version, typeName, plot_beginyear, plot_endyear, MONTH, nchannel, special_name, width, height):
    MM = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    every_point_plot_obs_pm25 = np.zeros([],dtype=np.float32)
    every_point_plot_pre_pm25 = np.zeros([],dtype=np.float32)
    for iyear in range(plot_endyear-plot_beginyear+1):
        if MONTH == 'Annual':
            for imonth in MM:
                every_point_plot_obs_pm25 = np.append(every_point_plot_obs_pm25,plot_obs_pm25[str(plot_beginyear+iyear)][imonth])
                every_point_plot_pre_pm25 = np.append(every_point_plot_pre_pm25,plot_pre_pm25[str(plot_beginyear+iyear)][imonth])
        else:
            every_point_plot_obs_pm25 = np.append(every_point_plot_obs_pm25,plot_obs_pm25[str(plot_beginyear+iyear)][MONTH])
            every_point_plot_pre_pm25 = np.append(every_point_plot_pre_pm25,plot_pre_pm25[str(plot_beginyear+iyear)][MONTH])
    

    fig_output_dir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-Regression/'.format(species, version)

    if not os.path.isdir(fig_output_dir):
        os.makedirs(fig_output_dir)
    
    fig_outfile =  fig_output_dir +  '{}-{}-EverypointRegression_Figure_{}-{}_{}_{}x{}_{}Channel{}.png'.format(typeName, species, plot_beginyear, plot_endyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    mask = ~np.isnan(every_point_plot_obs_pm25) & ~np.isnan(every_point_plot_pre_pm25)
    every_point_plot_obs_pm25 = every_point_plot_obs_pm25[mask]
    every_point_plot_pre_pm25 = every_point_plot_pre_pm25[mask]

    H, xedges, yedges = np.histogram2d(every_point_plot_obs_pm25, every_point_plot_pre_pm25, bins=100)
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(every_point_plot_obs_pm25, every_point_plot_pre_pm25))
    RMSE = round(RMSE, 1)

    R2 = linear_regression(every_point_plot_obs_pm25, every_point_plot_pre_pm25)
    R2 = np.round(R2, 2)

    ax = plt.axes([0.1,0.1,0.8,0.8])  # [left, bottom, width, height]
    cbar_ax = plt.axes([0.91,0.2,0.03,0.6])
    regression_Dic = regress2(_x=every_point_plot_obs_pm25,_y=every_point_plot_pre_pm25,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
    b0,b1 = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    b0 = round(b0, 2)
    b1 = round(b1, 2)

    extentlim = 6*np.mean(every_point_plot_obs_pm25)
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(every_point_plot_obs_pm25, every_point_plot_pre_pm25,
                   cmap='RdYlBu_r', norm=colors.LogNorm(vmin=1, vmax=100), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='blue', linestyle='-')
    #ax.set_title('Comparsion of Modeled $PM_{2.5}$ and observations for '+area_name+' '+beginyear+' '+endyear)
    ax.set_xlabel('Measured ' + r'$\mathrm{NO_{2}}$' + ' (ppb)', fontsize=32)
    ax.set_ylabel('Estimated '+ r'$\mathrm{NO_{2}}$' + ' (ppb)', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $ {:.2f}'.format(R2), style='italic', fontsize=32)
    ax.text(0, extentlim - (0.05 + 0.064) * extentlim, '$RMSE = $' + str(RMSE)+'ppb', style='italic', fontsize=32)
    if b1 > 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = {}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)
    elif b1 == 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0), style='italic',
            fontsize=32)
    else:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y=-{}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)

    ax.text(0, extentlim - (0.05 + 0.064 * 3) * extentlim, 'N = ' + str(len(every_point_plot_obs_pm25)), style='italic',
            fontsize=32)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100])
    cbar.ax.set_yticklabels(['1', '10', r'$10^2$'], fontsize=24)
    cbar.set_label('Number of points', fontsize=28)

    fig.savefig(fig_outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.close()


def regression_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    species, version, typeName, beginyear, MONTH, nchannel, special_name, width, height):
    
    fig_output_dir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-Regression/'.format(species, version)

    if not os.path.isdir(fig_output_dir):
        os.makedirs(fig_output_dir)
    
    fig_outfile =  fig_output_dir +  '{}-{}-LongtermRegression_Figure_{}_{}_{}x{}_{}Channel{}.png'.format(typeName, species, beginyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    mask = ~np.isnan(plot_obs_pm25) & ~np.isnan(plot_pre_pm25)
    plot_obs_pm25 = plot_obs_pm25[mask]
    plot_pre_pm25 = plot_pre_pm25[mask]
    
    H, xedges, yedges = np.histogram2d(plot_obs_pm25, plot_pre_pm25, bins=100)
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(plot_obs_pm25, plot_pre_pm25))
    RMSE = round(RMSE, 1)

    R2 = linear_regression(plot_obs_pm25, plot_pre_pm25)
    R2 = np.round(R2, 2)

    ax = plt.axes([0.1,0.1,0.8,0.8])  # [left, bottom, width, height]
    cbar_ax = plt.axes([0.91,0.2,0.03,0.6])
    regression_Dic = regress2(_x=plot_obs_pm25,_y=plot_pre_pm25,_method_type_1='ordinary least square',_method_type_2='reduced major axis',
    )
    b0,b1 = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    b0 = round(b0, 2)
    b1 = round(b1, 2)

    extentlim = 6*np.mean(plot_obs_pm25)
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(plot_obs_pm25, plot_pre_pm25,
                   cmap='RdYlBu_r', norm=colors.LogNorm(vmin=1, vmax=100), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='blue', linestyle='-')
    #ax.set_title('Comparsion of Modeled $PM_{2.5}$ and observations for '+area_name+' '+beginyear+' '+endyear)
    ax.set_xlabel('Measured '+ r'$\mathrm{NO_{2}}$' + 'mixing ratio (ppb)', fontsize=32)
    ax.set_ylabel('Estimated '+ r'$\mathrm{NO_{2}}$' + 'mixing ratio (ppb)', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $ {:.2f}'.format(R2), style='italic', fontsize=32)
    ax.text(0, extentlim - (0.05 + 0.064) * extentlim, '$RMSE = $' + str(RMSE)+'ppb', style='italic', fontsize=32)
    if b1 > 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = {}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)
    elif b1 == 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0), style='italic',
            fontsize=32)
    else:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y=-{}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)

    ax.text(0, extentlim - (0.05 + 0.064 * 3) * extentlim, 'N = ' + str(len(plot_pre_pm25)), style='italic',
            fontsize=32)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100])
    cbar.ax.set_yticklabels(['1', '10', r'$10^2$',], fontsize=24)
    cbar.set_label('Number of points', fontsize=28)

    fig.savefig(fig_outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.close()

def return_sign(number):
    if number < 0.0:
        return '-'
    elif number == 0.0:
        return ''
    else:
        return '+'   

def geo_every_point_regression_plot(plot_obs_pm25:np.array,plot_pre_pm25:np.array,
                    species, version, typeName, plot_beginyear, plot_endyear, MONTH, nchannel, special_name, width, height):
    MM = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    every_point_plot_obs_pm25 = np.zeros([],dtype=np.float32)
    every_point_plot_pre_pm25 = np.zeros([],dtype=np.float32)
    for iyear in range(plot_endyear-plot_beginyear+1):
        if MONTH == 'Annual':
            for imonth in MM:
                every_point_plot_obs_pm25 = np.append(every_point_plot_obs_pm25,plot_obs_pm25[str(plot_beginyear+iyear)][imonth])
                every_point_plot_pre_pm25 = np.append(every_point_plot_pre_pm25,plot_pre_pm25[str(plot_beginyear+iyear)][imonth])
        else:
            every_point_plot_obs_pm25 = np.append(every_point_plot_obs_pm25,plot_obs_pm25[str(plot_beginyear+iyear)][MONTH])
            every_point_plot_pre_pm25 = np.append(every_point_plot_pre_pm25,plot_pre_pm25[str(plot_beginyear+iyear)][MONTH])
    

    fig_output_dir = Loss_Accuracy_outdir + '{}/{}/Figures/figures-Regression/'.format(species, version)

    if not os.path.isdir(fig_output_dir):
        os.makedirs(fig_output_dir)
    
    fig_outfile =  fig_output_dir +  '{}-{}-GeoEverypointRegression_Figure_{}-{}_{}_{}x{}_{}Channel{}.png'.format(typeName, species, plot_beginyear, plot_endyear, MONTH
                                                                                            ,width, height, nchannel,special_name)
    
    mask = ~np.isnan(every_point_plot_obs_pm25) & ~np.isnan(every_point_plot_pre_pm25)
    every_point_plot_obs_pm25 = every_point_plot_obs_pm25[mask]
    every_point_plot_pre_pm25 = every_point_plot_pre_pm25[mask]

    H, xedges, yedges = np.histogram2d(every_point_plot_obs_pm25, every_point_plot_pre_pm25, bins=100)
    fig = plt.figure(figsize=(10, 10))
    #fig = plt.figure(figsize=(figwidth, figheight))
    extent = [0, max(xedges), 0, max(xedges)]
    RMSE = np.sqrt(mean_squared_error(every_point_plot_obs_pm25, every_point_plot_pre_pm25))
    RMSE = round(RMSE, 1)

    R2 = linear_regression(every_point_plot_obs_pm25, every_point_plot_pre_pm25)
    R2 = np.round(R2, 2)

    ax = plt.axes([0.1,0.1,0.8,0.8])  # [left, bottom, width, height]
    cbar_ax = plt.axes([0.91,0.2,0.03,0.6])
    regression_Dic = regress2(_x=every_point_plot_obs_pm25,_y=every_point_plot_pre_pm25,_method_type_1='ordinary least square',_method_type_2='reduced major axis',)
    b0,b1 = regression_Dic['intercept'], regression_Dic['slope']
    #b0, b1 = linear_slope(plot_obs_pm25,
    #                      plot_pre_pm25)
    b0 = round(b0, 2)
    b1 = round(b1, 2)

    extentlim = 6*np.mean(every_point_plot_obs_pm25)
    # im = ax.imshow(
    #    H, extent=extent,
    #    cmap= 'gist_rainbow',
    #   origin='lower',
    #  norm=colors.LogNorm(vmin=1, vmax=1e3))
    im = ax.hexbin(every_point_plot_obs_pm25, every_point_plot_pre_pm25,
                   cmap='RdYlBu_r', norm=colors.LogNorm(vmin=1, vmax=100), extent=(0, extentlim, 0, extentlim),
                   mincnt=1)
    ax.plot([0, extentlim], [0, extentlim], color='black', linestyle='--')
    ax.plot([0, extentlim], [b0, b0 + b1 * extentlim], color='blue', linestyle='-')
    #ax.set_title('Comparsion of Modeled $PM_{2.5}$ and observations for '+area_name+' '+beginyear+' '+endyear)
    ax.set_xlabel('Measured ' + r'$\mathrm{NO_{2}}$' + ' (ppb)', fontsize=32)
    ax.set_ylabel('Estimated '+ r'$\mathrm{NO_{2}}$' + ' (ppb)', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=28)

    ax.text(0, extentlim - 0.05 * extentlim, '$R^2 = $ {:.2f}'.format(R2), style='italic', fontsize=32)
    ax.text(0, extentlim - (0.05 + 0.064) * extentlim, '$RMSE = $' + str(RMSE)+'ppb', style='italic', fontsize=32)
    if b1 > 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = {}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)
    elif b1 == 0.0:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y = ' + str(b0), style='italic',
            fontsize=32)
    else:
        ax.text(0, extentlim - (0.05 + 0.064 * 2) * extentlim, 'y=-{}x {} {}'.format(abs(b1),return_sign(b0),abs(b0)) , style='italic',
            fontsize=32)

    ax.text(0, extentlim - (0.05 + 0.064 * 3) * extentlim, 'N = ' + str(len(every_point_plot_obs_pm25)), style='italic',
            fontsize=32)
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='vertical', shrink=1.0, ticks=[1, 10, 100])
    cbar.ax.set_yticklabels(['1', '10', r'$10^2$'], fontsize=24)
    cbar.set_label('Number of points', fontsize=28)

    fig.savefig(fig_outfile, dpi=1000,transparent = True,bbox_inches='tight' )
    plt.close()