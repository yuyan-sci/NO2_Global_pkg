from CEDSAnthroEmissions_pkg.Assemble_Func import *
from CEDSAnthroEmissions_pkg.visualization import plot_multiple_emissions

if __name__ == '__main__':
    YYYY = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
    # nametags = ['BC', 'OC', 'SO2', 'CO2', 'CH4', 'NO', 'NH3', 'OTHER_VOC', 'NMVOC'] 
    nametags = ['NO'] 
    calculate_save_anthro_emi_interpolate_indices()
    interpolate_anthro_emi_mapdata(YYYY=YYYY,nametags=nametags,Area='Global')
    # copy_YYYY = ['2019']
    # paste_YYYY = ['2023']
    # copy_paste_emi(copy_YYYY=copy_YYYY,paste_YYYY=paste_YYYY,nametags=nametags)

    plot_multiple_emissions(YYYY=YYYY[0], nametags=nametags, Area='Global', MONTH='01')