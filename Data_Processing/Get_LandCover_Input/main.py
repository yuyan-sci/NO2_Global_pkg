from LandCover_pkg.Assemble_Func import *
from LandCover_pkg.utils import IGBP_tagnames
from LandCover_nearest_distances_pkg.Assemble_Func import Calculate_save_LandCover_NearestDistances_forEachPixel

if __name__ == '__main__':
    calculate_save_LandCover_interpolate_indices()
    YEARS = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019','2020','2021','2022','2023']
    index = np.arange(0,15)
    nametags = [IGBP_tagnames[i] for i in index]
    for YYYY in YEARS:
        interpolate_LandCover_mapdata(YYYY=YYYY,nametags=nametags,Area='Global', index=index)
        Calculate_save_LandCover_NearestDistances_forEachPixel(init_nametags=nametags,Area='Global', YEAR=YYYY)