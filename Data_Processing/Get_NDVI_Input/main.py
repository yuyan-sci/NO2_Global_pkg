from NDVI_pkg.Assemble_Func import *
from NDVI_pkg.visualization import verify_NDVI_mapdata

if __name__ == '__main__':
    YEARS = ['2019']
    MM = ['12']
    for YYYY in YEARS:
        calculate_save_NDVI_interpolate_indices(YYYY=YYYY,MM=MM)
        for imonth in range(len(MM)):
            interpolate_NDVI_mapdata(YYYY=YYYY,MM=MM[imonth],Area='Global')
        verify_NDVI_mapdata(YYYY=YYYY,MM=MM)