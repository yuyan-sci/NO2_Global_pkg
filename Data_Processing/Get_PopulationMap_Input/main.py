import numpy as np
from Population_input_pkg.Assemble_Func import load_crop_interpolate_save_PopulationData, copy_paste_Population_data
from Population_input_pkg.visualization import verify_Population_mapdata


if __name__ == '__main__':
    Left_Point_YEARS  = [2000,2005,2010,2015]
    Right_Point_YEARS = [2005,2010,2015,2020]
    Copy_YEARS  = [2020,2020, 2020]
    Paste_YEARS = [2021,2022, 2023]
    Extent = [-59.995,69.995,-179.995,179.995]
    for Aimed_YEAR in list(range(2005, 2021)):
        load_crop_interpolate_save_PopulationData(Left_Point_YEARS=Left_Point_YEARS,Right_Point_YEARS=Right_Point_YEARS,
                                                Aimed_YEAR=Aimed_YEAR,Extent=Extent)
        verify_Population_mapdata(Aimed_YEAR)
    copy_paste_Population_data(Copy_YEARS=Copy_YEARS,Paste_YEARS=Paste_YEARS)
    verify_Population_mapdata(2021)
    verify_Population_mapdata(2022)
    verify_Population_mapdata(2023)