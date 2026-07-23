from GFED_input_pkg.Assemble_Func import interpolate_save_GFED_mapdata,copy_paste_Population_data
from GFED_input_pkg.visualization import verify_anthro_emi_mapdata

if __name__ == '__main__':
    YEARS = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    MONTHS = ['01','02','03','04','05','06','07','08','09','10','11','12']
    nametags = ['DM_TOTL']
    # get_indices = True
    # interpolate_save_GFED_mapdata(GetIndices=get_indices,YEARS=YEARS,MONTHS=MONTHS,Nametags=nametags)

    Copy_YEARS = [2010,2010,2010,2010,2010]
    Paste_YEARS = [2009, 2008, 2007, 2006, 2005]
    copy_paste_Population_data(Copy_YEARS=Copy_YEARS, Paste_YEARS=Paste_YEARS, MONTHS=MONTHS)
    
    for iyear in range(len(Paste_YEARS)):
        verify_anthro_emi_mapdata(YYYY=Paste_YEARS[iyear],MONTH=MONTHS[0])