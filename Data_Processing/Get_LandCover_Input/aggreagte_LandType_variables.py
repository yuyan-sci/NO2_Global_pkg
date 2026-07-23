from LandCover_nearest_distances_pkg.Assemble_Func import derive_aggregated_LandTypeVariablesMap, add_buffer


if __name__ == '__main__':
    YEARS = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019','2020','2021','2022','2023']
    for YEAR in YEARS:
        derive_aggregated_LandTypeVariablesMap(YEAR=YEAR)
        for buffer in [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11]:
            add_buffer(buffer=buffer, YEAR=YEAR)