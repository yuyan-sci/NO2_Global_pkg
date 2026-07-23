from Meteo_input_pkg.Assemble_Func import calculate_save_interpolate_indices,average_interpolate_meteorology_map
from Meteo_input_pkg.visualization import plot_multiple_METEO

if __name__ == '__main__':
    # calculate_save_interpolate_indices()
    YYYY = ['2005', '2006', '2007', '2008', '2009', '2010', '2011',
            '2012', '2013', '2014', '2015', '2016', '2017', '2018',
            '2019', '2020', '2021', '2022']
    MM = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']  
    filetag = ['A1','A1','A1','A1','A1','A1', 'A1', 'A1', 'A1', 'A3dyn','I3']
    nametag = ['LAI','GRN','USTAR','GWETTOP','PRECTOT','PBLH', 'T2M', 'V10M', 'U10M', 'RH','PS']
    lev = 0
    average_interpolate_meteorology_map(YYYY=YYYY, MM=MM, filetag=filetag, nametag=nametag, Area='GL', lev = lev)
    plot_multiple_METEO(tagname=nametag, Area='GL', YYYY=YYYY[0], MONTH=MM[0])