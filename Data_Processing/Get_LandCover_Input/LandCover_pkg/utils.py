LandCover_init_indir = '/path/to/MCD12C1_v061/'
LandCover_outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/LandCover_input/'

delta_x = 0.05
delta_y = 0.05

def get_LandCover_Percentage_Variables(data):
    Land_Cover_Type_1_Percent = data.variables['Land_Cover_Type_1_Percent'][:]
    return Land_Cover_Type_1_Percent

IGBP_tagnames = ['Water-Bodies', 'Evergreen-Needleleaf-Forests', 'Evergreen-Broadleaf-Forests', 'Deciduous-Needleleaf-Forests',
'Deciduous-Broadleaf-Forests', 'Mixed-Forests', 'Closed-Shrublands', 'Open-Shrublands', 'Woody-Shrublands', 'Savannas', 
'Grasslands', 'Permanent-Wetlands', 'Croplands', 'Urban-Builtup-Lands', 'Cropland-Natural-Vegetation-Mosaics', 'Permanent-Snow-Ice','Barren']