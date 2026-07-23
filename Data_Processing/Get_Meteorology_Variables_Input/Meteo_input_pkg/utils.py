

meteorology_indir = '/ExtData/GEOS_0.5x0.625/GEOS_IT/'
meteorology_mapdata_outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/Meteorology_input/'

delta_x = 0.5
delta_y = 0.625


def get_variables_A1_Is_files(A1_file, nametag, lev):
    variable = A1_file.variables[nametag][:]
    return variable

def get_variables_A3cld_A3dyn_A3mstE_A3mstC(file, nametag, lev):
    variable = file.variables[nametag][:,lev,:,:]
    return variable

def File_lookup_table():
    lookup_table = {
        'A1' : get_variables_A1_Is_files,
        'I3'  : get_variables_A1_Is_files,
        'A3cld'  : get_variables_A3cld_A3dyn_A3mstE_A3mstC,
        'A3dyn'  : get_variables_A3cld_A3dyn_A3mstE_A3mstC,
        'A3mstE'   : get_variables_A3cld_A3dyn_A3mstE_A3mstC,
        'A3mstC'   : get_variables_A3cld_A3dyn_A3mstE_A3mstC,
        
    }
    return lookup_table