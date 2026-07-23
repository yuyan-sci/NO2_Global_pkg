



CEDS_indir = '/path/to/CEDS/v2025-04_0.1/'

Anthropogenic_Emission_outdir = '/path/to/NO2_DL_global_2019/NO2_global_pkg/input_variables/CEDS_Anthro_Emissions_01_input/'

def get_total_emissions(data,nametag):
    agr_sector = data.variables['{}_agr'.format(nametag)][:]
    ene_sector = data.variables['{}_ene'.format(nametag)][:]
    ind_sector = data.variables['{}_ind'.format(nametag)][:]
    tra_sector = data.variables['{}_tra'.format(nametag)][:]
    rco_sector = data.variables['{}_rco'.format(nametag)][:]
    slv_sector = data.variables['{}_slv'.format(nametag)][:]
    wst_sector = data.variables['{}_wst'.format(nametag)][:]
    shp_sector = data.variables['{}_shp'.format(nametag)][:]
    total_emi = agr_sector + ene_sector + ind_sector + tra_sector + rco_sector + slv_sector + wst_sector + shp_sector 
    return total_emi