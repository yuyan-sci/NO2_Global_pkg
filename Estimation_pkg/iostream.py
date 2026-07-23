import torch
import numpy as np
import os
import csv
import netCDF4 as nc
import time
from datetime import datetime
from Estimation_pkg.utils import *
from Training_pkg.utils import *
import pickle


# WGS84 well-known text, used for the 'crs' scalar variable in CF grid_mapping
_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",'
    'SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],'
    'AUTHORITY["EPSG","6326"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4326"]]'
)


def _write_cf_estimation_nc(outfile, final_data, extent, SPECIES, YYYY, MM):
    """Write a CF/ACDD-compliant NetCDF4 file with NO2 estimation map.

    Adds global attributes, CRS grid_mapping, CF-standard variable/coordinate
    attributes, compression and chunking.
    """
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = 0.01
    lon_delta = 0.01

    MapData = nc.Dataset(outfile, 'w', format='NETCDF4')

    # ----- Global attributes (CF-1.8 / ACDD-1.3) -----
    MapData.Conventions       = 'CF-1.8, ACDD-1.3'
    MapData.title             = 'Process-based constarined LightGBM enhanced Monthly Global NO2 Estimation'
    MapData.contact           = 'Yu Yan <yany1@wustl.edu>'
    MapData.institution       = 'ACAG, Washington University in St. Louis'
    MapData.history           = '{} created by Yu Yan'.format(datetime.utcnow().strftime('%Y-%m-%d'))
    MapData.geospatial_extent = '[{:.3f},{:.3f},{:.3f},{:.3f}]'.format(extent[0], extent[1], extent[2], extent[3])
    MapData.time_coverage     = '{}-{}'.format(YYYY, MM)
    MapData.LAT_DELTA         = lat_delta
    MapData.LON_DELTA         = lon_delta

    # ----- Dimensions -----
    MapData.createDimension('lat', lat_size)
    MapData.createDimension('lon', lon_size)

    # ----- CRS grid mapping variable (EPSG:4326 / WGS84) -----
    crs = MapData.createVariable('crs', 'i4')
    crs.grid_mapping_name        = 'latitude_longitude'
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis          = 6378137.0
    crs.inverse_flattening       = 298.257223563
    crs.crs_wkt                  = _WGS84_WKT
    crs.epsg_code                = 'EPSG:4326'

    # ----- Coordinate variables -----
    latitudes = MapData.createVariable('lat', 'f4', ('lat',))
    longitudes = MapData.createVariable('lon', 'f4', ('lon',))
    latitudes[:]  = np.arange(extent[0], np.round(extent[1] + lat_delta, decimals=5), lat_delta)
    longitudes[:] = np.arange(extent[2], np.round(extent[3] + lon_delta, decimals=5), lon_delta)
    latitudes.units          = 'degrees_north'
    latitudes.standard_name  = 'latitude'
    latitudes.long_name      = 'latitude'
    latitudes.axis           = 'Y'
    longitudes.units         = 'degrees_east'
    longitudes.standard_name = 'longitude'
    longitudes.long_name     = 'longitude'
    longitudes.axis          = 'X'

    # ----- NO2 data variable -----
    # chunksizes chosen for efficient spatial subsetting; complevel=4 gives good size/speed trade-off
    chunk_lat = min(512, lat_size)
    chunk_lon = min(512, lon_size)
    # NaN is the fill value for any no-data pixel (ocean, missing channel,
    # etc.). The prediction code flags these as -999; we convert to NaN so
    # downstream consumers see real missing data with _FillValue=NaN.
    NO2 = MapData.createVariable(
        SPECIES, 'f4', ('lat', 'lon'),
        fill_value=np.float32(np.nan),
        zlib=True, complevel=4, shuffle=True,
        chunksizes=(chunk_lat, chunk_lon),
    )
    NO2.standard_name = 'mixing_ratio_of_nitrogen_dioxide_at_groud_level'
    NO2.units         = 'ppb'
    NO2.coordinates   = 'lat lon'
    NO2.grid_mapping  = 'crs'
    if LightGBM_setting:
        NO2.long_name = 'LightGBM derived Monthly {} [ppb]'.format(SPECIES)
    else:
        NO2.long_name = 'CNN derived Monthly {} [ppb]'.format(SPECIES)

    if np.ma.isMaskedArray(final_data):
        final_data = final_data.filled(np.nan)
    final_data = np.where(final_data == -999.0, np.nan, final_data).astype(np.float32)
    # Negative NO2 concentration is physically invalid (LightGBM can
    # produce small negatives near zero in remote areas). Treat them as
    # missing so downstream consumers see real NaN, not a spurious < 0.
    final_data = np.where(final_data < 0, np.nan, final_data)
    NO2[:] = final_data

    MapData.close()

def load_trained_month_based_model_forEstimation(model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear,month_index, width, height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    PATH = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear,Selected_MONTHS_str, nchannel,special_name)
    if LightGBM_setting:
        with open(PATH, 'rb') as f:
            model = pickle.load(f)
            print(f"Loaded LightGBM model from: {PATH}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = torch.load(PATH, map_location=torch.device(device)).eval()
        model.to(device)
        print(f"Loaded PyTorch model from: {PATH}")
    return model

def load_trained_model_forEstimation(model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, width, height):
    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    PATH = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear, nchannel,special_name)
    if LightGBM_setting:
        with open(PATH, 'rb') as f:
            model = pickle.load(f)
            print(f"Loaded LightGBM model from: {PATH}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = torch.load(PATH, map_location=torch.device(device)).eval()
        model.to(device)
        print(f"Loaded PyTorch model from: {PATH}")
    return model    

def load_map_data(channel_names, YYYY, MM):
    inputfiles = inputfiles_table(YYYY=YYYY,MM=MM)
    indir = '/path/to/NO2_DL_global/input_variables/'
    lat_infile = indir + 'tSATLAT_global.npy'
    lon_infile = indir + 'tSATLON_global.npy'
    SATLAT = np.load(lat_infile)
    SATLON = np.load(lon_infile)
    output = np.zeros((len(channel_names), len(SATLAT), len(SATLON)))
    loading_time_start = time.time()
    for i in range(len(channel_names)):
        infile = inputfiles[channel_names[i]]
        tempdata = np.load(infile)
        print('{} has been loaded!'.format(channel_names[i]))
        print('{} has shape {}'.format(channel_names[i], tempdata.shape))  # Added debug print
        output[i,:,:] = tempdata
    loading_time_end = time.time()
    print('Loading time cost: ', loading_time_end - loading_time_start, 's')
    return output

def load_estimation_map_data(YYYY:str, MM:str,SPECIES:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    infile = indir + '{}_{}_{}{}{}.nc'.format(SPECIES,version,YYYY,MM,special_name)
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables[SPECIES][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon

def load_ForcedSlopeUnity_estimation_map_data(YYYY:str, MM:str,SPECIES:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/ForcedSlopeUnity_Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    infile = indir + '{}_{}_{}{}{}_ForcedSlopeUnity.nc'.format(SPECIES,version,YYYY,MM,special_name)
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables[SPECIES][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon


def load_Annual_estimation_map_data(YYYY:str,SPECIES:str, version:str, special_name):
    indir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    infile = indir + '{}_{}_{}{}_AnnualMean.nc'.format(SPECIES,version,YYYY,special_name)
    MapData = nc.Dataset(infile)
    lat = MapData.variables['lat'][:]
    lon = MapData.variables['lon'][:]
    SPECIES_Map = MapData.variables[SPECIES][:]
    SPECIES_Map = np.array(SPECIES_Map)
    return SPECIES_Map, lat, lon

def load_ForcedSlope_forEstimation(model_indir, typeName, version, species, nchannel, special_name,beginyear, endyear, month_index,width, height):
    indir = model_indir + '{}/{}/Results/Estimation-ForcedSlopeUnity_Dicts/'.format(species, version)
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)
    dic_infile = indir + 'Estimation-ForcedSlopeUnity_Dicts_{}_{}_{}x{}_{}-{}_{}_{}Channel{}.npy'.format(typeName, species, width,height, beginyear, endyear,Selected_MONTHS_str, nchannel,special_name)
    ForcedSlopeUnity_Dictionary_forEstimation = np.load(dic_infile,allow_pickle=True).item()
    return ForcedSlopeUnity_Dictionary_forEstimation

def save_ForcedSlope_forEstimation(ForcedSlopeUnity_Dictionary_forEstimation, model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, month_index,width, height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)

    outdir = model_outdir + '{}/{}/Results/Estimation-ForcedSlopeUnity_Dicts/'.format(species, version)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    dic_outfile = outdir + 'Estimation-ForcedSlopeUnity_Dicts_{}_{}_{}x{}_{}-{}_{}_{}Channel{}.npy'.format(typeName, species, width,height, beginyear, endyear,Selected_MONTHS_str, nchannel,special_name)
    np.save(dic_outfile,ForcedSlopeUnity_Dictionary_forEstimation)
    return

def save_trained_model_forEstimation(model, model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, width, height):
    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    model_outfile = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear, nchannel,special_name)
    if LightGBM_setting:
        with open(model_outfile, 'wb') as f:
            pickle.dump(model, f)
            print(f"Saved LightGBM model to: {model_outfile}")
    else:
        torch.save(model, model_outfile)
        print(f"Saved PyTorch model to: {model_outfile}")
    return 

def save_trained_month_based_model_forEstimation(model, model_outdir, typeName, version, species, nchannel, special_name,beginyear, endyear, month_index,width, height):
    MONTH = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    Selected_MONTHS_list = [MONTH[i] for i in month_index]
    Selected_MONTHS_str = '-'.join(Selected_MONTHS_list)

    outdir = model_outdir + '{}/{}/Results/Estimation-Trained_Models/'.format(species, version)
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    model_outfile = outdir +  'Estimation_{}_{}_{}x{}_{}-{}_{}_{}Channel{}.pt'.format(typeName, species, width,height, beginyear, endyear,Selected_MONTHS_str, nchannel,special_name)
    if LightGBM_setting:
        with open(model_outfile, 'wb') as f:
            pickle.dump(model, f)
            print(f"Saved LightGBM model to: {model_outfile}")
    else:
        torch.save(model, model_outfile)
        print(f"Saved PyTorch model to: {model_outfile}")
    return 


def save_annual_final_map_data(final_data:np.array, YYYY:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    outfile = outdir + 'Annual_{}_{}_{}{}.nc'.format(SPECIES,version,YYYY,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = 0.01 #(extent[1]-extent[0])/(lat_size-1)
    lon_delta = 0.01 #(extent[3]-extent[2])/(lon_size-1)

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    if LightGBM_setting:
        MapData.TITLE = 'Process-based constarined LightGBM enhanced Global Annual {} Estimation.'.format(SPECIES)
    else:
        MapData.TITLE = 'LightGBM Global Annual {} Estimation.'.format(SPECIES)
    MapData.CONTACT = 'Yu Yan <yany1@wustl.edu>, ACAG-Washington University in St. Louis'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}'.format(YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    NO2 = MapData.createVariable(SPECIES,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = np.arange(extent[0],np.round(extent[1]+lat_delta,decimals=5),lat_delta)
    longitudes[:] = np.arange(extent[2],np.round(extent[3]+lon_delta,decimals=5),lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    NO2.units = 'ppb'
    if LightGBM_setting:
        NO2.long_name = 'LightGBM derived Annual {} [ppb]'.format(SPECIES)
    else:
        NO2.long_name = 'CNN derived Annual {} [ppb]'.format(SPECIES)
    NO2[:] = final_data
    return
    
def save_final_map_data(final_data:np.array, YYYY:str, MM:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}_{}_{}{}{}.nc'.format(SPECIES,version,YYYY,MM,special_name)
    _write_cf_estimation_nc(outfile, final_data, extent, SPECIES, YYYY, MM)
    return

def save_ForcedSlopeUnity_final_map_data(final_data:np.array, YYYY:str, MM:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/ForcedSlopeUnity_Map_Estimation/{}/'.format(SPECIES,version,YYYY)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outfile = outdir + '{}_{}_{}{}{}_ForcedSlopeUnity.nc'.format(SPECIES,version,YYYY,MM,special_name)
    _write_cf_estimation_nc(outfile, final_data, extent, SPECIES, YYYY, MM)
    return

def save_combinedGeo_map_data(final_data:np.array, YYYY:str, MM:str, extent:list, SPECIES:str, version:str, special_name):
    outdir = Estimation_outdir + '{}/{}/Map_Estimation/Combined_withGeo/{}/'.format(SPECIES,version,YYYY)
    
    if not os.path.isdir(outdir):
                os.makedirs(outdir)
    outfile = outdir + 'Combined-{}km-Geo{}_{}_{}_{}{}{}.nc'.format(Coefficient_start_distance,SPECIES,SPECIES,version,YYYY,MM,special_name)
    lat_size = final_data.shape[0]
    lon_size = final_data.shape[1]
    lat_delta = 0.01
    lon_delta = 0.01

    MapData = nc.Dataset(outfile,'w',format='NETCDF4')
    MapData.TITLE = 'LightGBM Monthly Global {} Estimation combined with Geophysical data.'.format(SPECIES)
    MapData.CONTACT = 'Yu Yan <y.yan@wustl.edu>'
    MapData.LAT_DELTA = lat_delta
    MapData.LON_DELTA = lon_delta
    MapData.TIMECOVERAGE    = '{}/{}'.format(MM,YYYY)

    lat = MapData.createDimension("lat",lat_size)
    lon = MapData.createDimension("lon",lon_size)
    NO2 = MapData.createVariable(SPECIES,'f4',('lat','lon',))
    latitudes = MapData.createVariable("lat","f4",("lat",))
    longitudes = MapData.createVariable("lon","f4",("lon",))
    latitudes[:] = np.arange(extent[0],np.round(extent[1]+lat_delta,decimals=5),lat_delta)
    longitudes[:] = np.arange(extent[2],np.round(extent[3]+lon_delta,decimals=5),lon_delta) 
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    latitudes.standard_name = 'latitude'
    latitudes.long_name = 'latitude'
    longitudes.standard_name = 'longitude'
    longitudes.long_name = 'longitude'
    NO2.units = 'ppb'
    NO2.long_name = 'LightGBM combined with Geophysical data Monthly {} [ppb]'.format(SPECIES)
    NO2[:] = final_data
    return

def Monthly_PWM_no2_output_text(PWM_PM_dic,species,YYYY,MM,outfile,areas_list):
    with open(outfile,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Area','Time Period','PWM {} [ppb]'.format(species)])
        for iarea in areas_list:
            for iyear in range(len(YYYY)):
                for imonth in range(len(MM)):
                    writer.writerow([iarea,'{}-{}'.format(YYYY[iyear],MM[imonth]),str(np.round(PWM_PM_dic[iarea][iyear*12+imonth],4))])
        
    return

def Annual_PWM_no2_output_text(PWM_PM_dic,species,YYYY,outfile,areas_list):
    with open(outfile,'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Area','Time Period','PWM {} [ppb]'.format(species)])
        for iarea in areas_list:
            for iyear in range(len(YYYY)):
                writer.writerow([iarea,'{}'.format(YYYY[iyear]),str(np.round(PWM_PM_dic[iarea][iyear],4))])  
    return
