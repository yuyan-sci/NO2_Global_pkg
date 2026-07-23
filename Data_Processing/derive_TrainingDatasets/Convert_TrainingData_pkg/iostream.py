import numpy as np
import netCDF4 as nc
import os

from Convert_TrainingData_pkg.utils import *


def load_mapdata(infile):
    mapdata = np.load(infile)
    if len(np.where(np.isnan(mapdata)==True)):
        mapdata[np.where(np.isnan(mapdata)==True)] = np.mean(mapdata[np.where(np.isnan(mapdata)==False)])
    return mapdata

def load_Global_GeoLatLon():
    indir = '/path/to/NO2_DL_global/input_variables/'
    lat_infile = indir + 'tSATLAT_global.npy'
    lon_infile = indir + 'tSATLON_global.npy'
    Global_GeoLAT = np.load(lat_infile)
    Global_GeoLON = np.load(lon_infile)
    return Global_GeoLAT, Global_GeoLON


def load_monthly_obs_LatLon():
    infile = Observation_indir + 'NO2_observation_corrected_v7_filtered_v5.13.nc'
    data = nc.Dataset(infile)
    SPECIES_OBS = data['NO2'][:]
    sites_number = SPECIES_OBS.shape[0]
    lat = data["latitude"][:]
    lon = data["longitude"][:]
    return sites_number, lat, lon


def load_lat_lon_indices():
    indir = TrainingData_outdir
    lat_index_infile = indir + 'lat_index.npy'
    lon_index_infile = indir + 'lon_index.npy'
    lat_index = np.load(lat_index_infile)
    lon_index = np.load(lon_index_infile)
    return lat_index, lon_index

def save_lat_lon_indices(lat_index, lon_index):
    outdir = TrainingData_outdir
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    lat_index = np.array(lat_index)
    lon_index = np.array(lon_index)
    lat_index_outfile = outdir + 'lat_index.npy'
    lon_index_outfile = outdir + 'lon_index.npy'
    np.save(lat_index_outfile, lat_index)
    np.save(lon_index_outfile, lon_index)
    return

def nc_createDimensions(outfile,nametags,start_YYYYMM,end_YYYYMM,start_YYYY,end_YYYY,sitesnumber,datenumber,total_number,width,height):
    data = nc.Dataset(outfile, 'w', format='NETCDF4')
    data.TITLE = 'Monthly Training Datasets for training for NO2 over Global'
    data.VariableList = nametags
    data.TIMERESOLUTION = 'Monthly'
    data.TIMECOVERAGE = '{}-{}'.format(start_YYYYMM,end_YYYYMM)
    data.createDimension('sites_number', sitesnumber)
    data.createDimension('date_number', datenumber)
    data.createDimension('Total_Number',total_number)
    data.createDimension('width', width)
    data.createDimension('height', height)
    data.createDimension('unity', 1)
    data.createVariable('width','i4',('unity'))[:] = width
    data.createVariable('height','i4',('unity'))[:] = height
    data.createVariable('sites_number','i4',('unity'))[:] = sitesnumber
    data.createVariable('Total_number','i4',('unity'))[:] = total_number
    data.createVariable('start_YYYY', 'i4', ('unity'))[:] = start_YYYY
    data.createVariable('end_YYYY', 'i4', ('unity'))[:] = end_YYYY
    data.close()
    return

def load_init_ncfile(init_training_infile,nametags):
    data = nc.Dataset(init_training_infile,'r')
    width = np.array(data.variables['width'][:])[0]
    height = np.array(data.variables['height'][:])[0]
    total_number = np.array(data.variables['Total_number'])[0]
    sitesnumber = np.array(data.variables['sites_number'])[0]
    start_YYYY  = np.array(data.variables['start_YYYY'])[0]
    TrainingDatasets = np.zeros((total_number,len(nametags),width,height),dtype=np.float64)
    for i in range(len(nametags)):
        TrainingDatasets[:,i,:,:] = np.array(data.variables[nametags[i]][:,:,:])
    return TrainingDatasets

def nc_saveTrainingVariables(outfile,training_data_array, nametag):
    
    data = nc.Dataset(outfile, 'a', format='NETCDF4')
    data.createVariable(nametag,'f8', ('Total_Number','width','height'))[:] = training_data_array
    data.close()
    # mode: access mode. r means read-only; no data can be modified. w means write;
    # a new file is created, an existing file with the same name is deleted. x means write, but fail if an existing file with the same name already exists.
    # a and r+ mean append; an existing file is opened for reading and writing, if file does not exist already, one is created. 
    # Appending s to modes r, w, r+ or a will enable unbuffered shared access to NETCDF3_CLASSIC, NETCDF3_64BIT_OFFSET or NETCDF3_64BIT_DATA formatted files. 
    # Unbuffered access may be useful even if you don't need shared access, since it may be faster for programs that don't access data sequentially. 
    # This option is ignored for NETCDF4 and NETCDF4_CLASSIC formatted files.
    return

