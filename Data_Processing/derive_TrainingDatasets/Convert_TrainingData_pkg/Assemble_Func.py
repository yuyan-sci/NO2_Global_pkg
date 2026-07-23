import numpy as np
from Convert_TrainingData_pkg.data_func import get_nearest_point_index,get_CNN_training_site_data
from Convert_TrainingData_pkg.iostream import *
from Convert_TrainingData_pkg.utils import *
import os
from multiprocessing import Pool, cpu_count, get_context
from functools import partial


def get_save_nearest_indices():
    GeoLAT, GeoLON = load_Global_GeoLatLon()
    sites_number, sitelat, sitelon = load_monthly_obs_LatLon()
    print('sites_number:', sites_number)
    lon_index, lat_index = get_nearest_point_index(sitelat=sitelat,sitelon=sitelon,lat_grid=GeoLAT,lon_grid=GeoLON)
    print('lat lon indices saved!')
    save_lat_lon_indices(lat_index=lat_index,lon_index=lon_index)
    return

def _process_single_channel(args):
    """Process a single channel - designed for parallel execution
    
    Args is a tuple of (channel_name, YEAR, MONTH, width, height, sites_number, 
                        lat_index, lon_index, total_number)
    """
    channel_name, YEAR, MONTH, width, height, sites_number, lat_index, lon_index, total_number = args
    
    temp_trainingdata = np.full((total_number, width, height), -999.0, dtype=np.float64)
    
    for iyear in range(len(YEAR)):
        for imonth in range(len(MONTH)):
            print('Channel: {}, YEAR: {}, MONTH: {}'.format(channel_name, YEAR[iyear], MONTH[imonth]))
            inputfiles_dic = inputfiles_table(YYYY=YEAR[iyear], MM=MONTH[imonth])
            infile = inputfiles_dic[channel_name]
            mapdata = load_mapdata(infile=infile)
            
            start_idx = sites_number * (iyear * 12 + imonth)
            end_idx = sites_number * (iyear * 12 + imonth + 1)
            temp_trainingdata[start_idx:end_idx, :, :] = get_CNN_training_site_data(
                initial_array=mapdata, Height=height, Width=width,
                lat_index=lat_index, lon_index=lon_index, nsite=sites_number
            )
    
    return channel_name, temp_trainingdata


def derive_TrainingDatasets(channel_names, width, height, YEAR, use_parallel=True, n_workers=None):
    """
    Optimized version with parallel processing support
    
    Parameters:
    -----------
    channel_names : list
        List of channel names to process
    width, height : int
        Dimensions of the training windows
    YEAR : list
        List of years to process
    use_parallel : bool, default=True
        Whether to use parallel processing for channels
    n_workers : int, optional
        Number of parallel workers (defaults to cpu_count - 1)
    """
    #### Set variables and load arrays for running function
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    start_YYYYMM = '{}{}'.format(YEAR[0],MONTH[0])
    end_YYYYMM   = '{}{}'.format(YEAR[-1],MONTH[-1])
    sites_number, sitelat, sitelon = load_monthly_obs_LatLon()
    datenumber = len(YEAR)*12
    total_number = sites_number * datenumber
    lat_index, lon_index  = load_lat_lon_indices()

    ### Set outfile
    outdir =  TrainingData_outdir
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    outfile = outdir + '{}{}_TrainingData_{}channels_{}x{}_{}01-{}12.nc'.format(Obs_version, special_name, len(channel_names),width,height,YEAR[0],YEAR[-1])

    ### Create the nc file and add primary information
    nc_createDimensions(outfile=outfile,nametags=channel_names,start_YYYYMM=start_YYYYMM,end_YYYYMM=end_YYYYMM,
                        start_YYYY=YEAR[0],end_YYYY=YEAR[-1],sitesnumber=sites_number,
                        datenumber=datenumber,total_number=total_number,width=width,height=height)
    
    if use_parallel and len(channel_names) > 1:
        # Parallel processing of channels
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
        
        print(f'Processing {len(channel_names)} channels in parallel using {n_workers} workers...')
        
        # Prepare arguments for each channel - pass everything as tuple to avoid pickling issues
        channel_args = [
            (channel_name, YEAR, MONTH, width, height, sites_number, 
             lat_index, lon_index, total_number)
            for channel_name in channel_names
        ]
        
        # Process channels in parallel using ThreadPoolExecutor (fork-safe and fast)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_args = {executor.submit(_process_single_channel, args): args for args in channel_args}
            for future in as_completed(future_to_args):
                try:
                    result = future.result()
                    results.append(result)
                    print(f'Completed channel: {result[0]}')
                except Exception as e:
                    args = future_to_args[future]
                    print(f'Error processing channel {args[0]}: {e}')
        
        # Save results to NetCDF
        for channel_name, temp_trainingdata in results:
            print(f'Saving channel: {channel_name}')
            nc_saveTrainingVariables(outfile=outfile, training_data_array=temp_trainingdata, nametag=channel_name)
    else:
        # Sequential processing (original behavior)
        print('Processing channels sequentially...')
        for itag in range(len(channel_names)):
            temp_trainingdata = np.full((total_number,width,height),-999.0,dtype=np.float64)
            for iyear in range(len(YEAR)):
                for imonth in range(len(MONTH)):
                    print('Channel: {}, YEAR: {}, MONTH: {}'.format(channel_names[itag],YEAR[iyear], MONTH[imonth]))
                    inputfiles_dic = inputfiles_table(YYYY=YEAR[iyear],MM=MONTH[imonth])
                    infile = inputfiles_dic[channel_names[itag]]
                    mapdata = load_mapdata(infile=infile)
                    
                    temp_trainingdata[sites_number*(iyear*12+imonth):sites_number*(iyear*12+imonth+1),:,:] = get_CNN_training_site_data(initial_array=mapdata,Height=height,Width=width,lat_index=lat_index,lon_index=lon_index,nsite=sites_number)
                    
            nc_saveTrainingVariables(outfile=outfile,training_data_array=temp_trainingdata,nametag=channel_names[itag])
    
    print('Training dataset generation complete!')
    return

def _process_yearmonth(args):
    """Process a single (year, month) pair for one channel — designed for parallel execution.
    
    Returns (start_idx, end_idx, site_data) so results can be assembled without ordering issues.
    """
    channel_name, year, month, width, height, sites_number, lat_index, lon_index, iyear, imonth = args
    inputfiles_dic = inputfiles_table(YYYY=year, MM=month)
    infile = inputfiles_dic[channel_name]
    mapdata = load_mapdata(infile=infile)
    site_data = get_CNN_training_site_data(
        initial_array=mapdata, Height=height, Width=width,
        lat_index=lat_index, lon_index=lon_index, nsite=sites_number
    )
    start_idx = sites_number * (iyear * 12 + imonth)
    end_idx   = sites_number * (iyear * 12 + imonth + 1)
    print(f'  Done: Channel={channel_name}, YEAR={year}, MONTH={month}')
    return start_idx, end_idx, site_data


def add_channels_on_Existed_TrainingFiles(init_training_file,init_channel_names,add_channel_names,
                                          width, height, YEAR,ExcludeIndustialSites, nonan_GCHPfilted,
                                          replace_channel_names=None,
                                          use_parallel=True, n_workers=None):
    """Add (and optionally replace) channels on an existing training file.

    Parameters
    ----------
    replace_channel_names : list or None
        Channels that already exist in ``init_channel_names`` but should be
        re-loaded from disk (replacing the old values).  They are NOT counted
        as extra channels — the output file has
        ``len(init_channel_names) + len(add_channel_names)`` unique channels.
    use_parallel : bool
        Whether to parallelise the year-month loop (default True).
    n_workers : int or None
        Number of parallel workers.  None = cpu_count - 1.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    if replace_channel_names is None:
        replace_channel_names = []
    replace_set = set(replace_channel_names)

    #### Set variables and load arrays for running function
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    start_YYYYMM = '{}{}'.format(YEAR[0],MONTH[0])
    end_YYYYMM   = '{}{}'.format(YEAR[-1],MONTH[-1])

    if ExcludeIndustialSites:
        sites_number, sitelat, sitelon = load_monthly_obs_ExcludeIndustrialSites_LatLon()
    elif nonan_GCHPfilted:
        sites_number, sitelat, sitelon = load_monthly_obs_nonan_GCHPfilted_LatLon()
    else:
        sites_number, sitelat, sitelon = load_monthly_obs_LatLon()
    datenumber = len(YEAR)*12
    total_number = sites_number * datenumber
    lat_index, lon_index  = load_lat_lon_indices()

    ### Set outfile
    outdir = TrainingData_outdir

    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    # Output channel count: init channels (replacements stay in-place) + truly new channels
    n_out_channels = len(init_channel_names) + len(add_channel_names)
    outfile = outdir + '{}{}_TrainingData{}_{}channels_{}x{}_{}01-{}12.nc'.format(Obs_version, special_name, training_version, n_out_channels,width,height,YEAR[0],YEAR[-1])

    t0 = time.time()
    print('Loading initial training file ...')
    # Load only the channels that won't be replaced (skip replaced ones to save memory)
    init_channels_to_copy = [c for c in init_channel_names if c not in replace_set]
    init_TrainingDatasets = load_init_ncfile(init_training_infile=init_training_file,nametags=init_channels_to_copy)
    print(f'  Loaded in {time.time()-t0:.1f}s')

    ### Create the nc file and add primary information
    # VariableList in metadata: init channels (in original order) + new channels
    channel_names = init_channel_names + add_channel_names
    nc_createDimensions(outfile=outfile,nametags=channel_names,start_YYYYMM=start_YYYYMM,end_YYYYMM=end_YYYYMM,
                        start_YYYY=YEAR[0],end_YYYY=YEAR[-1],sitesnumber=sites_number,
                        datenumber=datenumber,total_number=total_number,width=width,height=height)

    # --- Copy existing channels (skip replaced ones) ---
    t1 = time.time()
    print(f'Copying {len(init_channels_to_copy)} existing channels (skipping {len(replace_set)} replaced) ...')
    for itag in range(len(init_channels_to_copy)):
        nc_saveTrainingVariables(outfile=outfile,training_data_array=init_TrainingDatasets[:,itag,:,:],nametag=init_channels_to_copy[itag])
    print(f'  Copied in {time.time()-t1:.1f}s')

    # --- Process replaced + new channels (parallelise the year-month loop) ---
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    channels_to_process = replace_channel_names + add_channel_names
    if replace_channel_names:
        print(f'Will replace {len(replace_channel_names)} channel(s): {replace_channel_names}')

    for itag in range(len(channels_to_process)):
        t2 = time.time()
        ch_name = channels_to_process[itag]
        n_tasks = len(YEAR) * len(MONTH)
        print(f'Processing new channel: {ch_name}  ({n_tasks} year-month tasks)')

        temp_trainingdata = np.zeros((total_number, width, height), dtype=np.float64)

        if use_parallel and n_workers > 1:
            print(f'  Using {n_workers} parallel workers ...')
            # Build task list
            task_args = [
                (ch_name, YEAR[iy], MONTH[im], width, height,
                 sites_number, lat_index, lon_index, iy, im)
                for iy in range(len(YEAR))
                for im in range(len(MONTH))
            ]

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_yearmonth, a): a for a in task_args}
                for future in as_completed(futures):
                    try:
                        start_idx, end_idx, site_data = future.result()
                        temp_trainingdata[start_idx:end_idx, :, :] = site_data
                    except Exception as e:
                        a = futures[future]
                        print(f'  ERROR: Channel={a[0]}, YEAR={a[1]}, MONTH={a[2]}: {e}')
        else:
            print('  Processing sequentially ...')
            for iyear in range(len(YEAR)):
                for imonth in range(len(MONTH)):
                    print('  Channel: {}, YEAR: {}, MONTH: {}'.format(ch_name, YEAR[iyear], MONTH[imonth]))
                    inputfiles_dic = inputfiles_table(YYYY=YEAR[iyear], MM=MONTH[imonth])
                    infile = inputfiles_dic[ch_name]
                    mapdata = load_mapdata(infile=infile)
                    start_idx = sites_number * (iyear * 12 + imonth)
                    end_idx   = sites_number * (iyear * 12 + imonth + 1)
                    temp_trainingdata[start_idx:end_idx, :, :] = get_CNN_training_site_data(
                        initial_array=mapdata, Height=height, Width=width,
                        lat_index=lat_index, lon_index=lon_index, nsite=sites_number)

        temp_trainingdata = np.nan_to_num(temp_trainingdata, nan=0.0)
        nc_saveTrainingVariables(outfile=outfile, training_data_array=temp_trainingdata, nametag=ch_name)
        print(f'  Channel {ch_name} done in {time.time()-t2:.1f}s')

    print(f'Total add-channels time: {time.time()-t0:.1f}s')
    return