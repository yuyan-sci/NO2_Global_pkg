from data_func.iostream import load_RawObs_training_data,load_RawObservation, load_GeoLatLon, load_GeoLatLon_Map, load_BLISCO_data,load_mahalanobis_distance_data
from data_func.calculation import calculate_covariance_matrix, invert_matrix,calculate_mahalanobis_distance
from data_func.utils import neighbors_haversine_indices,Obs_version
from map_uncertainty_func.iostream import save_absoulute_uncertainty_map,load_absolute_uncertainty_map,load_estimation_map_data,load_rRMSE_map,save_rRMSE_map,load_bins_LOWESS_values,save_mahalanobis_distance_map,load_mahalanobis_distance_map,get_landtype, save_pixel_nearby_sites_index_map,load_pixels_nearest_sites_indices_map, save_local_reference_map, load_local_reference_map, load_mapdata, get_nearby_sites_index_map_path
from map_uncertainty_func.utils import inputfiles_table
from map_uncertainty_func.data_func import Get_Mahalanobis_Distances
import numpy as np
import numpy.lib.format as _np_format
import gc
from data_func.iostream import load_TrainingVariables

def Get_absolute_uncertainty_map(species,version,special_name,YYYY,MM,obs_version,nearby_sites_number,
                                 map_estimation_special_name,map_estimation_version):
    '''
    Get absolute uncertainty map for given species and date.
    '''
    print(f"Calculating absolute uncertainty map for species: {species}, version: {version}, date: {YYYY}-{MM}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']
    rRMSE_uncertainty_map = load_rRMSE_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                           obs_version=obs_version,nearby_sites_number=nearby_sites_number,
                                           special_name=special_name)

    if MM != 12:
        map_data, lat, lon = load_estimation_map_data(YYYY=YYYY, MM=MONTH[MM], SPECIES=species, version=map_estimation_version, special_name=map_estimation_special_name)
        padded_map = np.zeros(rRMSE_uncertainty_map.shape,dtype=np.float64)
        
        # Determine the padding offset dynamically based on the shape difference
        rh, rw = rRMSE_uncertainty_map.shape
        h, w = map_data.shape
        pad_h = (rh - h) // 2
        pad_w = (rw - w) // 2
        padded_map[pad_h:pad_h+h, pad_w:pad_w+w] = map_data
    else:
        padded_map = np.zeros(rRMSE_uncertainty_map.shape,dtype=np.float64)
        rh, rw = rRMSE_uncertainty_map.shape
        for m in range(12):
            monthly_map, lat, lon = load_estimation_map_data(YYYY=YYYY, MM=MONTH[m], SPECIES=species, version=map_estimation_version, special_name=map_estimation_special_name)
            h, w = monthly_map.shape
            pad_h = (rh - h) // 2
            pad_w = (rw - w) // 2
            padded_map[pad_h:pad_h+h, pad_w:pad_w+w] += monthly_map
        padded_map = padded_map / 12.0
    absolute_uncertainty_map = padded_map * rRMSE_uncertainty_map 
    save_absoulute_uncertainty_map(absolute_uncertainty_map=absolute_uncertainty_map,species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY=YYYY,MM=MONTH[MM],
                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    return

def Get_longterm_average_absolute_uncertainty_map(species,version,special_name,YYYY_list:np.int32,MM:np.int32,obs_version,nearby_sites_number,
                                 map_estimation_special_name,map_estimation_version):
    '''
    Get longterm average absolute uncertainty map for given species and years/months.
    '''
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','AllMonths']
    if MM != 12:
        print(f"Calculating longterm average absolute uncertainty map for species: {species}, version: {version}, month: {MONTH[MM]}, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    else:
        print(f"Calculating longterm average absolute uncertainty map for species: {species}, version: {version}, all months, obs_version: {obs_version}, nearby_sites_number: {nearby_sites_number}")
    longterm_average_absolute_uncertainty_map = None
    count = 0
    for YYYY in YYYY_list:
        if MM != 12:
            temp_absolute_uncertainty_map = load_absolute_uncertainty_map(species=species,version=version,special_name=special_name,YYYY=YYYY,MM=MONTH[MM],
                                                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number)
        else:
            temp_absolute_uncertainty_map = np.zeros((13000,36000),dtype=np.float64)
            for m in range(12):
                monthly_absolute_uncertainty_map = load_absolute_uncertainty_map(species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY=YYYY,MM=MONTH[m],
                                                                       obs_version=obs_version,nearby_sites_number=nearby_sites_number)
                temp_absolute_uncertainty_map += monthly_absolute_uncertainty_map
            temp_absolute_uncertainty_map = temp_absolute_uncertainty_map / 12.0
        if longterm_average_absolute_uncertainty_map is None:
            longterm_average_absolute_uncertainty_map = np.zeros_like(temp_absolute_uncertainty_map)
        longterm_average_absolute_uncertainty_map += temp_absolute_uncertainty_map
        count += 1
    longterm_average_absolute_uncertainty_map = longterm_average_absolute_uncertainty_map / count
    save_absoulute_uncertainty_map(absolute_uncertainty_map=longterm_average_absolute_uncertainty_map,
                                   species=species,version=map_estimation_version,special_name=map_estimation_special_name,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],
                                   obs_version=obs_version,nearby_sites_number=nearby_sites_number)
    return

def Convert_mahalanobis_distance_map_to_uncertainty(species,version,special_name,
                                                    Obs_version,nearby_sites_number,YYYY_list:np.int32,MM_list:np.int32,
                                                    nchannel,startyear,endyear):
    '''
    Convert Mahalanobis distance map to uncertainty map for given species and years/months.

    The LOWESS relationship is keyed by (nchannel, startyear-endyear) so the
    26ch (2019-2023) and 27ch (2005-2018) curves are kept separate.
    '''
    Mahalanobis_distance_bin_centers,WINTER_LOWESS_values,SPRING_LOWESS_values,SUMMER_LOWESS_values,AUTUMN_LOWESS_values,ALL_LOWESS_values = load_bins_LOWESS_values(species=species,version=version,special_name=special_name,nearby_sites_number=nearby_sites_number,nchannel=nchannel,startyear=startyear,endyear=endyear)
    
    ## Get Four Seasons and All data, and plot five lines in one figure

    WINTER_MONTHS = ['Jan', 'Feb',  'Dec']
    SPRING_MONTHS = ['Mar', 'Apr', 'May']
    SUMMER_MONTHS = ['Jun','Jul','Aug']
    AUTUMN_MONTHS = ['Sep','Oct','Nov']
    ALL_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # MONTHS = ['Jun','Jul','Aug']
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']
    
    for YYYY in YYYY_list:
        for MM in MM_list:
            print(f"Calculating rRMSE uncertainty map for species: {species}, version: {version}, date: {YYYY}-{MONTH[MM]}, obs_version: {Obs_version}, nearby_sites_number: {nearby_sites_number}")
            mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                             obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
            mahalanobis_distance_map = np.log(mahalanobis_distance_map + 1)
            map_uncertainty = np.zeros(mahalanobis_distance_map.shape,dtype=np.float64)
            
            if MM in [0,1,11]: # Winter
                LOWESS_values = WINTER_LOWESS_values
            elif MM in [2,3,4]: # Spring
                LOWESS_values = SPRING_LOWESS_values
            elif MM in [5,6,7]: # Summer
                LOWESS_values = SUMMER_LOWESS_values
            elif MM in [8,9,10]: # Autumn
                LOWESS_values = AUTUMN_LOWESS_values
            elif MM == 12:
                LOWESS_values = ALL_LOWESS_values
            valid_LOWESS_index = np.where(~np.isnan(LOWESS_values))[0]
            LOWESS_values = LOWESS_values[valid_LOWESS_index]
            temp_Mahalanobis_distance_bin_centers = [Mahalanobis_distance_bin_centers[i] for i in valid_LOWESS_index]
            for iradius in range(len(temp_Mahalanobis_distance_bin_centers)-1):
                d_left  = temp_Mahalanobis_distance_bin_centers[iradius]
                d_right = temp_Mahalanobis_distance_bin_centers[iradius+1]
                rRMSE_left  = LOWESS_values[iradius]
                rRMSE_right = LOWESS_values[iradius+1]
                pixels_index = np.where((mahalanobis_distance_map >= d_left) & (mahalanobis_distance_map < d_right))
                print('d_left: {}, d_right: {}, rRMSE_left: {}, rRMSE_right: {}'.format(d_left,d_right,rRMSE_left,rRMSE_right))
                slope = (rRMSE_right - rRMSE_left) / (d_right - d_left)
                map_uncertainty[pixels_index] = (mahalanobis_distance_map[pixels_index]-d_left)*slope + rRMSE_left

            d_left  = temp_Mahalanobis_distance_bin_centers[0]
            d_right = temp_Mahalanobis_distance_bin_centers[-1]
            rRMSE_left  = LOWESS_values[0]
            rRMSE_right = LOWESS_values[-1]
            outrange_pixels_index = np.where(mahalanobis_distance_map >= temp_Mahalanobis_distance_bin_centers[-1])
            
            mask_low = np.where(mahalanobis_distance_map < d_left)
            
            if LOWESS_values[0] <= LOWESS_values[1]:
                slope = abs(LOWESS_values[1]-LOWESS_values[0])/(temp_Mahalanobis_distance_bin_centers[1]-temp_Mahalanobis_distance_bin_centers[0])
            else:
                slope = 0.05
            map_uncertainty[mask_low] = slope*(mahalanobis_distance_map[mask_low]-temp_Mahalanobis_distance_bin_centers[0])+LOWESS_values[0]

            if LOWESS_values[-1] >= LOWESS_values[-2]:
                slope = abs(LOWESS_values[-1]-LOWESS_values[-2])/(temp_Mahalanobis_distance_bin_centers[-1]-temp_Mahalanobis_distance_bin_centers[-2])
                map_uncertainty[outrange_pixels_index] = slope*(mahalanobis_distance_map[outrange_pixels_index]-temp_Mahalanobis_distance_bin_centers[-1])+LOWESS_values[-1]
            else:
                #slope,intercept = m, b = np.polyfit(Mahalanobis_distance_bin_centers,LOWESS_values,1)#abs(BLCO_rRMSE_LOWESS_values[-1]-BLCO_rRMSE_LOWESS_values[0])/(distances_bins_array[-1]-distances_bins_array[0])
                map_uncertainty[outrange_pixels_index] = 0.1*(mahalanobis_distance_map[outrange_pixels_index]-temp_Mahalanobis_distance_bin_centers[-1])+LOWESS_values[-1]
                #map_uncertainty[outrange_pixels_index] = rRMSE_right #(map_distances[outrange_pixels_index]-d_left)/(d_right-d_left) * (rRMSE_right - rRMSE_left) +rRMSE_left
            map_uncertainty = np.minimum(map_uncertainty, 1.0)
            save_rRMSE_map(rRMSE_uncertainty_map=map_uncertainty,species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                            obs_version=Obs_version,nearby_sites_number=nearby_sites_number,special_name=special_name)
    return

def Get_nearby_sites_indices_map(species,version,nearby_sites_number,YYYY:np.int32,MM:np.int32):
    """Compute the (H, W, K) nearby-sites index map and stream it to a
    disk-backed .npy memmap. This avoids ever holding the full indices array
    (~56 GB at int32 for a global 0.01 deg grid with K=30) in RAM; the kernel
    pages written rows to disk and evicts them under memory pressure.
    """
    obs_data, obs_lat, obs_lon = load_RawObservation(species)
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    original_nearby_sites_number = nearby_sites_number
    if MM != 12:
        temp_obs_data = obs_data[:,(YYYY-2005)*12 + MM]
        nonan_index = np.where(~np.isnan(temp_obs_data))
        obs_index = np.arange(obs_lat.shape[0])
        # Pre-extract the compact lat/lon/index arrays, then release the large
        # obs_data (N_sites x N_months) so it does not live alongside the map.
        obs_lat_sel = np.ascontiguousarray(obs_lat[nonan_index])
        obs_lon_sel = np.ascontiguousarray(obs_lon[nonan_index])
        original_idx_lookup = obs_index[nonan_index]
        del obs_data, temp_obs_data, obs_lat, obs_lon, obs_index
        gc.collect()

        n_valid = obs_lat_sel.shape[0]
        if n_valid < nearby_sites_number:
            nearby_sites_number = n_valid
            print('The nearby sites number is set to {} due to the limited valid observations.'.format(nearby_sites_number))

        # int32 is sufficient (max obs index is nowhere near 2**31 - 1).
        # This halves the on-disk and in-RAM footprint vs default int64.
        out_dtype = np.int32

        GeoLAT_MAP, GeoLON_MAP = load_GeoLatLon_Map()
        H, W = GeoLAT_MAP.shape
        landtype = get_landtype(extent=[-59.995,69.995,-179.995,179.995])

        # Allocate the output as a disk-backed memmap .npy. open_memmap writes
        # a valid numpy header, so the file can be loaded transparently with
        # np.load(..., mmap_mode='r') by downstream steps (no format changes).
        outfile = get_nearby_sites_index_map_path(species=species, version=version,
                                                  YYYY=YYYY, MM=MONTH[MM],
                                                  obs_version=Obs_version,
                                                  nearby_sites_number=original_nearby_sites_number)
        nearby_sites_training_data_indices = _np_format.open_memmap(
            outfile, mode='w+', dtype=out_dtype,
            shape=(H, W, nearby_sites_number),
        )

        for ix in range(H):
            land_index = np.where(landtype[ix, :] != 0)
            print('It is procceding ' + str(np.round(100*(ix/H), 2))+'%.')
            if len(land_index[0]) == 0:
                print('No lands.')
                continue
            idx = neighbors_haversine_indices(
                obs_lat_sel, obs_lon_sel,
                GeoLAT_MAP[ix, land_index[0]], GeoLON_MAP[ix, land_index[0]],
                nearby_sites_number,
            )
            idx = np.asarray(idx)
            original_idx = original_idx_lookup[idx].astype(out_dtype, copy=False)
            nearby_sites_training_data_indices[ix, land_index[0], :] = original_idx
            # Periodically flush so dirty pages do not accumulate unbounded in
            # the page cache (LSF accounts these against the job).
            if (ix & 0x1FF) == 0x1FF:  # every 512 rows
                nearby_sites_training_data_indices.flush()

        nearby_sites_training_data_indices.flush()
        del nearby_sites_training_data_indices
        gc.collect()

    return

def Get_local_reference_for_channels(channel_lists,species,version,
                                   Obs_version,nearby_sites_number,YYYY:np.int32,MM:np.int32):
    """Compute per-pixel local reference (mean of K nearest sites' training
    values) for every channel.

    Performance: the nearby-indices file is a 56-112 GB mmap of shape (H, W, K)
    with K as the fastest-changing axis, so the old ``[:, :, ik]`` slice did
    K strided passes over the whole file per channel. This rewrite reads the
    mmap once in contiguous row blocks and folds over K and all channels
    inside each block, turning hundreds of strided passes into one sequential
    scan.
    """
    if MM != 12:
        MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
        width_nc, height_nc, sites_number, start_YYYY_training, _ = load_TrainingVariables(channel_lists)
        RawObs_training_data = load_RawObs_training_data(channel_lists=channel_lists)

        nearby_sites_training_data_indices = load_pixels_nearest_sites_indices_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                                                obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
        H, W, K = nearby_sites_training_data_indices.shape

        GeoLAT_MAP, GeoLON_MAP = load_GeoLatLon_Map()

        # Per-channel 1D training slice for this (YYYY, MM), cast to float32
        # once so the gather produces float32 directly (skips a later copy).
        ch_start = ((YYYY - 2005) * 12 + MM) * sites_number
        ch_end   = ch_start + sites_number
        per_channel_training = {
            ch: np.ascontiguousarray(RawObs_training_data[ch][ch_start:ch_end], dtype=np.float32)
            for ch in channel_lists
        }

        # Output accumulators, one per channel.
        local_reference_for_channels_map = {
            ch: np.zeros((H, W), dtype=np.float32) for ch in channel_lists
        }

        # Row-block streaming. CHUNK tuned so one block of indices is a few GB
        # at most: CHUNK * W * K * sizeof(dtype). For W=3600, K=30, int32 that
        # is 512 * 3600 * 30 * 4 B ~= 220 MB; for int64 ~440 MB. Safe either
        # way within the 150 GB per-job budget.
        CHUNK = 512
        inv_K = np.float32(1.0 / K)
        print('Processing {} - {} local reference for {} channels ({} row blocks of {} rows each)'.format(
            species, MONTH[MM], len(channel_lists), (H + CHUNK - 1) // CHUNK, CHUNK))
        for i0 in range(0, H, CHUNK):
            i1 = min(i0 + CHUNK, H)
            # Contiguous read from the mmap: one sequential scan serves all
            # K neighbors AND all channels in this block.
            idx_block = np.asarray(nearby_sites_training_data_indices[i0:i1])  # (b, W, K)
            for ch in channel_lists:
                # Gather once, sum over the K axis once, scale by 1/K once.
                gathered = per_channel_training[ch][idx_block]  # (b, W, K) float32
                local_reference_for_channels_map[ch][i0:i1] = gathered.sum(
                    axis=-1, dtype=np.float32
                ) * inv_K
            del idx_block
            if ((i0 // CHUNK) & 0x1F) == 0:  # ~every 32 blocks
                print('  rows {}/{} done'.format(i1, H))

        save_local_reference_map(local_reference_for_channels_map=local_reference_for_channels_map,
                                species=species,version=version,YYYY=YYYY,MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
    return

 
def Calculate_Mahalanobis_distance(channel_lists,species,version,
                                   Obs_version,nearby_sites_number,YYYY:np.int32,MM:np.int32,
                                   longterm_average:bool=False):
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','Annual']

    if MM != 12:
        # Cov matrix on the (N_sites*12_months, N_channels) training table.
        # Small relative to the map (n_channels x n_channels), so keep as-is.
        RawObs_training_data = load_RawObs_training_data(channel_lists=channel_lists)
        total_channels_training_site_data_list = np.stack(
            [RawObs_training_data[channel] for channel in channel_lists], axis=1
        )
        covariance_matrix = calculate_covariance_matrix(total_channels_training_site_data_list)
        inverted_covariance_matrix = invert_matrix(covariance_matrix).astype(np.float32, copy=False)
        del total_channels_training_site_data_list, RawObs_training_data
        gc.collect()

        # Load the per-channel local-reference map (dict of (H, W) float32).
        local_reference_for_channels_map = load_local_reference_map(
            species=species, version=version, YYYY=YYYY, MM=MONTH[MM],
            obs_version=Obs_version, nearby_sites_number=nearby_sites_number,
        )

        # Load per-channel map arrays (dict of (H, W) float32). We intentionally
        # do NOT stack them into an (n, H, W) array: two stacked (n, H, W)
        # float32 arrays at global 0.01 deg are ~50 GB each, and the subsequent
        # diff/left einsum temporaries would push peak memory past the 150 GB
        # job budget. Keeping them as dicts + streaming row blocks below avoids
        # that extra 100+ GB.
        input_file_dic = inputfiles_table(YYYY=YYYY, MM=MONTH[MM])
        channel_maps = {}
        for channel in channel_lists:
            print('Processing {} - {} - {} Mahalanobis distance for channel: {}'.format(species, YYYY, MONTH[MM], channel))
            channel_maps[channel] = load_mapdata(input_file_dic[channel]).astype(np.float32, copy=False)

        # Row-block streaming of the Mahalanobis distance. For each block,
        # build diff = X - mu with shape (n, block, W), then
        # dist2 = sum_ij diff[i] * inv[i,j] * diff[j].
        n = len(channel_lists)
        H, W = channel_maps[channel_lists[0]].shape
        mahalanobis_distance_data_map = np.zeros((H, W), dtype=np.float32)
        inv = inverted_covariance_matrix  # (n, n) float32

        CHUNK = 512
        n_blocks = (H + CHUNK - 1) // CHUNK
        print('Computing Mahalanobis distance in {} row blocks of {} rows each ({} channels)'.format(n_blocks, CHUNK, n))
        for i0 in range(0, H, CHUNK):
            i1 = min(i0 + CHUNK, H)
            block_h = i1 - i0
            diff = np.empty((n, block_h, W), dtype=np.float32)
            for ichannel, ch in enumerate(channel_lists):
                np.subtract(channel_maps[ch][i0:i1], local_reference_for_channels_map[ch][i0:i1], out=diff[ichannel])
            # left = inv @ diff along the channel axis; then sum_i left[i] * diff[i]
            left = np.einsum('ij,jhw->ihw', inv, diff, optimize=True)
            block_dist2 = np.einsum('ihw,ihw->hw', left, diff, optimize=True)
            np.sqrt(np.maximum(block_dist2, 0.0, out=block_dist2), out=block_dist2)
            mahalanobis_distance_data_map[i0:i1] = block_dist2
            del diff, left, block_dist2
            if ((i0 // CHUNK) & 0x1F) == 0:
                print('  MD rows {}/{} done'.format(i1, H))

        del channel_maps, local_reference_for_channels_map
        gc.collect()

        Mahalanobis_distance_data_map = mahalanobis_distance_data_map
        save_mahalanobis_distance_map(mahalanobis_distance_map=Mahalanobis_distance_data_map,
                                    species=species,version=version,YYYY=YYYY,MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
        Mahalanobis_distance_data = Mahalanobis_distance_data_map
    elif MM == 12:
        local_reference_for_channels_map = load_local_reference_map(species=species,version=version,YYYY=YYYY,MM=MONTH[0],
                                                                    obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
        Mahalanobis_distance_data = np.zeros((local_reference_for_channels_map[channel_lists[0]].shape[0],local_reference_for_channels_map[channel_lists[0]].shape[1]), dtype=np.float32)
        for imonth in range(12):
            temp_Mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[imonth],
                                                            obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
            Mahalanobis_distance_data += temp_Mahalanobis_distance_map
        Mahalanobis_distance_data = Mahalanobis_distance_data / 12.0
        save_mahalanobis_distance_map(mahalanobis_distance_map=Mahalanobis_distance_data,
                                    species=species,version=version,YYYY=YYYY,MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)

        
                
    return Mahalanobis_distance_data

def get_longterm_average_mahalanobis_distance_map(species,version,Obs_version,nearby_sites_number,YYYY_list:np.int32,MM:np.int32):
    '''
    Get longterm average Mahalanobis distance map for given species and years/months.
    '''
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12','AllMonths']
    if MM != 12:
        print(f"Calculating longterm average Mahalanobis distance map for species: {species}, version: {version}, month: {MONTH[MM]}, obs_version: {Obs_version}, nearby_sites_number: {nearby_sites_number}")
        longterm_average_mahalanobis_distance_map = None
        count = 0
        for YYYY in YYYY_list:
            temp_mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[MM],
                                                            obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
            if longterm_average_mahalanobis_distance_map is None:
                longterm_average_mahalanobis_distance_map = np.zeros_like(temp_mahalanobis_distance_map)
            longterm_average_mahalanobis_distance_map += temp_mahalanobis_distance_map
            count += 1
        longterm_average_mahalanobis_distance_map = longterm_average_mahalanobis_distance_map / count
        save_mahalanobis_distance_map(mahalanobis_distance_map=longterm_average_mahalanobis_distance_map,
                                    species=species,version=version,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
    elif MM == 12:
        print(f"Calculating longterm average Mahalanobis distance map for species: {species}, version: {version}, month: All Months, obs_version: {Obs_version}, nearby_sites_number: {nearby_sites_number}")
        longterm_average_mahalanobis_distance_map = None
        count = 0
        for YYYY in YYYY_list:
            for imonth in range(12):
                temp_mahalanobis_distance_map = load_mahalanobis_distance_map(species=species,version=version,YYYY=YYYY,MM=MONTH[imonth],
                                                            obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
                if longterm_average_mahalanobis_distance_map is None:
                    longterm_average_mahalanobis_distance_map = np.zeros_like(temp_mahalanobis_distance_map)
                longterm_average_mahalanobis_distance_map += temp_mahalanobis_distance_map
                count += 1
        longterm_average_mahalanobis_distance_map = longterm_average_mahalanobis_distance_map / count
        save_mahalanobis_distance_map(mahalanobis_distance_map=longterm_average_mahalanobis_distance_map,
                                    species=species,version=version,YYYY='Longterm_{}-{}'.format(YYYY_list[0], YYYY_list[-1]),MM=MONTH[MM],obs_version=Obs_version,nearby_sites_number=nearby_sites_number)
    return