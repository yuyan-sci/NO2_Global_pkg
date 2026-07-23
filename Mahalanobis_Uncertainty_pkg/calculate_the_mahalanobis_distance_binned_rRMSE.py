import argparse
from data_func.Assemble_func import Assemble_Mahalanobis_distance_data, derive_local_reference_for_channels
from data_func.utils import Get_typeName

# Phase U3: Build the (Mahalanobis distance, rRMSE) relationship from resampled
# BLISCO training data. Runs per year with channel scheme matching the BLISCO run.
#
#   2005-2018: 27 channels including ML_NO2_2019_Map
#   2019-2023: 26 channels (no ML_NO2)

BASE_CHANNELS = ['GeoNO2', 'GCHP_NO2', 'Population',
                 'NO_emi', 'Total_DM',
                 'NDVI', 'ISA',
                 'TSW', 'USTAR', 'V10M', 'U10M', 'T2M', 'RH', 'PBLH', 'TP', 'PS',
                 'elevation', 'lat', 'lon',
                 'log_major_roads', 'log_minor_roads_new',
                 'forests_density', 'shrublands_distance', 'croplands_distance',
                 'urban_builtup_lands_buffer-6500', 'water_bodies_distance']


def parse_args():
    p = argparse.ArgumentParser(description='Phase U3: Mahalanobis distance <-> rRMSE binned relationship over a multi-year period.')
    p.add_argument('--startyear', type=int, required=True, help='Start year inclusive (e.g. 2019).')
    p.add_argument('--endyear', type=int, required=True, help='End year inclusive (e.g. 2023).')
    p.add_argument('--version', type=str, default='v5')
    p.add_argument('--special-name', type=str, default='')
    p.add_argument('--seeds-number', type=int, default=10)
    p.add_argument('--kfold', type=int, default=10)
    p.add_argument('--width', type=int, default=5)
    p.add_argument('--height', type=int, default=5)
    p.add_argument('--local-nearby-sites-number', type=int, default=30)
    p.add_argument('--skip-local-reference', action='store_true',
                   help='Skip derive_local_reference_for_channels if already computed.')
    p.add_argument('--skip-mahalanobis', action='store_true',
                   help='Skip Assemble_Mahalanobis_distance_data.')
    return p.parse_args()


def main():
    args = parse_args()
    startyear = args.startyear
    endyear = args.endyear
    if startyear > endyear:
        raise ValueError(f'startyear ({startyear}) must be <= endyear ({endyear})')

    if endyear <= 2018:
        channel_lists = BASE_CHANNELS + ['ML_NO2_2019_Map']
    elif startyear >= 2019:
        channel_lists = list(BASE_CHANNELS)
    else:
        raise ValueError(
            f'Period {startyear}-{endyear} crosses the 2018/2019 channel-scheme boundary; '
            'run 2005-2018 and 2019-2023 as separate jobs.'
        )
    nchannel = len(channel_lists)

    species = 'NO2'
    typeName = Get_typeName(bias=False, normalize_bias=False, normalize_species=False,
                            absolute_species=True, log_species=False, species=species)

    buffer_radius_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                          110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    desire_year_list = [str(y) for y in range(startyear, endyear + 1)]

    print(f'[U3] startyear={startyear} endyear={endyear}  version={args.version}  '
          f'special_name={args.special_name!r}  nchannel={nchannel}  '
          f'seeds={args.seeds_number}  kfold={args.kfold}')

    if not args.skip_local_reference:
        derive_local_reference_for_channels(channel_lists, species, args.version, typeName,
                                            startyear, endyear, nchannel, args.special_name,
                                            args.width, args.height,
                                            buffer_radius_list, args.kfold,
                                            desire_year_list, 'mean',
                                            args.local_nearby_sites_number)

    if not args.skip_mahalanobis:
        Assemble_Mahalanobis_distance_data(species, args.version, typeName, startyear, endyear,
                                           nchannel, args.special_name, args.width, args.height,
                                           buffer_radius_list, args.kfold,
                                           channel_lists, desire_year_list,
                                           args.local_nearby_sites_number)


if __name__ == '__main__':
    main()
