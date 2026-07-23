import argparse
import numpy as np
import os
from data_func.Assemble_func import derive_corresponding_training_data_BLISCO_data
from data_func.utils import Get_typeName

# Phase U2: Derive resampled training data from BLISCO output, per year.
#
# Channel scheme (consistent with SCV + BLISCO runs):
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
    p = argparse.ArgumentParser(description='Phase U2: derive resampled BLISCO training data for a multi-year period.')
    p.add_argument('--startyear', type=int, required=True, help='Start year inclusive (e.g. 2019).')
    p.add_argument('--endyear', type=int, required=True, help='End year inclusive (e.g. 2023).')
    p.add_argument('--version', type=str, default='v5')
    p.add_argument('--special-name', type=str, default='')
    p.add_argument('--seeds-number', type=int, default=10)
    p.add_argument('--kfold', type=int, default=10)
    p.add_argument('--width', type=int, default=5)
    p.add_argument('--height', type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    startyear = args.startyear
    endyear = args.endyear
    if startyear > endyear:
        raise ValueError(f'startyear ({startyear}) must be <= endyear ({endyear})')
    # Channel scheme must match the BLISCO run for this period.
    # 2005-2018 → 27ch (with ML_NO2_2019_Map); 2019-2023 → 26ch (no ML_NO2).
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

    print(f'[U2] startyear={startyear} endyear={endyear}  version={args.version}  '
          f'special_name={args.special_name!r}  nchannel={nchannel}  '
          f'seeds={args.seeds_number}  kfold={args.kfold}')

    derive_corresponding_training_data_BLISCO_data(
        typeName=typeName, species=species, version=args.version,
        startyear=startyear, endyear=endyear, nchannel=nchannel,
        special_name=args.special_name, width=args.width, height=args.height,
        buffer_radius_list=buffer_radius_list,
        BLCO_kfold=args.kfold, BLCO_seeds_number=args.seeds_number,
        channel_lists=channel_lists,
        desire_year_list=desire_year_list,
    )


if __name__ == '__main__':
    main()
