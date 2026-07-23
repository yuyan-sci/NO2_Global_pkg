import argparse
import numpy as np

from map_uncertainty_func.Assemble_func import (
    get_longterm_average_mahalanobis_distance_map,
    Get_longterm_average_absolute_uncertainty_map,
    Get_absolute_uncertainty_map,
    Convert_mahalanobis_distance_map_to_uncertainty,
    Calculate_Mahalanobis_distance,
    Get_nearby_sites_indices_map,
    Get_local_reference_for_channels,
)
from data_func.utils import Get_typeName, Obs_version
from visualization_pkg.Assemble_func import (
    plot_longterm_average_absolute_uncertainty_map,
    plot_longterm_average_mahalanobis_distance_map,
    plot_longterm_average_map_estimation_data,
    plot_mahalanobis_distance_map,
    plot_rRMSE_uncertainty_map,
    plot_absolute_uncertainty_map,
    plot_map_estimation_data,
)

# Phases U5/U6: build Mahalanobis-distance uncertainty maps for a given year.
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
    p = argparse.ArgumentParser(description='Phases U5/U6: map Mahalanobis distance -> uncertainty.')
    p.add_argument('--startyear', type=int, required=True,
                   help='Start year inclusive. For per-year U5 work, set startyear=endyear.')
    p.add_argument('--endyear', type=int, required=True,
                   help='End year inclusive. U6 uses this (with --startyear) to load the '
                        'matching LOWESS relationship file.')
    p.add_argument('--version', type=str, default='v5')
    p.add_argument('--special-name', type=str, default='')
    p.add_argument('--seeds-number', type=int, default=10)
    p.add_argument('--kfold', type=int, default=10)
    p.add_argument('--width', type=int, default=5)
    p.add_argument('--height', type=int, default=5)
    p.add_argument('--local-nearby-sites-number', type=int, default=30)
    p.add_argument('--months', type=str, default='0,1,2,3,4,5,6,7,8,9,10,11,12',
                   help='Comma-separated months to process (12 = annual).')
    p.add_argument('--years', type=str, default=None,
                   help='Comma-separated years to iterate over. Overrides the default '
                        '`range(startyear, endyear+1)`. The LOWESS-file lookup still '
                        'uses --startyear/--endyear/--nchannel, so this lets you run '
                        'a single year (e.g. --years 2005) against the multi-year '
                        'LOWESS relationship (--startyear 2005 --endyear 2018).')

    # Switches (off by default; enable what you need)
    p.add_argument('--get-nearby-indices', action='store_true')
    p.add_argument('--get-local-reference', action='store_true')
    p.add_argument('--get-mahalanobis-map', action='store_true')
    p.add_argument('--plot-mahalanobis-map', action='store_true')
    p.add_argument('--get-uncertainty-rrmse-map', action='store_true')
    p.add_argument('--plot-uncertainty-rrmse-map', action='store_true')
    p.add_argument('--get-absolute-uncertainty-map', action='store_true')
    p.add_argument('--plot-absolute-uncertainty-map', action='store_true')
    p.add_argument('--plot-map-estimation', action='store_true')

    # Convenience bundles
    p.add_argument('--pipeline-u5', action='store_true',
                   help='Shortcut: --get-nearby-indices --get-local-reference --get-mahalanobis-map')
    p.add_argument('--pipeline-u6', action='store_true',
                   help='Shortcut: --get-uncertainty-rrmse-map --get-absolute-uncertainty-map')
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

    plot_months = [int(m) for m in args.months.split(',') if m.strip() != '']
    if args.years:
        desire_year_list = [int(y) for y in args.years.split(',') if y.strip() != '']
        # Validate years fall in the LOWESS-range so the channel scheme matches.
        for y in desire_year_list:
            if not (startyear <= y <= endyear):
                raise ValueError(
                    f'--years contains {y} which is outside the LOWESS range '
                    f'[{startyear}, {endyear}]; channel scheme would mismatch.'
                )
    else:
        desire_year_list = list(range(startyear, endyear + 1))
    local_nearby_sites_number = args.local_nearby_sites_number

    # Resolve bundle shortcuts
    if args.pipeline_u5:
        args.get_nearby_indices = True
        args.get_local_reference = True
        args.get_mahalanobis_map = True
    if args.pipeline_u6:
        args.get_uncertainty_rrmse_map = True
        args.get_absolute_uncertainty_map = True

    print(f'[U5/U6] startyear={startyear} endyear={endyear}  version={args.version}  '
          f'special_name={args.special_name!r}  nchannel={nchannel}  months={plot_months}')

    # ---- U5 block --------------------------------------------------------
    if args.get_nearby_indices:
        for YYYY in desire_year_list:
            for MM in plot_months:
                Get_nearby_sites_indices_map(species=species, version=args.version,
                                             nearby_sites_number=local_nearby_sites_number,
                                             YYYY=YYYY, MM=MM)

    if args.get_local_reference:
        for YYYY in desire_year_list:
            for MM in plot_months:
                Get_local_reference_for_channels(channel_lists=channel_lists,
                                                 species=species, version=args.version,
                                                 Obs_version=Obs_version,
                                                 nearby_sites_number=local_nearby_sites_number,
                                                 YYYY=YYYY, MM=MM)

    if args.get_mahalanobis_map:
        for YYYY in desire_year_list:
            for MM in plot_months:
                Calculate_Mahalanobis_distance(channel_lists=channel_lists,
                                               species=species, version=args.version,
                                               Obs_version=Obs_version,
                                               nearby_sites_number=local_nearby_sites_number,
                                               YYYY=YYYY, MM=MM)

    if args.plot_mahalanobis_map:
        for YYYY in desire_year_list:
            for MM in plot_months:
                plot_mahalanobis_distance_map(species=species, version=args.version,
                                              YYYY=YYYY, MM=MM,
                                              obs_version=Obs_version,
                                              nearby_sites_number=local_nearby_sites_number)

    # ---- U6 block --------------------------------------------------------
    if args.get_uncertainty_rrmse_map:
        Convert_mahalanobis_distance_map_to_uncertainty(
            species=species, version=args.version, special_name=args.special_name,
            Obs_version=Obs_version,
            nearby_sites_number=local_nearby_sites_number,
            YYYY_list=desire_year_list, MM_list=plot_months,
            nchannel=nchannel, startyear=startyear, endyear=endyear,
        )

    if args.plot_uncertainty_rrmse_map:
        vmin_list = [0.30, 0.35, 0.30, 0.30]
        vmax_list = [0.50, 0.50, 0.50, 0.50]
        for YYYY in desire_year_list:
            for MM in plot_months:
                if MM in [0, 1, 11]:
                    vmin, vmax = vmin_list[0], vmax_list[0]
                elif MM in [2, 3, 4]:
                    vmin, vmax = vmin_list[1], vmax_list[1]
                elif MM in [5, 6, 7]:
                    vmin, vmax = vmin_list[2], vmax_list[2]
                else:
                    vmin, vmax = vmin_list[3], vmax_list[3]
                plot_rRMSE_uncertainty_map(species=species, version=args.version,
                                           YYYY=YYYY, MM=MM,
                                           obs_version=Obs_version,
                                           nearby_sites_number=local_nearby_sites_number,
                                           vmin=vmin, vmax=vmax,
                                           special_name=args.special_name)

    if args.get_absolute_uncertainty_map:
        for YYYY in desire_year_list:
            for MM in plot_months:
                Get_absolute_uncertainty_map(species=species, version=args.version,
                                             special_name=args.special_name,
                                             obs_version=Obs_version,
                                             nearby_sites_number=local_nearby_sites_number,
                                             YYYY=YYYY, MM=MM,
                                             map_estimation_special_name=args.special_name,
                                             map_estimation_version=args.version)

    if args.plot_absolute_uncertainty_map:
        absolute_uncertainty_vmin_list = [0, 0, 0, 0, 0]
        absolute_uncertainty_vmax_list = [5, 5, 5, 5, 5]
        for YYYY in desire_year_list:
            for MM in plot_months:
                if MM in [0, 1, 11]:
                    vmin, vmax = absolute_uncertainty_vmin_list[0], absolute_uncertainty_vmax_list[0]
                elif MM in [2, 3, 4]:
                    vmin, vmax = absolute_uncertainty_vmin_list[1], absolute_uncertainty_vmax_list[1]
                elif MM in [5, 6, 7]:
                    vmin, vmax = absolute_uncertainty_vmin_list[2], absolute_uncertainty_vmax_list[2]
                elif MM in [8, 9, 10]:
                    vmin, vmax = absolute_uncertainty_vmin_list[3], absolute_uncertainty_vmax_list[3]
                else:
                    vmin, vmax = absolute_uncertainty_vmin_list[4], absolute_uncertainty_vmax_list[4]
                plot_absolute_uncertainty_map(species=species,
                                              map_estimation_version=args.version,
                                              map_estimation_special_name=args.special_name,
                                              YYYY=YYYY, MM=MM,
                                              obs_version=Obs_version,
                                              nearby_sites_number=local_nearby_sites_number,
                                              vmin=vmin, vmax=vmax)

    if args.plot_map_estimation:
        for YYYY in desire_year_list:
            for MM in plot_months:
                plot_map_estimation_data(species=species,
                                         map_estimation_version=args.version,
                                         YYYY=YYYY, MM=MM,
                                         map_estimation_special_name=args.special_name)


if __name__ == '__main__':
    main()
