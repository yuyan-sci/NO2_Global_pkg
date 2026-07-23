from Convert_TrainingData_pkg.Assemble_Func import get_save_nearest_indices, add_channels_on_Existed_TrainingFiles
from Convert_TrainingData_pkg.utils import *
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--width',  type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--parallel', action='store_true', default=True,
                    help='Use parallel year-month processing (default: True)')
parser.add_argument('--no-parallel', dest='parallel', action='store_false',
                    help='Disable parallel processing')
parser.add_argument('--workers', type=int, default=None,
                    help='Number of parallel workers (default: cpu_count - 1)')
parser.add_argument('--skip-indices', action='store_true',
                    help='Skip computing nearest indices (use existing)')

if __name__ == '__main__':
    args = parser.parse_args()

    YEAR = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019','2020', '2021', '2022', '2023']
    width  = args.width
    height = args.height
    init_channel_names = ['GeoNO2',
                          'GCHP_NO2',
                          'SatColNO2_TM', 'SatColNO2_GC',
                    'Population',
                    'NO_emi',
                    'Total_DM',
                    'NDVI', 'ISA',
                    'TSW', 'USTAR', 'V10M', 'U10M', 'T2M', 'RH', 'PBLH', 'TP', 'PS',
                    'lat', 'lon', 'elevation', 
                    'log_major_roads', 'log_minor_roads_new',
                    'forests_density',  'shrublands_distance', 'croplands_distance', 'water_bodies_distance',
                    'urban_builtup_lands_buffer-6500',
]
    Exclusion_Industrial_Sites = False
    nonan_GCHPfilted = False
    #### Set variables and load arrays for running function
    MONTH = ['01','02','03','04','05','06','07','08','09','10','11','12']
    print('YEAR:', YEAR)
    print(f'Parallel: {args.parallel}, Workers: {args.workers or "auto"}')
    start_YYYYMM = '{}{}'.format(YEAR[0],MONTH[0])
    end_YYYYMM   = '{}{}'.format(YEAR[-1],MONTH[-1])

    replace_channel_names = []           # replace in-place (no extra channel count)
    add_channel_names = ['GeoNO2-v2']  # truly new channels to append
    outdir = TrainingData_outdir
    init_infile = outdir + f'v7_filtered_TrainingData_28channels_5x5_{start_YYYYMM}-{end_YYYYMM}.nc'

    start_time = time.time()

    if not args.skip_indices:
        print('Computing and saving nearest indices ...')
        get_save_nearest_indices()
    else:
        print('Skipping indices computation (using existing)')

    add_channels_on_Existed_TrainingFiles(
        init_training_file=init_infile,
        init_channel_names=init_channel_names,
        add_channel_names=add_channel_names,
        replace_channel_names=replace_channel_names,
        width=width, height=height,
        YEAR=YEAR,
        nonan_GCHPfilted=nonan_GCHPfilted,
        ExcludeIndustialSites=Exclusion_Industrial_Sites,
        use_parallel=args.parallel,
        n_workers=args.workers,
    )

    elapsed = time.time() - start_time
    print(f'\nTotal elapsed: {elapsed:.1f}s  ({elapsed/60:.1f} min)')
