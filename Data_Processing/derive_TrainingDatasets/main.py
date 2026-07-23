from Convert_TrainingData_pkg.Assemble_Func import get_save_nearest_indices,derive_TrainingDatasets
import argparse
import time

parser=argparse.ArgumentParser()
parser.add_argument('--width',type=int, required=True, help='Width of training window')
parser.add_argument('--height',type=int, required=True, help='Height of training window')
parser.add_argument('--parallel', action='store_true', default=True, help='Use parallel processing (default: True)')
parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='Disable parallel processing')
parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: auto)')
parser.add_argument('--skip-indices', action='store_true', help='Skip computing nearest indices (use existing)')


if __name__ == '__main__':
    args = parser.parse_args()
    YEAR = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019','2020', '2021', '2022', '2023']
    width = args.width
    height = args.height
    
    channel_names = ['GeoNO2', 'GCHP_NO2',
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
    
    print(f'Starting training dataset derivation...')
    print(f'  Window size: {width}x{height}')
    print(f'  Channels: {len(channel_names)}')
    print(f'  Years: {len(YEAR)} ({YEAR[0]}-{YEAR[-1]})')
    print(f'  Parallel processing: {args.parallel}')
    if args.parallel and args.workers:
        print(f'  Workers: {args.workers}')
    print()
    
    start_time = time.time()
    
    if not args.skip_indices:
        print('Computing and saving nearest indices...')
        indices_start = time.time()
        get_save_nearest_indices()
        indices_time = time.time() - indices_start
        print(f'Indices computation completed in {indices_time:.2f} seconds\n')
    else:
        print('Skipping indices computation (using existing)\n')
    
    print('Deriving training datasets...')
    derive_start = time.time()
    derive_TrainingDatasets(
        channel_names=channel_names,
        width=width, 
        height=height,
        YEAR=YEAR,
        use_parallel=args.parallel,
        n_workers=args.workers
    )
    derive_time = time.time() - derive_start
    
    total_time = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'Training dataset derivation time: {derive_time:.2f} seconds ({derive_time/60:.2f} minutes)')
    print(f'Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)')
    print(f'{"="*60}')
    