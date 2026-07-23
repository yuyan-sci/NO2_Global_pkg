import pandas as pd
import numpy as np
import calendar
import glob
import re
import time
from pathlib import Path
from io import StringIO
from pandas.errors import ParserError
import unicodedata

def robust_read_csv(path, **kwargs):
    # Use the python engine + skip-bad-lines for EVERY encoding attempt. Some TD
    # files (e.g. 2014) contain malformed rows that make the default C parser
    # raise ParserError under the correct encoding (cp932/shift_jis); the old
    # code then fell through to a latin1 read that mojibake'd the Japanese column
    # names, so name-based column lookup silently dropped the whole year. Skipping
    # bad lines lets the correct encoding win with intact column names.
    for enc in ('utf-8','utf-8-sig','shift_jis','cp932','euc_jp','iso2022_jp','latin1'):
        try:
            return pd.read_csv(path, encoding=enc, engine='python',
                               on_bad_lines='skip', **kwargs)
        except (UnicodeDecodeError, ParserError):
            continue

    # last resort
    return pd.read_csv(path, encoding='latin1', engine='python', on_bad_lines='skip', **kwargs)


# -----------------------------------------------------------------------------
# Full-width numerals mapping for months 1-9 (e.g. '1' -> '１')
FW = {str(i): chr(ord('０') + i) for i in range(1,10)}

if __name__ == '__main__':
    start_time = time.time()

    raw_dir   = Path('/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/japan/TD20250704_014123_18')
    sites_dir = Path('/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/japan/TM20250704_014354_18')
    out_dir   = Path('/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/japan')

    all_years = []
    
    # Track statistics across all years
    total_raw_sites = 0
    total_filtered_sites = 0
    total_excluded_sites = 0
    year_stats = []
    
    # Track all unique sites (for calculating unique exclusions)
    all_unique_raw_sites = set()
    all_unique_filtered_sites = set()

    # Process each TD file
    for td_file in sorted(raw_dir.glob('TD*.csv')):
        m = re.match(r'TD(\d{4})', td_file.stem)
        if not m:
            print(f"Skipping unexpected file: {td_file.name}")
            continue
        year = int(m.group(1))
        print(f"\n=== YEAR: {year} ===")

        # Load site metadata TM file for this year
        tm_file = sites_dir / f'TM{year}0000.csv'
        if not tm_file.exists():
            print(f"  Metadata not found: {tm_file.name}")
            continue
        meta = robust_read_csv(tm_file, dtype=str)
        meta.columns = [
            unicodedata.normalize('NFKC', col).strip()
            for col in meta.columns
        ]
        # Ensure station_id is string
        meta['station_id'] = meta['国環研局番'].astype(str).str.strip()
        # Compute decimal lat/lon
        for col in ['緯度_度','緯度_分','緯度_秒','経度_度','経度_分','経度_秒']:
            meta[col] = pd.to_numeric(meta[col], errors='coerce')
        meta['lat'] = meta['緯度_度'] + meta['緯度_分']/60 + meta['緯度_秒']/3600
        meta['lon'] = meta['経度_度'] + meta['経度_分']/60 + meta['経度_秒']/3600
        site_meta = meta[['station_id','lon','lat']].drop_duplicates()
        site_meta['station_id'] = site_meta['station_id'].astype(str)

        # Load TD data
        df = robust_read_csv(td_file, dtype=str, header=0)
        
        # Use column index (12th column, index 11) for station_id instead of column name
        # This is more robust when dealing with encoding issues
        station_col_index = 11  # 12th column (0-indexed)
        
        # Extract station_id from the 12th column by index
        print(f"  Using column index {station_col_index} for station_id")
        df['station_id'] = df.iloc[:, station_col_index].astype(str).str.strip()
        df['station_id'] = df['station_id'].astype(str)

        raw_sites = df['station_id'].nunique()
        print(f"  Sites (raw): {raw_sites}")
        
        # Track unique raw sites
        all_unique_raw_sites.update(df['station_id'].unique())

        monthly_list = []

        for month in range(1,13):
            # Select columns by NAME for all years (incl. 2014). The annual TD
            # files use Japanese-fiscal-year column ORDER (Apr->Mar) but the
            # column NAMES are consistent across years, so name lookup is robust.
            # (A previous hardcoded-index special case for 2014 was off by one and
            #  made March read April's hourly-max column instead of the monthly
            #  mean -- corrupting Japan 2014. Name-based selection avoids this.)
            m_char = FW[str(month)] if month < 10 else str(month)
            valid_col = f'有効測定日数(日)_{m_char}月'
            mean_col  = f'月平均値(ppm)_{m_char}月'
            if valid_col not in df.columns or mean_col not in df.columns:
                continue

            sub = df[['station_id', '都道府県_ローマ字', '市区町村名_ローマ字', valid_col, mean_col]].copy()
            sub.columns = ['station_id', 'province', 'county', 'valid_days', 'mean_ppm']

            sub['year'] = year
            sub['month'] = month

            sub['valid_days'] = pd.to_numeric(sub['valid_days'], errors='coerce')
            sub['mean_ppm'] = pd.to_numeric(sub['mean_ppm'], errors='coerce')

            dim = calendar.monthrange(year, month)[1]
            sub = sub[sub['valid_days'] >= 0.75 * dim]
            if sub.empty:
                continue

            # Convert from ppm to ppb with higher decimal precision
            sub['no2_ppb'] = pd.to_numeric(sub['mean_ppm'], errors='coerce').astype('float64') * 1000.0
            sub['country'] = 'Japan'
            monthly_list.append(sub)

        if not monthly_list:
            print(f"  No valid monthly data for {year}")
            continue

        year_df = pd.concat(monthly_list, ignore_index=True)
        # Ensure string key for merge
        year_df['station_id'] = year_df['station_id'].astype(str)

        # Merge in metadata
        merged = year_df.merge(site_meta, on='station_id', how='left')
        missing_loc = merged[['lon','lat']].isna().any(axis=1).sum()
        if missing_loc:
            print(f"  Dropping {missing_loc} rows missing lon/lat metadata")
            merged = merged.dropna(subset=['lon','lat'])

        filt_sites = merged['station_id'].nunique()
        avg_no2    = merged['no2_ppb'].mean()
        excluded_sites = raw_sites - filt_sites
        
        # Track unique filtered sites
        all_unique_filtered_sites.update(merged['station_id'].unique())
        
        print(f"  Sites after month filter: {filt_sites}")
        print(f"  Sites excluded: {excluded_sites} ({excluded_sites/raw_sites*100:.1f}%)")
        print(f"  Avg NO2: {avg_no2:.2f} ppb")
        
        # Accumulate totals
        total_raw_sites += raw_sites
        total_filtered_sites += filt_sites
        total_excluded_sites += excluded_sites
        year_stats.append({
            'year': year,
            'raw_sites': raw_sites,
            'filtered_sites': filt_sites,
            'excluded_sites': excluded_sites
        })

        # Save per-year CSV
        out_cols = ['year', 'month', 'station_id', 'county','province','country','lat','lon','no2_ppb']
        out_file = out_dir / f'japan_monthly_{year}.csv'
        merged.to_csv(out_file, columns=out_cols, index=False, float_format='%.3f')
        print(f"  Saved {len(merged)} records to {out_file}")

        all_years.append(merged)

    # Combine all years
    if all_years:
        combined = pd.concat(all_years, ignore_index=True)
        combo_file = out_dir / 'JP_monthly_2005_2023_combined.csv'
        combined.to_csv(combo_file, columns=out_cols, index=False, float_format='%.3f')
        print(f"\nCombined file: {combo_file} ({len(combined)} rows)")
        print("Annual mean NO2 (ppb):")
        print(combined.groupby('year')['no2_ppb'].mean().to_string())
        
        # Print summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS - SITE EXCLUSION")
        print("="*70)
        print(f"Total raw sites (sum across years):      {total_raw_sites:,}")
        print(f"Total filtered sites (sum across years): {total_filtered_sites:,}")
        print(f"Total excluded sites (sum):              {total_excluded_sites:,}")
        print(f"Overall exclusion rate (by sum):         {total_excluded_sites/total_raw_sites*100:.1f}%")
        
        # Calculate UNIQUE sites across all years
        unique_raw_count = len(all_unique_raw_sites)
        unique_filtered_count = len(all_unique_filtered_sites)
        unique_excluded_count = unique_raw_count - unique_filtered_count
        unique_exclusion_rate = (unique_excluded_count / unique_raw_count * 100) if unique_raw_count > 0 else 0
        
        print(f"\nUNIQUE SITES (across all years):")
        print(f"  Unique raw sites (before filtering):        {unique_raw_count:,}")
        print(f"  Unique filtered sites (after filtering):    {unique_filtered_count:,}")
        print(f"  Unique EXCLUDED sites:                      {unique_excluded_count:,}")
        print(f"  Unique exclusion rate:                      {unique_exclusion_rate:.1f}%")
        
        print("\nYearly breakdown:")
        stats_df = pd.DataFrame(year_stats)
        stats_df['exclusion_rate_%'] = (stats_df['excluded_sites'] / stats_df['raw_sites'] * 100).round(1)
        print(stats_df.to_string(index=False))
    else:
        print("No Japan data processed.")

    duration = time.time() - start_time
    print(f"\nDone in {duration:.1f}s")
