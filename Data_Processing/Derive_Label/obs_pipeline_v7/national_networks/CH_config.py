import pandas as pd
import numpy as np
import calendar
import glob
import time
from pathlib import Path

# --- START TIMER ---
start_time = time.time()

# --- CONFIG ---
InDir  = Path("/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/china")
OutDir = Path("/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled")
OutDir.mkdir(parents=True, exist_ok=True)
years = range(2014, 2024)

# Read metadata
meta_dir = InDir / "全国空气质量" / "_站点列表"
xlsx_path = meta_dir / "站点列表-2022.02.13起.xlsx"
csv_path  = meta_dir / "站点列表-2022.02.13起.csv"
try:
    sites_info = pd.read_excel(xlsx_path, dtype={"监测点编码": str})
except Exception:
    sites_info = pd.read_csv(csv_path, dtype={"监测点编码": str}, encoding='utf-8')
# Standardize site_code key
sites_info = (
    sites_info
    .rename(columns={"监测点编码": "site_code", "城市": "city", "经度": "lon", "纬度": "lat"})
)
sites_info['site_code'] = sites_info['site_code'].str.strip()
sites_info = sites_info.loc[:, ["site_code", "city", "lon", "lat"]]

# Helper: days per month

def days_in_month(year, month):
    return calendar.monthrange(year, month)[1]

results_by_year = {}

# Track statistics across all years
total_raw_sites = 0
total_after_day_filter = 0
total_after_month_filter = 0
total_excluded_sites = 0
year_stats = []

# Track all unique site_codes (for calculating unique exclusions)
all_unique_raw_sites = set()
all_unique_filtered_sites = set()

for year in years:
    print(f"\n=== YEAR: {year} ===")
    folder = f"站点_{year}0513-{year}1231" if year==2014 else f"站点_{year}0101-{year}1231"
    folder_path = InDir / "全国空气质量" / folder
    files = sorted(glob.glob(str(folder_path / "*.csv")))
    if not files:
        print(f"  No CSV files for {year} in {folder_path}")
        continue

    all_melted = []
    for fn in files:
        df = pd.read_csv(fn, encoding='utf-8')
        # Strip col names and map id columns
        df.columns = df.columns.str.strip()
        col_map = {c: 'date' for c in df.columns if c.lower()=='date'}
        col_map.update({c: 'hour' for c in df.columns if c.lower()=='hour'})
        col_map.update({c: 'type' for c in df.columns if c.lower()=='type'})
        df.rename(columns=col_map, inplace=True)
        # Ensure id columns
        if not {'date','hour'}.issubset(df.columns):
            print(f"  Missing date/hour in {fn}, skipping")
            continue
        # Filter NO2
        if 'type' in df.columns:
            df = df[df['type'].astype(str).str.upper()=='NO2']
        # Melt station columns
        exclude = {'date','hour','type'}
        station_cols = [c for c in df.columns if c not in exclude]
        melted = df.melt(id_vars=['date','hour'], value_vars=station_cols,
                         var_name='site_code', value_name='no2')
        # Standardize site_code
        melted['site_code'] = melted['site_code'].str.strip()
        all_melted.append(melted)

    if not all_melted:
        print(f"  No data for {year}")
        continue

    df_year = pd.concat(all_melted, ignore_index=True)
    raw_sites = df_year['site_code'].nunique()
    print(f"  Sites (raw): {raw_sites}")
    
    # Track unique raw site_codes
    all_unique_raw_sites.update(df_year['site_code'].unique())

    # Drop missing no2, parse dates
    df_year = df_year.dropna(subset=['no2'])
    df_year['date'] = pd.to_datetime(df_year['date'].astype(str), format='%Y%m%d')
    df_year['Year']  = year
    df_year['Month'] = df_year['date'].dt.month
    df_year['Day']   = df_year['date'].dt.day

    # Day filter
    daily = df_year.groupby(['site_code','Year','Month','Day'], as_index=False)
    daily = daily.agg(valid_hours=('no2','count'), daily_mean=('no2','mean'))
    daily = daily[daily['valid_hours']>=18]
    sites_after_day = daily['site_code'].nunique()
    print(f"  Sites after day filter: {sites_after_day}")

    # Month filter
    monthly = daily.groupby(['site_code','Year','Month'], as_index=False)
    monthly = monthly.agg(valid_days=('Day','count'), monthly_mean=('daily_mean','mean'))
    monthly['days_in_month'] = monthly.apply(lambda r: days_in_month(r.Year,r.Month), axis=1)
    monthly = monthly[monthly['valid_days']>=0.75*monthly['days_in_month']]

    # Stats
    sites_after_month = monthly['site_code'].nunique()
    excluded_sites = raw_sites - sites_after_month
    print(f"  Sites after month filter: {sites_after_month}")
    print(f"  Sites excluded: {excluded_sites} ({excluded_sites/raw_sites*100:.1f}%)")
    print(f"  Avg NO2: {monthly['monthly_mean'].mean():.2f} ppb")
    
    # Accumulate totals
    total_raw_sites += raw_sites
    total_after_day_filter += sites_after_day
    total_after_month_filter += sites_after_month
    total_excluded_sites += excluded_sites
    year_stats.append({
        'year': year,
        'raw_sites': raw_sites,
        'after_day_filter': sites_after_day,
        'after_month_filter': sites_after_month,
        'excluded_sites': excluded_sites
    })

    # Merge metadata
    monthly = monthly.merge(sites_info, on='site_code', how='left')
    # Drop records missing metadata
    missing_meta = monthly[['lon','lat','city']].isna().any(axis=1).sum()
    if missing_meta>0:
        print(f"  Dropping {missing_meta} records lacking metadata")
        monthly = monthly.dropna(subset=['lon','lat','city'])

    # Track unique filtered site_codes
    all_unique_filtered_sites.update(monthly['site_code'].unique())
    
    # Rename and add country
    monthly.rename(columns={'monthly_mean':'no2_ug/m3'}, inplace=True)
    monthly['no2_ppb'] = monthly['no2_ug/m3'] * 24.45 / 46
    monthly['country'] = 'China'

    # Save
    out_csv = InDir / f"china_monthly_{year}.csv"
    monthly[['lon','lat','city','country','Year','Month','no2_ug/m3', 'no2_ppb']].to_csv(out_csv, index=False)
    print(f"  Saved {len(monthly)} records to {out_csv}")

    results_by_year[year] = monthly

# Combine all
if results_by_year:
    combined = pd.concat(results_by_year.values(), ignore_index=True)
    final = combined[['lon','lat','city','country','Year','Month','no2_ppb','no2_ug/m3']]
    combined_csv = OutDir / "CH_monthly_2014-2023_combined.csv"
    # final.to_csv(combined_csv, index=False)
    print(f"\nCombined: {len(final)} rows to {combined_csv}")
    print("\nAnnual mean NO2:")
    print(final.groupby('Year')['no2_ppb'].mean().to_string())
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS - SITE EXCLUSION")
    print("="*70)
    print(f"Total raw sites (sum across years):           {total_raw_sites:,}")
    print(f"Total after day filter (≥18 hrs):             {total_after_day_filter:,}")
    print(f"Total after month filter (≥75% days):         {total_after_month_filter:,}")
    print(f"Total excluded sites (sum):                   {total_excluded_sites:,}")
    print(f"Overall exclusion rate (by sum):              {total_excluded_sites/total_raw_sites*100:.1f}%")
    
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
    print("\nNo data to combine.")

print(f"\nCompleted in {time.time()-start_time:.1f}s.")