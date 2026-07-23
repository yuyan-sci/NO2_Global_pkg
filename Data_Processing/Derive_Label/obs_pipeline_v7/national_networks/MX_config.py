import os
import glob
import pandas as pd
import calendar
import numpy as np

# --- USER PARAMETERS ---
raw_dir      = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/mexico'
station_file = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/mexico/cat_estacion.csv'
unit_file    = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/mexico/cat_unidades.csv'
out_dir      = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/mexico'
os.makedirs(out_dir, exist_ok=True)
output_csv   = os.path.join(out_dir, 'Mexico_monthly_no2_2005-2023.csv')

# --- 1) Read station metadata ---
stations = pd.read_csv(station_file, header=1, encoding='latin1')
stations.columns = (
    stations.columns
            .str.strip()
            .str.replace('\ufeff','')
            .str.lower()
            .str.replace(' ', '_')
)
stations = stations.rename(columns={
    'cve_estac':   'site',
    'id_estacion': 'site',
    'latitud':     'lat',
    'longitud':    'lon'
})[['site','lat','lon']]

# --- 2) Read unit catalog & build converter ---
units_df = pd.read_csv(unit_file, encoding='latin1')
units_df.columns = units_df.columns.str.strip().str.lower()
unit_map = {
    int(r['id_unidad']): r['clave_unidad'].strip().lower()
    for _,r in units_df.iterrows()
}

def to_ppb(val, u):
    if pd.isna(val) or pd.isna(u): 
        return np.nan
    if u == 'ppb':
        return val
    if u == 'ppm':
        return val * 1000
    if u in ('µg/m³','ug/m3','microgramos_por_metro_cubico'):
        return val * (24.465 / 46.0055)
    return np.nan

# --- 3) Keywords for dynamic detection ---
date_keys  = ['date','fecha']
site_keys  = ['estac','station']
param_keys = ['param','contaminante','pollut']
value_keys = ['value','valor']
unit_keys  = ['unit','unidad','id_unidad','cve_unidad']

daily_frames = []

# Track all unique sites (for calculating unique exclusions)
all_unique_raw_sites = set()
all_unique_filtered_sites = set()

# --- 4) Loop through contaminantes files ---
for fpath in sorted(glob.glob(os.path.join(raw_dir, 'contaminantes_*.csv'))):
    fname = os.path.basename(fpath)
    df = None

    # Try skiprows from 0..10 to locate header row
    for skip in range(0, 11):
        try:
            tmp = pd.read_csv(
                fpath,
                skiprows=skip,
                encoding='latin1',
                dtype={},
                low_memory=False
            )
        except Exception:
            continue

        # clean & normalize column names
        cols = (
            pd.Series(tmp.columns)
              .str.strip()
              .str.replace(r'[\ufeffï»¿]', '', regex=True)
              .str.lower()
              .str.replace(' ', '_')
        )
        tmp.columns = cols

        # helper to pick column by substring
        def pick(keys):
            return next((c for c in cols if any(k == c or k in c for k in keys)), None)

        cd = pick(date_keys)
        cs = pick(site_keys)
        cp = pick(param_keys)
        cv = pick(value_keys)
        cu = pick(unit_keys)

        if None not in (cd, cs, cp, cv, cu):
            df = tmp
            col_date, col_site, col_param, col_val, col_unit = cd, cs, cp, cv, cu
            print(f"\n{fname} → skiprows={skip}, columns={list(cols)}")
            break

    if df is None:
        print(f"⚠️  Couldn’t locate header in {fname}, skipping")
        continue

    # work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # filter to NO2
    df[col_param] = df[col_param].astype(str).str.upper().str.strip()
    df = df[df[col_param] == 'NO2']
    if df.empty:
        print(f" ⚠️ no NO₂ rows in {fname}, skipping")
        continue

    # parse calendar date
    df['date'] = pd.to_datetime(
        df[col_date].astype(str).str.strip(),
        errors='coerce'
    ).dt.date
    df = df.dropna(subset=['date'])
    if df.empty:
        print(f" ⚠️ date parse failed in {fname}, skipping")
        continue

    # unit conversion to ppb
    df['unit_code']    = pd.to_numeric(df[col_unit], errors='coerce').astype('Int64')
    df['clave_unidad'] = df['unit_code'].map(unit_map)
    df['value']        = pd.to_numeric(df[col_val], errors='coerce')
    df['no2_ppb']      = df.apply(lambda r: to_ppb(r['value'], r['clave_unidad']), axis=1)
    df = df.dropna(subset=['no2_ppb'])
    if df.empty:
        print(f" ⚠️ unit→ppb conversion failed in {fname}, skipping")
        continue

    # merge latitude/longitude
    df = df.merge(stations, left_on=col_site, right_on='site', how='left')
    df['country'] = 'Mexico'
    
    # Track unique raw sites (before daily filter)
    all_unique_raw_sites.update(df['site'].dropna().unique())

    # daily aggregation + ≥18 h filter
    daily = (
        df.groupby(['site','date','lat','lon','country'], as_index=False)
          .agg(
              hours=('no2_ppb','count'),
              daily_mean=('no2_ppb','mean')
          )
    )
    daily = daily[daily['hours'] >= 12]
    if daily.empty:
        print(f" ⚠️ no valid days in {fname}, skipping")
        continue

    daily['year']  = pd.DatetimeIndex(daily['date']).year
    daily['month'] = pd.DatetimeIndex(daily['date']).month
    daily_frames.append(daily)

# --- 5) Combine daily frames & count before monthly filter ---
if not daily_frames:
    raise RuntimeError("No daily data collected from any file!")

daily_all    = pd.concat(daily_frames, ignore_index=True)
before_count = daily_all.groupby('year')['site'].nunique()

# --- 6) Monthly aggregation & 75 % filter with coverage diagnostics ---
monthly_all = (
    daily_all
    .groupby(['site','lat','lon','country','year','month'], as_index=False)
    .agg(
        valid_days   = ('daily_mean','count'),
        monthly_mean = ('daily_mean','mean')
    )
)
monthly_all['days_in_month'] = monthly_all.apply(
    lambda r: calendar.monthrange(r['year'], r['month'])[1],
    axis=1
)
monthly_all['coverage'] = monthly_all['valid_days'] / monthly_all['days_in_month']

print("\nMonthly coverage stats by year:")
print(
    monthly_all
      .groupby('year')['coverage']
      .agg(['min','mean','max'])
)

monthly = monthly_all[monthly_all['coverage'] >= 0.3].copy()
after_count = monthly.groupby('year')['site'].nunique()

# Track unique filtered sites
all_unique_filtered_sites.update(monthly['site'].dropna().unique())

# --- 7) Summary & write out ---
print("\nStations by year (before → after):")
for y in sorted(before_count.index):
    print(f"  {y}: {before_count[y]:3d} → {after_count.get(y,0):3d}")

monthly = monthly.rename(columns={'monthly_mean':'no2_ppb'})
final_cols = ['lat','lon','no2_ppb','year','month','site','country']
monthly[final_cols].to_csv(output_csv, index=False)

print(f"\n▶️  Written monthly averages to: {output_csv}")

# Calculate UNIQUE sites across all years
unique_raw_count = len(all_unique_raw_sites)
unique_filtered_count = len(all_unique_filtered_sites)
unique_excluded_count = unique_raw_count - unique_filtered_count
unique_exclusion_rate = (unique_excluded_count / unique_raw_count * 100) if unique_raw_count > 0 else 0

print("\n" + "="*70)
print("UNIQUE SITES (across all years):")
print("="*70)
print(f"  Unique raw sites (before filtering):        {unique_raw_count:,}")
print(f"  Unique filtered sites (after filtering):    {unique_filtered_count:,}")
print(f"  Unique EXCLUDED sites:                      {unique_excluded_count:,}")
print(f"  Unique exclusion rate:                      {unique_exclusion_rate:.1f}%")
print("="*70)
