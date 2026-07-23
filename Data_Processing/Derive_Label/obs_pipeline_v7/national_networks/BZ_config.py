import os
import glob
import unicodedata
import pandas as pd
import calendar
import numpy as np
from difflib import get_close_matches

# --- USER PARAMETERS ---
InDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/brazil'
site_info_file = os.path.join(InDir, 'Annual air quality assessment map_data.csv')
state_list = ['BA','CE','ES','MG','PE','PR','RJ','RS','SP']
OutDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/brazil'
os.makedirs(OutDir, exist_ok=True)
output_csv = os.path.join(OutDir, 'Brazil_monthly_no2_2005-2013.csv')

# --- normalize function (strip BOM, accents, casefold) ---
def normalize_text(s):
    if not isinstance(s, str):
        return ''
    s = unicodedata.normalize('NFC', s)
    for ch in ('\ufeff','\u200b','\r','\n','\t'):
        s = s.replace(ch,'')
    s = s.strip()
    # remove accents
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s.casefold()

# --- load metadata and pre-normalize station names ---
meta = pd.read_csv(site_info_file, encoding='utf-16le', sep='\t')
meta.columns = [c.replace('\ufeff','').strip() for c in meta.columns]
meta['Est_norm'] = meta['Estacao1'].map(normalize_text)

# Track all unique sites (for calculating unique exclusions)
all_unique_raw_sites = set()
all_unique_filtered_sites = set()

daily_frames = []

for state in state_list:
    for fpath in sorted(glob.glob(os.path.join(InDir, state, '*.csv'))):
        fname = os.path.basename(fpath)
        # read & clean columns
        df = pd.read_csv(fpath, encoding='latin1', sep=',', engine='python')
        df.columns = [c.replace('\ufeff','').replace('ï»¿','').strip() for c in df.columns]

        # filter NO2
        df['Poluente'] = df['Poluente'].astype(str).str.strip()
        df = df[df['Poluente'].str.upper()=='NO2']
        if df.empty:
            continue

        # parse date
        s = (df['Data'].astype(str)
               .str.replace(r'^[\ufeff\r\n]+|[\r\n\s]+$', '', regex=True)
               .str.replace('/', '-', regex=False)
               .str.strip()
            )
        df['date'] = pd.to_datetime(s, errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        if df.empty:
            continue

        # convert units -> ppb. Normalize the unit string first so all µg/m3
        # spellings match: casefold, unify micro sign (µ U+00B5 vs μ U+03BC) and
        # superscript ³ -> 3. (Previously 'µg/m³' with the superscript was NOT in
        # the match list, so those rows became NaN and were silently dropped.)
        unit = (df['Unidade'].astype(str).str.strip().str.lower()
                .str.replace('μ', 'µ', regex=False)   # greek mu -> micro
                .str.replace('³', '3', regex=False))        # superscript 3 -> 3
        valor = pd.to_numeric(df['Valor'], errors='coerce')
        df['no2_ppb'] = np.where(unit == 'ppb', valor,
                        np.where(unit == 'ppm', valor * 1000,
                        np.where(unit.isin(['µg/m3', 'ug/m3']), valor * (24.45 / 46), np.nan)))
        df = df.dropna(subset=['no2_ppb'])
        # value sanity cap: raw Valor contains impossible spikes (up to 2.5e12).
        df = df[(df['no2_ppb'] > 0) & (df['no2_ppb'] < 500)]
        if df.empty:
            continue

        # normalize station name
        raw_station = df['Estacao'].iloc[0]
        norm_station = normalize_text(raw_station)

        # try exact match
        candidates = meta.loc[meta['Estado1']==state, 'Est_norm']
        match_meta = meta[
            (meta['Estado1']==state) & (meta['Est_norm']==norm_station)
        ]

        # if no exact, try fuzzy
        if match_meta.empty:
            choices = candidates.unique().tolist()
            close = get_close_matches(norm_station, choices, n=1, cutoff=0.7)
            if close:
                print(f"ℹ️  Fuzzy-matched '{raw_station}' → '{close[0]}'")
                match_meta = meta[
                    (meta['Estado1']==state) & (meta['Est_norm']==close[0])
                ]
            else:
                print(f"⚠️  No metadata for {state} / '{raw_station}' (norm='{norm_station}'), skipping")
                print("    available:", choices[:10], "…")
                continue

        lat, lon = match_meta['Latitude'].iat[0], match_meta['Longitude'].iat[0]

        # annotate
        df['site']    = raw_station
        df['state']   = state
        df['lat']     = lat
        df['lon']     = lon
        df['country'] = 'Brazil'
        
        # Track unique raw site
        all_unique_raw_sites.add(raw_station)

        # daily aggregate + filter
        daily = (df.groupby(['site','date','lat','lon','state','country'], as_index=False)
                   .agg(hours=('no2_ppb','count'), daily_mean=('no2_ppb','mean')))
        daily = daily[daily['hours']>=18]
        if daily.empty:
            continue

        daily['year']  = pd.DatetimeIndex(daily['date']).year
        daily['month'] = pd.DatetimeIndex(daily['date']).month
        daily_frames.append(daily)

# combine daily
daily_all = pd.concat(daily_frames, ignore_index=True)
before = daily_all.groupby('year')['site'].nunique()

# monthly aggregate + 75% filter
monthly = (daily_all.groupby(['site','lat','lon','state','country','year','month'], as_index=False)
               .agg(valid_days=('daily_mean','count'), monthly_mean=('daily_mean','mean')))
monthly['days_in_month'] = monthly.apply(
    lambda r: calendar.monthrange(r['year'], r['month'])[1], axis=1)
monthly = monthly[monthly['valid_days']>=0.75*monthly['days_in_month']].copy()
after = monthly.groupby('year')['site'].nunique()

# Track unique filtered sites
all_unique_filtered_sites.update(monthly['site'].unique())

# summary
print("\nStations by year (before → after):")
for y in sorted(before.index):
    print(f" {y}: {before[y]} → {after.get(y,0)}")

# write
monthly = monthly.rename(columns={'monthly_mean':'no2_ppb'})
monthly[['lat','lon','no2_ppb','year','month','state','country']]\
       .to_csv(output_csv, index=False)
print("Written:", output_csv)

# Print unique site statistics with exclusions
print("\n" + "="*70)
print("UNIQUE SITE STATISTICS - BRAZIL")
print("="*70)
unique_raw_count = len(all_unique_raw_sites)
unique_filtered_count = len(all_unique_filtered_sites)
unique_excluded_count = unique_raw_count - unique_filtered_count
unique_exclusion_rate = (unique_excluded_count / unique_raw_count * 100) if unique_raw_count > 0 else 0

print(f"Unique raw sites (before filtering):        {unique_raw_count:,}")
print(f"Unique filtered sites (after filtering):    {unique_filtered_count:,}")
print(f"Unique EXCLUDED sites:                      {unique_excluded_count:,}")
print(f"Unique exclusion rate:                      {unique_exclusion_rate:.1f}%")
print("="*70)
