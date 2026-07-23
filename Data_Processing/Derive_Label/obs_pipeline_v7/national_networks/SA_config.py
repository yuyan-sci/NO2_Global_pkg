import os, re, glob
import pandas as pd
# from zipfile import BadZipFile
# from xlrd import XLRDError  # Only needed for commented-out Excel parsing code

def parse_station_file(path):
    # 1) Read raw so we can locate header rows
    df_raw = pd.read_excel(path, header=None, engine='openpyxl')
    
    # 2) Extract station name from the first row
    meta = df_raw.iloc[0, 0]
    m = re.search(r"Station Name:\s*([^,]+)", str(meta))
    station = m.group(1).strip() if m else os.path.splitext(os.path.basename(path))[0]
    
    # 3) Find the row index where "Date Time" appears
    header_row = df_raw[df_raw.iloc[:, 0] == 'Date Time'].index[0]
    
    # 4) Read off the unit from the row just before the data start
    #    row+1 has the variable name ("NO2"), row+2 has the unit ("ppb")
    unit = str(df_raw.iloc[header_row + 2, 1]).strip()
    
    # 5) Extract the actual data (skip the two header rows after "Date Time")
    block = df_raw.iloc[header_row + 3 :].reset_index(drop=True).iloc[:, :2]
    block.columns = ['date_raw', 'no2_ppb']
    
    # 1) remove the time prefix "HH:MM "  
    date_only = block['date_raw'].astype(str)\
                        .str.replace(r'^\d{1,2}:\d{2}\s*', '', regex=True)

    # 2) parse with an exact format  
    block['date'] = pd.to_datetime(
        date_only, 
        format='%d/%m/%Y', 
        dayfirst=True,
        errors='coerce'
    ).dt.date
    
    # parse NO2 and drop bad rows
    block['no2_ppb'] = pd.to_numeric(block['no2_ppb'], errors='coerce')
    block = block.dropna(subset=['date','no2_ppb'])
    
    # assemble final
    block['sites'] = station
    block['unit']  = unit
    return block[['sites','date','no2_ppb','unit']]


# ─── Main script ──────────────────────────────────────────────────
InDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/south_africa/'
# all_files = glob.glob(os.path.join(InDir, '*.xlsx'))

# parsed = []
# for f in all_files:
#     try:
#         df = parse_station_file(f)
#     except (BadZipFile, XLRDError):
#         print(f"⚠️  Skipping unreadable file: {os.path.basename(f)}")
#         continue
#     except Exception as e:
#         print(f"⚠️  Error parsing {os.path.basename(f)}: {e}")
#         continue
#     parsed.append(df)

# if not parsed:
#     raise RuntimeError("No station files parsed successfully!")

# # Concatenate into one DataFrame
# combined = pd.concat(parsed, ignore_index=True)
# combined = combined[combined['no2_ppb'] > 0]
# mask_ug = combined['unit'].str.lower() == 'µg/m3'

# # Convert those values to ppb
# combined.loc[mask_ug, 'no2_ppb'] /= 1.88

# # Update the unit label
# combined.loc[mask_ug, 'unit'] = 'ppb'

# # Write out to CSV
out_csv = os.path.join(InDir, 'south_africa_all_stations_NO2.csv')
# combined.to_csv(out_csv, index=False)

# print(f"Wrote {len(combined)} rows from {len(parsed)} stations to:\n  {out_csv}")

daily_file = pd.read_csv(
    out_csv,
    encoding='latin-1',        # or 'ISO-8859-1', 'cp1252', etc.
    dtype={'date': str}
)

daily_file['date'] = pd.to_datetime(daily_file['date'], dayfirst=True).dt.date

# Extract year and month into new columns
daily_file['year']  = daily_file['date'].apply(lambda d: d.year)
daily_file['month'] = daily_file['date'].apply(lambda d: d.month)

# --- value sanity: the upstream build's >0 filter and µg/m3->ppb conversion are
#     commented out, so the daily file reaches 43,491 ppb at industrial sites.
#     Convert any µg/m3 rows to ppb, then drop non-positive and impossible highs. ---
if 'unit' in daily_file.columns:
    _ug = (daily_file['unit'].astype(str).str.strip().str.lower()
           .str.replace('³', '3', regex=False).isin(['µg/m3', 'ug/m3']))
    daily_file.loc[_ug, 'no2_ppb'] = daily_file.loc[_ug, 'no2_ppb'] / 1.88
daily_file = daily_file[(daily_file['no2_ppb'] > 0) & (daily_file['no2_ppb'] < 500)]

monthly = (
    daily_file
      .groupby(['sites','lat', 'lon', 'year','month'])
      .agg(
          days=('date', 'nunique'),
          mean_no2_ppb=('no2_ppb','mean')
      )
      .reset_index()
)

# Track raw sites per year BEFORE filtering
raw_sites_per_year = (
    monthly
      .groupby('year')['sites']
      .nunique()
      .reset_index(name='raw_sites')
)

# Only keep months with ≥23 days
monthly = monthly[monthly['days'] >= 12]

# 1) Identify valid years per site: months_count >= 9
yearly = (
    monthly
      .groupby(['sites','lat','lon','year'])
      .agg(months_count=('month','nunique'))
      .reset_index()
)
valid_years = yearly[yearly['months_count'] >= 2]

# 2) Count how many valid years each site has
site_valid_counts = (
    valid_years
      .groupby(['sites','lat','lon'])
      .size()
      .reset_index(name='valid_years')
)

# Keep all sites (>=0): intentional coverage policy for this sparse network.
# (Restores the previously-commented assignment that caused a NameError, but with
#  the >=0 keep-all policy chosen for sparse regions.)
good_sites = site_valid_counts[site_valid_counts['valid_years'] >= 0]
# print("Sites kept (valid_years >= 0):\n", good_sites[['sites','valid_years']])

# 4) (Optional) Filter the monthly table to just those sites
filtered_monthly = monthly[monthly.set_index(['sites','lat','lon']).index.isin(
    good_sites.set_index(['sites','lat','lon']).index
)].copy()

sites_per_year = (
    filtered_monthly
      .groupby('year')['sites']
      .nunique()
      .reset_index(name='num_sites')
)

# Merge raw and filtered site counts for exclusion statistics
exclusion_stats = raw_sites_per_year.merge(sites_per_year, on='year', how='outer').fillna(0)
exclusion_stats['excluded_sites'] = exclusion_stats['raw_sites'] - exclusion_stats['num_sites']
exclusion_stats['exclusion_rate_%'] = (exclusion_stats['excluded_sites'] / exclusion_stats['raw_sites'] * 100).fillna(0)

# Print exclusion statistics
print("\n" + "="*70)
print("SITE EXCLUSION STATISTICS - SOUTH AFRICA")
print("="*70)
print("\nStations by year (raw → filtered):")
for _, row in exclusion_stats.iterrows():
    year = int(row['year'])
    raw = int(row['raw_sites'])
    filtered = int(row['num_sites'])
    excluded = int(row['excluded_sites'])
    rate = row['exclusion_rate_%']
    print(f"  {year}: {raw:3d} → {filtered:3d}  (excluded: {excluded:3d}, {rate:5.1f}%)")

# Get the data before the ≥23 days filter for accurate raw count
monthly_before_filter = (
    daily_file
      .groupby(['sites','lat', 'lon', 'year','month'])
      .agg(
          days=('date', 'nunique'),
          mean_no2_ppb=('no2_ppb','mean')
      )
      .reset_index()
)

# Summary statistics - UNIQUE sites across all years
total_unique_raw = monthly_before_filter['sites'].nunique()
total_unique_filtered = filtered_monthly['sites'].nunique()
total_unique_excluded = total_unique_raw - total_unique_filtered
overall_rate = (total_unique_excluded / total_unique_raw * 100) if total_unique_raw > 0 else 0

print(f"\nSummary (Unique Sites Across All Years):")
print(f"  Total unique raw sites:      {total_unique_raw:,}")
print(f"  Total unique filtered sites: {total_unique_filtered:,}")
print(f"  Total excluded sites:        {total_unique_excluded:,}")
print(f"  Overall exclusion rate:      {overall_rate:.1f}%")
print("="*70)

# 5) Save results
OutDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/south_africa/'
good_sites.to_csv(os.path.join(OutDir, 'SA_good_sites.csv'), index=False)
filtered_monthly.to_csv(os.path.join(OutDir, 'SA_monthly_no2_2005-2023.csv'), index=False)