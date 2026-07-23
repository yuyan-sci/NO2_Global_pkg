# import os
# import pandas as pd
# from pyproj import Transformer

# # 1) Inputs & outputs
# InDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/new zealand/'
# xlsx  = os.path.join(InDir, "Ambient-air-quality-nitrogen-dioxide-monitoring-2007-2023-monthly-summary.xlsx")
# out_csv = os.path.join(InDir, "NZ_sites_latlon.csv")

# df = pd.read_excel(
#     xlsx,
#     sheet_name="2007-2023 Metadata",
#     header=[9, 10],        # zero-based: 9→row10, 10→row11
#     engine="openpyxl"
# )

# # Flatten the MultiIndex into nice single names:
# df.columns = [
#     f"{lvl0.strip()}_{lvl1.strip()}"
#     for lvl0, lvl1 in df.columns
# ]

# # You’ll now have columns like:
# #   'Site identification_'   (lvl1 is blank)
# #   'NZMG_Easting'           (lvl0='NZMG', lvl1='Easting')
# #   'NZMG_Northing'
# #   'NZTM_Easting'
# #   'NZTM_Northing'
# #   … + all your other fields

# # Rename the ones we care about:
# df = df.rename(columns={
#     "Site Identification_Unnamed: 0_level_1": "site_ID",
#     "Easting_Easting.1":                    "Easting_NZTM",
#     "Northing_Northing.1":                  "Northing_NZTM",
#     "City_Unnamed: 8_level_1":              "City",
# })

# # verify everything is in place
# print("RENAMED COLUMNS:", df.columns.tolist())
# # Reproject NZTM → WGS84:
# transformer = Transformer.from_crs(2193, 4326, always_xy=True)
# df["lon"], df["lat"] = transformer.transform(
#     df["Easting_NZTM"], df["Northing_NZTM"]
# )

# # Subset & write:
# clean = df[["site_ID","lat","lon","City"]].copy()
# clean["Country"] = "New Zealand"
# clean.to_csv(InDir + "NZ_sites_latlon.csv", index=False)

# import os
# import re
# import pandas as pd
# from datetime import datetime

# # ── 1) Paths ─────────────────────────────────────────────────────
# monthly_xlsx = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/new zealand/' \
#                'Ambient-air-quality-nitrogen-dioxide-monitoring-2007-2023-monthly-summary.xlsx'
# sites_csv    = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/new zealand/NZ_sites_latlon.csv'
# out_csv      = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/new_zealand/' \
#                'NZ_monthly_no2_2007-2023_combined_code.csv'

# # ─── Load site metadata (with lat/lon, City, Country) ────────────────────
# sites = pd.read_csv(sites_csv, dtype={'site_ID': str})

# # ─── Identify all “Monthly Summary YYYY” sheets ─────────────────────────
# xls = pd.ExcelFile(monthly_xlsx, engine='openpyxl')
# summary_sheets = sorted(
#     [s for s in xls.sheet_names if re.match(r"^Monthly Summary \d{4}$", s)],
#     key=lambda s: int(re.search(r"\d{4}", s).group())
# )

# records = []
# for sheet in summary_sheets:
#     year = int(re.search(r"(\d{4})", sheet).group())
#     print(f"Processing {sheet}…")

#     # 1) Read the sheet with no header so we can detect where the real header is
#     raw = pd.read_excel(monthly_xlsx, sheet_name=sheet, header=None, engine='openpyxl')

#     # 2) Find the row index where one of the cells == "Site ID"
#     is_header = raw.apply(
#         lambda row: row.astype(str).str.strip().eq("Site ID").any(),
#         axis=1
#     )
#     header_idx = is_header.idxmax()

#     # 3) Extract column names and the data below
#     col_names = raw.iloc[header_idx].astype(str).str.strip().tolist()
#     data = raw.iloc[header_idx + 1 :].copy().reset_index(drop=True)
#     data.columns = col_names

#     # 4) Drop any empty or unnamed columns
#     data = data.loc[:, [
#         c for c in data.columns
#         if (isinstance(c, str) and c.strip()) or isinstance(c, (datetime, pd.Timestamp))
#     ]]

#     # 5) Rename "Site ID" → "site_ID"
#     data = data.rename(columns={"Site ID": "site_ID"})

#     # 6) Identify the monthly columns (they’re datetime objects)
#     month_cols = [c for c in data.columns if isinstance(c, (datetime, pd.Timestamp))]

#     # 7) Melt into long format
#     df_long = data.melt(
#         id_vars=["site_ID"],
#         value_vars=month_cols,
#         var_name="month_ts",
#         value_name="NO2"
#     )
#     df_long["year"] = year

#     # 8) Convert the timestamp to a month integer
#     df_long["month"] = pd.to_datetime(df_long["month_ts"]).dt.month

#     # 9) Merge in lat/lon/City/Country
#     merged = df_long.merge(sites, on="site_ID", how="left")
#     records.append(merged)

# # ─── Concatenate all years and write out ────────────────────────────────
# if not records:
#     raise RuntimeError("No data collected – check your sheet names & header detection!")

# combined = pd.concat(records, ignore_index=True)
# combined = combined[[
#     "site_ID", "year", "month", "NO2",
#     "lat", "lon", "City", "Country"
# ]]
# combined.to_csv(out_csv, index=False)
# print(f"Wrote {len(combined)} rows to {out_csv}")

import os, re, glob
import pandas as pd

nz_csv = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/new_zealand/NZ_monthly_no2_2007-2023_combined.csv'

monthly_file = pd.read_csv(
    nz_csv,
    dtype={'date': str}
)
monthly_file = monthly_file.dropna(subset=['NO2'])

# Track raw sites per year BEFORE any filtering
raw_sites_per_year = (
    monthly_file
      .groupby('year')['site_ID']
      .nunique()
      .reset_index(name='raw_sites')
)

# 1) Identify valid years per site: months_count >= 9
yearly = (
    monthly_file
      .groupby(['site_ID','lat','lon','year','City', 'Country'])
      .agg(months_count=('month','nunique'))
      .reset_index()
)
valid_years = yearly[yearly['months_count'] >= 2]

# 2) Count how many valid years each site has
site_valid_counts = (
    valid_years
      .groupby(['site_ID','lat','lon'])
      .size()
      .reset_index(name='valid_years')
)

# Keep all sites (>=0): intentional coverage policy for this sparse network.
good_sites = site_valid_counts[site_valid_counts['valid_years'] >= 0]
print("Sites kept (valid_years >= 0):\n", good_sites[['site_ID','valid_years']])

# 4) (Optional) Filter the monthly table to just those sites
filtered_monthly = monthly_file[monthly_file.set_index(['site_ID','lat','lon']).index.isin(
    good_sites.set_index(['site_ID','lat','lon']).index
)].copy()

# source is µg/m3; convert to ppb and name the column `no2_ppb` so it matches
# what combine_no2_dataset.py expects (it reads val_col='no2_ppb' for NZ).
filtered_monthly['no2_ppb'] = filtered_monthly['NO2'] * 24.45 / 46
filtered_monthly = filtered_monthly[filtered_monthly['no2_ppb'] > 0]

sites_per_year = (
    filtered_monthly
      .groupby('year')['site_ID']
      .nunique()
      .reset_index(name='num_sites')
)

# Merge raw and filtered site counts for exclusion statistics
exclusion_stats = raw_sites_per_year.merge(sites_per_year, on='year', how='outer').fillna(0)
exclusion_stats['excluded_sites'] = exclusion_stats['raw_sites'] - exclusion_stats['num_sites']
exclusion_stats['exclusion_rate_%'] = (exclusion_stats['excluded_sites'] / exclusion_stats['raw_sites'] * 100).fillna(0)

# Print exclusion statistics
print("\n" + "="*70)
print("SITE EXCLUSION STATISTICS - NEW ZEALAND")
print("="*70)
print("\nStations by year (raw → filtered):")
for _, row in exclusion_stats.iterrows():
    year = int(row['year'])
    raw = int(row['raw_sites'])
    filtered = int(row['num_sites'])
    excluded = int(row['excluded_sites'])
    rate = row['exclusion_rate_%']
    print(f"  {year}: {raw:3d} → {filtered:3d}  (excluded: {excluded:3d}, {rate:5.1f}%)")

# Summary statistics - UNIQUE sites across all years
total_unique_raw = monthly_file['site_ID'].nunique()
total_unique_filtered = filtered_monthly['site_ID'].nunique()
total_unique_excluded = total_unique_raw - total_unique_filtered
overall_rate = (total_unique_excluded / total_unique_raw * 100) if total_unique_raw > 0 else 0

print(f"\nSummary (Unique Sites Across All Years):")
print(f"  Total unique raw sites:      {total_unique_raw:,}")
print(f"  Total unique filtered sites: {total_unique_filtered:,}")
print(f"  Total excluded sites:        {total_unique_excluded:,}")
print(f"  Overall exclusion rate:      {overall_rate:.1f}%")
print("="*70)

# 5) Save results
OutDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/new_zealand/'
good_sites.to_csv(os.path.join(OutDir, 'NZ_good_sites.csv'), index=False)
filtered_monthly.to_csv(os.path.join(OutDir, 'NZ_monthly_no2_ppb_2007-2023.csv'), index=False)