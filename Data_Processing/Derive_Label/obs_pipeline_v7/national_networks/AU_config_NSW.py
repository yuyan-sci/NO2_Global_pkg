import pandas as pd
import numpy as np
import os

# --- inputs ---
TABLE_PATH = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/australia/NSW/tmp_table_24857_1756828451.csv"
SITES_INFO_PATH = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/australia/NSW/air-quality-api-excel-power-query.xlsx"
SITES_SHEET_NAME = None   # set if you know it
OUT_PATH = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/australia/NSW/no2_processed_ppb.csv"

def find_col(df, names):
    m = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        c = m.get(n.lower())
        if c:
            return c
    return None

def find_date_col(df):
    # try common names
    c = find_col(df, ["date", "datetime", "time", "timestamp", "date_local", "date_gmt", "obs_date", "date_time"])
    if c: return c
    # heuristic: first column where >=80% parse as datetime
    for col in df.columns:
        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().mean() > 0.8:
            return col
    return None

def find_site_col(df):
    return find_col(df, ["Column1.SiteName"])

def clean_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

def load_sites_info(path, sheet=None):
    if not os.path.exists(path):
        return None
    xl = pd.ExcelFile(path)
    sheets = [sheet] if sheet else xl.sheet_names
    print('sheet in site_info:', sheets)
    dfi = xl.parse('SiteDetails')
    site_c = find_site_col(dfi)
    lat_c  = find_col(dfi, ["Column1.Latitude"])
    lon_c  = find_col(dfi, ["Column1.Longitude"])
    if site_c and lat_c and lon_c:
        out = dfi[[site_c, lat_c, lon_c]].copy()
        out.columns = ["site", "lat", "lon"]
        out["site"] = out["site"].astype(str).str.strip()
        out["lat"] = clean_num(out["lat"])
        out["lon"] = clean_num(out["lon"])
        return out.dropna(subset=["site"]).drop_duplicates("site")
    return None

# --- process wide table ---
df = pd.read_csv(TABLE_PATH, skiprows=2)

# 1) find date column
date_col = find_date_col(df)
if date_col is None:
    raise ValueError("Couldn't detect the date/time column in the table.")

# 2) melt wide -> long
id_vars = [date_col]
# treat all other columns as site columns (exclude fully empty cols)
value_vars = [c for c in df.columns if c not in id_vars]
# if there are non-site metadata columns, you can explicitly drop them here
# e.g., value_vars = [c for c in value_vars if c not in ["Units", "Notes"]]
long_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="site", value_name="value")

# 3) clean site names, parse date, derive year/mon

long_df["site"] = (
    long_df["site"]
    .astype(str)
    .str.replace(r"\s*NO2 monthly average \[pphm\]", "", regex=True)
    .str.strip()
)
dt = pd.to_datetime(long_df[date_col], errors="coerce")
long_df["year"] = dt.dt.year
long_df["mon"]  = dt.dt.month

# 4) numeric values and unit conversion to ppb (×10)
long_df["value_ppb"] = clean_num(long_df["value"]) * 10.0

# 5) filter: keep rows with proper date and numeric values
out = long_df.dropna(subset=["year", "mon", "value_ppb"]).loc[:, ["year", "mon", "site", "value_ppb"]]

# 6) join lat/lon from sites_info
sites_info = load_sites_info(SITES_INFO_PATH, sheet=SITES_SHEET_NAME)
if sites_info is not None:
    out = out.merge(sites_info, on="site", how="left")

# 7) save
# out.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH} — rows: {len(out)}")

# 8) Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS - NSW AUSTRALIA")
print("="*70)

# Unique sites
total_unique_sites = out['site'].nunique()
print(f"Total unique sites:                      {total_unique_sites:,}")

if "lat" in out.columns:
    sites_with_coords = out[out['lat'].notna()]['site'].nunique()
    sites_missing_coords = total_unique_sites - sites_with_coords
    print(f"Sites with coordinates (lat/lon):        {sites_with_coords:,}")
    print(f"Sites missing coordinates:               {sites_missing_coords:,}")
    print(f"Coordinate coverage:                     {sites_with_coords/total_unique_sites*100:.1f}%")
    
    # Row-level stats
    rows_with_coords = out['lat'].notna().sum()
    rows_missing_coords = out['lat'].isna().sum()
    print(f"\nRow-level statistics:")
    print(f"Records with lat/lon:                    {rows_with_coords:,}")
    print(f"Records missing lat/lon:                 {rows_missing_coords:,}")

# Year coverage
if 'year' in out.columns:
    year_range = f"{int(out['year'].min())} - {int(out['year'].max())}"
    years_covered = out['year'].nunique()
    print(f"\nTemporal coverage:")
    print(f"Year range:                              {year_range}")
    print(f"Number of years:                         {years_covered}")
    
    # Sites per year
    sites_per_year = out.groupby('year')['site'].nunique().reset_index()
    print(f"\nSites per year:")
    for _, row in sites_per_year.iterrows():
        print(f"  {int(row['year'])}: {row['site']:3d} sites")

print("="*70)
