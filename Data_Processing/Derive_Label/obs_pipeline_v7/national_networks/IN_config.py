import pandas as pd
import calendar


InDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/india/'

# Track all unique sites (for calculating unique exclusions)
all_unique_raw_sites = set()
all_unique_filtered_sites = set()

# 1) load your flat ppb‐CSV
df = pd.read_csv(
    InDir + 'cpcb_india_no2_2016_2023.csv',
    dtype={"local_datetime": str},  # read as text
    parse_dates=[]
)
df["local_datetime"] = pd.to_datetime(
    df["local_datetime"].str.slice(0, 19),
    format="%Y-%m-%dT%H:%M:%S",
    errors="raise"   # or "coerce" if you want to drop bad strings
)
# derive date, year, month
df["date"]  = df["local_datetime"].dt.date
df["year"]  = pd.to_datetime(df["date"]).dt.year.astype(int)
df["month"] = pd.to_datetime(df["date"]).dt.month.astype(int)

# --- value & year sanity filter (raw CPCB file contains sentinel fills
#     (-10000), zeros, impossible highs up to 1e26, and out-of-range years
#     incl. 2025). Drop these BEFORE any aggregation. ---
df = df[(df["no2_ppb"] > 0) & (df["no2_ppb"] < 500)]
df = df[(df["year"] >= 2016) & (df["year"] <= 2023)]

# 2) count unique sensors per year before any filtering
sites_before = (
    df.groupby("year")["sensor_id"]
    .nunique()
    .rename("sites_before")
)
print("\nSites before filtering by year:")
print(sites_before.to_string())

# Track all unique raw sites
all_unique_raw_sites.update(df["sensor_id"].unique())

# 3) daily aggregation: count & mean
daily = (
    df.groupby(["sensor_id","year","month","date","latitude", "longitude"])["no2_ppb"]
    .agg(count="count", daily_mean="mean")
    .reset_index()
)

# 4) keep days with ≥18 hours
daily_valid = daily[daily["count"] >= 18]

# 5) monthly aggregation on those valid days
monthly = (
    daily_valid.groupby(["sensor_id","year","month","latitude", "longitude"])
                .agg(valid_days=("date","nunique"),
                    monthly_avg=("daily_mean","mean"))
                .reset_index()
)

# 6) compute days in each month and % valid
monthly["days_in_month"] = monthly.apply(
    lambda r: calendar.monthrange(int(r["year"]), int(r["month"]))[1],
    axis=1
)
monthly["valid_frac"] = monthly["valid_days"] / monthly["days_in_month"]

# 7) keep only months with ≥75% valid days
monthly_valid = monthly[monthly["valid_frac"] >= 0.75]

# 8) count unique sensors per year after filtering
sites_after = (
    monthly_valid.groupby("year")["sensor_id"]
                .nunique()
                .rename("sites_after")
)
print("\nSites after filtering by year:")
print(sites_after.to_string())

# Track all unique filtered sites
all_unique_filtered_sites.update(monthly_valid["sensor_id"].unique())

# 9) write out your monthly means
#    (rename monthly_avg -> no2_ppb and add country so the column names match
#     what combine_no2_dataset.py expects: latitude, longitude, year, month,
#     no2_ppb, country)
out = monthly_valid[
    ["sensor_id","latitude", "longitude", "year","month","monthly_avg"]
].rename(columns={"monthly_avg": "no2_ppb"})
out["country"] = "India"
out.to_csv(
    "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/india/"
    "IN_monthly_no2_2016_2023_combined.csv",
    index=False
)

print("\n✔ Saved filtered monthly averages.")

# Print unique site statistics with exclusions
print("\n" + "="*70)
print("UNIQUE SITE STATISTICS - INDIA")
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