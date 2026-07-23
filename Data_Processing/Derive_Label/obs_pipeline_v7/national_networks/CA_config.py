import pandas as pd
import numpy as np
import calendar
import time

# Start timing
start_time = time.time()

InDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/canada/'
OutDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/'

years_to_process = range(2005, 2024)  # 2005 to 2023

# Dictionary to store results by year
results_by_year = {year: [] for year in years_to_process}

# Dictionary to track unique sites by year
unique_sites_by_year = {year: set() for year in years_to_process}

# Track all unique sites (for calculating unique exclusions)
all_unique_raw_sites = set()
all_unique_filtered_sites = set()

# Function to get days in month accounting for leap years
def get_days_in_month(year):
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

hour_cols = [f"H{h:02d}" for h in range(1,25)]
years = range(2005, 2024)

all_monthly = []  # collect each year's filtered monthly data

for year in years:
    fn = InDir + f"NO2_{year}.csv"
    print(f"\n=== {year} ===")
    # load, skipping header rows if necessary
    df = pd.read_csv(fn, skiprows=7, dtype={c: float for c in hour_cols})
    
    # rename columns to simpler names if they have "//" in them
    df = (df
        .rename(columns=lambda s: s.split("//")[0].strip())
        .assign(
            Date=lambda d: pd.to_datetime(d["Date"], format="%Y-%m-%d"),
            Year=lambda d: d["Date"].dt.year,
            Month=lambda d: d["Date"].dt.month
        ))
    
    # replace missing code (-999) with NaN, then count valid hours
    df[hour_cols] = df[hour_cols].replace(-999, np.nan)
    df["valid_hours"] = df[hour_cols].notna().sum(axis=1)
    
    # before filtering: how many unique sites?
    sites_before = df["NAPS ID"].nunique()
    print("Unique sites (raw):", sites_before)
    
    # Track unique raw sites
    all_unique_raw_sites.update(df["NAPS ID"].unique())
    
    # 1) day‐filter: drop any day with <18 valid hours
    df_day = df[df["valid_hours"] >= 18].copy()
    sites_after_day = df_day["NAPS ID"].nunique()
    print("Sites after day‐filter:", sites_after_day)
    
    # compute daily mean NO₂ (ppb) over the valid hours
    df_day["no2_pb"] = df_day[hour_cols].mean(axis=1, skipna=True)
    
    # 2) month‐filter: require ≥75% of days in that month
    # first, count valid days per site‐month
    month_counts = (
        df_day
        .groupby(["NAPS ID", "City", "Latitude", "Longitude", "Year", "Month"])
        .agg(valid_days=("Date", "nunique"),monthly_mean=("no2_pb", "mean"))
        .reset_index()
    )
    # compute days in each month
    month_counts["days_in_month"] = month_counts.apply(
        lambda r: calendar.monthrange(int(r.Year), int(r.Month))[1],
        axis=1
    )
    # keep only those with valid_days ≥ 0.75 × days_in_month
    month_ok = month_counts[
        month_counts["valid_days"] >= 0.75 * month_counts["days_in_month"]
    ].copy()
    sites_after_month = month_ok["NAPS ID"].nunique()
    print("Sites after month‐filter:", sites_after_month)
    
    # Track unique filtered sites
    all_unique_filtered_sites.update(month_ok["NAPS ID"].unique())
    
    # keep only the columns you wanted and add country column
    monthly_data = month_ok[[
        "Latitude", "Longitude", "City", "Year", "Month", "monthly_mean"
    ]].rename(columns={"Latitude": "lat","Longitude": "lon", "Year": "year", "Month": "month","City": "city", "monthly_mean": "no2_ppb"})
    
    # Add country column with 'Canada' for all rows
    monthly_data['country'] = 'Canada'
    
    all_monthly.append(monthly_data)

# concatenate all years and save
combined = pd.concat(all_monthly, ignore_index=True)
# combined.to_csv(OutDir + 'CA_monthly_no2_2005-2023_combined.csv', index=False)
print(f"\nWrote {len(combined)} site‐months to {OutDir + 'CA_monthly_no2_2005-2023_combined.csv'}")

# compute & display annual mean NO₂ (across all sites)
annual_mean = combined.groupby("year")["no2_ppb"].mean()
print("\nAnnual mean NO₂ (ppb) for all sites:")
print(annual_mean.to_string())

# Print unique site statistics with exclusions
print("\n" + "="*70)
print("UNIQUE SITE STATISTICS - CANADA")
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

# Print execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")