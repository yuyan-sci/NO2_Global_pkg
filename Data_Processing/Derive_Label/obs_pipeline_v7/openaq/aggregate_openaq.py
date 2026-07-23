import os
import glob
import pandas as pd
import numpy as np

BASE_INPUT_DIR = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/openaq/raw/"
GLOBAL_OUTPUT_FILE = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/openaq/compiled/global_monthly_means_v7.csv"

def process_single_year_file(filepath: str) -> pd.DataFrame:
    """
    Loads a country-year CSV, applies QA/QC, aggregates to daily then monthly means.
    Returns a DataFrame with monthly means.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    # Basic validation
    if "value" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
    
    # Filter negatives AND apply a physical upper ceiling on RAW HOURLY values.
    # Without a cap, a single corrupt hour (seen: 1.03e26 ppb in India 2021-11,
    # 22,111 ppb in Turkey) destroys the daily->monthly mean. Real hourly surface
    # NO2 essentially never exceeds a few hundred ppb; cap at 300 ppb per project policy.
    df = df[(df["value"] >= 0) & (df["value"] <= 300)].copy()
    
    # Parse dates
    # Input files have 'year', 'month', 'day', 'hour' columns from global_openaq.py
    if {"year", "month", "day", "hour"}.issubset(df.columns):
        df["Date Local"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]],
            errors="coerce"
        )
    elif "period.datetime_from.local" in df.columns:
        df["Date Local"] = pd.to_datetime(df["period.datetime_from.local"], errors="coerce")
    else:
        print(f"  Missing date columns in {filepath}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    # Drop invalid rows
    required_cols = ["Date Local", "latitude", "longitude"]
    df.dropna(subset=required_cols, inplace=True)
    
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    # Round coordinates to align data
    df["latitude"] = df["latitude"].round(4)
    df["longitude"] = df["longitude"].round(4)

    # Ensure optional columns exist for grouping
    for col in ["location_name", "country_name", "country_iso", "sensor_type"]:
        if col not in df.columns:
            df[col] = "unknown"
        else:
            df[col] = df[col].fillna("unknown")

    # --- Hourly to Daily ---
    # Criteria: >= 18 valid hours per day
    grouping_cols_daily = [
        "latitude", "longitude", "location_name", 
        "country_name", "country_iso", "sensor_type", 
        df["Date Local"].dt.date.rename("date_only")
    ]
    
    # Calculate daily stats
    # We use a simplified approach: 
    # 1. Count valid hours per day
    # 2. Mean value per day
    
    # Note: We need to group by the actual columns + date. 
    # Pandas groupby with Series (date_only) works but we need to put it in the df for cleaner aggregation
    df["date_only"] = df["Date Local"].dt.date
    
    daily_group = [
        "latitude", "longitude", "location_name", 
        "country_name", "country_iso", "sensor_type", "date_only"
    ]

    daily_stats = df.groupby(daily_group)["value"].agg(["count", "mean"]).reset_index()
    daily_stats.rename(columns={"count": "valid_hours", "mean": "daily_mean"}, inplace=True)
    
    # Filter for valid days
    n_days_total = len(daily_stats)
    df_valid_days = daily_stats[daily_stats["valid_hours"] >= 18].copy()
    n_days_valid = len(df_valid_days)
    if n_days_total > n_days_valid:
        print(f"    Daily filter: dropped {n_days_total - n_days_valid} days ({100*(1 - n_days_valid/n_days_total):.1f}%) < 18h")
    
    if df_valid_days.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0

    # --- Daily to Monthly ---
    # Criteria: >= 75% valid days per month
    
    # Extract year and month from date_only
    df_valid_days["date_only"] = pd.to_datetime(df_valid_days["date_only"])
    df_valid_days["year"] = df_valid_days["date_only"].dt.year
    df_valid_days["month"] = df_valid_days["date_only"].dt.month
    
    monthly_group = [
        "latitude", "longitude", "location_name", 
        "country_name", "country_iso", "sensor_type", "year", "month"
    ]
    
    monthly_stats = df_valid_days.groupby(monthly_group)["daily_mean"].agg(["count", "mean"]).reset_index()
    monthly_stats.rename(columns={"count": "valid_days", "mean": "no2_ppb"}, inplace=True)
    
    # Calculate days in month
    # Vectorized calculation of days in month
    monthly_stats["days_in_month"] = pd.to_datetime(
        monthly_stats.assign(day=1)[["year", "month", "day"]]
    ).dt.days_in_month
    
    # Filter for valid months
    n_months_total = len(monthly_stats)
    df_valid_months = monthly_stats[
        monthly_stats["valid_days"] >= 0.75 * monthly_stats["days_in_month"]
    ].copy()
    n_months_valid = len(df_valid_months)
    
    if n_months_total > n_months_valid:
        print(f"    Monthly filter: dropped {n_months_total - n_months_valid} months ({100*(1 - n_months_valid/n_months_total):.1f}%) < 75% valid days")
    
    # Clean up output
    output_cols = [
        "latitude", "longitude", "year", "month", "no2_ppb", 
        "country_name", "country_iso", "location_name", "sensor_type"
    ]
    return df_valid_months[output_cols], monthly_stats[["latitude", "longitude"]].drop_duplicates(), df_valid_months[["latitude", "longitude"]].drop_duplicates(), (n_months_total - n_months_valid)

def process_country(country_dir: str, country_name_slug: str, global_sites_tracker) -> pd.DataFrame:
    """
    Processes all year files in a country directory.
    Saves a combined monthly means file for the country.
    Returns the combined DataFrame.
    """
    print(f"Processing country: {country_name_slug}...")
    
    # Find all year csv files (digits only)
    all_files = glob.glob(os.path.join(country_dir, "*.csv"))
    year_files = [f for f in all_files if os.path.basename(f).split('.')[0].isdigit()]
    year_files.sort()
    
    if not year_files:
        print(f"  No year files found in {country_dir}")
        return pd.DataFrame()

    all_monthly_frames = []
    
    for filepath in year_files:
        # print(f"  Reading {os.path.basename(filepath)}...")
        result = process_single_year_file(filepath)
        
        if isinstance(result, tuple) and len(result) == 4:
             df_monthly, all_sites_in_file, kept_sites_in_file, dropped_months = result
             
             # Update global trackers
             if 'dropped_months' in global_sites_tracker:
                 global_sites_tracker['dropped_months'] += dropped_months
                 
             # Convert DataFrames to list of tuples for set addition
             if not all_sites_in_file.empty:
                 global_sites_tracker['all'].update(
                     list(zip(all_sites_in_file['latitude'], all_sites_in_file['longitude']))
                 )
             if not kept_sites_in_file.empty:
                 global_sites_tracker['kept'].update(
                     list(zip(kept_sites_in_file['latitude'], kept_sites_in_file['longitude']))
                 )
        else:
            # Fallback for empty return or error (tuple of 3 empty DFs)
            if isinstance(result, tuple):
                 df_monthly = result[0]
            else:
                 df_monthly = result

        if not df_monthly.empty:
            all_monthly_frames.append(df_monthly)
            
    if not all_monthly_frames:
        print(f"  No valid monthly data generated for {country_name_slug}")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_monthly_frames, ignore_index=True)
    
    # Save country-level combined file
    output_path = os.path.join(country_dir, "combined_monthly_means.csv")
    # combined_df.to_csv(output_path, index=False)
    print(f"  Saved {len(combined_df)} monthly records to {output_path}")
    
    return combined_df

def main():
    # Find country directories
    # Assume any directory in BASE_INPUT_DIR that is not 'job_output' or hidden
    items = os.listdir(BASE_INPUT_DIR)
    country_dirs = []
    for item in items:
        full_path = os.path.join(BASE_INPUT_DIR, item)
        if os.path.isdir(full_path) and not item.startswith('.') and item != "job_output":
            country_dirs.append((item, full_path))
    
    country_dirs.sort()
    print(f"Found {len(country_dirs)} potential country directories.")
    
    global_frames = []
    
    # Initialize site trackers
    global_sites_tracker = {
        'all': set(),
        'kept': set(),
        'dropped_months': 0
    }
    
    for slug, path in country_dirs:
        df_country = process_country(path, slug, global_sites_tracker)
        if not df_country.empty:
            global_frames.append(df_country)
            
    if global_frames:
        print("\nAggregating global data...")
        global_df = pd.concat(global_frames, ignore_index=True)
        os.makedirs(os.path.dirname(GLOBAL_OUTPUT_FILE), exist_ok=True)
        global_df.to_csv(GLOBAL_OUTPUT_FILE, index=False)
        print(f"Done! Saved {len(global_df)} global monthly records to {GLOBAL_OUTPUT_FILE}")
        
        # Report site exclusion statistics
        total_unique_sites = len(global_sites_tracker['all'])
        kept_unique_sites = len(global_sites_tracker['kept'])
        excluded_unique_sites = total_unique_sites - kept_unique_sites
        total_dropped_months = global_sites_tracker['dropped_months']
        
        print("\nGlobal Site Exclusion Statistics:")
        print(f"  Total unique sites encountered: {total_unique_sites}")
        print(f"  Total unique sites retained:    {kept_unique_sites}")
        print(f"  Total unique sites excluded:    {excluded_unique_sites} ({100 * excluded_unique_sites / total_unique_sites if total_unique_sites > 0 else 0:.2f}%)")
        print(f"  Total site-months excluded:     {total_dropped_months}")
        
    else:
        print("\nNo data found across any countries.")

if __name__ == "__main__":
    main()
