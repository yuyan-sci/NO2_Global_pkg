import pandas as pd
import os
import glob

# Base paths
COMPILED_DIR = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled"
OPENAQ_FILE = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/openaq/compiled/global_monthly_means_v7.csv"
OUTPUT_FILE = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/combined/combined_global_no2_2005-2023_v7.csv"

# Target columns
TARGET_COLS = ["lat", "lon", "year", "mon", "no2", "country"]

def standardize_and_round(df, lat_col, lon_col, year_col, mon_col, val_col, country_col=None, country_name=None):
    """
    Standardizes DataFrame columns and rounds coordinates.
    """
    rename_map = {
        lat_col: "lat",
        lon_col: "lon",
        year_col: "year",
        mon_col: "mon",
        val_col: "no2"
    }
    if country_col:
        rename_map[country_col] = "country"
    
    df = df.rename(columns=rename_map)
    
    if country_name and "country" not in df.columns:
        df["country"] = country_name
        
    # Ensure columns exist
    for col in TARGET_COLS:
        if col not in df.columns:
            # Fallback for country if still missing
            if col == "country":
                df["country"] = "Unknown"
            else:
                print(f"Warning: Missing column {col}")
                return pd.DataFrame() # Skip invalid

    # Select and copy
    df = df[TARGET_COLS].copy()
    
    # Ensure numeric coordinates
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    
    # Drop invalid coordinates
    df = df.dropna(subset=["lat", "lon"])

    # Round coordinates
    df["lat"] = df["lat"].round(4)
    df["lon"] = df["lon"].round(4)
    
    return df

def load_compiled_datasets():
    dfs = []
    
    # 1. Australia
    print("Processing Australia...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "australia", "AU_monthly_no2_2005_2023_combined.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_name="Australia"))
    except Exception as e: print(f"Error AU: {e}")

    # 2. Brazil
    print("Processing Brazil...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "brazil", "Brazil_monthly_no2_2015-2022.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_col="country"))
    except Exception as e: print(f"Error BR: {e}")

    # 3. Canada
    print("Processing Canada...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "canada", "CA_monthly_no2_2005-2023_combined.csv"))
        # Check if Country or country column exists
        c_col = "country" if "country" in df.columns else "Country"
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_col=c_col, country_name="Canada"))
    except Exception as e: print(f"Error CA: {e}")

    # 4. China (CH file)
    print("Processing China...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "china", "CH_monthly_2014-2023_combined.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_col="country"))
    except Exception as e: print(f"Error CH: {e}")

    # 5. Europe
    print("Processing Europe...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "europe", "EU_monthly_no2_2005-2023_combined.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_col="site_country"))
    except Exception as e: print(f"Error EU: {e}")

    # 6. India
    print("Processing India...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "india", "IN_monthly_no2_2016_2023_combined.csv"))
        dfs.append(standardize_and_round(df, "latitude", "longitude", "year", "month", "no2_ppb", country_col="country"))
    except Exception as e: print(f"Error IN: {e}")

    # 7. Japan
    print("Processing Japan...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "japan", "JP_monthly_2005_2023_combined.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_col="country"))
    except Exception as e: print(f"Error JP: {e}")

    # 8. Mexico
    print("Processing Mexico...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "mexico", "Mexico_monthly_no2_2005-2023.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_col="country"))
    except Exception as e: print(f"Error MX: {e}")

    # 9. New Zealand
    print("Processing New Zealand...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "new_zealand", "NZ_monthly_no2_2007-2023.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_col="Country"))
    except Exception as e: print(f"Error NZ: {e}")

    # 10. South Africa
    print("Processing South Africa...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "south_africa", "SA_monthly_no2_2005-2023.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_name="South Africa"))
    except Exception as e: print(f"Error SA: {e}")

    # 11. United States
    print("Processing United States...")
    try:
        df = pd.read_csv(os.path.join(COMPILED_DIR, "usa", "US_monthly_no2_2005-2023_combined.csv"))
        dfs.append(standardize_and_round(df, "lat", "lon", "year", "month", "no2_ppb", country_col="country"))
    except Exception as e: print(f"Error US: {e}")

    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def main():
    # 1. Load and Merge Compiled Data
    print("Loading compiled ground datasets...")
    df_compiled = load_compiled_datasets()
    if df_compiled.empty:
        print("No compiled data loaded.")
        return
    
    print(f"Total compiled records: {len(df_compiled)}")
    
    # 2. Load OpenAQ Data
    print("Loading OpenAQ global dataset...")
    try:
        df_openaq = pd.read_csv(OPENAQ_FILE)
        df_openaq = standardize_and_round(
            df_openaq, 
            "latitude", "longitude", "year", "month", "no2_ppb", "country_name"
        )
        print(f"Total OpenAQ records: {len(df_openaq)}")
    except Exception as e:
        print(f"Error loading OpenAQ: {e}")
        return

    # 3. Filter OpenAQ (Prioritize Compiled)
    # We want to exclude any site (lat, lon) in OpenAQ that already exists in df_compiled
    
    # Create unique list of compiled locations
    print("Identifying overlapping locations...")
    compiled_locs = df_compiled[["lat", "lon"]].drop_duplicates()
    
    # Merge to find overlaps
    # We use a left join on openaq with compiled_locs, filtering out matches
    merged = df_openaq.merge(
        compiled_locs, 
        on=["lat", "lon"], 
        how="left", 
        indicator=True
    )
   
    # Analyze overlaps
    overlaps = merged[merged["_merge"] == "both"]
    if not overlaps.empty:
        print("\nOverlapping Data Analysis:")
        overlap_counts = overlaps.groupby("country")["lat"].count().sort_values(ascending=False)
        print("Overlapping records by country:")
        print(overlap_counts)
        
        # Unique sites overlap
        unique_overlap_sites = overlaps[["lat", "lon", "country"]].drop_duplicates()
        site_counts = unique_overlap_sites["country"].value_counts()
        print("\nOverlapping unique sites by country:")
        print(site_counts)
    
    # Keep only rows that exist in left_only (OpenAQ only)
    df_openaq_filtered = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    
    print(f"OpenAQ records after filtering duplicates: {len(df_openaq_filtered)}")
    print(f"Removed {len(df_openaq) - len(df_openaq_filtered)} overlapping records.")

    # 4. Combine
    print("Combining datasets...")
    final_df = pd.concat([df_compiled, df_openaq_filtered], ignore_index=True)

    # Drop non-positive monthly means: zeros and negatives are sentinel/fill
    # values, not measurements. Combine previously kept these, so ~1084 <=0
    # values reached the training label.
    n_nonpos = int((final_df['no2'] <= 0).sum())
    final_df = final_df[final_df['no2'] > 0].copy()
    print(f"Removed {n_nonpos} records with no2 <= 0 (sentinel/fill).")

    # Pre-filter extreme impossible values (> 500 ppb) before calculating percentile
    initial_count = len(final_df)
    extreme_mask = final_df['no2'] > 500
    if extreme_mask.any():
        n_extreme = extreme_mask.sum()
        print(f"\n--- Pre-filtering Impossible Values (> 500 ppb) ---")
        print(f"Removed {n_extreme} records > 500 ppb before percentile calculation.")
        final_df = final_df[~extreme_mask].copy()

    # Remove extreme outliers based on 99.99th percentile
    print("\n--- Outlier Filtering (99.99th percentile) ---")
    
    # Calculate 99.99th percentile on the full dataset
    no2_values = final_df['no2']
    percentile_9999 = no2_values.quantile(0.9999)
    print(f"Calculated 99.99th percentile threshold: {percentile_9999:.4f} ppb")
    
    initial_count = len(final_df)
    outliers = final_df[final_df['no2'] > percentile_9999]
    
    if not outliers.empty:
        print(f"Total outliers found (> {percentile_9999:.4f} ppb): {len(outliers)}")
        
        # Group by country
        outlier_counts = outliers['country'].value_counts()
        print("\nOutliers by Country:")
        print(outlier_counts)
        
        # Show all outliers sorted by NO2 value
        print("\nAll Extreme Values (sorted):")
        pd.set_option('display.max_rows', None)  # Ensure all rows are printed
        print(outliers[['country', 'year', 'mon', 'no2', 'lat', 'lon']].sort_values('no2', ascending=False).to_string(index=False))
        pd.reset_option('display.max_rows')      # Reset to default
        print("---------------------------------------------------\n")

    # Filter out outliers
    final_df = final_df[final_df['no2'] <= 300].copy()
    print(f"Removed {initial_count - len(final_df)} records (> 300 ppb).")
    
    # Remove data for 2024 and 2025
    initial_count = len(final_df)
    final_df = final_df[~final_df['year'].isin([2024, 2025])].copy()
    print(f"Removed {initial_count - len(final_df)} records for years 2024 and 2025.")

    # --- Temporal-spike QC: flag site-months > 2.5x the mean of the ADJACENT
    #     months (prev/next) at the same site. This catches in-range network
    #     errors (e.g. a wrong-column month inflating a region ~3-4x but staying
    #     under 300 ppb) that the magnitude cut cannot. Flagged rows are logged
    #     and dropped. ---
    print("\n--- Temporal-spike QC (> 2.5x adjacent-month mean) ---")
    RATIO, FLOOR = 2.5, 3.0
    sp = final_df.copy()
    sp['t'] = sp['year'].astype(int) * 12 + sp['mon'].astype(int)
    sp = sp.sort_values(['lat', 'lon', 't'])
    g = sp.groupby(['lat', 'lon'], sort=False)
    prev_t, prev_v = g['t'].shift(1), g['no2'].shift(1)
    next_t, next_v = g['t'].shift(-1), g['no2'].shift(-1)
    pv = prev_v.where(sp['t'] - prev_t == 1)   # only true adjacent months
    nv = next_v.where(next_t - sp['t'] == 1)
    neighbor = pd.concat([pv, nv], axis=1).mean(axis=1)
    spike = (neighbor > FLOOR) & (sp['no2'] > RATIO * neighbor)
    n_spike = int(spike.sum())
    if n_spike:
        log_path = os.path.join(os.path.dirname(OUTPUT_FILE), 'flagged_spikes_v7.csv')
        sp.loc[spike].assign(neighbor_mean=neighbor[spike]).to_csv(log_path, index=False)
        print(f"Flagged & dropped {n_spike} temporal-spike site-months (logged to {log_path}).")
        print(sp.loc[spike, 'country'].value_counts().to_string())
    else:
        print("No temporal spikes found.")
    final_df = sp[~spike].drop(columns='t').copy()

    # Review-only log (NOT dropped): plausibly-high monthly means 150-300 ppb.
    n_hi = int(((final_df['no2'] > 150) & (final_df['no2'] <= 300)).sum())
    print(f"Note: {n_hi} site-months in 150-300 ppb retained (review if unexpected).")

    # Normalize duplicate country labels (national vs OpenAQ English/local names).
    # 'Turkey' (OpenAQ) -> 'Türkiye' (national/EEA). National sites already win via
    # the (lat,lon) dedup above (national is in df_compiled, so any OpenAQ record at
    # a coincident coordinate was dropped); this only unifies the label so Türkiye
    # appears once while OpenAQ-unique Turkish sites are retained.
    COUNTRY_NORMALIZE = {"Turkey": "Türkiye"}
    final_df["country"] = final_df["country"].replace(COUNTRY_NORMALIZE)

    # Verify Countries
    countries = final_df["country"].unique()
    countries.sort()
    print(f"\nFinal dataset contains data for {len(countries)} countries:")
    print(", ".join(str(c) for c in countries))
    print(f"China record count: {len(final_df[final_df['country'] == 'China'])}")
    
    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved {len(final_df)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

