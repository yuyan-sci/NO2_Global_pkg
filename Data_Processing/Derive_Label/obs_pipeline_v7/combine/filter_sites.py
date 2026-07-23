import pandas as pd
import os

# Paths
COMBINED_FILE = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/combined/combined_global_no2_2005-2023_v7.csv"
US_SITES_FILE = "/path/to/NO2_DL_global/aws/no2_ground/sites_info/us_sites.csv"
EU_SITES_FILE = "/path/to/NO2_DL_global/aws/no2_ground/sites_info/europe_sites.csv"
OUTPUT_FILE = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/combined/combined_global_no2_2005-2023_v7_filtered.csv"
EXCLUDED_SITES_FILE = "/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/combined/excluded_sites_v7.csv"

def main():
    print(f"Loading combined data from {COMBINED_FILE}...")
    try:
        df = pd.read_csv(COMBINED_FILE)
    except FileNotFoundError:
        print(f"Error: File not found at {COMBINED_FILE}")
        return

    print(f"Original record count: {len(df)}")
    
    # Ensure lat/lon are rounded to 4 decimals for consistent matching
    # The combined file should already be rounded, but we ensure it here.
    df["lat_rnd"] = pd.to_numeric(df["lat"], errors="coerce").round(4)
    df["lon_rnd"] = pd.to_numeric(df["lon"], errors="coerce").round(4)
    
    # 1. Identify US Mobile Sites
    print("Loading US sites...")
    try:
        us_df = pd.read_csv(US_SITES_FILE)
    except FileNotFoundError:
        print(f"Warning: US sites file not found at {US_SITES_FILE}")
        us_df = pd.DataFrame()
    
    # Filter for MOBILE (case insensitive just in case)
    if not us_df.empty and "Land Use" in us_df.columns:
        mobile_us = us_df[us_df["Land Use"].astype(str).str.upper() == "MOBILE"].copy()
    else:
        if not us_df.empty:
            print("Warning: 'Land Use' column not found in US sites.")
        mobile_us = pd.DataFrame()
        
    if not mobile_us.empty:
        mobile_us["lat_rnd"] = pd.to_numeric(mobile_us["Latitude"], errors="coerce").round(4)
        mobile_us["lon_rnd"] = pd.to_numeric(mobile_us["Longitude"], errors="coerce").round(4)
        mobile_us.dropna(subset=["lat_rnd", "lon_rnd"], inplace=True)
        
        # Create a set of (lat, lon) to exclude
        us_exclude_coords = set(zip(mobile_us["lat_rnd"], mobile_us["lon_rnd"]))
        print(f"Found {len(us_exclude_coords)} unique US MOBILE site coordinates.")
    else:
        us_exclude_coords = set()
        print("No US MOBILE sites found.")

    # 2. Identify EU Traffic Sites
    print("Loading EU sites...")
    try:
        eu_df = pd.read_csv(EU_SITES_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"Warning: EU sites file not found at {EU_SITES_FILE}")
        eu_df = pd.DataFrame()
    
    # Filter for Traffic
    if not eu_df.empty and "Air Quality Station Type" in eu_df.columns:
        # The user specified "Traffic", the data has "traffic". We use case-insensitive match.
        traffic_eu = eu_df[eu_df["Air Quality Station Type"].astype(str).str.lower() == "traffic"].copy()
    else:
        if not eu_df.empty:
            print("Warning: 'Air Quality Station Type' column not found in EU sites.")
        traffic_eu = pd.DataFrame()
        
    if not traffic_eu.empty:
        traffic_eu["lat_rnd"] = pd.to_numeric(traffic_eu["Latitude"], errors="coerce").round(4)
        traffic_eu["lon_rnd"] = pd.to_numeric(traffic_eu["Longitude"], errors="coerce").round(4)
        traffic_eu.dropna(subset=["lat_rnd", "lon_rnd"], inplace=True)
        
        eu_exclude_coords = set(zip(traffic_eu["lat_rnd"], traffic_eu["lon_rnd"]))
        print(f"Found {len(eu_exclude_coords)} unique EU Traffic site coordinates.")
    else:
        eu_exclude_coords = set()
        print("No EU Traffic sites found.")
        
    # 3. Apply Filter
    print("Filtering combined dataset...")
    
    # Remove extreme outliers
    initial_count = len(df)
    df = df[df['no2'] <= 300].copy()
    print(f"Removed {initial_count - len(df)} records with NO2 > 300 ppb.")
    
    # Remove data for 2024 and 2025
    initial_count = len(df)
    df = df[~df['year'].isin([2024, 2025])].copy()
    print(f"Removed {initial_count - len(df)} records for years 2024 and 2025.")
    
    # Combine all exclusions
    all_exclusions = us_exclude_coords.union(eu_exclude_coords)
    
    if not all_exclusions:
        print("No sites to exclude.")
        df.drop(columns=["lat_rnd", "lon_rnd"], inplace=True)
        df.to_csv(OUTPUT_FILE, index=False)
        return

    # Identify rows to keep
    # We check if (lat, lon) is in the exclusion set
    print("Matching coordinates...")
    
    # Using a list comprehension with zip is efficient
    mask = [ (lat, lon) in all_exclusions for lat, lon in zip(df["lat_rnd"], df["lon_rnd"]) ]
    
    # Select rows that are NOT in the mask
    df_filtered = df[~pd.Series(mask, index=df.index)].copy()
    
    excluded_count = sum(mask)
    print(f"Excluded {excluded_count} records matching filtered sites.")
    print(f"Remaining record count: {len(df_filtered)}")
    
    # Count unique sites excluded from the actual data
    removed_df = df[pd.Series(mask, index=df.index)].copy()
    
    # Get unique sites with metadata (lat, lon, country)
    # We use the original lat/lon (first occurrence) for the output
    unique_removed_sites = removed_df.drop_duplicates(subset=["lat_rnd", "lon_rnd"])[["lat", "lon", "country"]]
    
    print(f"Unique sites actually removed from dataset: {len(unique_removed_sites)}")
    
    # Save excluded sites
    if not unique_removed_sites.empty:
        print(f"Saving excluded sites to {EXCLUDED_SITES_FILE}...")
        unique_removed_sites.to_csv(EXCLUDED_SITES_FILE, index=False)
    
    # 4. Save
    # Drop the temporary rounding columns
    df_filtered.drop(columns=["lat_rnd", "lon_rnd"], inplace=True)
    
    print(f"Saving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_filtered.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

