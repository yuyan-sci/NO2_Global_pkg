import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import time

# Start timing
start_time = time.time()

InDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/europe/'
OutDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/europe/'

os.makedirs(OutDir, exist_ok=True)

# Load site information once
print("Loading site information...")
sites_info = pd.read_csv(InDir + 'DataExtract.csv')

# Get all parquet files
path = glob.glob(InDir + '/*.parquet')
print(f"Found {len(path)} parquet files")

# Constants
Mw = 46  # g/mol for NO2
AirMolVol = 24.45  # mol/L at standard condition
years_to_process = range(2005, 2024)  # 2005 to 2023

# Dictionary to store results by year
results_by_year = {year: [] for year in years_to_process}

# Dictionary to track unique sites by year
unique_sites_by_year = {year: set() for year in years_to_process}

# Track all unique sites across all years (for calculating unique exclusions)
all_unique_raw_sites = set()
all_unique_filtered_sites = set()

# Function to get days in month accounting for leap years
def get_days_in_month(year):
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    return [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Process each parquet file
for parquet_file in path:
    try:
        print(f"Processing {os.path.basename(parquet_file)}")
        
        # Read the parquet file
        file = pd.read_parquet(parquet_file)
        
        # Skip if no data or Start column is missing
        if file.empty or 'Start' not in file.columns:
            print(f"No valid data in {os.path.basename(parquet_file)}, skipping")
            continue
        
        # Get site code once
        try:
            site_code = file['Samplingpoint'].unique()[0].split('/')[1]
        except (IndexError, KeyError) as e:
            print(f"Could not extract site code from {os.path.basename(parquet_file)}: {e}")
            continue
            
        # Try to get lat/lon once
        try:
            lat = sites_info.loc[sites_info['Sampling Point Id'] == site_code, 'Latitude'].values[0]
            lon = sites_info.loc[sites_info['Sampling Point Id'] == site_code, 'Longitude'].values[0]
            site_country = sites_info.loc[sites_info['Sampling Point Id'] == site_code, 'Country'].values[0]
            site_city = sites_info.loc[sites_info['Sampling Point Id'] == site_code, 'City'].values[0]
        except:
            print(f"Could not find lat/lon for {site_code}, skipping")
            continue
            
        # Add year column for filtering
        file['year'] = file['Start'].dt.year
        
        # Process each year for this site
        for year in years_to_process:
            # Filter for current year data
            file_year = file[file['year'] == year]
            
            # Skip if no data for this year
            if file_year.empty:
                continue
                
            # Add this site to unique sites for this year
            unique_sites_by_year[year].add(site_code)
            all_unique_raw_sites.add(site_code)
            
            # Reset index for clean processing
            file_year = file_year.reset_index(drop=True)
            
            # Calculate NO2 in ppb
            file_year['no2_ppb'] = file_year['Value'].astype(np.float32) * AirMolVol / Mw
            
            # Add date columns
            file_year['month'] = file_year['Start'].dt.month
            file_year['date'] = file_year['Start'].dt.date
            file_year['hour'] = file_year['Start'].dt.hour
            
            # Get days in month for current year
            days_in_month = get_days_in_month(year)
            
            # Keep only EEA-valid observations and drop negatives.
            # The EEA Validity flag (>=1 = valid; -1/-99 = not valid) is the robust
            # discriminator: invalid records carry fill values that are NOT all
            # negative -- 99999 is a common POSITIVE fill (Validity == -1) that a
            # plain `Value >= 0` filter lets through (-> 99999*24.45/46 ~= 53000 ppb).
            if 'Validity' in file_year.columns:
                data_positive = file_year[(file_year['Validity'] >= 1) & (file_year['Value'] >= 0)]
            else:
                data_positive = file_year[file_year['Value'] >= 0]
            
            # Skip if no positive values
            if data_positive.empty:
                continue
            
            # Count samples per day to check for required hours (at least 18 hours per day)
            samples_per_day = data_positive.groupby('date').size()
            days_with_sufficient_hours = samples_per_day[samples_per_day >= 18].index
            
            # Skip if no days have enough hourly samples
            if len(days_with_sufficient_hours) == 0:
                continue
            
            # Filter for only days with sufficient hourly data
            filtered_data = data_positive[data_positive['date'].isin(days_with_sufficient_hours)]
            
            # Count valid days per month (at least 75% of days in month)
            month_days = filtered_data.groupby('month')['date'].nunique()
            valid_months = []
            
            for month, count in month_days.items():
                if month < 1 or month > 12:  # Skip invalid months
                    continue
                day_filter = days_in_month[month-1] * 0.75
                if count >= day_filter:
                    valid_months.append(month)
            
            # Skip if no months have enough days
            if len(valid_months) == 0:
                continue
            
            # Filter for months with enough sampling days
            final_filtered_data = filtered_data[filtered_data['month'].isin(valid_months)]
            
            # Calculate monthly averages of no2_ppb
            monthly_avg_no2 = final_filtered_data.groupby('month')['no2_ppb'].mean()
            
            # Store results for this site and year
            for month, value in monthly_avg_no2.items():
                results_by_year[year].append({
                    'year': year,
                    'month': month,
                    'site_code': site_code,
                    'country': site_country,
                    'city': site_city,
                    'lat': lat,
                    'lon': lon,
                    'no2_ppb': value
                })
            
            # Track filtered sites
            all_unique_filtered_sites.add(site_code)
            
            print(f"Processed {site_country} {site_city} for year {year} with {len(monthly_avg_no2)} valid months")
            
    except Exception as e:
        print(f"Error processing {os.path.basename(parquet_file)}: {e}")
        continue

# Process results for each year
for year in years_to_process:
    if results_by_year[year]:
        # Create DataFrame for this year
        results_df = pd.DataFrame(results_by_year[year])
        
        # Save to CSV
        csv_filename = f"{InDir}EU_monthly_no2_{year}.csv"
        results_df.to_csv(csv_filename, index=False)
        
        # Calculate statistics
        unique_sites_results = results_df['site_code'].nunique()
        avg_no2_ppb = results_df['no2_ppb'].mean()
        
        print(f"\nSummary Statistics for {year}:")
        print(f"Number of unique sites in original {year} data: {len(unique_sites_by_year[year])}")
        print(f"Number of unique sites in filtered results {year}: {unique_sites_results}")
        print(f"Average NO2 concentration {year}: {avg_no2_ppb:.2f} ppb")
        print(f"Data for {len(results_df)} site-months saved to {csv_filename}")
    else:
        print(f"\nNo valid data for {year}")

# Create a combined DataFrame with all years
all_results = []
for year in years_to_process:
    all_results.extend(results_by_year[year])

if all_results:
    all_results_df = pd.DataFrame(all_results)
    combined_csv_filename = f"{OutDir}EU_monthly_no2_2005-2023_combined_v1.csv"
    all_results_df.to_csv(combined_csv_filename, index=False)
    
    # Overall statistics
    total_sites = len(set(all_results_df['site_code']))
    total_site_months = len(all_results_df)
    overall_avg_no2 = all_results_df['no2_ppb'].mean()
    
    print("\nOverall Summary Statistics (2005-2023):")
    print(f"Total unique sites across all years: {total_sites}")
    print(f"Total site-months: {total_site_months}")
    print(f"Overall average NO2 concentration: {overall_avg_no2:.2f} ppb")
    print(f"Combined data saved to {combined_csv_filename}")
    
    # Print unique site statistics with exclusions
    print("\n" + "="*70)
    print("UNIQUE SITE STATISTICS - EUROPE")
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