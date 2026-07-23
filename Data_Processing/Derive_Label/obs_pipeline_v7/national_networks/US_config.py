import pandas as pd
import numpy as np
import calendar
import time

# Start timing
start_time = time.time()

InDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/usa/'
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

all_monthly = []  # collect each year's filtered monthly data

for year in years_to_process:
    try:
        print(f"\n=== {year} ===")
        fn = InDir + f'hourly_{year}.csv'
        
        # Load data with low_memory=False to avoid DtypeWarning
        df = pd.read_csv(fn, low_memory=False)
        
        # Convert date to datetime
        df['Date Local'] = pd.to_datetime(df['Date Local'])
        df['Year'] = df['Date Local'].dt.year
        df['Month'] = df['Date Local'].dt.month
        
        # before filtering: how many unique sites?
        sites_before = df[['Latitude', 'Longitude']].drop_duplicates().shape[0]
        print("Unique sites (raw):", sites_before)
        
        # Track unique raw sites
        for _, row in df[['Latitude', 'Longitude']].drop_duplicates().iterrows():
            all_unique_raw_sites.add((row['Latitude'], row['Longitude']))
        
        # Count valid measurements per day
        df_valid = df[df['Sample Measurement'] >= 0].copy()  # Filter out negative values
        
        # Group by site and date to count measurements per day
        day_counts = df_valid.groupby(['Latitude', 'Longitude', 'State Name', 'County Name', 'Date Local'])['Sample Measurement'].count().reset_index()
        day_counts.rename(columns={'Sample Measurement': 'valid_hours'}, inplace=True)
        
        # 1) day-filter: require at least 18 valid hours per day
        df_day = day_counts[day_counts['valid_hours'] >= 18].copy()
        sites_after_day = df_day[['Latitude', 'Longitude']].drop_duplicates().shape[0]
        print("Sites after day-filter:", sites_after_day)
        
        # Calculate daily mean NO2 values
        daily_means = df_valid.groupby(['Latitude', 'Longitude', 'State Name', 'County Name', 'Date Local'])['Sample Measurement'].mean().reset_index()
        daily_means.rename(columns={'Sample Measurement': 'no2_ppb'}, inplace=True)
        
        # Merge with filtered days
        df_day = pd.merge(df_day, daily_means, on=['Latitude', 'Longitude', 'State Name', 'County Name', 'Date Local'], how='left')
        
        # Extract year and month for monthly filtering
        df_day['Year'] = df_day['Date Local'].dt.year
        df_day['Month'] = df_day['Date Local'].dt.month
        
        # 2) month-filter: require ≥75% of days in that month
        # Count valid days per site-month
        month_counts = df_day.groupby(['Latitude', 'Longitude', 'State Name', 'County Name', 'Year', 'Month']).agg(
            valid_days=('Date Local', 'nunique'),
            monthly_mean=('no2_ppb', 'mean')
        ).reset_index()
        
        # Compute days in each month
        month_counts['days_in_month'] = month_counts.apply(
            lambda r: calendar.monthrange(int(r.Year), int(r.Month))[1],
            axis=1
        )
        
        # Keep only those with valid_days ≥ 0.75 × days_in_month
        month_ok = month_counts[
            month_counts['valid_days'] >= 0.75 * month_counts['days_in_month']
        ].copy()
        sites_after_month = month_ok[['Latitude', 'Longitude']].drop_duplicates().shape[0]
        print("Sites after month-filter:", sites_after_month)
        
        # Track unique filtered sites
        for _, row in month_ok[['Latitude', 'Longitude']].drop_duplicates().iterrows():
            all_unique_filtered_sites.add((row['Latitude'], row['Longitude']))
        
        # Keep only the columns you want and add country column
        monthly_data = month_ok[[
            'Latitude', 'Longitude', 'County Name', 'State Name', 'Year', 'Month', 'monthly_mean'
        ]].rename(columns={
            'Latitude': 'lat',
            'Longitude': 'lon',
            'County Name': 'city',
            'State Name': 'state',
            'Year': 'year',
            'Month': 'month',
            'monthly_mean': 'no2_ppb'
        })
        
        # Add country column with 'USA' for all rows
        monthly_data['country'] = 'USA'
        
        all_monthly.append(monthly_data)
        
    except FileNotFoundError:
        print(f"File not found for year {year}, skipping.")
    except Exception as e:
        print(f"Error processing year {year}: {e}")

# Concatenate all years and save
if all_monthly:
    combined = pd.concat(all_monthly, ignore_index=True)
    # combined.to_csv(OutDir + 'US_monthly_no2_2005-2023_combined.csv', index=False)
    print(f"\nWrote {len(combined)} site-months to {OutDir + 'US_monthly_no2_2005-2023_combined.csv'}")
    
    # Compute & display annual mean NO₂ (across all sites)
    annual_mean = combined.groupby('year')['no2_ppb'].mean()
    print("\nAnnual mean NO₂ (ppb) for all sites:")
    print(annual_mean.to_string())
    
    # Print unique site statistics with exclusions
    print("\n" + "="*70)
    print("UNIQUE SITE STATISTICS - UNITED STATES")
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
else:
    print("No data was processed. Check file paths and data availability.")

# Print execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution completed in {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")