import pandas as pd
import numpy as np
from datetime import datetime
import os

def process_new_format_excel(xlsx_file, min_hours_per_day=18):
    """
    Process 2021-2023 multi-sheet Excel format:
    - Metadata sheet: contains site info in two sections (Parameter Info + Location Info)
    - Individual sheets: one per site with hourly measurements
    """
    print(f"Processing new format Excel: {os.path.basename(xlsx_file)}")
    
    # Read the Excel file to get all sheet names
    excel_file = pd.ExcelFile(xlsx_file)
    sheet_names = excel_file.sheet_names
    print(f"Found {len(sheet_names)} sheets: {sheet_names[:10]}...")
    
    # Read metadata sheet without header to find the sections
    metadata_raw = pd.read_excel(xlsx_file, sheet_name='Metadata', header=None)
    print(f"Raw metadata shape: {metadata_raw.shape}")
    
    # Find the Location Information section
    location_start_row = None
    for i, row in metadata_raw.iterrows():
        if any('Location Information' in str(cell) for cell in row if pd.notna(cell)):
            location_start_row = i + 2  # Skip the header row
            break
    
    if location_start_row is None:
        print("Could not find 'Location Information' section")
        return None
    
    print(f"Found Location Information starting at row {location_start_row}")
    
    # Read location information with proper headers
    location_data = pd.read_excel(xlsx_file, sheet_name='Metadata', 
                                 header=location_start_row, skiprows=0)
    
    # Clean up column names and data
    location_data = location_data.dropna(how='all')  # Remove empty rows
    location_data.columns = location_data.columns.astype(str)
    
    print(f"Location data columns: {list(location_data.columns)}")
    print(f"Location data shape: {location_data.shape}")
    print(f"Sample location data:")
    print(location_data.head())
    
    # Find the parameter information section for NO2
    param_start_row = None
    for i, row in metadata_raw.iterrows():
        if any('Parameter Information' in str(cell) for cell in row if pd.notna(cell)):
            param_start_row = i + 2  # Skip the header row
            break
    
    no2_sites = set()
    if param_start_row is not None:
        # Read parameter information
        param_data = pd.read_excel(xlsx_file, sheet_name='Metadata', 
                                  header=param_start_row, nrows=20)  # Limit rows to parameter section
        
        # Look for NO2 row
        for i, row in param_data.iterrows():
            if 'NO2' in str(row.iloc[0]):
                print(f"Found NO2 parameter row: {i}")
                # Get all sites that have 'Present' for NO2
                for col_idx, val in enumerate(row.iloc[1:], 1):  # Skip first column (parameter name)
                    if str(val).strip().lower() == 'present':
                        try:
                            # The column headers should correspond to location_ids
                            site_id = param_data.columns[col_idx]
                            no2_sites.add(str(site_id).strip())
                        except:
                            pass
                break
    
    print(f"Sites with NO2 monitoring: {len(no2_sites)} sites")
    print(f"NO2 site IDs: {list(no2_sites)[:10]}...")  # Show first 10
    
    # Match location data with NO2 sites
    location_data['location_id'] = location_data.iloc[:, 0].astype(str)  # First column is location_id
    no2_locations = location_data[location_data['location_id'].isin(no2_sites)].copy()
    
    if len(no2_locations) == 0:
        print("No matching locations found for NO2 sites. Processing all sites...")
        no2_locations = location_data.copy()
    
    print(f"Processing {len(no2_locations)} locations")
    
    # Process each site
    all_site_data = []
    successful_sites = 0
    failed_sites = 0
    
    for idx, site_row in no2_locations.iterrows():
        location_id = str(site_row.iloc[0]).strip()  # First column
        site_name = str(site_row.iloc[1]).strip()   # Second column  
        
        # Try to get coordinates (assuming they're in columns 2 and 3)
        try:
            lat = float(site_row.iloc[2]) if pd.notna(site_row.iloc[2]) else None
            lon = float(site_row.iloc[3]) if pd.notna(site_row.iloc[3]) else None
        except:
            lat, lon = None, None
        
        print(f"Processing site: {site_name} (ID: {location_id}, Lat: {lat}, Lon: {lon})")
        
        # Find matching sheet - look for sheet names containing the site name or location ID
        matching_sheet = None
        for sheet_name in sheet_names:
            if sheet_name not in ['Data Info Panel', 'Metadata']:
                # Check if sheet name contains site name or location ID
                if (site_name.replace(' ', '').lower() in sheet_name.replace(' ', '').lower() or
                    location_id in sheet_name):
                    matching_sheet = sheet_name
                    break
        
        if not matching_sheet:
            # Try alternative matching - look for sheets that start with site name
            site_name_clean = site_name.replace(' ', '').replace('-', '').lower()
            for sheet_name in sheet_names:
                if sheet_name not in ['Data Info Panel', 'Metadata']:
                    sheet_name_clean = sheet_name.split('_')[0].replace(' ', '').replace('-', '').lower()
                    if site_name_clean.startswith(sheet_name_clean) or sheet_name_clean.startswith(site_name_clean):
                        matching_sheet = sheet_name
                        break
        
        if not matching_sheet:
            print(f"  Warning: No matching sheet found for '{site_name}', skipping...")
            failed_sites += 1
            continue
        
        try:
            # Read site data
            site_data = pd.read_excel(xlsx_file, sheet_name=matching_sheet)
            print(f"  Reading sheet '{matching_sheet}' with {site_data.shape[0]} rows")
            
            # Check if NO2 column exists
            no2_columns = [col for col in site_data.columns if 'NO2' in str(col).upper()]
            if not no2_columns:
                print(f"  Warning: No NO2 column found, skipping...")
                failed_sites += 1
                continue
            
            no2_col = no2_columns[0]
            print(f"  Found NO2 column: {no2_col}")
            
            # Process the site data
            site_processed = process_single_site_data(site_data, site_name, lat, lon, 'ppb', no2_col)
            
            if site_processed is not None and len(site_processed) > 0:
                all_site_data.append(site_processed)
                successful_sites += 1
                print(f"  ✓ Success: {len(site_processed)} valid records")
            else:
                failed_sites += 1
                print(f"  ✗ No valid data")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_sites += 1
    
    print(f"\nFinal summary:")
    print(f"Successfully processed: {successful_sites} sites")
    print(f"Failed to process: {failed_sites} sites")
    
    if not all_site_data:
        print("No site data successfully processed!")
        return None
    
    # Combine all site data
    combined_data = pd.concat(all_site_data, ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")
    return combined_data

def process_single_site_data(site_data, site_name, lat, lon, unit, no2_col):
    """
    Process hourly data for a single site
    """
    # Assume first column is datetime
    datetime_col = site_data.columns[0]
    
    # Create clean dataframe
    df_clean = pd.DataFrame({
        'site_name': site_name,
        'latitude': lat,
        'longitude': lon,
        'unit': unit,
        'datetime': pd.to_datetime(site_data[datetime_col]),
        'value': pd.to_numeric(site_data[no2_col], errors='coerce')
    })
    
    # Remove invalid data
    df_clean = df_clean.dropna(subset=['value', 'datetime'])
    df_clean = df_clean[df_clean['value'] >= 0]
    
    if len(df_clean) == 0:
        return None
    
    # Extract date components
    df_clean['date'] = df_clean['datetime'].dt.date
    df_clean['year'] = df_clean['datetime'].dt.year
    df_clean['month'] = df_clean['datetime'].dt.month
    df_clean['hour'] = df_clean['datetime'].dt.hour
    
    return df_clean

def calculate_monthly_averages_new_format(df_combined, min_hours_per_day=18):
    """
    Calculate monthly averages from combined site data
    Returns: (result_dataframe, raw_sites_set, filtered_sites_set)
    """
    print(f"Calculating monthly averages from {len(df_combined)} hourly records...")
    
    # Track raw sites before any filtering
    raw_sites = set(df_combined['site_name'].unique())
    
    # Step 1: Calculate daily averages
    daily_stats = df_combined.groupby(['site_name', 'date', 'latitude', 'longitude', 
                                      'year', 'month', 'unit']).agg({
        'value': ['mean', 'count'],
        'hour': 'nunique'
    }).reset_index()
    
    # Flatten column names
    daily_stats.columns = ['site_name', 'date', 'latitude', 'longitude', 'year', 'month', 
                          'unit', 'daily_mean', 'hourly_count', 'unique_hours']
    
    # Filter for days with sufficient hourly coverage
    daily_valid = daily_stats[daily_stats['unique_hours'] >= min_hours_per_day].copy()
    print(f"Valid daily averages (>={min_hours_per_day} hours): {len(daily_valid)}")
    
    # Step 2: Calculate monthly averages
    monthly_stats = daily_valid.groupby(['site_name', 'year', 'month', 'latitude', 
                                        'longitude', 'unit']).agg({
        'daily_mean': ['mean', 'count'],
        'date': 'nunique'
    }).reset_index()
    
    # Flatten column names
    monthly_stats.columns = ['site_name', 'year', 'month', 'latitude', 'longitude', 
                            'unit', 'monthly_mean', 'daily_count', 'unique_days']
    
    # Filter for months with >75% daily coverage
    monthly_stats['year_month'] = pd.to_datetime(monthly_stats[['year', 'month']].assign(day=1))
    monthly_stats['days_in_month'] = monthly_stats['year_month'].dt.days_in_month
    monthly_stats['min_days_required'] = np.ceil(monthly_stats['days_in_month'] * 0.75).astype(int)
    
    monthly_valid = monthly_stats[monthly_stats['unique_days'] >= monthly_stats['min_days_required']].copy()
    print(f"Valid monthly averages (>=75% days): {len(monthly_valid)}")
    
    # Format final output
    result = monthly_valid[['latitude', 'longitude', 'year', 'month', 'monthly_mean', 
                           'unit', 'site_name']].copy()
    result = result.rename(columns={
        'latitude': 'lat',
        'longitude': 'lon', 
        'month': 'mon',
        'monthly_mean': 'value',
        'site_name': 'site'
    })
    
    # Round values
    result['value'] = result['value'].round(2)
    result['lat'] = result['lat'].round(6)
    result['lon'] = result['lon'].round(6)
    
    # Track filtered sites (sites that made it through all filters)
    filtered_sites = set(result['site'].unique())
    
    return result, raw_sites, filtered_sites

def process_single_year_old_format(year, InDir, OutDir):
    """
    Process 2005-2020 format (single sheet with all data)
    """
    xlsx_name = InDir + f'{year}_All_sites_air_quality_hourly_avg_AIR-I-F-V-VH-O-S1-DB-M2-4-0.xlsx'
    
    if not os.path.exists(xlsx_name):
        print(f"File not found: {xlsx_name}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Processing year {year} (OLD FORMAT)")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    sheet_names = pd.ExcelFile(xlsx_name).sheet_names
    valid_sheets = [s for s in sheet_names if s.strip().lower() != "data info panel".lower()]
    
    if not valid_sheets:
        print(f"No valid data sheets found in {xlsx_name}")
        return None
    
    # Load first valid sheet
    print("Loading data...")
    df = pd.read_excel(xlsx_name, sheet_name=valid_sheets[0])
    print(f"Original dataset shape: {df.shape}")
    print(f"Available columns: {list(df.columns)[:10]}...")
    
    # Check if this is really the old format
    if 'param_id' in df.columns and 'sp_name' in df.columns:
        # Old format confirmed
        print("Confirmed old format with param_id and sp_name")
        result_data = process_old_format_data(df)
    else:
        print("Unexpected column structure for old format")
        print(f"Columns: {list(df.columns)}")
        return None, None, None
    
    if result_data is None:
        print(f"No valid data for year {year}")
        return None, None, None
    
    result, raw_sites, filtered_sites = result_data
    
    if result is None or len(result) == 0:
        print(f"No valid data for year {year}")
        return None, None, None
    
    # Show summary
    print(f"\nYear {year} Summary:")
    print(f"  Raw sites (before filtering): {len(raw_sites)}")
    print(f"  Filtered sites (after QC): {len(filtered_sites)}")
    print(f"  Sites excluded: {len(raw_sites) - len(filtered_sites)}")
    print(f"  Total monthly averages: {len(result)}")
    print(f"  Units: {result['unit'].value_counts().to_dict()}")
    
    # Save results
    # output_file = OutDir + f'AU_monthly_no2_{year}.csv'
    # result.to_csv(output_file, index=False)
    # print(f"Results saved to: {output_file}")
    
    return result, raw_sites, filtered_sites

def choose_no2_value_column(df):
    """
    Return the name of the NO2 numeric column:
    - Prefer 'PV' if present (any case)
    - Else use 'value' (any case)
    - Returns None if neither exists
    """
    cols = {c.strip().lower(): c for c in df.columns}
    if 'pv' in cols:
        return cols['pv']
    if 'value' in cols:
        return cols['value']
    return None

def process_old_format_data(df):
    """
    Process the old format (2005-2020) single-sheet data
    """
    # Filter for NO2 data
    df_no2 = df[df['param_id'] == 'NO2'].copy()
    
    if len(df_no2) == 0:
        print("No NO2 data found")
        return None
    
    print(f"Found {len(df_no2)} NO2 records from {df_no2['sp_name'].nunique()} sites")
    
    value_col = choose_no2_value_column(df_no2)
    if value_col is None:
        print("Neither 'PV' nor 'value' column found")
        return None

    # Unify to 'value'
    df_no2['value'] = pd.to_numeric(df_no2[value_col], errors='coerce')
    df_clean = df_no2.dropna(subset=['value'])
    df_clean = df_clean[df_clean['value'] >= 0]
    
    # Extract date components
    df_clean['datetime'] = pd.to_datetime(df_clean['sample_datetime'])
    df_clean['date'] = df_clean['datetime'].dt.date
    df_clean['year'] = df_clean['datetime'].dt.year
    df_clean['month'] = df_clean['datetime'].dt.month
    df_clean['hour'] = df_clean['datetime'].dt.hour
    
    # Rename columns to match new format
    df_clean['site_name'] = df_clean['sp_name']
    df_clean['unit'] = df_clean['param_std_unit_of_measure']
    
    # Calculate monthly averages
    result, raw_sites, filtered_sites = calculate_monthly_averages_new_format(df_clean)
    return result, raw_sites, filtered_sites

def process_single_year_new_format_flexible(year, InDir, OutDir):
    """
    Process new format (2021-2023) - handles both with and without metadata sheet
    Returns: (result_dataframe, raw_sites_set, filtered_sites_set)
    """
    xlsx_name = InDir + f'{year}_All_sites_air_quality_hourly_avg_AIR-I-F-V-VH-O-S1-DB-M2-4-0.xlsx'
    
    if not os.path.exists(xlsx_name):
        print(f"File not found: {xlsx_name}")
        return None, None, None
    
    print(f"\n{'='*60}")
    print(f"Processing year {year} (NEW FORMAT)")
    print(f"{'='*60}")
    
    # Read the Excel file to get sheet names
    excel_file = pd.ExcelFile(xlsx_name)
    sheet_names = excel_file.sheet_names
    print(f"Found sheets: {sheet_names}")
    
    # Check if metadata sheet exists
    if 'Metadata' in sheet_names:
        print("Found Metadata sheet - using multi-sheet approach")
        return process_with_metadata_sheet(xlsx_name, year, OutDir)
    else:
        print("No Metadata sheet - using single sheet approach")
        # Look for a sheet named after the year or similar
        year_sheet = str(year)
        if year_sheet in sheet_names:
            return process_single_year_sheet(xlsx_name, year_sheet, year, OutDir)
        else:
            # Try to find the main data sheet (not 'Data Info Panel')
            data_sheets = [s for s in sheet_names if s not in ['Data Info Panel']]
            if len(data_sheets) == 1:
                return process_single_year_sheet(xlsx_name, data_sheets[0], year, OutDir)
            else:
                print(f"Cannot determine main data sheet from: {data_sheets}")
                return None, None, None

def process_with_metadata_sheet(xlsx_file, year, OutDir):
    """Process files that have a metadata sheet (2021-2023 style)"""
    combined_site_data = process_new_format_excel(xlsx_file)
    
    if combined_site_data is None:
        print(f"No valid data for year {year}")
        return None, None, None
    
    # Calculate monthly averages
    result, raw_sites, filtered_sites = calculate_monthly_averages_new_format(combined_site_data)
    
    if len(result) == 0:
        print(f"No valid monthly averages for year {year}")
        return None, None, None
    
    # Show summary and save
    print(f"\nYear {year} Summary:")
    print(f"  Raw sites (before filtering): {len(raw_sites)}")
    print(f"  Filtered sites (after QC): {len(filtered_sites)}")
    print(f"  Sites excluded: {len(raw_sites) - len(filtered_sites)}")
    print(f"  Exclusion rate: {(len(raw_sites)-len(filtered_sites))/len(raw_sites)*100:.1f}%" if len(raw_sites) > 0 else "N/A")
    print(f"  Total monthly averages: {len(result)}")
    print(f"  Units: {result['unit'].value_counts().to_dict()}")
    
    # Save results
    # output_file = OutDir + f'AU_monthly_no2_{year}.csv'
    # result.to_csv(output_file, index=False)
    # print(f"Results saved to: {output_file}")
    
    return result, raw_sites, filtered_sites

def process_single_year_sheet(xlsx_file, sheet_name, year, OutDir):
    """Process files with a single data sheet (2019-2020 style)"""
    print(f"Reading sheet '{sheet_name}'...")
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}...")
    
    # Check if this has the new format structure
    if 'parameter_name' in df.columns and 'location_name' in df.columns:
        # Filter for NO2
        df_no2 = df[df['parameter_name'] == 'NO2'].copy()
        if len(df_no2) == 0:
            print("No NO2 data found")
            return None
        
        print(f"Found {len(df_no2)} NO2 records from {df_no2['location_name'].nunique()} sites")
        
        # Process similar to new format
        df_clean = df_no2.dropna(subset=['value']).copy()
        df_clean = df_clean[df_clean['value'] >= 0]
        
        # Extract date components
        df_clean['datetime'] = pd.to_datetime(df_clean['datetime_AEST'])
        df_clean['date'] = df_clean['datetime'].dt.date
        df_clean['year'] = df_clean['datetime'].dt.year
        df_clean['month'] = df_clean['datetime'].dt.month
        df_clean['hour'] = df_clean['datetime'].dt.hour
        
        # Rename columns to match expected format
        df_clean['site_name'] = df_clean['location_name']
        df_clean['unit'] = df_clean['unit_of_measure']
        
        result, raw_sites, filtered_sites = calculate_monthly_averages_new_format(df_clean)
        
    else:
        print("Unknown format for single sheet data")
        print(f"Available columns: {list(df.columns)}")
        return None, None, None
    
    if result is None or len(result) == 0:
        print(f"No valid data for year {year}")
        return None, None, None
    
    # Show summary and save
    print(f"\nYear {year} Summary:")
    print(f"  Raw sites (before filtering): {len(raw_sites)}")
    print(f"  Filtered sites (after QC): {len(filtered_sites)}")
    print(f"  Sites excluded: {len(raw_sites) - len(filtered_sites)}")
    print(f"  Total monthly averages: {len(result)}")
    print(f"  Units: {result['unit'].value_counts().to_dict()}")
    
    #Save results
    # output_file = OutDir + f'AU_monthly_no2_{year}.csv'
    # result.to_csv(output_file, index=False)
    # print(f"Results saved to: {output_file}")
    
    return result, raw_sites, filtered_sites

def process_all_years_mixed_format(start_year=2005, end_year=2023, format_change_year=2021):
    """
    Process all years handling multiple formats:
    - 2005-2020: Old single-sheet format
    - 2021-2023: Multi-sheet format (some may not have metadata sheet)
    """
    InDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/raw/australia/EPA/'
    OutDir = '/path/to/NO2_DL_global/TrainingDatasets/Global_NO2_v7/no2_ground/national_networks/compiled/'
    
    os.makedirs(OutDir, exist_ok=True)
    
    all_results = []
    successful_years = []
    failed_years = []
    year_exclusion_stats = []  # Track raw vs filtered sites per year
    
    # Track unique sites across all years
    all_unique_raw_sites = set()
    all_unique_filtered_sites = set()
    
    for year in range(start_year, end_year + 1):
        try:
            if year < format_change_year:
                # 2005-2020: Old format
                result, raw_sites, filtered_sites = process_single_year_old_format(year, InDir, OutDir)
            else:
                # 2021-2023: New format
                result, raw_sites, filtered_sites = process_single_year_new_format_flexible(year, InDir, OutDir)
            
            if result is not None and len(result) > 0:
                all_results.append(result)
                successful_years.append(year)
                
                # Track unique sites
                if raw_sites is not None:
                    all_unique_raw_sites.update(raw_sites)
                if filtered_sites is not None:
                    all_unique_filtered_sites.update(filtered_sites)
                    
                # Track per-year stats
                year_exclusion_stats.append({
                    'year': year,
                    'raw_sites': len(raw_sites) if raw_sites else 0,
                    'filtered_sites': len(filtered_sites) if filtered_sites else 0,
                    'excluded_sites': (len(raw_sites) - len(filtered_sites)) if (raw_sites and filtered_sites) else 0
                })
            else:
                failed_years.append(year)
                
        except Exception as e:
            print(f"Error processing year {year}: {e}")
            failed_years.append(year)
    
    # Combine all years
    if all_results:
        print(f"\n{'='*60}")
        print("COMBINING ALL YEARS")
        print(f"{'='*60}")
        
        combined_data = pd.concat(all_results, ignore_index=True)
        combined_data = combined_data.sort_values(['year', 'mon', 'site']).reset_index(drop=True)
        
        # Save combined results
        combined_file = OutDir + 'AU_monthly_no2_2005_2023_combined.csv'
        # combined_data.to_csv(combined_file, index=False)
        
        print(f"Combined dataset:")
        print(f"Total records: {len(combined_data)}")
        print(f"Years: {sorted(combined_data['year'].unique())}")
        print(f"Sites: {combined_data['site'].nunique()}")
        print(f"Year range: {combined_data['year'].min()} - {combined_data['year'].max()}")
        print(f"Combined file saved to: {combined_file}")
        
        # Calculate unique site statistics
        unique_raw_count = len(all_unique_raw_sites)
        unique_filtered_count = len(all_unique_filtered_sites)
        unique_excluded_count = unique_raw_count - unique_filtered_count
        unique_exclusion_rate = (unique_excluded_count / unique_raw_count * 100) if unique_raw_count > 0 else 0
        
        # Calculate sum statistics
        total_raw_sites = sum(stat['raw_sites'] for stat in year_exclusion_stats)
        total_filtered_sites = sum(stat['filtered_sites'] for stat in year_exclusion_stats)
        total_excluded_sites = total_raw_sites - total_filtered_sites
        
        print("\n" + "="*70)
        print("SUMMARY STATISTICS - SITE EXCLUSION (AUSTRALIA)")
        print("="*70)
        print(f"Total raw sites (sum across years):      {total_raw_sites:,}")
        print(f"Total filtered sites (sum across years): {total_filtered_sites:,}")
        print(f"Total excluded sites (sum):              {total_excluded_sites:,}")
        print(f"Overall exclusion rate (by sum):         {total_excluded_sites/total_raw_sites*100:.1f}%" if total_raw_sites > 0 else "N/A")
        
        print(f"\nUNIQUE SITES (across all years):")
        print(f"  Unique raw sites (before filtering):        {unique_raw_count:,}")
        print(f"  Unique filtered sites (after QC):           {unique_filtered_count:,}")
        print(f"  Unique EXCLUDED sites:                      {unique_excluded_count:,}")
        print(f"  Unique exclusion rate:                      {unique_exclusion_rate:.1f}%")
        
        print("\nQuality Control Filters Applied:")
        print("  - Daily filter: ≥18 hours of measurements per day")
        print("  - Monthly filter: ≥75% of days with valid daily averages")
        
        print("\nYearly breakdown:")
        stats_df = pd.DataFrame(year_exclusion_stats)
        if len(stats_df) > 0:
            stats_df['exclusion_rate_%'] = (stats_df['excluded_sites'] / stats_df['raw_sites'] * 100).round(1)
            print(stats_df.to_string(index=False))
        print("="*70)
        
    print(f"\nProcessing Summary:")
    print(f"Successful years: {successful_years}")
    print(f"Failed years: {failed_years}")
    
    return combined_data if all_results else None

if __name__ == "__main__":
    print("\nProcessing all years...")
    combined_data = process_all_years_mixed_format(2005, 2023, 2021)