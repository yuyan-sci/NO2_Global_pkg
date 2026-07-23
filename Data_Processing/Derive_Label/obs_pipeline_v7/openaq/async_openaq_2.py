import argparse
import os
import sys
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import pandas as pd
from openaq import OpenAQ


BASE_OUTPUT_DIR = "/path/to/openaq"
DEFAULT_YEARS = range(2005, 2024)  # Inclusive of 2023
PARAMETER_IDS = [5, 7, 15]  # NO2 ug/m3, ppm + ppb
BASE_DELAY_SECONDS = 5
CHECKPOINT_INTERVAL = 50


@dataclass
class Country:
    id: int
    name: str
    iso: str

COUNTRIES: List[Country] = [
    # --- Africa ---
    Country(122, "Algeria", "DZ"),
    Country(150, "Burkina Faso", "BF"),
    Country(147, "Cameroon", "CM"),
    Country(115, "Chad", "TD"),
    Country(96, "Côte d'Ivoire", "CI"),
    Country(32, "Democratic Republic of the Congo", "CD"),
    Country(162, "Egypt", "EG"),
    Country(14, "Ethiopia", "ET"),
    Country(152, "Ghana", "GH"),
    Country(83, "Guinea", "GN"),
    Country(17, "Kenya", "KE"),
    Country(182, "Madagascar", "MG"),
    Country(18, "Malawi", "MW"),
    Country(98, "Mali", "ML"),
    Country(219, "Mauritius", "MU"),
    Country(27, "Morocco", "MA"),
    Country(123, "Mozambique", "MZ"),
    Country(100, "Nigeria", "NG"),
    Country(222, "Republic of Cabo Verde", "CV"),
    Country(126, "Rwanda", "RW"),
    Country(99, "Senegal", "SN"),
    Country(37, "South Africa", "ZA"),
    Country(15, "South Sudan", "SS"),
    Country(86, "Sudan", "SD"),
    Country(166, "The Gambia", "GM"),
    Country(73, "Tunisia", "TN"),
    Country(133, "Uganda", "UG"),
    Country(81, "Zambia", "ZM"),
    Country(108, "Zimbabwe", "ZW"),

    # --- Asia ---
    Country(130, "Afghanistan", "AF"),
    Country(70, "Armenia", "AM"),
    Country(64, "Azerbaijan", "AZ"),
    Country(250, "Bahrain", "BH"),
    Country(128, "Bangladesh", "BD"),
    Country(57, "Cambodia", "KH"),
    Country(10, "China", "CN"),
    Country(8, "Cyprus", "CY"),  # Geographically Asia, politically often EU
    Country(167, "Hong Kong", "HK"),
    Country(9, "India", "IN"),
    Country(1, "Indonesia", "ID"),
    Country(90, "Iraq", "IQ"),
    Country(11, "Israel", "IL"),
    Country(190, "Japan", "JP"),
    Country(144, "Jordan", "JO"),
    Country(42, "Kazakhstan", "KZ"),
    Country(116, "Kuwait", "KW"),
    Country(69, "Kyrgyzstan", "KG"),
    Country(68, "Lao PDR", "LA"),
    Country(2, "Malaysia", "MY"),
    Country(239, "Maldives", "MV"),
    Country(47, "Mongolia", "MN"),
    Country(127, "Myanmar", "MM"),
    Country(145, "Nepal", "NP"),
    Country(40, "Oman", "OM"),
    Country(109, "Pakistan", "PK"),
    Country(12, "Palestine", "PS"),
    Country(183, "Philippines", "PH"),
    Country(105, "Qatar", "QA"),
    Country(25, "Republic of Korea", "KR"),
    Country(106, "Saudi Arabia", "SA"),
    Country(231, "Singapore", "SG"),
    Country(184, "Sri Lanka", "LK"),
    Country(189, "Taiwan", "TW"),
    Country(43, "Tajikistan", "TJ"),
    Country(111, "Thailand", "TH"),
    Country(66, "Turkey", "TR"),  # Transcontinental
    Country(143, "Turkmenistan", "TM"),
    Country(59, "United Arab Emirates", "AE"),
    Country(41, "Uzbekistan", "UZ"),
    Country(56, "Vietnam", "VN"),

    # --- Europe ---
    Country(129, "Andorra", "AD"),
    Country(89, "Austria", "AT"),
    Country(60, "Belgium", "BE"),
    Country(132, "Bosnia and Herzegovina", "BA"),
    Country(110, "Bulgaria", "BG"),
    Country(103, "Croatia", "HR"),
    Country(49, "Czech Republic", "CZ"),
    Country(71, "Denmark", "DK"),
    Country(51, "Estonia", "EE"),
    Country(55, "Finland", "FI"),
    Country(22, "France", "FR"),
    Country(50, "Germany", "DE"),
    Country(154, "Gibraltar", "GI"),
    Country(80, "Greece", "GR"),
    Country(75, "Hungary", "HU"),
    Country(192, "Iceland", "IS"),
    Country(78, "Ireland", "IE"),
    Country(91, "Italy", "IT"),
    Country(224, "Jersey", "JE"),
    Country(65, "Kosovo", "XK"),
    Country(52, "Latvia", "LV"),
    Country(44, "Lithuania", "LT"),
    Country(58, "Luxembourg", "LU"),
    Country(223, "Malta", "MT"),
    Country(142, "Moldova", "MD"),
    Country(121, "Monaco", "MC"),
    Country(131, "Montenegro", "ME"),
    Country(94, "Netherlands", "NL"),
    Country(62, "North Macedonia", "MK"),
    Country(53, "Norway", "NO"),
    Country(77, "Poland", "PL"),
    Country(141, "Portugal", "PT"),
    Country(74, "Romania", "RO"),
    Country(48, "Russian Federation", "RU"),
    Country(112, "San Marino", "SM"),
    Country(97, "Serbia", "RS"),
    Country(76, "Slovakia", "SK"),
    Country(104, "Slovenia", "SI"),
    Country(67, "Spain", "ES"),
    Country(54, "Sweden", "SE"),
    Country(92, "Switzerland", "CH"),
    Country(34, "Ukraine", "UA"),
    Country(79, "United Kingdom", "GB"),
    Country(7, "Dhekelia", "-99"),  # Sovereign Base Area (Cyprus/UK)

    # --- North America (incl. Central America & Caribbean) ---
    Country(158, "Belize", "BZ"),
    Country(156, "Canada", "CA"),
    Country(29, "Costa Rica", "CR"),
    Country(185, "Curaçao", "CW"),
    Country(118, "Guatemala", "GT"),
    Country(136, "Honduras", "HN"),
    Country(157, "Mexico", "MX"),
    Country(211, "Puerto Rico", "PR"),
    Country(38, "Saint-Martin", "MF"),
    Country(199, "Trinidad and Tobago", "TT"),
    Country(155, "United States", "US"),

    # --- Oceania ---
    Country(177, "Australia", "AU"),
    Country(180, "New Zealand", "NZ"),

    # --- South America ---
    Country(6, "Argentina", "AR"),
    Country(45, "Brazil", "BR"),
    Country(3, "Chile", "CL"),
    Country(138, "Colombia", "CO"),
    Country(137, "Ecuador", "EC"),
    Country(24, "Guyana", "GY"),
    Country(139, "Paraguay", "PY"),
    Country(5, "Peru", "PE"),
    Country(46, "Uruguay", "UY"),

    # --- Antarctica ---
    Country(176, "Antarctica", "AQ"),
]

EXCLUDED_NAMES = {
    "United States",
    "Canada",
    "China",
    "New Zealand",
    "South Africa",
}

EUROPEAN_COUNTRIES = {
    "Andorra",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Gibraltar",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Italy",
    "Kosovo",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Moldova",
    "Monaco",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Russian Federation",
    "San Marino",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Ukraine",
    "United Kingdom",
    "Jersey",
    "Armenia",
    "Azerbaijan",
    "Turkey",
}


def filtered_countries() -> List[Country]:
    excluded = EXCLUDED_NAMES | EUROPEAN_COUNTRIES
    return [c for c in COUNTRIES if c.name not in excluded]


def select_countries(iso_filters: Optional[Set[str]]) -> List[Country]:
    if not iso_filters:
        return filtered_countries()
    iso_upper = {iso.upper() for iso in iso_filters}
    selected = [c for c in COUNTRIES if c.iso.upper() in iso_upper]
    missing = iso_upper - {c.iso.upper() for c in selected}
    if missing:
        print(f"Warning: unknown ISO codes ignored: {', '.join(sorted(missing))}")
    return selected


def slugify(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
    )


def ensure_output_dir(country: Country) -> str:
    country_slug = slugify(country.name)
    output_dir = os.path.join(BASE_OUTPUT_DIR, country_slug)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def determine_sensor_type(row: pd.Series) -> str:
    # Heuristic similar to loc_openaq.py
    if row.get("is_mobile") and not row.get("is_monitor"):
        return "low_cost_sensor"
    if row.get("is_monitor") and not row.get("is_mobile"):
        return "stationary_sensor"
    return "unknown"


def create_sensor_location_map(df_locs: pd.DataFrame, country: Country) -> Dict[int, Dict[str, Optional[str]]]:
    sensor_map: Dict[int, Dict[str, Optional[str]]] = {}
    for _, row in df_locs.iterrows():
        location_id = row.get("id")
        location_name = row.get("name")
        lat = row.get("coordinates.latitude")
        lon = row.get("coordinates.longitude")
        # Fallback to bounds if coordinates missing
        if pd.isna(lat) or pd.isna(lon):
            bounds = row.get("bounds")
            if isinstance(bounds, list) and len(bounds) >= 2:
                lon, lat = bounds[0], bounds[1]
        sensor_type = determine_sensor_type(row)

        for sensor in row.get("sensors", []):
            parameter = sensor.get("parameter", {})
            param_id = parameter.get("id")
            if param_id not in PARAMETER_IDS:
                continue
            if param_id == 7:
                unit = "ppm"
            elif param_id == 15:
                unit = "ppb"
            elif param_id == 5:
                unit = "µg/m³"
            else:
                continue
            sensor_map[sensor["id"]] = {
                "sensor_id": sensor["id"],
                "sensor_type": sensor_type,
                "location_id": location_id,
                "location_name": location_name,
                "latitude": lat,
                "longitude": lon,
                "unit": unit,
                "country_name": country.name,
                "country_iso": country.iso,
            }
    return sensor_map


def fetch_locations_for_country(client: OpenAQ, country: Country) -> Dict[int, Dict[str, Optional[str]]]:
    combined_map: Dict[int, Dict[str, Optional[str]]] = {}
    for parameter_id in PARAMETER_IDS:
        print(f"Fetching locations for {country.name} (parameter {parameter_id})...")
        sys.stdout.flush()
        all_results = []
        page = 1
        while True:
            locations = client.locations.list(
                countries_id=country.id,
                parameters_id=parameter_id,
                limit=100,
                page=page,
            )
            results = locations.dict().get("results", [])
            if not results:
                break
            all_results.extend(results)
            if len(results) < 100:
                break
            page += 1
        if not all_results:
            continue
        df_locations = pd.json_normalize(all_results)
        # Some responses may lack coordinates columns until normalized; ensure they exist
        for col in ["coordinates.latitude", "coordinates.longitude"]:
            if col not in df_locations.columns:
                df_locations[col] = None
        combined_map.update(create_sensor_location_map(df_locations, country))
    return combined_map


def get_measurements_for_year(
    client: OpenAQ,
    sensor_id: int,
    year: int,
    max_retries: int = 5,
    initial_wait: int = 5,
) -> Optional[List[dict]]:
    all_results: List[dict] = []
    page = 1
    date_from = f"{year}-01-01"
    date_to = f"{year}-12-31"
    retries = max_retries
    wait_time = initial_wait
    while retries > 0:
        try:
            resp = client.measurements.list(
                sensors_id=sensor_id,
                datetime_from=date_from,
                datetime_to=date_to,
                data="hours",
                limit=1000,
                page=page,
            )
            results = resp.dict().get("results", [])
            if not results:
                break
            all_results.extend(results)
            if len(results) < 1000:
                break
            page += 1
            retries = max_retries
            wait_time = initial_wait
        except Exception as exc:
            exc_str = str(exc)
            if "Too many requests" in exc_str or "429" in exc_str:
                wait_match = re.search(r"resets in (\d+) seconds?", exc_str, re.IGNORECASE)
                if wait_match:
                    extracted = int(wait_match.group(1))
                    wait_time = max(wait_time, extracted + 5) # Add 5s buffer
                else:
                    wait_time = max(wait_time, 30) # Default to 30s if parsing fails
                
                print(f"    Rate limit for sensor {sensor_id}, year {year}. Sleeping {wait_time}s... (Retry {retries})")
                sys.stdout.flush()
                time.sleep(wait_time)
                retries -= 1
                # Don't double wait_time here, just trust the API's reset time or our backoff
                # wait_time = min(wait_time * 2, 60) 
            else:
                print(f"    Error fetching sensor {sensor_id} year {year}: {exc}")
                sys.stdout.flush()
                # Don't return None immediately on other errors, retry a few times?
                # For now, let's just sleep and retry
                time.sleep(wait_time)
                retries -= 1
                wait_time = min(wait_time * 2, 60)
    if not all_results:
        print(f"    No measurements found for sensor {sensor_id} year {year}.") # Optional: too verbose
        return None
    return all_results


def convert_units_to_ppb(df: pd.DataFrame) -> pd.DataFrame:
    units_col = None
    for candidate in ["units", "parameter.units"]:
        if candidate in df.columns:
            units_col = candidate
            break
    if units_col is None:
        # No unit field: do NOT guess from magnitude (the old `*1000 if v<1`
        # heuristic silently corrupted clean-air ppb readings, e.g. 0.4 ppb->400).
        # Mark unknown so these rows can be reviewed/dropped downstream.
        df["units"] = "unknown"
        return df

    def to_ppb(row):
        # Normalize the unit string: strip, lowercase, unify micro sign
        # (µ U+00B5 vs μ U+03BC) and superscript ³ -> 3, so all µg/m3 spellings
        # match. Unrecognized units -> NaN (dropped downstream) rather than
        # crashing the whole country-year (the old code returned None -> KeyError).
        unit = (str(row[units_col]).strip().lower()
                .replace("μ", "µ").replace("³", "3"))
        v = row["value"]
        if unit == "ppm":
            return v * 1000.0, "ppb"
        elif unit == "ppb":
            return v, "ppb"
        elif unit in ("µg/m3", "ug/m3"):
            return v * 24.45 / 46, "ppb"
        else:
            return float("nan"), "unknown"

    converted_values = df.apply(to_ppb, axis=1, result_type="expand")
    df["value"] = converted_values[0]
    df[units_col] = converted_values[1]
    return df


def process_country_year(
    client: OpenAQ,
    country: Country,
    year: int,
    sensor_map: Dict[int, Dict[str, Optional[str]]],
) -> Optional[pd.DataFrame]:
    measurements: List[dict] = []
    failed = []
    sensor_ids = list(sensor_map.keys())
    start_time = time.time()
    
    # Dynamic delay calculation:
    # If few sensors, wait longer to avoid hitting rate limit cycle.
    # If many sensors, we can go faster (0.3s).
    # Target: Allow ~60s reset time.
    # If N sensors, loop takes N * delay seconds. We want N * delay > 60 (mostly).
    # So delay > 60 / N.
    # Let's use a safer buffer: max(0.3, 100 / len(sensor_ids)) capped at say 10s.
    num_sensors = len(sensor_ids)
    if num_sensors > 0:
        calculated_delay = max(0.3, 150.0 / num_sensors)
        delay = min(calculated_delay, 10.0)
    else:
        delay = BASE_DELAY_SECONDS
        
    print(f"    Dynamic delay set to {delay:.2f}s for {num_sensors} sensors.")
    sys.stdout.flush()

    for idx, sensor_id in enumerate(sensor_ids, 1):
        if idx % 50 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"    Processed {idx}/{len(sensor_ids)} sensors ({elapsed:.1f} min elapsed)")
            sys.stdout.flush()
        result = get_measurements_for_year(client, sensor_id, year)
        if not result:
            failed.append(sensor_id)
            continue
        info = sensor_map.get(sensor_id, {})
        for measurement in result:
            measurement["latitude"] = info.get("latitude")
            measurement["longitude"] = info.get("longitude")
            measurement["location_id"] = info.get("location_id")
            measurement["location_name"] = info.get("location_name")
            measurement["sensor_type"] = info.get("sensor_type", "unknown")
            measurement["country_name"] = info.get("country_name")
            measurement["country_iso"] = info.get("country_iso")
            measurements.append(measurement)
        time.sleep(delay)

    if failed:
        print(f"    Warning: {len(failed)} sensors failed for {country.name} {year}: {failed[:10]}...")
    if not measurements:
        return None

    df_year = pd.json_normalize(measurements)
    if "period.datetime_from.local" in df_year.columns:
        df_year["datetime_local"] = pd.to_datetime(df_year["period.datetime_from.local"], errors="coerce")
    else:
        df_year["datetime_local"] = pd.NaT

    def parse_components(row):
        dt = row["datetime_local"]
        if pd.isna(dt):
            return pd.Series([None, None, None, None])
        return pd.Series([dt.year, dt.month, dt.day, dt.hour])

    df_year[["year", "month", "day", "hour"]] = df_year.apply(parse_components, axis=1)
    df_year.dropna(subset=["year", "month", "day", "hour"], inplace=True)
    df_year[["year", "month", "day", "hour"]] = df_year[["year", "month", "day", "hour"]].astype(int)
    df_year = convert_units_to_ppb(df_year)
    df_year["latitude"] = df_year["latitude"].round(6)
    df_year["longitude"] = df_year["longitude"].round(6)

    column_mapping = {
        "parameter.name": "parameter",
        "parameter.units": "units",
        "sensors_id": "sensors_id",
        "coverage.percent_coverage": "percent_coverage",
    }
    df_year = df_year.rename(columns=column_mapping)
    desired_cols = [
        "year",
        "month",
        "day",
        "hour",
        "latitude",
        "longitude",
        "value",
        "parameter",
        "units",
        "percent_coverage",
        "location_id",
        "location_name",
        "sensors_id",
        "sensor_type",
        "country_name",
        "country_iso",
    ]
    existing = [col for col in desired_cols if col in df_year.columns]
    df_year = df_year[existing]
    return df_year


def parse_args():
    parser = argparse.ArgumentParser(description="Download OpenAQ NO2 data by country/year.")
    parser.add_argument(
        "--countries",
        nargs="+",
        help="ISO codes of countries to process (default: all non-excluded countries).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=min(DEFAULT_YEARS),
        help="First year to download (default: 2005).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=max(DEFAULT_YEARS),
        help="Last year to download (default: 2023).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    years = range(args.start_year, args.end_year + 1)
    print("Initializing OpenAQ client...")
    sys.stdout.flush()
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAQ_API_KEY environment variable is not set.")
    client = OpenAQ(api_key=api_key)

    selected_countries = select_countries(set(args.countries) if args.countries else None)
    if not selected_countries:
        print("No countries selected after filtering. Exiting.")
        return
    print(f"Countries to process: {len(selected_countries)} ({', '.join(c.iso for c in selected_countries)})")
    print(f"Years: {years.start} - {years.stop - 1}")
    sys.stdout.flush()

    for country in selected_countries:
        country_dir = ensure_output_dir(country)
        print(f"\n=== {country.name} ({country.iso}) ===")
        sys.stdout.flush()
        sensor_map = fetch_locations_for_country(client, country)
        if not sensor_map:
            print(f"  No NO₂ sensors found for {country.name}. Skipping.")
            continue
        print(f"  Sensors mapped: {len(sensor_map)}")

        for year in years:
            output_file = os.path.join(country_dir, f"{year}.csv")
            if os.path.exists(output_file):
                print(f"  {country.name} {year} already processed. Skipping.")
                continue
            print(f"  Processing year {year}...")
            sys.stdout.flush()
            df_year = process_country_year(client, country, year, sensor_map)
            if df_year is None or df_year.empty:
                print(f"    No data for {country.name} {year}.")
                continue
            df_year.to_csv(output_file, index=False)
            print(f"    Saved {len(df_year)} records to {output_file}")
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(1)

