"""
Data Collection Script for Food Desert Prediction Project

This script downloads and organizes all required datasets from public sources.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import zipfile
import json
import time
from dotenv import load_dotenv
from census import Census

# Load environment variables
load_dotenv()

# Set up paths - organized data folders
BASE_DIR = Path(__file__).parent.parent
CENSUS_DIR = BASE_DIR / "data" / "01_census_demographics"
FOOD_ACCESS_DIR = BASE_DIR / "data" / "02_food_access"
HEALTH_DIR = BASE_DIR / "data" / "03_health_outcomes"
HOUSING_DIR = BASE_DIR / "data" / "04_housing_economics"
INFRASTRUCTURE_DIR = BASE_DIR / "data" / "05_infrastructure_transportation"

# Create directories
for dir_path in [CENSUS_DIR, FOOD_ACCESS_DIR, HEALTH_DIR, HOUSING_DIR, INFRASTRUCTURE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def download_file(url, filepath, description="file", resume=True):
    """
    Download a file with progress bar and resume support.
    
    Args:
        url: URL to download from
        filepath: Path where file should be saved
        description: Description for progress bar
        resume: Whether to attempt resuming partial downloads
    """
    # Check if file exists and get its size for resume
    existing_size = 0
    if filepath.exists() and resume:
        existing_size = filepath.stat().st_size
        if existing_size > 0:
            print(f"  Found partial file ({existing_size:,} bytes), attempting resume...")
    
    # Prepare headers for resume
    headers = {}
    if existing_size > 0 and resume:
        headers['Range'] = f'bytes={existing_size}-'
    
    try:
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        
        # Check if server supports resume (206 Partial Content)
        if existing_size > 0 and response.status_code == 206:
            mode = 'ab'  # Append mode for resume
            # Parse Content-Range header to get total size
            content_range = response.headers.get('content-range', '')
            if content_range:
                total_size = int(content_range.split('/')[-1])
            else:
                # Fallback: use content-length + existing_size
                total_size = existing_size + int(response.headers.get('content-length', 0))
            initial_pos = existing_size
            print(f"  Resuming download from byte {existing_size:,}...")
        elif existing_size > 0 and response.status_code == 200:
            # Server doesn't support resume, restart download
            print(f"  Server doesn't support resume, restarting download...")
            filepath.unlink()  # Delete partial file
            mode = 'wb'
            total_size = int(response.headers.get('content-length', 0))
            initial_pos = 0
        else:
            # New download
            mode = 'wb'
            total_size = int(response.headers.get('content-length', 0))
            initial_pos = 0
            if response.status_code != 200:
                response.raise_for_status()
        
        # Download with progress bar
        with open(filepath, mode) as f, tqdm(
            desc=description,
            total=total_size,
            initial=initial_pos,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"✓ Downloaded {description}")
        
    except requests.exceptions.RequestException as e:
        print(f"⚠ Error downloading {description}: {e}")
        # If download failed and we have a partial file, keep it for resume
        if existing_size == 0 and filepath.exists():
            print(f"  Partial file saved at {filepath}, can resume later")
        raise

def collect_food_access_atlas():
    """Download USDA Food Access Research Atlas data."""
    print("\n=== Collecting USDA Food Access Research Atlas ===")
    
    # Check for 2019 data
    filepath_2019 = FOOD_ACCESS_DIR / "FoodAccessResearchAtlasData2019.xlsx"
    if filepath_2019.exists() and filepath_2019.stat().st_size > 1024 * 1024:
        print(f"✓ Food Access Atlas 2019 already exists ({filepath_2019.stat().st_size / (1024*1024):.1f} MB)")
    else:
        url_2019 = 'https://www.ers.usda.gov/webdocs/DataFiles/80591/FoodAccessResearchAtlasData2019.xlsx'
        try:
            download_file(url_2019, filepath_2019, "Food Access Atlas 2019")
        except Exception as e:
            print(f"⚠ Could not download 2019 data: {e}")
            print(f"  Manual download: {url_2019}")
            print(f"  Save to: {filepath_2019}")
    
    # Check for 2015 data - first check local folder, then try download
    filepath_2015 = FOOD_ACCESS_DIR / "FoodAccessResearchAtlasData2015.xlsx"
    local_2015 = FOOD_ACCESS_DIR / "USDA Food Environment Atlas" / "2015 Food Access Research Atlas" / "FoodAccessResearchAtlasData2015.xlsx"
    
    if filepath_2015.exists() and filepath_2015.stat().st_size > 1024 * 1024:
        print(f"✓ Food Access Atlas 2015 already exists ({filepath_2015.stat().st_size / (1024*1024):.1f} MB)")
    elif local_2015.exists() and local_2015.stat().st_size > 1024 * 1024:
        print(f"✓ Food Access Atlas 2015 found in local folder ({local_2015.stat().st_size / (1024*1024):.1f} MB)")
        print(f"  Copying to main food access folder...")
        import shutil
        shutil.copy2(local_2015, filepath_2015)
        print(f"  ✓ Copied to {filepath_2015}")
    else:
        url_2015 = 'https://www.ers.usda.gov/webdocs/DataFiles/80591/FoodAccessResearchAtlasData2015.xlsx'
        try:
            download_file(url_2015, filepath_2015, "Food Access Atlas 2015")
        except Exception as e:
            print(f"⚠ Could not download 2015 data: {e}")
            print(f"  Manual download options:")
            print(f"    1. Download from: {url_2015}")
            print(f"    2. Or place file at: {local_2015}")
            print(f"    3. Or place file at: {filepath_2015}")

def collect_census_acs():
    """Download Census ACS data via API for all US tracts using direct Census API."""
    print("\n=== Collecting Census ACS Data via API ===")
    
    # Use provided API key or environment variable
    api_key = os.getenv('CENSUS_API_KEY', '309fb7c3fdce86563a88d7124877c7c7f3c9d91c')
    if not api_key:
        print("⚠ CENSUS_API_KEY not found")
        print("  Get a free key at: https://api.census.gov/key.html")
        print("  For now, skipping Census ACS collection")
        return None
    
    # Check if data already exists
    acs_file = CENSUS_DIR / "census_acs_api_data.csv"
    if acs_file.exists() and acs_file.stat().st_size > 1024 * 1024:  # > 1MB
        print(f"✓ Census ACS API data already exists ({acs_file.stat().st_size / (1024*1024):.1f} MB)")
        print("  To refresh, delete the file and run again")
        return acs_file
    
    print(f"Using Census API key: {api_key[:10]}...")
    print("Collecting ACS 5-year estimates for all US census tracts...")
    print("  Using 2023 ACS 5-year data (most recent)")
    print("  This will take several minutes (state-by-state collection)")
    
    # Use 2023 ACS 5-year data (most recent available)
    year = 2023
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    
    # All ACS variables we need to collect
    # Note: Variable codes ending in 'E' are estimates
    variables = {
        # Income and Poverty
        'B19013_001E': 'MEDIANHHI',  # Median household income
        'B17001_002E': 'POVERTYN',  # Population below poverty level
        'B17001_001E': 'POVDENOM',  # Population for whom poverty status determined
        
        # Education
        'B15003_022E': 'BACHELORS',  # Bachelor's degree
        'B15003_023E': 'MASTERS',  # Master's degree
        'B15003_024E': 'PROFESSIONAL',  # Professional degree
        'B15003_025E': 'DOCTORATE',  # Doctorate degree
        'B15003_001E': 'EDU_TOTAL',  # Total population 25+ for education
        
        # Housing
        'B25064_001E': 'MEDGRENT',  # Median gross rent
        'B25077_001E': 'MEDHOMEVAL',  # Median home value
        'B25003_002E': 'OWNEROCC',  # Owner-occupied housing units
        'B25003_003E': 'RENTEROCC',  # Renter-occupied housing units
        'B25001_001E': 'HUTOTAL',  # Total housing units
        
        # Transportation
        'B08301_021E': 'HH_NOVEH',  # Households with no vehicle available
        'B08301_001E': 'HH_TOTAL',  # Total households for vehicle calculation
        
        # Population
        'B01001_001E': 'POPTOTAL',  # Total population
        'B01001_020E': 'M_65_66',  # Male 65-66
        'B01001_021E': 'M_67_69',  # Male 67-69
        'B01001_022E': 'M_70_74',  # Male 70-74
        'B01001_023E': 'M_75_79',  # Male 75-79
        'B01001_024E': 'M_80_84',  # Male 80-84
        'B01001_025E': 'M_85PLUS',  # Male 85+
        'B01001_044E': 'F_65_66',  # Female 65-66
        'B01001_045E': 'F_67_69',  # Female 67-69
        'B01001_046E': 'F_70_74',  # Female 70-74
        'B01001_047E': 'F_75_79',  # Female 75-79
        'B01001_048E': 'F_80_84',  # Female 80-84
        'B01001_049E': 'F_85PLUS',  # Female 85+
        
        # Demographics
        'B03002_003E': 'WHITENH',  # White alone, not Hispanic
        'B03002_004E': 'BLACKNH',  # Black or African American alone
        'B03002_005E': 'AMINDNH',  # American Indian and Alaska Native alone
        'B03002_006E': 'ASIANNH',  # Asian alone
        'B03002_012E': 'HISPPOP',  # Hispanic or Latino population
    }
    
    # Get list of all states (FIPS codes 01-56)
    state_fips = [f"{i:02d}" for i in range(1, 57)]  # 01 to 56
    
    all_data = []
    failed_states = []
    
    print(f"Collecting data for {len(state_fips)} states/territories...")
    
    # Build variable list for API call
    var_list = list(variables.keys())
    var_list.append('NAME')  # Include NAME for verification
    
    for state_fips_code in tqdm(state_fips, desc="Collecting by state"):
        try:
            # Build API URL for tract-level data
            # Format: get=NAME,var1,var2,...&for=tract:*&in=state:XX&key=API_KEY
            var_string = ','.join(var_list)
            params = {
                'get': var_string,
                'for': 'tract:*',  # All tracts
                'in': f'state:{state_fips_code}',  # Specific state
                'key': api_key
            }
            
            # Make API request
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            if len(data) < 2:  # Header row + at least one data row
                continue
            
            # First row is headers
            headers = data[0]
            data_rows = data[1:]
            
            # Process each tract
            for row_data in data_rows:
                # Create dictionary from headers and row data
                tract_dict = dict(zip(headers, row_data))
                
                # Extract geographic identifiers
                state = tract_dict.get('state', '')
                county = tract_dict.get('county', '')
                tract = tract_dict.get('tract', '')
                
                if state and county and tract:
                    # Remove decimal point from tract if present
                    tract_clean = tract.replace('.', '')
                    # Pad tract to 6 digits
                    tract_padded = tract_clean.zfill(6)
                    census_tract = f"{state}{county}{tract_padded}"
                    
                    # Create row with all variables
                    row = {
                        'GEOID': f"1400000US{state}{county}{tract_padded}",
                        'GEOID2': census_tract,
                        'STATE': state,
                        'COUNTY': county,
                        'TRACT': tract_clean,
                        'CensusTract': census_tract,
                    }
                    
                    # Add all variable values
                    for var_code, var_name in variables.items():
                        value = tract_dict.get(var_code, None)
                        # Convert to numeric, handling special values
                        try:
                            if value is not None:
                                value = int(value)
                                # Handle -666666666 (suppressed), -222222222 (not available), -999999999 (error)
                                if value in [-666666666, -222222222, -999999999]:
                                    row[var_name] = None
                                else:
                                    row[var_name] = value
                            else:
                                row[var_name] = None
                        except (ValueError, TypeError):
                            row[var_name] = None
                    
                    all_data.append(row)
            
            # Be polite to the API - small delay between states
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            failed_states.append((state_fips_code, f"API Error: {str(e)}"))
            print(f"  ⚠ State {state_fips_code}: {e}")
            time.sleep(2)  # Longer delay on error
            continue
        except Exception as e:
            failed_states.append((state_fips_code, str(e)))
            print(f"  ⚠ State {state_fips_code}: {e}")
            time.sleep(1)
            continue
    
    if not all_data:
        print("⚠ No data collected. Check API key and network connection.")
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD INSTRUCTIONS FOR CENSUS ACS DATA")
        print("="*60)
        print("If API connection fails, download manually:")
        print("  1. Visit: https://data.census.gov/cedsci/")
        print("  2. Search for: 'ACS 5-Year Estimates Data Profiles'")
        print("  3. Select: 2023 ACS 5-year estimates (or most recent)")
        print("  4. Geography: Census Tract")
        print("  5. Download all states and save to:")
        print(f"     {acs_file}")
        print("  6. Required variables:")
        for var_code, var_name in list(variables.items())[:10]:
            print(f"     - {var_code}: {var_name}")
        print(f"     ... and {len(variables)-10} more variables")
        print("="*60)
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Calculate derived variables
    print("Calculating derived variables...")
    
    # Poverty rate
    if 'POVERTYN' in df.columns and 'POVDENOM' in df.columns:
        df['POV100RATE'] = (df['POVERTYN'] / df['POVDENOM'].replace(0, pd.NA)) * 100
    
    # Education rate (bachelor's or higher)
    if 'BACHELORS' in df.columns and 'MASTERS' in df.columns and 'PROFESSIONAL' in df.columns and 'DOCTORATE' in df.columns and 'EDU_TOTAL' in df.columns:
        df['BACHELORS_PLUS'] = (
            df['BACHELORS'].fillna(0) + 
            df['MASTERS'].fillna(0) + 
            df['PROFESSIONAL'].fillna(0) + 
            df['DOCTORATE'].fillna(0)
        )
        df['BACHELORS'] = df['BACHELORS_PLUS']  # Update to include all higher ed
        df = df.drop(columns=['BACHELORS_PLUS'], errors='ignore')
    
    # Total households (if not available, use housing units)
    if 'HH_TOTAL' not in df.columns and 'HUTOTAL' in df.columns:
        df['HH_TOTAL'] = df['HUTOTAL']
    
    # Homeownership rate
    if 'OWNEROCC' in df.columns and 'HUTOTAL' in df.columns:
        df['HOMEOWNPCT'] = (df['OWNEROCC'] / df['HUTOTAL'].replace(0, pd.NA)) * 100
    
    # Population 65+
    age_65_cols = ['M_65_66', 'M_67_69', 'M_70_74', 'M_75_79', 'M_80_84', 'M_85PLUS',
                   'F_65_66', 'F_67_69', 'F_70_74', 'F_75_79', 'F_80_84', 'F_85PLUS']
    if all(col in df.columns for col in age_65_cols):
        df['AGE65UP'] = df[age_65_cols].sum(axis=1)
    
    # Average household size (approximate)
    if 'POPTOTAL' in df.columns and 'HH_TOTAL' in df.columns:
        df['AVGHHSIZE'] = df['POPTOTAL'] / df['HH_TOTAL'].replace(0, pd.NA)
    
    # Save to CSV
    df.to_csv(acs_file, index=False)
    
    print(f"\n✓ Collected Census ACS data for {len(df)} tracts")
    print(f"  Saved to: {acs_file}")
    print(f"  File size: {acs_file.stat().st_size / (1024*1024):.1f} MB")
    
    if failed_states:
        print(f"\n⚠ Failed to collect data for {len(failed_states)} states:")
        for state, error in failed_states[:5]:  # Show first 5
            print(f"  State {state}: {error}")
        if len(failed_states) > 5:
            print(f"  ... and {len(failed_states) - 5} more")
    
    return acs_file

def collect_cdc_places():
    """Download CDC PLACES Local Data for Better Health via API."""
    print("\n=== Collecting CDC PLACES Data ===")
    
    filepath = HEALTH_DIR / "CDC_PLACES_Tract.csv"
    
    # Check if file exists and is complete (CDC PLACES health outcomes should be < 500MB)
    # If file is > 5GB, it's likely COVID data, not health outcomes
    if filepath.exists():
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        if file_size_mb > 5000:
            print(f"⚠ Existing file is very large ({file_size_mb:.1f} MB) - likely COVID data, not health outcomes")
            print("  Will attempt to download correct health outcomes file...")
        elif file_size_mb > 100:
            print(f"✓ CDC PLACES data already exists ({file_size_mb:.1f} MB)")
            return filepath
    
    # Use CDC PLACES API endpoint for Local Data for Better Health
    # Direct CSV download endpoint (downloads full dataset ~726MB)
    csv_url = "https://data.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD"
    json_url = "https://data.cdc.gov/resource/cwsq-ngmh.json"
    
    print("  Attempting to download CDC PLACES health outcomes data...")
    print(f"  Using direct CSV download endpoint (full dataset ~726MB)...")
    
    try:
        # Direct CSV download - downloads complete dataset
        print("  Downloading full dataset (this may take a few minutes)...")
        
        response = requests.get(csv_url, timeout=600, stream=True)
        
        if response.status_code == 200:
            # Download CSV directly
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                print(f"  Downloading CSV file ({total_size / (1024*1024):.1f} MB)...")
            else:
                print(f"  Downloading CSV file (size unknown, streaming)...")
            
            with open(filepath, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="    Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # If size unknown, download without progress bar
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (10 * 1024 * 1024) == 0:  # Print every 10MB
                                print(f"    Downloaded: {downloaded / (1024*1024):.1f} MB")
            
            # Verify it's health outcomes, not COVID data
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  Downloaded file size: {file_size_mb:.1f} MB")
            
            if file_size_mb > 5000:
                print(f"  ⚠ Downloaded file is very large ({file_size_mb:.1f} MB) - likely wrong file")
                filepath.unlink()  # Delete it
                raise Exception("Downloaded file appears to be COVID data, not health outcomes")
            
            # Load and filter for census tract level
            print("  Loading and filtering for census tract level data...")
            print("  Note: This may take a few minutes for large file...")
            df = pd.read_csv(filepath, low_memory=False)
            print(f"  Total records downloaded: {len(df):,}")
            
            # Filter for census tracts (LocationName should be 11 digits)
            # Note: Column name might be 'LocationName' (capitalized) or 'locationname' (lowercase)
            location_col = None
            for col in ['LocationName', 'locationname', 'Locationname']:
                if col in df.columns:
                    location_col = col
                    break
            
            if location_col:
                initial_count = len(df)
                df[location_col] = df[location_col].astype(str)
                df = df[df[location_col].str.len() == 11].copy()
                print(f"  Filtered to census tracts: {len(df):,} records (from {initial_count:,})")
            else:
                print(f"  ⚠ No location name column found, keeping all records")
            
            # Verify it contains health outcome indicators
            health_indicators = ['diabetes', 'obesity', 'hypertension', 'physical_inactivity', 'bphigh', 'cancer', 'measure', 'Measure']
            has_health_data = any(ind.lower() in str(df.columns).lower() for ind in health_indicators)
            
            if not has_health_data:
                print(f"  ⚠ File does not contain expected health outcome indicators")
                print(f"  Columns found: {list(df.columns)[:10]}")
                filepath.unlink()
                raise Exception("Downloaded file does not contain health outcomes data")
            
            # Save filtered data
            print("  Saving filtered data...")
            df.to_csv(filepath, index=False)
            final_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            print(f"✓ Downloaded and processed CDC PLACES health outcomes data")
            print(f"  Final file size: {final_size_mb:.1f} MB")
            print(f"  Census tract records: {len(df):,}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Sample columns: {list(df.columns)[:5]}")
            return filepath
        else:
            raise Exception(f"CSV download failed with status {response.status_code}")
            
    except Exception as csv_error:
        print(f"  CSV download failed: {csv_error}")
        print("  Trying JSON API as fallback...")
        
        try:
            # Fallback to JSON API using correct Socrata endpoint
            # Socrata returns 1000 records by default, need to paginate
            all_data = []
            offset = 0
            batch_size = 1000  # Socrata default limit
            
            print("  Using JSON API with pagination...")
            
            while True:
                params = {
                    '$limit': batch_size,
                    '$offset': offset
                }
                
                print(f"  Fetching records {offset:,} to {offset + batch_size:,}...")
                
                # Add headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
                
                response = requests.get(json_url, params=params, headers=headers, timeout=120)
                
                if response.status_code == 403:
                    raise Exception("API returned 403 Forbidden - may require authentication")
                
                response.raise_for_status()
                
                data = response.json()
                
                if not data or len(data) == 0:
                    break
                
                all_data.extend(data)
                print(f"    Retrieved {len(data)} records (total: {len(all_data):,})")
                
                if len(data) < batch_size:
                    break
                
                offset += batch_size
                time.sleep(0.5)  # Be polite to API
            
            if not all_data:
                print("⚠ No data retrieved from API")
                raise Exception("No data returned from CDC PLACES API")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            print(f"  ✓ Retrieved {len(df)} records from API")
            
            # Filter for census tracts (locationname should be 11 digits)
            if 'locationname' in df.columns:
                initial_count = len(df)
                df['locationname'] = df['locationname'].astype(str)
                df = df[df['locationname'].str.len() == 11].copy()
                print(f"  Filtered to census tracts: {len(df):,} records (from {initial_count:,})")
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            print(f"✓ Saved CDC PLACES data to: {filepath}")
            print(f"  File size: {filepath.stat().st_size / (1024*1024):.1f} MB")
            print(f"  Columns: {len(df.columns)}")
            
            return filepath
            
        except Exception as json_error:
            print(f"  JSON API also failed: {json_error}")
            raise
        
    except Exception as e:
        print(f"⚠ Could not download CDC PLACES data via API: {e}")
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD INSTRUCTIONS FOR CDC PLACES DATA")
        print("="*60)
        print("Download CDC PLACES Local Data for Better Health:")
        print("  1. Visit: https://data.cdc.gov/browse?category=Places")
        print("  2. Select: 'PLACES: Local Data for Better Health, Census Tract Data 2023 release'")
        print("  3. Or direct link: https://data.cdc.gov/Places/PLACES-Local-Data-for-Better-Health-Census-Tract-D/cwsq-ngmh")
        print("  4. Click 'Export' -> 'CSV'")
        print(f"  5. Save to: {filepath}")
        print("  6. Make sure it's HEALTH OUTCOMES data, not COVID case data")
        print("     (File should be < 500MB, not > 5GB)")
        print("="*60)
        return None

def collect_gtfs_feeds():
    """Download GTFS transit feeds for major cities."""
    print("\n=== Collecting GTFS Transit Feeds ===")
    
    # Major transit agencies with public GTFS feeds
    gtfs_sources = {
        'NYC_MTA': 'http://web.mta.info/developers/data/nyct/subway/google_transit.zip',
        'LA_Metro': 'https://gitlab.com/LACMTA/gtfs_rail/-/archive/master/gtfs_rail-master.zip',
        # Add more as needed
    }
    
    gtfs_dir = INFRASTRUCTURE_DIR / "gtfs"
    gtfs_dir.mkdir(exist_ok=True)
    
    for agency, url in gtfs_sources.items():
        filepath = gtfs_dir / f"{agency}.zip"
        # Check if file exists and is complete (GTFS files are typically > 100KB)
        if filepath.exists() and filepath.stat().st_size > 100 * 1024:
            print(f"✓ {agency} GTFS already exists ({filepath.stat().st_size / (1024*1024):.1f} MB)")
        else:
            try:
                download_file(url, filepath, f"GTFS feed for {agency}")
            except Exception as e:
                print(f"⚠ Could not download {agency} GTFS: {e}")
                print(f"  You can resume this download by running the script again")

def collect_zillow_data():
    """Check for Zillow housing data files."""
    print("\n=== Collecting Zillow Housing Data ===")
    
    zillow_dir = HOUSING_DIR / "Zillow"
    
    if not zillow_dir.exists():
        print(f"⚠ Zillow data directory not found: {zillow_dir}")
        print("\n" + "="*60)
        print("MANUAL DOWNLOAD INSTRUCTIONS FOR ZILLOW DATA")
        print("="*60)
        print("Download Zillow data from: https://www.zillow.com/research/data/")
        print(f"Save files to: {zillow_dir}")
        print("\nRequired files:")
        print("  - Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
        print("  - Metro_zori_uc_sfrcondomfr_sm_month.csv")
        print("  - Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
        print("  - Metro_new_renter_income_needed_uc_sfrcondomfr_sm_sa_month.csv")
        print("  - Metro_new_renter_affordability_uc_sfrcondomfr_sm_sa_month.csv")
        print("  - Metro_market_temp_index_uc_sfrcondo_month.csv")
        print("="*60)
        return
    
    # Check for required files
    required_files = [
        "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "Metro_zori_uc_sfrcondomfr_sm_month.csv",
        "Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "Metro_new_renter_income_needed_uc_sfrcondomfr_sm_sa_month.csv",
        "Metro_new_renter_affordability_uc_sfrcondomfr_sm_sa_month.csv",
        "Metro_market_temp_index_uc_sfrcondo_month.csv"
    ]
    
    found_files = []
    missing_files = []
    
    for filename in required_files:
        filepath = zillow_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            found_files.append(f"  ✓ {filename} ({size_mb:.1f} MB)")
        else:
            missing_files.append(f"  ✗ {filename}")
    
    if found_files:
        print(f"✓ Found {len(found_files)}/{len(required_files)} Zillow data files:")
        for f in found_files:
            print(f)
    
    if missing_files:
        print(f"\n⚠ Missing {len(missing_files)} files:")
        for f in missing_files:
            print(f)
        print("\nDownload from: https://www.zillow.com/research/data/")
        print(f"Save to: {zillow_dir}")

def query_overpass_api(overpass_query, max_retries=3, retry_delay=5):
    """
    Query OpenStreetMap Overpass API with retry logic.
    
    Args:
        overpass_query: Overpass QL query string
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        JSON response from API or None if failed
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                overpass_url,
                data={'data': overpass_query},
                timeout=300  # 5 minute timeout for large queries
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"  ⚠ Query timeout, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("  ⚠ Query failed after multiple attempts (timeout)")
                return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ Request error: {e}, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"  ⚠ Request failed after multiple attempts: {e}")
                return None
    
    return None

def collect_grocery_store_data():
    """Collect grocery store location data using OpenStreetMap Overpass API."""
    print("\n=== Collecting Grocery Store Data (OpenStreetMap) ===")
    
    # Check if data already exists
    grocery_file = INFRASTRUCTURE_DIR / "grocery_stores_osm.csv"
    if grocery_file.exists() and grocery_file.stat().st_size > 1024:
        print(f"✓ Grocery store data already exists ({grocery_file.stat().st_size / 1024:.1f} KB)")
        print("  To refresh, delete the file and run again")
        return
    
    # OSM tags for grocery stores and supermarkets
    # shop=supermarket: Large supermarkets
    # shop=grocery: Smaller grocery stores
    # shop=convenience: Convenience stores (optional, may want to exclude)
    # amenity=marketplace: Markets
    
    print("Querying OpenStreetMap for grocery stores and supermarkets...")
    print("  This may take several minutes for the entire US...")
    
    # Query for supermarkets and grocery stores
    # Using bounding box for continental US (to avoid timeout, we'll query in chunks)
    # US bounding box: North: 49.5, South: 24.5, West: -125.0, East: -66.5
    
    # Query in regional chunks to avoid timeout
    regions = [
        {"name": "Northeast", "bbox": [49.5, -85.0, 40.0, -66.5]},  # North, West, South, East
        {"name": "Southeast", "bbox": [40.0, -85.0, 24.5, -75.0]},
        {"name": "Midwest", "bbox": [49.5, -105.0, 35.0, -85.0]},
        {"name": "Southwest", "bbox": [40.0, -125.0, 31.0, -105.0]},
        {"name": "West", "bbox": [49.5, -125.0, 31.0, -105.0]},
    ]
    
    all_stores = []
    
    for region in tqdm(regions, desc="Querying regions"):
        bbox = region["bbox"]
        # Overpass QL query: get all nodes and ways with shop=supermarket or shop=grocery
        overpass_query = f"""
        [out:json][timeout:300];
        (
          node["shop"="supermarket"]({bbox[2]},{bbox[1]},{bbox[0]},{bbox[3]});
          node["shop"="grocery"]({bbox[2]},{bbox[1]},{bbox[0]},{bbox[3]});
          way["shop"="supermarket"]({bbox[2]},{bbox[1]},{bbox[0]},{bbox[3]});
          way["shop"="grocery"]({bbox[2]},{bbox[1]},{bbox[0]},{bbox[3]});
        );
        out center meta;
        """
        
        result = query_overpass_api(overpass_query)
        
        if result and 'elements' in result:
            region_stores = 0
            for element in result['elements']:
                store_data = {
                    'osm_id': element.get('id'),
                    'osm_type': element.get('type'),
                    'name': element.get('tags', {}).get('name', ''),
                    'shop_type': element.get('tags', {}).get('shop', ''),
                    'brand': element.get('tags', {}).get('brand', ''),
                    'operator': element.get('tags', {}).get('operator', ''),
                }
                
                # Get coordinates
                if element.get('type') == 'node':
                    store_data['latitude'] = element.get('lat')
                    store_data['longitude'] = element.get('lon')
                elif element.get('type') == 'way' and 'center' in element:
                    store_data['latitude'] = element['center'].get('lat')
                    store_data['longitude'] = element['center'].get('lon')
                else:
                    continue  # Skip if no coordinates
                
                # Additional metadata
                if 'timestamp' in element:
                    store_data['last_updated'] = element['timestamp']
                
                all_stores.append(store_data)
                region_stores += 1
            
            print(f"  ✓ {region['name']}: Found {region_stores} stores")
        else:
            print(f"  ⚠ {region['name']}: Query failed or returned no results")
        
        # Be polite to the API - add delay between queries
        time.sleep(2)
    
    if not all_stores:
        print("⚠ No grocery stores found. This may indicate:")
        print("  - Network connectivity issues")
        print("  - Overpass API timeout (try running again)")
        print("  - Query syntax issues")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_stores)
    
    # Remove duplicates based on coordinates (within small tolerance)
    if len(df) > 0:
        # Round coordinates to 4 decimal places (~11 meters) to identify duplicates
        df['lat_rounded'] = df['latitude'].round(4)
        df['lon_rounded'] = df['longitude'].round(4)
        df = df.drop_duplicates(subset=['lat_rounded', 'lon_rounded'], keep='first')
        df = df.drop(columns=['lat_rounded', 'lon_rounded'])
    
    # Save to CSV
    df.to_csv(grocery_file, index=False)
    print(f"✓ Collected {len(df)} unique grocery stores")
    print(f"  Saved to: {grocery_file}")
    print(f"  File size: {grocery_file.stat().st_size / 1024:.1f} KB")
    
    # Print summary statistics
    if len(df) > 0:
        print("\n  Summary:")
        print(f"    Supermarkets: {len(df[df['shop_type'] == 'supermarket'])}")
        print(f"    Grocery stores: {len(df[df['shop_type'] == 'grocery'])}")
        print(f"    With names: {df['name'].notna().sum()}")
        print(f"    With brands: {df['brand'].notna().sum()}")

def main():
    """Main data collection function."""
    print("=" * 60)
    print("Food Desert Prediction - Data Collection")
    print("=" * 60)
    print("Note: Downloads can be stopped and resumed.")
    print("      Partial files will be automatically detected and resumed.")
    print("=" * 60)
    
    # Collect all datasets
    try:
        collect_food_access_atlas()
        collect_census_acs()
        collect_cdc_places()
        collect_gtfs_feeds()
        collect_zillow_data()
        collect_grocery_store_data()
        
        print("\n" + "=" * 60)
        print("Data collection complete!")
        print(f"Data saved to organized folders:")
        print(f"  - Census: {CENSUS_DIR}")
        print(f"  - Food Access: {FOOD_ACCESS_DIR}")
        print(f"  - Health: {HEALTH_DIR}")
        print(f"  - Housing: {HOUSING_DIR}")
        print(f"  - Infrastructure: {INFRASTRUCTURE_DIR}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user.")
        print("  You can resume by running this script again.")
        print("  Partial downloads will be automatically continued.")
        print(f"  Check the organized data folders for partial files.")
    except Exception as e:
        print(f"\n\n⚠ Error during data collection: {e}")
        print("  You can resume by running this script again.")
        print("  Partial downloads will be automatically continued.")

if __name__ == "__main__":
    main()

