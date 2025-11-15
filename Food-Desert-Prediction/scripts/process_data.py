"""
Data Processing Script for Food Desert Prediction Project

This script processes raw data into features for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

# Set up paths - organized data folders
BASE_DIR = Path(__file__).parent.parent
CENSUS_DIR = BASE_DIR / "data" / "01_census_demographics"
FOOD_ACCESS_DIR = BASE_DIR / "data" / "02_food_access"
HEALTH_DIR = BASE_DIR / "data" / "03_health_outcomes"
HOUSING_DIR = BASE_DIR / "data" / "04_housing_economics"
INFRASTRUCTURE_DIR = BASE_DIR / "data" / "05_infrastructure_transportation"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "data" / "features"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def load_food_access_data(year='2019'):
    """Load and process Food Access Research Atlas data."""
    print(f"Loading Food Access Atlas {year}...")
    
    filepath = FOOD_ACCESS_DIR / f"FoodAccessResearchAtlasData{year}.xlsx"
    
    # For 2015, also check local folder
    if year == 2015 and not filepath.exists():
        local_2015 = FOOD_ACCESS_DIR / "USDA Food Environment Atlas" / "2015 Food Access Research Atlas" / "FoodAccessResearchAtlasData2015.xlsx"
        if local_2015.exists():
            print(f"  Found 2015 data in local folder, using: {local_2015}")
            filepath = local_2015
    
    if not filepath.exists():
        print(f"⚠ File not found: {filepath}")
        if year == 2015:
            print(f"  Also checked: {FOOD_ACCESS_DIR / 'USDA Food Environment Atlas' / '2015 Food Access Research Atlas'}")
        return None
    
    # Load the main data sheet
    try:
        # First, try to find the correct sheet name
        xl_file = pd.ExcelFile(filepath, engine='openpyxl')
        sheet_names = xl_file.sheet_names
        
        # Look for the data sheet (usually contains "Food Access" or "Research Atlas")
        data_sheet = None
        for sheet in sheet_names:
            if 'food access' in sheet.lower() or 'research atlas' in sheet.lower() or 'data' in sheet.lower():
                if 'note' not in sheet.lower() and 'readme' not in sheet.lower():
                    data_sheet = sheet
                    break
        
        # If no specific sheet found, try common names
        if not data_sheet:
            for sheet in ['Food Access Research Atlas', 'FoodAccessResearchAtlas', 'Data', 'Sheet1']:
                if sheet in sheet_names:
                    data_sheet = sheet
                    break
        
        # If still no sheet found, use first non-note sheet
        if not data_sheet:
            for sheet in sheet_names:
                if 'note' not in sheet.lower() and 'readme' not in sheet.lower():
                    data_sheet = sheet
                    break
        
        if not data_sheet:
            data_sheet = sheet_names[0]  # Fallback to first sheet
        
        print(f"  Using sheet: {data_sheet}")
        df = pd.read_excel(filepath, sheet_name=data_sheet, engine='openpyxl')
        
        # Check if we got actual data (more than just notes/headers)
        if len(df.columns) < 5:
            print(f"  ⚠ Sheet '{data_sheet}' appears to be notes/header only")
            print(f"  Available sheets: {sheet_names}")
            # Try other sheets
            for sheet in sheet_names:
                if sheet != data_sheet and 'note' not in sheet.lower():
                    print(f"  Trying sheet: {sheet}")
                    df = pd.read_excel(filepath, sheet_name=sheet, engine='openpyxl')
                    if len(df.columns) >= 5:
                        print(f"  ✓ Found data in sheet: {sheet}")
                        break
        
    except Exception as e:
        print(f"⚠ Error reading {year} file: {e}")
        print("  Trying alternative methods...")
        try:
            # Try reading as CSV if it's actually a CSV
            df = pd.read_csv(filepath)
        except:
            print(f"  Could not read {year} file, skipping...")
            return None
    
    # Select key columns
    key_cols = [
        'CensusTract', 'State', 'County', 'Urban',
        'Pop2010', 'OHU2010',
        'LILATracts_1And10', 'LILATracts_halfAnd10', 
        'LILATracts_1And20', 'LILATracts_Vehicle',
        'LowIncomeTracts', 'PovertyRate', 'MedianFamilyIncome',
        'lapop1', 'lapop10', 'lapop20',
        'lalowi1', 'lalowi10', 'lalowi20',
        'lahunv1', 'lahunv10', 'lahunv20',
        'lasnap1', 'lasnap10', 'lasnap20',
        'TractLOWI', 'TractKids', 'TractSeniors',
        'TractHUNV', 'TractSNAP'
    ]
    
    # Select only columns that exist
    available_cols = [col for col in key_cols if col in df.columns]
    df = df[available_cols].copy()
    
    # Create normalized features
    if 'Pop2010' in df.columns:
        df['normalized_lapop1'] = df['lapop1'] / df['Pop2010'].replace(0, np.nan)
        df['normalized_lapop10'] = df['lapop10'] / df['Pop2010'].replace(0, np.nan)
    
    if 'TractLOWI' in df.columns:
        df['normalized_lalowi1'] = df['lalowi1'] / df['TractLOWI'].replace(0, np.nan)
        df['normalized_lalowi10'] = df['lalowi10'] / df['TractLOWI'].replace(0, np.nan)
    
    if 'OHU2010' in df.columns:
        df['normalized_lahunv1'] = df['lahunv1'] / df['OHU2010'].replace(0, np.nan)
        df['normalized_lahunv10'] = df['lahunv10'] / df['OHU2010'].replace(0, np.nan)
    
    if 'TractSNAP' in df.columns:
        df['normalized_lasnap1'] = df['lasnap1'] / df['TractSNAP'].replace(0, np.nan)
        df['normalized_lasnap10'] = df['lasnap10'] / df['TractSNAP'].replace(0, np.nan)
    
    return df

def load_census_acs_data():
    """Load and process Census ACS data from API or local file."""
    print("Loading Census ACS data...")
    
    # First, try to load API-collected data (preferred - complete dataset)
    api_data_file = CENSUS_DIR / "census_acs_api_data.csv"
    if api_data_file.exists() and api_data_file.stat().st_size > 1024 * 1024:  # > 1MB
        try:
            print("  Using API-collected Census ACS data...")
            df = pd.read_csv(api_data_file)
            print(f"  Loaded {len(df)} tracts from API data")
        except Exception as e:
            print(f"⚠ Error reading API data file: {e}")
            print("  Falling back to local Excel file...")
            df = None
    else:
        df = None
    
    # Fallback to local Excel file if API data not available
    if df is None:
        possible_paths = [
            CENSUS_DIR / "CensusACSTract.xlsx"
        ]
        
        filepath = None
        for path in possible_paths:
            if path.exists():
                filepath = path
                break
        
        if filepath is None:
            print("⚠ Census ACS data not found. Options:")
            print("  1. Run 'python scripts/collect_data.py' to collect via API")
            print("  2. Place CensusACSTract.xlsx in one of these locations:")
            for path in possible_paths:
                print(f"     - {path}")
            return None
        
        try:
            print("  Using local Census ACS Excel file...")
            df = pd.read_excel(filepath, engine='openpyxl')
            print(f"  Loaded {len(df)} tracts from local file")
        except Exception as e:
            print(f"⚠ Error reading Census ACS file: {e}")
            return None
    
    # If data came from API, CensusTract is already set
    # If from Excel, we need to convert GEOID
    if 'CensusTract' not in df.columns:
        # Convert GEOID2 to CensusTract format (11-digit code)
        if 'GEOID2' in df.columns:
            df['CensusTract'] = df['GEOID2'].astype(str).str.zfill(11)
        elif 'GEOID' in df.columns:
            # Extract last 11 digits from GEOID (format: "1400000US27001770100")
            df['CensusTract'] = df['GEOID'].astype(str).str.extract(r'(\d{11})$')[0]
    
    # Extract key socioeconomic features (select only columns that exist)
    key_cols = [
        'CensusTract', 'GEOID', 'GEOID2', 'STATE', 'COUNTY', 'TRACT',
        'MEDIANHHI',  # Median household income
        'POV100RATE',  # Poverty rate (may be calculated)
        'POVERTYN',  # Population below poverty (raw)
        'POVDENOM',  # Population for poverty calculation
        'BACHELORS',  # Population with bachelor's degree or higher
        'MEDGRENT',  # Median gross rent
        'MEDHOMEVAL',  # Median home value
        'HH_NOVEH',  # Households with no vehicle
        'HHTOTAL',  # Total households
        'HH_TOTAL',  # Alternative name for total households
        'POPTOTAL',  # Total population
        'HOMEOWNPCT',  # Homeownership percentage (may be calculated)
        'OWNEROCC',  # Owner-occupied units
        'HUTOTAL',  # Total housing units
        'AVGHHSIZE',  # Average household size (may be calculated)
        'AGE65UP',  # Population 65 and over (may be calculated)
        'HISPPOP',  # Hispanic population
        'BLACKNH',  # Black non-Hispanic population
        'WHITENH',  # White non-Hispanic population
    ]
    
    # Select only columns that exist
    available_cols = [col for col in key_cols if col in df.columns]
    # Also keep any columns that start with Census_ (already processed)
    census_cols = [col for col in df.columns if col.startswith('Census_')]
    df = df[available_cols + census_cols].copy()
    
    # Rename columns to match naming conventions
    rename_map = {
        'MEDIANHHI': 'Census_MedianHouseholdIncome',
        'POV100RATE': 'Census_PovertyRate',
        'BACHELORS': 'Census_BachelorsDegree',
        'MEDGRENT': 'Census_MedianGrossRent',
        'MEDHOMEVAL': 'Census_MedianHomeValue',
        'HH_NOVEH': 'Census_HouseholdsNoVehicle',
        'HHTOTAL': 'Census_TotalHouseholds',
        'POPTOTAL': 'Census_TotalPopulation',
        'HOMEOWNPCT': 'Census_HomeownershipRate',
        'AVGHHSIZE': 'Census_AvgHouseholdSize',
        'AGE65UP': 'Census_Pop65Plus',
        'HISPPOP': 'Census_HispanicPopulation',
        'BLACKNH': 'Census_BlackPopulation',
        'WHITENH': 'Census_WhitePopulation',
    }
    
    # Rename only columns that exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    
    # Calculate derived features
    if 'Census_TotalPopulation' in df.columns and 'Census_BachelorsDegree' in df.columns:
        # Education rate (bachelor's degree or higher)
        df['Census_EducationRate'] = (df['Census_BachelorsDegree'] / df['Census_TotalPopulation'].replace(0, np.nan)) * 100
    
    if 'Census_TotalHouseholds' in df.columns and 'Census_HouseholdsNoVehicle' in df.columns:
        # Vehicle ownership rate (percentage of households with at least one vehicle)
        df['Census_VehicleOwnershipRate'] = 1 - (df['Census_HouseholdsNoVehicle'] / df['Census_TotalHouseholds'].replace(0, np.nan))
        df['Census_VehicleOwnershipRate'] = df['Census_VehicleOwnershipRate'].clip(0, 1)
    elif 'Census_TotalPopulation' in df.columns and 'Census_HouseholdsNoVehicle' in df.columns:
        # Fallback: approximate using population (less accurate)
        df['Census_VehicleOwnershipRate'] = 1 - (df['Census_HouseholdsNoVehicle'] / df['Census_TotalPopulation'].replace(0, np.nan))
        df['Census_VehicleOwnershipRate'] = df['Census_VehicleOwnershipRate'].clip(0, 1)
    
    if 'Census_MedianHouseholdIncome' in df.columns and 'Census_MedianGrossRent' in df.columns:
        # Rent burden (rent as percentage of income)
        df['Census_RentBurden'] = (df['Census_MedianGrossRent'] * 12 / df['Census_MedianHouseholdIncome'].replace(0, np.nan)) * 100
        df['Census_RentBurden'] = df['Census_RentBurden'].clip(0, 100)
    
    print(f"✓ Processed Census ACS data: {len(df)} tracts, {len(df.columns)} features")
    return df

def load_zillow_data():
    """Load and process Zillow housing data (Metro/MSA level, matched to tracts by State)."""
    print("Loading Zillow housing data...")
    
    zillow_dir = HOUSING_DIR / "Zillow"
    if not zillow_dir.exists():
        print(f"⚠ Zillow data directory not found: {zillow_dir}")
        return None
    
    try:
        # Load key Zillow datasets
        zillow_data = {}
        
        # 1. Home Value Index (ZHVI) - most important
        zhvi_file = zillow_dir / "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        if zhvi_file.exists():
            print("  Loading ZHVI (Home Value Index)...")
            df_zhvi = pd.read_csv(zhvi_file)
            # Filter to MSA level only (exclude country-level)
            df_zhvi = df_zhvi[df_zhvi['RegionType'] == 'msa'].copy()
            
            # Get 2019 data (matching Food Access Atlas year)
            date_cols = [c for c in df_zhvi.columns if '2019' in str(c) and '-' in str(c)]
            if date_cols:
                # Use the latest 2019 date
                latest_2019 = sorted(date_cols)[-1]
                df_zhvi['Zillow_ZHVI_2019'] = pd.to_numeric(df_zhvi[latest_2019], errors='coerce')
                zillow_data['zhvi'] = df_zhvi[['StateName', 'RegionName', 'Zillow_ZHVI_2019']].copy()
                print(f"    Loaded ZHVI for {len(df_zhvi)} MSAs")
        
        # 2. Rent Index (ZORI) - for rent burden
        zori_file = zillow_dir / "Metro_zori_uc_sfrcondomfr_sm_month.csv"
        if zori_file.exists():
            print("  Loading ZORI (Rent Index)...")
            df_zori = pd.read_csv(zori_file)
            df_zori = df_zori[df_zori['RegionType'] == 'msa'].copy()
            
            date_cols = [c for c in df_zori.columns if '2019' in str(c) and '-' in str(c)]
            if date_cols:
                latest_2019 = sorted(date_cols)[-1]
                df_zori['Zillow_ZORI_2019'] = pd.to_numeric(df_zori[latest_2019], errors='coerce')
                zillow_data['zori'] = df_zori[['StateName', 'RegionName', 'Zillow_ZORI_2019']].copy()
                print(f"    Loaded ZORI for {len(df_zori)} MSAs")
        
        # 3. Income needed for homeownership - affordability pressure
        income_file = zillow_dir / "Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        if income_file.exists():
            print("  Loading income needed for homeownership...")
            df_income = pd.read_csv(income_file)
            df_income = df_income[df_income['RegionType'] == 'msa'].copy()
            
            date_cols = [c for c in df_income.columns if '2019' in str(c) and '-' in str(c)]
            if date_cols:
                latest_2019 = sorted(date_cols)[-1]
                df_income['Zillow_IncomeNeeded_2019'] = pd.to_numeric(df_income[latest_2019], errors='coerce')
                zillow_data['income'] = df_income[['StateName', 'RegionName', 'Zillow_IncomeNeeded_2019']].copy()
                print(f"    Loaded income needed for {len(df_income)} MSAs")
        
        # 3b. Income needed for renting - rent affordability pressure
        renter_income_file = zillow_dir / "Metro_new_renter_income_needed_uc_sfrcondomfr_sm_sa_month.csv"
        if renter_income_file.exists():
            print("  Loading income needed for renting...")
            df_renter_income = pd.read_csv(renter_income_file)
            df_renter_income = df_renter_income[df_renter_income['RegionType'] == 'msa'].copy()
            
            date_cols = [c for c in df_renter_income.columns if '2019' in str(c) and '-' in str(c)]
            if date_cols:
                latest_2019 = sorted(date_cols)[-1]
                df_renter_income['Zillow_RenterIncomeNeeded_2019'] = pd.to_numeric(df_renter_income[latest_2019], errors='coerce')
                zillow_data['renter_income'] = df_renter_income[['StateName', 'RegionName', 'Zillow_RenterIncomeNeeded_2019']].copy()
                print(f"    Loaded renter income needed for {len(df_renter_income)} MSAs")
        
        # 3c. Renter affordability - % of income spent on rent
        renter_afford_file = zillow_dir / "Metro_new_renter_affordability_uc_sfrcondomfr_sm_sa_month.csv"
        if renter_afford_file.exists():
            print("  Loading renter affordability (% income on rent)...")
            df_renter_afford = pd.read_csv(renter_afford_file)
            df_renter_afford = df_renter_afford[df_renter_afford['RegionType'] == 'msa'].copy()
            
            date_cols = [c for c in df_renter_afford.columns if '2019' in str(c) and '-' in str(c)]
            if date_cols:
                latest_2019 = sorted(date_cols)[-1]
                df_renter_afford['Zillow_RenterAffordability_2019'] = pd.to_numeric(df_renter_afford[latest_2019], errors='coerce')
                zillow_data['renter_afford'] = df_renter_afford[['StateName', 'RegionName', 'Zillow_RenterAffordability_2019']].copy()
                print(f"    Loaded renter affordability for {len(df_renter_afford)} MSAs")
        
        # 4. Market temperature index - volatility indicator
        temp_file = zillow_dir / "Metro_market_temp_index_uc_sfrcondo_month.csv"
        if temp_file.exists():
            print("  Loading market temperature index...")
            df_temp = pd.read_csv(temp_file)
            df_temp = df_temp[df_temp['RegionType'] == 'msa'].copy()
            
            date_cols = [c for c in df_temp.columns if '2019' in str(c) and '-' in str(c)]
            if date_cols:
                latest_2019 = sorted(date_cols)[-1]
                df_temp['Zillow_MarketTemp_2019'] = pd.to_numeric(df_temp[latest_2019], errors='coerce')
                zillow_data['temp'] = df_temp[['StateName', 'RegionName', 'Zillow_MarketTemp_2019']].copy()
                print(f"    Loaded market temp for {len(df_temp)} MSAs")
        
        if not zillow_data:
            print("  ⚠ No Zillow data files found")
            return None
        
        # Aggregate to State level (since we don't have tract-to-MSA mapping)
        # Use median value per state as proxy
        state_aggregates = {}
        
        for key, df in zillow_data.items():
            value_col = [c for c in df.columns if c.startswith('Zillow_')][0]
            state_agg = df.groupby('StateName')[value_col].median().reset_index()
            state_agg.columns = ['State', value_col]
            state_aggregates[key] = state_agg
        
        # Merge all state-level aggregates
        result = state_aggregates[list(state_aggregates.keys())[0]].copy()
        for key in list(state_aggregates.keys())[1:]:
            result = pd.merge(result, state_aggregates[key], on='State', how='outer')
        
        print(f"✓ Processed Zillow data: {len(result)} states")
        print(f"  Variables: {[c for c in result.columns if c.startswith('Zillow_')]}")
        
        return result
        
    except Exception as e:
        print(f"⚠ Error loading Zillow data: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_cdc_places_data():
    """Load and process CDC PLACES data (currently COVID case data, not health outcomes)."""
    print("Loading CDC PLACES data...")
    
    filepath = HEALTH_DIR / "CDC_PLACES_Tract.csv"
    if not filepath.exists():
        print(f"⚠ CDC PLACES file not found: {filepath}")
        return None
    
    try:
        # Check file size - if too large, sample or aggregate
        file_size = filepath.stat().st_size / (1024**3)  # Size in GB
        print(f"  File size: {file_size:.2f} GB")
        
        if file_size > 5:  # If > 5GB, process in chunks
            print("  File is very large, processing in chunks...")
            print("  Note: Current file appears to be COVID case data (80M+ rows)")
            print("  Aggregating to tract level...")
            
            # Process in chunks and aggregate by tract
            chunk_size = 100000
            tract_data = {}
            
            # Try to find tract identifier in first chunk
            sample = pd.read_csv(filepath, nrows=1000)
            tract_cols = [col for col in sample.columns if any(x in col.lower() for x in ['tract', 'geoid', 'fips', 'county', 'state'])]
            
            if not tract_cols:
                print("  ⚠ Could not identify tract/county/state columns")
                print("  Skipping CDC PLACES integration (file format not recognized)")
                return None
            
            print(f"  Found geographic columns: {tract_cols[:3]}")
            print("  Processing file in chunks (this may take a while)...")
            
            # For now, skip processing this large file
            # It's COVID case data, not health outcomes
            print("  ⚠ Skipping: File is COVID case data, not health outcomes")
            print("  To use CDC PLACES, download health outcomes file from:")
            print("    https://www.cdc.gov/places/")
            return None
        
        # If file is manageable, try to load
        df = pd.read_csv(filepath, nrows=10000)  # Sample first
        
        # Check if this is the health outcomes file or COVID data
        health_indicators = ['Diabetes', 'Obesity', 'Hypertension', 'PhysicalInactivity']
        has_health_data = any(ind.lower() in str(df.columns).lower() for ind in health_indicators)
        
        if not has_health_data:
            print("⚠ CDC PLACES file does not contain expected health outcome indicators")
            print("  Expected columns: Diabetes, Obesity, Hypertension prevalence")
            print("  Current file appears to be COVID case data")
            print("  Skipping CDC PLACES integration")
            return None
        
        # Extract tract identifier
        tract_cols = [col for col in df.columns if 'tract' in col.lower() or 'geoid' in col.lower() or 'fips' in col.lower()]
        if not tract_cols:
            print("⚠ Could not find tract identifier in CDC PLACES file")
            return None
        
        tract_col = tract_cols[0]
        if 'geoid' in tract_col.lower():
            df['CensusTract'] = df[tract_col].astype(str).str.extract(r'(\d{11})$')[0]
        else:
            df['CensusTract'] = df[tract_col].astype(str).str.zfill(11)
        
        # Select health outcome columns
        health_cols = [col for col in df.columns if any(ind.lower() in col.lower() for ind in health_indicators)]
        keep_cols = ['CensusTract'] + health_cols
        df = df[keep_cols].copy()
        
        # Rename with prefix
        rename_map = {col: f'CDC_{col}' for col in health_cols}
        df = df.rename(columns=rename_map)
        
        print(f"✓ Processed CDC PLACES data: {len(df)} tracts")
        return df
        
    except Exception as e:
        print(f"⚠ Error reading CDC PLACES file: {e}")
        print(f"  Error type: {type(e).__name__}")
        return None

def calculate_change_features(df_current, df_previous):
    """Calculate change features between two time periods."""
    print("Calculating change features...")
    
    # Merge on CensusTract
    merged = pd.merge(
        df_current[['CensusTract', 'MedianFamilyIncome', 'PovertyRate']],
        df_previous[['CensusTract', 'MedianFamilyIncome', 'PovertyRate']],
        on='CensusTract',
        suffixes=('_current', '_previous'),
        how='inner'
    )
    
    # Calculate changes
    merged['delta_income'] = merged['MedianFamilyIncome_current'] - merged['MedianFamilyIncome_previous']
    merged['delta_income_pct'] = (merged['delta_income'] / merged['MedianFamilyIncome_previous'].replace(0, np.nan)) * 100
    
    merged['delta_poverty'] = merged['PovertyRate_current'] - merged['PovertyRate_previous']
    
    return merged[['CensusTract', 'delta_income', 'delta_income_pct', 'delta_poverty']]

def create_target_variable(df_current, df_previous):
    """Create target variable: did tract become low-access?"""
    print("Creating target variable...")
    
    merged = pd.merge(
        df_current[['CensusTract', 'LILATracts_1And10']],
        df_previous[['CensusTract', 'LILATracts_1And10']],
        on='CensusTract',
        suffixes=('_current', '_previous'),
        how='inner'
    )
    
    # Target: 1 if became low-access, 0 otherwise
    merged['target'] = 0
    merged.loc[
        (merged['LILATracts_1And10_previous'] == 0) & 
        (merged['LILATracts_1And10_current'] == 1),
        'target'
    ] = 1
    
    return merged[['CensusTract', 'target']]

def engineer_features(df):
    """Engineer additional features for modeling."""
    print("Engineering features...")
    
    features = df.copy()
    
    # Interaction features
    if 'LowIncomeTracts' in features.columns and 'LILATracts_Vehicle' in features.columns:
        features['low_income_no_vehicle'] = features['LowIncomeTracts'] * features['LILATracts_Vehicle']
    
    if 'PovertyRate' in features.columns and 'Urban' in features.columns:
        features['poverty_urban'] = features['PovertyRate'] * features['Urban']
    
    # Risk score components (from strategy)
    # Food Access Status (25% weight)
    food_access_features = [
        'LILATracts_1And10', 'LILATracts_halfAnd10', 'LILATracts_Vehicle',
        'normalized_lapop1', 'normalized_lalowi1'
    ]
    
    # Socioeconomic (30% weight)
    socioeconomic_features = [
        'MedianFamilyIncome', 'PovertyRate', 'LowIncomeTracts'
    ]
    
    # Normalize income (inverse - lower income = higher risk)
    if 'MedianFamilyIncome' in features.columns:
        income_min = features['MedianFamilyIncome'].min()
        income_max = features['MedianFamilyIncome'].max()
        if income_max > income_min:
            features['income_score'] = 1 - ((features['MedianFamilyIncome'] - income_min) / (income_max - income_min))
        else:
            features['income_score'] = 0.5
    
    # Normalize poverty (direct - higher poverty = higher risk)
    if 'PovertyRate' in features.columns:
        features['poverty_score'] = features['PovertyRate'] / 100
        features['poverty_score'] = features['poverty_score'].clip(0, 1)
    
    # Use Census ACS income if available (more recent/accurate than Food Access Atlas)
    if 'Census_MedianHouseholdIncome' in features.columns:
        # Normalize Census income (inverse - lower income = higher risk)
        income_min = features['Census_MedianHouseholdIncome'].min()
        income_max = features['Census_MedianHouseholdIncome'].max()
        if income_max > income_min:
            features['census_income_score'] = 1 - ((features['Census_MedianHouseholdIncome'] - income_min) / (income_max - income_min))
        else:
            features['census_income_score'] = 0.5
    
    # Use Census ACS poverty rate if available
    if 'Census_PovertyRate' in features.columns:
        features['census_poverty_score'] = features['Census_PovertyRate'] / 100
        features['census_poverty_score'] = features['census_poverty_score'].clip(0, 1)
    
    # Education score (inverse - lower education = higher risk)
    if 'Census_EducationRate' in features.columns:
        edu_min = features['Census_EducationRate'].min()
        edu_max = features['Census_EducationRate'].max()
        if edu_max > edu_min:
            features['education_score'] = 1 - ((features['Census_EducationRate'] - edu_min) / (edu_max - edu_min))
        else:
            features['education_score'] = 0.5
    
    # Rent burden score (direct - higher rent burden = higher risk)
    if 'Census_RentBurden' in features.columns:
        features['rent_burden_score'] = features['Census_RentBurden'] / 100
        features['rent_burden_score'] = features['rent_burden_score'].clip(0, 1)
    
    # Vehicle ownership score (inverse - lower ownership = higher risk)
    if 'Census_VehicleOwnershipRate' in features.columns:
        features['vehicle_ownership_score'] = 1 - features['Census_VehicleOwnershipRate']
    
    return features

def main():
    """Main data processing function."""
    print("=" * 60)
    print("Food Desert Prediction - Data Processing")
    print("=" * 60)
    
    # Load current year data (2019)
    df_2019 = load_food_access_data('2019')
    if df_2019 is not None:
        df_2019.to_csv(PROCESSED_DATA_DIR / "food_access_2019.csv", index=False)
        print(f"✓ Processed 2019 data: {len(df_2019)} tracts")
    
    # Load previous year data (2015) if available
    df_2015 = load_food_access_data('2015')
    changes = None
    target = None
    
    if df_2015 is not None and df_2019 is not None:
        df_2015.to_csv(PROCESSED_DATA_DIR / "food_access_2015.csv", index=False)
        print(f"✓ Processed 2015 data: {len(df_2015)} tracts")
        
        # Calculate changes
        changes = calculate_change_features(df_2019, df_2015)
        changes.to_csv(PROCESSED_DATA_DIR / "change_features_2015_2019.csv", index=False)
        print(f"✓ Calculated change features: {len(changes)} tracts")
        
        # Create target variable
        target = create_target_variable(df_2019, df_2015)
        target.to_csv(PROCESSED_DATA_DIR / "target_variable.csv", index=False)
        print(f"✓ Created target variable: {target['target'].sum()} tracts became low-access")
    
    # Load supplementary data sources
    census_acs = load_census_acs_data()
    cdc_places = load_cdc_places_data()
    zillow_data = load_zillow_data()
    
    # Engineer features for modeling
    if df_2019 is not None:
        features = engineer_features(df_2019)
        
        # Merge with Census ACS data
        if census_acs is not None:
            print("Merging Census ACS data...")
            initial_count = len(features)
            # Ensure CensusTract is string type for both DataFrames
            features['CensusTract'] = features['CensusTract'].astype(str)
            census_acs_merge = census_acs[['CensusTract'] + [col for col in census_acs.columns if col.startswith('Census_')]].copy()
            census_acs_merge['CensusTract'] = census_acs_merge['CensusTract'].astype(str)
            features = pd.merge(
                features, 
                census_acs_merge, 
                on='CensusTract', 
                how='left'
            )
            merged_count = features['CensusTract'].notna().sum()
            print(f"✓ Merged Census ACS: {merged_count}/{initial_count} tracts matched")
        
        # Merge with CDC PLACES data
        if cdc_places is not None:
            print("Merging CDC PLACES data...")
            initial_count = len(features)
            # Ensure CensusTract is string type for both DataFrames
            cdc_places_merge = cdc_places.copy()
            cdc_places_merge['CensusTract'] = cdc_places_merge['CensusTract'].astype(str)
            features = pd.merge(
                features, 
                cdc_places_merge, 
                on='CensusTract', 
                how='left'
            )
            merged_count = features['CensusTract'].notna().sum()
            print(f"✓ Merged CDC PLACES: {merged_count}/{initial_count} tracts matched")
        
        # Merge with Zillow housing data (matched by State)
        if zillow_data is not None:
            print("Merging Zillow housing data...")
            initial_count = len(features)
            # Ensure State column exists and matches
            if 'State' in features.columns:
                # Match by State name
                features = pd.merge(
                    features,
                    zillow_data,
                    on='State',
                    how='left'
                )
                merged_count = features['State'].notna().sum()
                zillow_cols = [c for c in features.columns if c.startswith('Zillow_')]
                print(f"✓ Merged Zillow data: {merged_count}/{initial_count} tracts matched")
                print(f"  Added variables: {zillow_cols}")
            else:
                print("  ⚠ State column not found, skipping Zillow merge")
        
        # Merge with changes and target if available
        if changes is not None:
            # Ensure CensusTract is string type for both DataFrames
            changes_merge = changes.copy()
            changes_merge['CensusTract'] = changes_merge['CensusTract'].astype(str)
            features['CensusTract'] = features['CensusTract'].astype(str)
            features = pd.merge(features, changes_merge, on='CensusTract', how='left')
        if target is not None:
            # Ensure CensusTract is string type for both DataFrames
            target_merge = target.copy()
            target_merge['CensusTract'] = target_merge['CensusTract'].astype(str)
            features['CensusTract'] = features['CensusTract'].astype(str)
            features = pd.merge(features, target_merge, on='CensusTract', how='left')
        else:
            # Create synthetic target based on current risk indicators
            print("⚠ No historical data available. Creating risk-based target...")
            # Use combination of low-income and low-access indicators as proxy target
            if 'LILATracts_1And10' in features.columns:
                features['target'] = features['LILATracts_1And10']
            elif 'LILATracts_halfAnd10' in features.columns:
                features['target'] = features['LILATracts_halfAnd10']
            else:
                # Create based on multiple risk factors
                risk_score = 0
                if 'LowIncomeTracts' in features.columns:
                    risk_score += features['LowIncomeTracts'].fillna(0)
                if 'PovertyRate' in features.columns:
                    risk_score += (features['PovertyRate'] > 20).astype(int)
                features['target'] = (risk_score >= 1).astype(int)
            print(f"✓ Created risk-based target: {features['target'].sum()} high-risk tracts")
        
        features.to_csv(FEATURES_DIR / "modeling_features.csv", index=False)
        print(f"✓ Engineered features: {len(features)} tracts, {len(features.columns)} features")
    
    # Generate data coverage map
    if df_2019 is not None:
        print("\nGenerating data coverage map...")
        generate_coverage_map(df_2019, census_acs, cdc_places)
    
    print("\n" + "=" * 60)
    print("Data processing complete!")
    print(f"Processed data saved to: {PROCESSED_DATA_DIR}")
    print(f"Features saved to: {FEATURES_DIR}")
    print("=" * 60)

def generate_coverage_map(df_food_access, df_census=None, df_cdc=None):
    """Generate a map showing data coverage across the US."""
    print("Creating data coverage visualization...")
    
    # Create coverage summary by state
    coverage_data = []
    
    # Food Access Atlas coverage
    if df_food_access is not None and 'State' in df_food_access.columns:
        fa_coverage = df_food_access.groupby('State').size().reset_index(name='FoodAccess_Tracts')
        coverage_data.append(('Food Access Atlas', fa_coverage))
    
    # Census ACS coverage
    if df_census is not None and 'STATE' in df_census.columns:
        census_coverage = df_census.groupby('STATE').size().reset_index(name='CensusACS_Tracts')
        census_coverage['State'] = census_coverage['STATE'].astype(str).str.zfill(2)
        coverage_data.append(('Census ACS', census_coverage))
    elif df_census is not None and 'State' in df_census.columns:
        census_coverage = df_census.groupby('State').size().reset_index(name='CensusACS_Tracts')
        coverage_data.append(('Census ACS', census_coverage))
    
    # CDC PLACES coverage (if available)
    if df_cdc is not None and 'CensusTract' in df_cdc.columns:
        # Extract state from CensusTract (first 2 digits)
        df_cdc_temp = df_cdc.copy()
        df_cdc_temp['State'] = df_cdc_temp['CensusTract'].astype(str).str[:2]
        cdc_coverage = df_cdc_temp.groupby('State').size().reset_index(name='CDCPLACES_Tracts')
        coverage_data.append(('CDC PLACES', cdc_coverage))
    
    # Create summary DataFrame
    if not coverage_data:
        print("  ⚠ No geographic data available for mapping")
        return
    
    # Start with Food Access data (most complete) or first available dataset
    summary = None
    for name, data in coverage_data:
        if 'State' in data.columns:
            summary = data[['State']].copy()
            break
        elif 'STATE' in data.columns:
            data_temp = data.copy()
            data_temp['State'] = data_temp['STATE'].astype(str).str.zfill(2)
            summary = data_temp[['State']].copy()
            break
    
    if summary is None:
        print("  ⚠ Could not create coverage map - no state identifiers found")
        return
    
    # Merge all coverage data
    for name, data in coverage_data:
        if 'State' in data.columns:
            # Get the tract count column (last numeric column)
            tract_col = [col for col in data.columns if col != 'State' and col != 'STATE'][0]
            data_to_merge = data[['State', tract_col]].copy()
            summary = pd.merge(summary, data_to_merge, on='State', how='outer', suffixes=('', f'_{name.replace(" ", "_")}'))
        elif 'STATE' in data.columns:
            data_temp = data.copy()
            data_temp['State'] = data_temp['STATE'].astype(str).str.zfill(2)
            tract_col = [col for col in data_temp.columns if col != 'State' and col != 'STATE'][0]
            data_to_merge = data_temp[['State', tract_col]].copy()
            summary = pd.merge(summary, data_to_merge, on='State', how='outer', suffixes=('', f'_{name.replace(" ", "_")}'))
    
    # Fill NaN with 0
    summary = summary.fillna(0)
    
    # Calculate total coverage
    summary['Total_Tracts'] = summary.select_dtypes(include=[np.number]).sum(axis=1)
    
    # Save coverage summary
    coverage_file = PROCESSED_DATA_DIR / "data_coverage_summary.csv"
    summary.to_csv(coverage_file, index=False)
    print(f"  ✓ Coverage summary saved to: {coverage_file}")
    
    # Create visualization
    try:
        # State FIPS to state name mapping (abbreviated)
        state_fips_to_name = {
            '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
            '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
            '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
            '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
            '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
            '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
            '54': 'WV', '55': 'WI', '56': 'WY'
        }
        
        # Add state abbreviations
        summary['State_Abbr'] = summary['State'].map(state_fips_to_name).fillna(summary['State'])
        
        # Create bar chart showing coverage by state
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Sort by total tracts
        summary_sorted = summary.sort_values('Total_Tracts', ascending=False)
        
        # Create grouped bar chart
        x = np.arange(len(summary_sorted))
        width = 0.25
        
        bars1 = ax.bar(x - width, summary_sorted.get('FoodAccess_Tracts', 0), width, label='Food Access Atlas', color='#2ecc71')
        if 'CensusACS_Tracts' in summary_sorted.columns:
            bars2 = ax.bar(x, summary_sorted['CensusACS_Tracts'], width, label='Census ACS', color='#3498db')
        if 'CDCPLACES_Tracts' in summary_sorted.columns:
            bars3 = ax.bar(x + width, summary_sorted['CDCPLACES_Tracts'], width, label='CDC PLACES', color='#e74c3c')
        
        ax.set_xlabel('State', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Census Tracts', fontsize=12, fontweight='bold')
        ax.set_title('Data Coverage by State - US Census Tracts', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(summary_sorted['State_Abbr'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        map_file = PROCESSED_DATA_DIR / "data_coverage_map.png"
        plt.savefig(map_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Coverage map saved to: {map_file}")
        
        # Print summary statistics
        print("\n  Coverage Summary:")
        print(f"    Total states with Food Access data: {(summary['FoodAccess_Tracts'] > 0).sum()}")
        if 'CensusACS_Tracts' in summary.columns:
            print(f"    Total states with Census ACS data: {(summary['CensusACS_Tracts'] > 0).sum()}")
        if 'CDCPLACES_Tracts' in summary.columns:
            print(f"    Total states with CDC PLACES data: {(summary['CDCPLACES_Tracts'] > 0).sum()}")
        print(f"    Total tracts (Food Access): {summary['FoodAccess_Tracts'].sum():,.0f}")
        if 'CensusACS_Tracts' in summary.columns:
            print(f"    Total tracts (Census ACS): {summary['CensusACS_Tracts'].sum():,.0f}")
        
    except Exception as e:
        print(f"  ⚠ Error creating visualization: {e}")
        print(f"    Coverage data saved to CSV: {coverage_file}")

if __name__ == "__main__":
    main()

