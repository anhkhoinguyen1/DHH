"""
Generate Top 100 Highest Risk Census Tracts

Creates a formatted list of the top 100 census tracts most likely to become food deserts.
Outputs CSV file matching the required format:
- tract_id, lat, lon, risk_probability, demand_mean, demand_std, svi_score (optional)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import os

# Set up paths
BASE_DIR = Path(__file__).parent.parent
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
OUTPUT_DIR = BASE_DIR / "data" / "predictions"
TIGER_CACHE_DIR = BASE_DIR / "data" / "01_census_demographics" / "tiger_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIGER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_predictions():
    """Load predictions."""
    filepath = PREDICTIONS_DIR / "predictions.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Predictions file not found: {filepath}\nRun generate_predictions.py first!")
    
    df = pd.read_csv(filepath)
    return df

def download_tiger_shapefile(state_fips, year=2022):
    """
    Download Census TIGER/Line shapefile for a state.
    
    Args:
        state_fips: 2-digit state FIPS code (e.g., "02" for Alaska)
        year: Year of TIGER data (default: 2022)
    
    Returns:
        Path to downloaded shapefile directory, or None if failed
    """
    state_dir = TIGER_CACHE_DIR / f"state_{state_fips}"
    shapefile_path = state_dir / f"tl_{year}_{state_fips}_tract.shp"
    
    # Check if already downloaded
    if shapefile_path.exists():
        return state_dir
    
    print(f"    Downloading TIGER shapefile for state {state_fips}...")
    
    try:
        # Census TIGER/Line download URL
        # Format: https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
        
        zip_path = state_dir / f"tl_{year}_{state_fips}_tract.zip"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        # Download zip file
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=f"      Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(state_dir)
        
        # Remove zip file to save space
        zip_path.unlink()
        
        print(f"    ✓ Downloaded and extracted TIGER shapefile for state {state_fips}")
        return state_dir
        
    except Exception as e:
        print(f"    ⚠ Error downloading TIGER shapefile for state {state_fips}: {e}")
        return None

def load_tract_coordinates_from_csv():
    """
    Load tract coordinates from CSV file if provided by user.
    
    Returns:
        Dictionary mapping tract_id to (lat, lon) tuple, or None if file doesn't exist
    """
    coords_file = BASE_DIR / "data" / "01_census_demographics" / "tract_coordinates.csv"
    if not coords_file.exists():
        return None
    
    try:
        print("  Checking for user-provided coordinates file...")
        df = pd.read_csv(coords_file)
        
        # Validate required columns
        required_cols = ['tract_id', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠ Missing required columns: {missing_cols}")
            return None
        
        # Convert tract_id to string and pad to 11 digits
        df['tract_id'] = df['tract_id'].astype(str).str.zfill(11)
        
        # Remove rows with missing coordinates
        df = df.dropna(subset=['lat', 'lon'])
        
        # Create dictionary
        coords_dict = {}
        for _, row in df.iterrows():
            tract_id = str(row['tract_id']).zfill(11)
            lat = float(row['lat'])
            lon = float(row['lon'])
            coords_dict[tract_id] = (lat, lon)
        
        print(f"  ✓ Loaded {len(coords_dict)} tract coordinates from CSV file")
        return coords_dict
        
    except Exception as e:
        print(f"  ⚠ Error loading coordinates from CSV: {e}")
        return None

def get_tract_coordinates(tract_data):
    """
    Get latitude/longitude for census tracts.
    
    First tries to load from user-provided CSV file.
    Falls back to Census TIGER/Line shapefiles if CSV not available.
    
    Args:
        tract_data: DataFrame with 'tract_id' column (11-digit codes)
    
    Returns:
        Dictionary mapping tract_id to (lat, lon) tuple
    """
    # First, try to load from user-provided CSV file
    csv_coords = load_tract_coordinates_from_csv()
    if csv_coords:
        # Check if we have all the tracts we need
        tract_ids = tract_data['tract_id'].astype(str).str.zfill(11).unique().tolist()
        missing = [tid for tid in tract_ids if tid not in csv_coords]
        if len(missing) == 0:
            print("  ✓ All tract coordinates found in CSV file")
            return csv_coords
        else:
            print(f"  ⚠ CSV file missing {len(missing)} tracts, will supplement with TIGER data")
            # Continue to TIGER download for missing tracts
    
    print("  Getting tract coordinates from Census TIGER/Line shapefiles...")
    
    # Extract unique tract IDs
    tract_ids = tract_data['tract_id'].astype(str).str.zfill(11).unique().tolist()
    
    # Group tracts by state (first 2 digits of tract ID)
    tracts_by_state = {}
    for tract_id in tract_ids:
        state_fips = tract_id[:2]
        if state_fips not in tracts_by_state:
            tracts_by_state[state_fips] = []
        tracts_by_state[state_fips].append(tract_id)
    
    coords_dict = {}
    year = 2022  # Use 2022 TIGER data (most recent)
    
    print(f"  Processing {len(tracts_by_state)} states...")
    
    for state_fips, state_tracts in tracts_by_state.items():
        print(f"  State {state_fips}: {len(state_tracts)} tracts")
        
        # Download TIGER shapefile if needed
        state_dir = download_tiger_shapefile(state_fips, year)
        if state_dir is None:
            # If download fails, mark all tracts for this state as missing
            for tract_id in state_tracts:
                coords_dict[tract_id] = (None, None)
            continue
        
        # Load shapefile
        shapefile_path = state_dir / f"tl_{year}_{state_fips}_tract.shp"
        if not shapefile_path.exists():
            print(f"    ⚠ Shapefile not found: {shapefile_path}")
            for tract_id in state_tracts:
                coords_dict[tract_id] = (None, None)
            continue
        
        try:
            # Read shapefile with geopandas
            gdf = gpd.read_file(shapefile_path)
            
            # Extract GEOID from shapefile (format: SSCCCTTTTTT)
            # TIGER shapefiles use GEOID column
            if 'GEOID' in gdf.columns:
                gdf['TRACT_GEOID'] = gdf['GEOID'].astype(str).str.zfill(11)
            elif 'TRACTCE' in gdf.columns and 'STATEFP' in gdf.columns and 'COUNTYFP' in gdf.columns:
                # Construct GEOID from components
                gdf['TRACT_GEOID'] = (
                    gdf['STATEFP'].astype(str).str.zfill(2) +
                    gdf['COUNTYFP'].astype(str).str.zfill(3) +
                    gdf['TRACTCE'].astype(str).str.zfill(6)
                )
            else:
                print(f"    ⚠ Could not find GEOID column in shapefile")
                for tract_id in state_tracts:
                    coords_dict[tract_id] = (None, None)
                continue
            
            # Calculate centroids (in WGS84 / EPSG:4326)
            if gdf.crs is None:
                # Assume it's in WGS84 if no CRS specified
                gdf.set_crs(epsg=4326, inplace=True)
            elif gdf.crs.to_string() != 'EPSG:4326':
                # Reproject to WGS84 if needed
                gdf = gdf.to_crs(epsg=4326)
            
            # Get centroids
            centroids = gdf.geometry.centroid
            gdf['lat'] = centroids.y
            gdf['lon'] = centroids.x
            
            # Match tracts
            state_success_count = 0
            for tract_id in state_tracts:
                tract_match = gdf[gdf['TRACT_GEOID'] == tract_id]
                if len(tract_match) > 0:
                    lat = float(tract_match.iloc[0]['lat'])
                    lon = float(tract_match.iloc[0]['lon'])
                    coords_dict[tract_id] = (lat, lon)
                    state_success_count += 1
                else:
                    coords_dict[tract_id] = (None, None)
            
            print(f"    ✓ Found coordinates for {state_success_count}/{len(state_tracts)} tracts")
            
        except Exception as e:
            print(f"    ⚠ Error processing shapefile for state {state_fips}: {e}")
            import traceback
            traceback.print_exc()
            for tract_id in state_tracts:
                coords_dict[tract_id] = (None, None)
    
    # Merge with CSV coordinates if available
    if csv_coords:
        for tract_id, coords in csv_coords.items():
            if tract_id not in coords_dict or coords_dict[tract_id] == (None, None):
                coords_dict[tract_id] = coords
    
    # Count successful geocodes
    successful = sum(1 for v in coords_dict.values() if v[0] is not None and v[1] is not None)
    print(f"  ✓ Successfully geocoded {successful}/{len(coords_dict)} tracts")
    
    return coords_dict

def calculate_svi_score(tract_data):
    """
    Calculate Social Vulnerability Index (SVI) score for each tract.
    
    SVI is a composite measure of social vulnerability based on:
    1. Socioeconomic Status (30%): Poverty, income, education, employment
    2. Household Composition (25%): Age, disability, single-parent households
    3. Minority Status & Language (25%): Race/ethnicity, language barriers
    4. Housing & Transportation (20%): Crowding, no vehicle, group quarters
    
    Returns normalized score [0-1] where higher values indicate higher vulnerability.
    
    Based on CDC/ATSDR Social Vulnerability Index methodology, adapted to available data.
    """
    svi_scores = []
    
    for idx, row in tract_data.iterrows():
        component_scores = []
        weights = []
        
        # 1. Socioeconomic Status (30% weight)
        socioeconomic_score = 0.0
        socioeconomic_weight = 0.0
        
        # Poverty rate (0-1, higher = more vulnerable)
        if 'Census_PovertyRate' in row and pd.notna(row['Census_PovertyRate']):
            poverty_score = min(row['Census_PovertyRate'] / 100.0, 1.0)  # Cap at 100%
            socioeconomic_score += poverty_score * 0.4
            socioeconomic_weight += 0.4
        elif 'PovertyRate' in row and pd.notna(row['PovertyRate']):
            poverty_score = min(row['PovertyRate'] / 100.0, 1.0)
            socioeconomic_score += poverty_score * 0.4
            socioeconomic_weight += 0.4
        
        # Income (0-1, lower income = more vulnerable)
        if 'Census_MedianHouseholdIncome' in row and pd.notna(row['Census_MedianHouseholdIncome']):
            # Normalize income: lower = higher vulnerability
            # Use percentile-based approach: bottom 20% = high vulnerability
            income = row['Census_MedianHouseholdIncome']
            if income < 30000:  # Very low income
                income_score = 1.0
            elif income < 50000:  # Low income
                income_score = 0.7
            elif income < 75000:  # Moderate income
                income_score = 0.4
            else:  # Higher income
                income_score = 0.1
            socioeconomic_score += income_score * 0.3
            socioeconomic_weight += 0.3
        elif 'MedianFamilyIncome' in row and pd.notna(row['MedianFamilyIncome']):
            income = row['MedianFamilyIncome']
            if income < 30000:
                income_score = 1.0
            elif income < 50000:
                income_score = 0.7
            elif income < 75000:
                income_score = 0.4
            else:
                income_score = 0.1
            socioeconomic_score += income_score * 0.3
            socioeconomic_weight += 0.3
        
        # Education (0-1, lower education = more vulnerable)
        if 'Census_EducationRate' in row and pd.notna(row['Census_EducationRate']):
            # Education rate is % with bachelor's degree or higher
            # Lower education = higher vulnerability
            edu_score = 1.0 - (row['Census_EducationRate'] / 100.0)
            edu_score = max(0.0, min(1.0, edu_score))  # Clip to [0, 1]
            socioeconomic_score += edu_score * 0.2
            socioeconomic_weight += 0.2
        
        # Rent burden (0-1, higher rent burden = more vulnerable)
        if 'Census_RentBurden' in row and pd.notna(row['Census_RentBurden']):
            # Rent burden is % of income spent on rent
            # >30% is considered burdened, >50% is severely burdened
            rent_burden = row['Census_RentBurden'] / 100.0
            if rent_burden > 0.5:
                rent_score = 1.0
            elif rent_burden > 0.3:
                rent_score = 0.7
            else:
                rent_score = 0.3
            socioeconomic_score += rent_score * 0.1
            socioeconomic_weight += 0.1
        
        if socioeconomic_weight > 0:
            socioeconomic_normalized = socioeconomic_score / socioeconomic_weight
            component_scores.append(socioeconomic_normalized)
            weights.append(0.30)
        
        # 2. Household Composition (25% weight)
        household_score = 0.0
        household_weight = 0.0
        
        # Age 65+ (proxy for elderly vulnerability)
        # We don't have direct age data, but can infer from other indicators
        # For now, use low-income + no vehicle as proxy for vulnerable households
        if 'LowIncomeTracts' in row and pd.notna(row['LowIncomeTracts']):
            household_score += float(row['LowIncomeTracts']) * 0.5
            household_weight += 0.5
        
        # No vehicle households (indicates transportation vulnerability)
        if 'Census_VehicleOwnershipRate' in row and pd.notna(row['Census_VehicleOwnershipRate']):
            # Lower vehicle ownership = higher vulnerability
            no_vehicle_score = 1.0 - row['Census_VehicleOwnershipRate']
            household_score += no_vehicle_score * 0.5
            household_weight += 0.5
        elif 'Census_HouseholdsNoVehicle' in row and 'Census_TotalHouseholds' in row:
            if pd.notna(row['Census_TotalHouseholds']) and row['Census_TotalHouseholds'] > 0:
                if pd.notna(row['Census_HouseholdsNoVehicle']):
                    no_vehicle_rate = row['Census_HouseholdsNoVehicle'] / row['Census_TotalHouseholds']
                    household_score += no_vehicle_rate * 0.5
                    household_weight += 0.5
        
        if household_weight > 0:
            household_normalized = household_score / household_weight
            component_scores.append(household_normalized)
            weights.append(0.25)
        
        # 3. Minority Status & Language (25% weight)
        # Note: We don't have detailed race/ethnicity data in current features
        # Use low-income tracts as proxy (historically correlated with minority status)
        # This is a limitation - ideally would use Census race/ethnicity data
        minority_score = 0.0
        if 'LowIncomeTracts' in row and pd.notna(row['LowIncomeTracts']):
            # Low-income areas often correlate with minority communities
            # This is a simplified proxy
            minority_score = float(row['LowIncomeTracts']) * 0.6
        
        # Use poverty as additional indicator
        if 'Census_PovertyRate' in row and pd.notna(row['Census_PovertyRate']):
            poverty_indicator = min(row['Census_PovertyRate'] / 100.0, 1.0)
            minority_score += poverty_indicator * 0.4
        
        if minority_score > 0:
            minority_normalized = min(minority_score, 1.0)  # Cap at 1.0
            component_scores.append(minority_normalized)
            weights.append(0.25)
        
        # 4. Housing & Transportation (20% weight)
        housing_score = 0.0
        housing_weight = 0.0
        
        # No vehicle (transportation vulnerability)
        if 'Census_VehicleOwnershipRate' in row and pd.notna(row['Census_VehicleOwnershipRate']):
            no_vehicle_score = 1.0 - row['Census_VehicleOwnershipRate']
            housing_score += no_vehicle_score * 0.5
            housing_weight += 0.5
        elif 'Census_HouseholdsNoVehicle' in row and 'Census_TotalHouseholds' in row:
            if pd.notna(row['Census_TotalHouseholds']) and row['Census_TotalHouseholds'] > 0:
                if pd.notna(row['Census_HouseholdsNoVehicle']):
                    no_vehicle_rate = row['Census_HouseholdsNoVehicle'] / row['Census_TotalHouseholds']
                    housing_score += no_vehicle_rate * 0.5
                    housing_weight += 0.5
        
        # Housing crowding (average household size as proxy)
        if 'Census_AvgHouseholdSize' in row and pd.notna(row['Census_AvgHouseholdSize']):
            # Higher household size = potential crowding
            avg_size = row['Census_AvgHouseholdSize']
            if avg_size > 3.0:  # Above average
                crowding_score = min((avg_size - 2.5) / 2.0, 1.0)  # Normalize
            else:
                crowding_score = 0.2
            housing_score += crowding_score * 0.3
            housing_weight += 0.3
        
        # Rent burden (housing stress)
        if 'Census_RentBurden' in row and pd.notna(row['Census_RentBurden']):
            rent_burden = row['Census_RentBurden'] / 100.0
            if rent_burden > 0.5:
                rent_score = 1.0
            elif rent_burden > 0.3:
                rent_score = 0.7
            else:
                rent_score = 0.3
            housing_score += rent_score * 0.2
            housing_weight += 0.2
        
        if housing_weight > 0:
            housing_normalized = housing_score / housing_weight
            component_scores.append(housing_normalized)
            weights.append(0.20)
        
        # Calculate weighted average SVI score
        if len(component_scores) > 0 and sum(weights) > 0:
            # Normalize weights to sum to 1.0
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            svi = sum(score * weight for score, weight in zip(component_scores, normalized_weights))
            svi = max(0.0, min(1.0, svi))  # Clip to [0, 1]
        else:
            # If no data available, use default moderate vulnerability
            svi = 0.5
        
        svi_scores.append(round(svi, 2))
    
    return svi_scores

def estimate_demand(tract_data):
    """
    Estimate weekly demand (households) based on available data.
    Uses population/household data to estimate weekly grocery shopping demand.
    """
    demand_mean = []
    demand_std = []
    
    for idx, row in tract_data.iterrows():
        # Estimate based on households or population
        households = None
        for col in ['HH_TOTAL', 'Census_TotalHouseholds', 'OHU2010', 'HUTOTAL']:
            if col in row and pd.notna(row[col]) and row[col] > 0:
                households = row[col]
                break
        
        if households is None:
            # Try population-based estimate
            for col in ['Pop2010', 'Census_TotalPopulation']:
                if col in row and pd.notna(row[col]) and row[col] > 0:
                    # Approximate: 2.5 people per household
                    households = row[col] / 2.5
                    break
        
        if households is None or households <= 0:
            households = 500  # Default estimate
        
        # Weekly demand: assume each household shops once per week on average
        # Adjust based on low-access status (higher demand if low access)
        base_demand = households * 0.8  # 80% of households shop weekly
        if 'LILATracts_1And10' in row and pd.notna(row['LILATracts_1And10']) and row['LILATracts_1And10'] == 1:
            # Low access areas may have higher demand per household (travel further, shop less frequently)
            base_demand = households * 1.2
        
        demand_mean.append(round(base_demand, 1))
        # Standard deviation: ~20% of mean
        demand_std.append(round(base_demand * 0.2, 1))
    
    return demand_mean, demand_std

def generate_top100():
    """Generate top 100 highest risk tracts in required format."""
    print("=" * 60)
    print("Generating Top 100 Highest Risk Census Tracts")
    print("=" * 60)
    
    # Load predictions
    df = load_predictions()
    print(f"✓ Loaded {len(df)} total tracts")
    
    # Load features for additional data (households, population, SVI calculation, etc.)
    features_file = BASE_DIR / "data" / "features" / "modeling_features.csv"
    features_df = None
    if features_file.exists():
        try:
            # Read all columns first to see what's available
            all_features = pd.read_csv(features_file, nrows=1)
            available_cols = ['CensusTract']
            
            # Check which columns exist for demand estimation
            for col in ['HH_TOTAL', 'OHU2010', 'Pop2010', 'Census_TotalHouseholds', 
                       'Census_TotalPopulation', 'LILATracts_1And10', 'HUTOTAL']:
                if col in all_features.columns:
                    available_cols.append(col)
            
            # Check which columns exist for SVI calculation
            svi_cols = [
                'Census_PovertyRate', 'PovertyRate', 'Census_MedianHouseholdIncome', 
                'MedianFamilyIncome', 'Census_EducationRate', 'Census_RentBurden',
                'LowIncomeTracts', 'Census_VehicleOwnershipRate', 
                'Census_HouseholdsNoVehicle', 'Census_TotalHouseholds',
                'Census_AvgHouseholdSize'
            ]
            for col in svi_cols:
                if col in all_features.columns and col not in available_cols:
                    available_cols.append(col)
            
            features_df = pd.read_csv(features_file, usecols=available_cols)
            features_df['CensusTract'] = features_df['CensusTract'].astype(str)
            print(f"✓ Loaded features for demand estimation and SVI calculation")
        except Exception as e:
            print(f"⚠ Could not load features: {e}")
    
    # Sort by probability (highest risk first)
    df_sorted = df.sort_values('probability', ascending=False)
    
    # Get top 100
    top100 = df_sorted.head(100).copy()
    
    # Add rank
    top100.insert(0, 'Rank', range(1, len(top100) + 1))
    
    # Ensure CensusTract is string and properly formatted (11 digits)
    top100['CensusTract'] = top100['CensusTract'].astype(str).str.zfill(11)
    
    # Merge with features for demand estimation
    if features_df is not None:
        top100 = pd.merge(top100, features_df, on='CensusTract', how='left')
    
    # Create output DataFrame with required columns in correct order
    output = pd.DataFrame()
    
    # Required: tract_id (11-digit format as string)
    output['tract_id'] = top100['CensusTract'].astype(str).str.zfill(11)
    
    # Required: lat, lon (latitude, longitude) - NOW USING TIGER/Line shapefiles
    print("\nGeocoding tract coordinates using Census TIGER/Line shapefiles...")
    tract_coords = get_tract_coordinates(output[['tract_id']])
    
    # Map coordinates to output
    lat_list = []
    lon_list = []
    for tract_id in output['tract_id']:
        lat, lon = tract_coords.get(tract_id, (None, None))
        if lat is not None and lon is not None:
            lat_list.append(f"{lat:.6f}")
            lon_list.append(f"{lon:.6f}")
        else:
            lat_list.append("")
            lon_list.append("")
    
    output['lat'] = lat_list
    output['lon'] = lon_list
    
    geocoded_count = sum(1 for lat, lon in zip(lat_list, lon_list) if lat and lon)
    print(f"✓ Geocoded {geocoded_count}/{len(output)} tracts\n")
    
    # Required: risk_probability (0-1)
    output['risk_probability'] = top100['probability'].clip(0, 1).round(4)
    
    # Required: demand_mean, demand_std (estimated weekly demand)
    demand_mean, demand_std = estimate_demand(top100)
    output['demand_mean'] = demand_mean
    output['demand_std'] = demand_std
    
    # Required: svi_score (Social Vulnerability Index [0-1])
    print("\nCalculating Social Vulnerability Index (SVI) scores...")
    svi_scores = calculate_svi_score(top100)
    output['svi_score'] = svi_scores
    print(f"✓ Calculated SVI scores for {len(svi_scores)} tracts")
    print(f"  SVI range: {min(svi_scores):.2f} - {max(svi_scores):.2f}")
    print(f"  Average SVI: {sum(svi_scores)/len(svi_scores):.2f}")
    
    # Reorder columns to match example: tract_id, lat, lon, risk_probability, demand_mean, demand_std, svi_score
    output = output[['tract_id', 'lat', 'lon', 'risk_probability', 'demand_mean', 'demand_std', 'svi_score']]
    
    # Save required format CSV (no quotes, standard CSV format)
    csv_file = OUTPUT_DIR / "top100_highest_risk_tracts.csv"
    # Convert tract_id to string explicitly before saving
    output['tract_id'] = output['tract_id'].astype(str)
    output.to_csv(csv_file, index=False, quoting=0)  # quoting=0 means QUOTE_MINIMAL (only when needed)
    print(f"✓ Saved CSV to: {csv_file}")
    print(f"  Columns: {list(output.columns)}")
    print(f"  Rows: {len(output)}")
    print(f"  Sample tract_id: {output['tract_id'].iloc[0]} (length: {len(output['tract_id'].iloc[0])})")
    
    # Also save detailed version with additional info
    detailed_file = OUTPUT_DIR / "top100_detailed.csv"
    detailed_output = top100[['CensusTract', 'State', 'County', 'probability', 'risk_level']].copy()
    detailed_output['tract_id'] = detailed_output['CensusTract'].astype(str).str.zfill(11)
    detailed_output = detailed_output[['tract_id', 'State', 'County', 'probability', 'risk_level']]
    detailed_output.to_csv(detailed_file, index=False)
    print(f"✓ Saved detailed CSV to: {detailed_file}")
    
    # Save as formatted text (using top100 for detailed info)
    txt_file = OUTPUT_DIR / "top100_highest_risk_tracts.txt"
    with open(txt_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TOP 100 CENSUS TRACTS MOST LIKELY TO BECOME FOOD DESERTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Based on predictive modeling using demographic, socioeconomic, retail,\n")
        f.write("and transportation access indicators.\n\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, row in top100.iterrows():
            f.write(f"Rank {row['Rank']:3d}: Census Tract {row['CensusTract']}\n")
            f.write(f"  Location: {row['County']}, {row['State']}\n")
            f.write(f"  Risk Probability: {row['probability']:.4f}\n")
            f.write(f"  Risk Level: {row['risk_level']}\n")
            if 'currently_low_access' in row and pd.notna(row['currently_low_access']):
                status = "Yes" if row['currently_low_access'] == 1 else "No"
                f.write(f"  Currently Low Access: {status}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Tracts Analyzed: {len(df):,}\n")
        f.write(f"Top 100 Risk Probability Range: {output['risk_probability'].min():.4f} - {output['risk_probability'].max():.4f}\n")
        f.write(f"Average Risk Probability (Top 100): {output['risk_probability'].mean():.4f}\n")
        f.write(f"Median Risk Probability (Top 100): {output['risk_probability'].median():.4f}\n\n")
        
        # State breakdown
        state_counts = top100['State'].value_counts()
        f.write("Top 100 by State:\n")
        for state, count in state_counts.items():
            f.write(f"  {state}: {count} tracts\n")
        f.write("\n")
        
        # County breakdown (top 10)
        county_counts = top100['County'].value_counts().head(10)
        f.write("Top 10 Counties in Top 100:\n")
        for county, count in county_counts.items():
            f.write(f"  {county}: {count} tracts\n")
    
    print(f"✓ Saved formatted text to: {txt_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Top 100 Summary")
    print("=" * 60)
    print(f"Risk Probability Range: {output['risk_probability'].min():.4f} - {output['risk_probability'].max():.4f}")
    print(f"Average Risk Probability: {output['risk_probability'].mean():.4f}")
    print(f"Average Weekly Demand: {output['demand_mean'].mean():.1f} households")
    print(f"\nTop 5 States:")
    for state, count in top100['State'].value_counts().head(5).items():
        print(f"  {state}: {count} tracts")
    
    print("\n" + "=" * 60)
    print("Top 100 list generation complete!")
    print(f"Files saved to: {OUTPUT_DIR}")
    print("=" * 60)
    if geocoded_count < len(output):
        print(f"\n⚠ Note: {len(output) - geocoded_count} tracts could not be geocoded")
        print("  Missing coordinates may be due to:")
        print("  - TIGER shapefile download issues")
        print("  - Tract ID format mismatches")
        print("  - Missing tracts in TIGER data")
    
    return output

if __name__ == "__main__":
    output = generate_top100()
    print("\nFirst 10 tracts (required format):")
    print(output.head(10).to_string(index=False))

