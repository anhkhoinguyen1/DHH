"""
Generate Top 1000 Highest Risk Census Tracts

Creates a formatted list of the top 1000 census tracts most likely to become food deserts.
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
import time

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
        
        # Download zip file with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url, 
                    stream=True, 
                    timeout=(30, 300),  # (connect timeout, read timeout) - 5 min read timeout
                    allow_redirects=True
                )
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=f"      Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192 * 4):  # Larger chunk size
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                break  # Success, exit retry loop
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff
                    print(f"      ⚠ Timeout (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    if zip_path.exists():
                        zip_path.unlink()  # Remove partial download
                else:
                    raise
        
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
    Get latitude/longitude for census tracts with robust fallback strategies.
    
    Strategy 1: Load from comprehensive CSV file (fastest, most complete)
    Strategy 2: Download and process TIGER/Line shapefiles for missing tracts
    Strategy 3: Try alternative tract ID formats and matching methods
    
    Args:
        tract_data: DataFrame with 'tract_id' column (11-digit codes)
    
    Returns:
        Dictionary mapping tract_id to (lat, lon) tuple
    """
    # Extract unique tract IDs and normalize format
    tract_ids = tract_data['tract_id'].astype(str).str.zfill(11).unique().tolist()
    coords_dict = {}
    
    # Strategy 1: Load from comprehensive CSV file first
    csv_coords = load_tract_coordinates_from_csv()
    if csv_coords:
        print(f"  Loaded {len(csv_coords)} coordinates from CSV file")
        for tract_id in tract_ids:
            if tract_id in csv_coords:
                coords_dict[tract_id] = csv_coords[tract_id]
        
        missing_from_csv = [tid for tid in tract_ids if tid not in coords_dict or coords_dict[tid] == (None, None)]
        if len(missing_from_csv) == 0:
            print("  ✓ All tract coordinates found in CSV file")
            return coords_dict
        else:
            print(f"  ⚠ CSV file missing {len(missing_from_csv)} tracts, supplementing with TIGER data")
    
    # Strategy 2: Get missing tracts from TIGER shapefiles
    missing = [tid for tid in tract_ids if tid not in coords_dict or coords_dict[tid] == (None, None)]
    if missing:
        print(f"  Getting coordinates for {len(missing)} missing tracts from Census TIGER/Line shapefiles...")
        
        # Group tracts by state (first 2 digits of tract ID)
        tracts_by_state = {}
        for tract_id in missing:
            state_fips = tract_id[:2]
            if state_fips not in tracts_by_state:
                tracts_by_state[state_fips] = []
            tracts_by_state[state_fips].append(tract_id)
        
        year = 2022  # Use 2022 TIGER data (most recent)
        print(f"  Processing {len(tracts_by_state)} states...")
        
        for state_fips, state_tracts in tracts_by_state.items():
            print(f"  State {state_fips}: {len(state_tracts)} tracts")
            
            # Download TIGER shapefile if needed
            state_dir = download_tiger_shapefile(state_fips, year)
            if state_dir is None:
                print(f"    ⚠ Failed to download TIGER shapefile for state {state_fips}")
                continue
            
            # Load shapefile
            shapefile_path = state_dir / f"tl_{year}_{state_fips}_tract.shp"
            if not shapefile_path.exists():
                print(f"    ⚠ Shapefile not found: {shapefile_path}")
                continue
            
            try:
                # Read shapefile with geopandas
                gdf = gpd.read_file(shapefile_path)
                
                # Strategy 2a: Try multiple GEOID extraction methods
                if 'GEOID' in gdf.columns:
                    gdf['TRACT_GEOID'] = gdf['GEOID'].astype(str).str.zfill(11)
                elif all(col in gdf.columns for col in ['STATEFP', 'COUNTYFP', 'TRACTCE']):
                    # Construct GEOID from components
                    gdf['TRACT_GEOID'] = (
                        gdf['STATEFP'].astype(str).str.zfill(2) +
                        gdf['COUNTYFP'].astype(str).str.zfill(3) +
                        gdf['TRACTCE'].astype(str).str.zfill(6)
                    )
                else:
                    print(f"    ⚠ Could not extract GEOID from shapefile columns: {list(gdf.columns)}")
                    continue
                
                # Calculate centroids with proper projection
                if gdf.crs is None:
                    gdf.set_crs(epsg=4326, inplace=True)
                elif gdf.crs.to_string() != 'EPSG:4326':
                    # Project to Albers Equal Area for accurate centroids, then back to WGS84
                    try:
                        gdf_projected = gdf.to_crs(epsg=5070)  # US Albers Equal Area
                        centroids_projected = gdf_projected.geometry.centroid
                        centroids_gdf = gpd.GeoDataFrame(geometry=centroids_projected, crs='epsg:5070')
                        centroids = centroids_gdf.to_crs(epsg=4326).geometry
                        gdf['lat'] = centroids.y
                        gdf['lon'] = centroids.x
                    except:
                        # Fallback to simple centroid if projection fails
                        gdf = gdf.to_crs(epsg=4326)
                        centroids = gdf.geometry.centroid
                        gdf['lat'] = centroids.y
                        gdf['lon'] = centroids.x
                else:
                    # Already in WGS84
                    centroids = gdf.geometry.centroid
                    gdf['lat'] = centroids.y
                    gdf['lon'] = centroids.x
                
                # Match tracts with multiple matching strategies
                state_success_count = 0
                for tract_id in state_tracts:
                    # Try exact match first
                    tract_match = gdf[gdf['TRACT_GEOID'] == tract_id]
                    
                    if len(tract_match) == 0:
                        # Try matching without leading zeros
                        tract_match = gdf[gdf['TRACT_GEOID'].astype(str).str.lstrip('0') == tract_id.lstrip('0')]
                    
                    if len(tract_match) == 0:
                        # Try matching last 11 digits (in case of prefix)
                        tract_suffix = tract_id[-11:] if len(tract_id) > 11 else tract_id
                        tract_match = gdf[gdf['TRACT_GEOID'].astype(str).str[-11:] == tract_suffix]
                    
                    if len(tract_match) > 0:
                        try:
                            lat = float(tract_match.iloc[0]['lat'])
                            lon = float(tract_match.iloc[0]['lon'])
                            # Validate coordinates are reasonable (within US bounds)
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                coords_dict[tract_id] = (lat, lon)
                                state_success_count += 1
                            else:
                                coords_dict[tract_id] = (None, None)
                        except (ValueError, KeyError) as e:
                            coords_dict[tract_id] = (None, None)
                    else:
                        coords_dict[tract_id] = (None, None)
                
                print(f"    ✓ Found coordinates for {state_success_count}/{len(state_tracts)} tracts")
                
            except Exception as e:
                print(f"    ⚠ Error processing shapefile for state {state_fips}: {e}")
                import traceback
                traceback.print_exc()
                # Don't mark as missing yet - might be in CSV
    
    # Strategy 3: Final merge with CSV coordinates (in case CSV had updates)
    if csv_coords:
        for tract_id, coords in csv_coords.items():
            if tract_id in tract_ids and (tract_id not in coords_dict or coords_dict[tract_id] == (None, None)):
                coords_dict[tract_id] = coords
    
    # Final validation and reporting
    successful = sum(1 for v in coords_dict.values() if v[0] is not None and v[1] is not None)
    missing_final = [tid for tid in tract_ids if coords_dict.get(tid) == (None, None)]
    
    print(f"  ✓ Successfully geocoded {successful}/{len(tract_ids)} tracts")
    
    if missing_final:
        print(f"  ⚠ WARNING: {len(missing_final)} tracts still missing coordinates:")
        for tid in missing_final[:10]:  # Show first 10
            print(f"    - {tid}")
        if len(missing_final) > 10:
            print(f"    ... and {len(missing_final) - 10} more")
        print(f"  Recommendation: Run 'python scripts/generate_all_tract_coordinates.py' to generate complete coordinate file")
    
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

def generate_top1000():
    """Generate top 1000 highest risk tracts in required format."""
    print("=" * 60)
    print("Generating Top 1000 Highest Risk Census Tracts")
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
    
    # Get coordinates for tracts to filter out those without coordinates
    print("\nGetting tract coordinates to filter tracts with valid lat/lon...")
    
    # Start with top 1200 to ensure we get 1000 with coordinates (some may be missing)
    candidate_tracts = df_sorted.head(1200).copy()
    candidate_tracts['CensusTract'] = candidate_tracts['CensusTract'].astype(str).str.zfill(11)
    
    # Get coordinates for candidate tracts
    tract_coords = get_tract_coordinates(candidate_tracts[['CensusTract']].rename(columns={'CensusTract': 'tract_id'}))
    
    # Filter to only tracts with valid coordinates
    tracts_with_coords = []
    for idx, row in candidate_tracts.iterrows():
        tract_id = row['CensusTract']
        lat, lon = tract_coords.get(tract_id, (None, None))
        if lat is not None and lon is not None:
            # Validate coordinates are reasonable
            if -180 <= lon <= 180 and -90 <= lat <= 90:
                tracts_with_coords.append(idx)
    
    if len(tracts_with_coords) < 1000:
        print(f"  ⚠ Only {len(tracts_with_coords)} tracts have coordinates out of top 1200")
        print(f"  Expanding search to get 1000 tracts with coordinates...")
        
        # Expand search to get more tracts with coordinates
        expanded_tracts = df_sorted.head(2000).copy()  # Check top 2000
        expanded_tracts['CensusTract'] = expanded_tracts['CensusTract'].astype(str).str.zfill(11)
        
        # Get coordinates for expanded set
        expanded_coords = get_tract_coordinates(expanded_tracts[['CensusTract']].rename(columns={'CensusTract': 'tract_id'}))
        
        # Filter to tracts with valid coordinates, sorted by probability
        tracts_with_coords = []
        for idx, row in expanded_tracts.iterrows():
            tract_id = row['CensusTract']
            lat, lon = expanded_coords.get(tract_id, (None, None))
            if lat is not None and lon is not None:
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    tracts_with_coords.append(idx)
                    if len(tracts_with_coords) >= 1000:
                        break
        
        # Use expanded set
        candidate_tracts = expanded_tracts
        tract_coords = expanded_coords
    
    # Get top 1000 tracts with valid coordinates
    top1000_with_coords = candidate_tracts.loc[tracts_with_coords[:1000]].copy()
    
    if len(top1000_with_coords) < 1000:
        print(f"  ⚠ WARNING: Only {len(top1000_with_coords)} tracts have valid coordinates")
        print(f"  Output will contain {len(top1000_with_coords)} tracts instead of 1000")
    
    # Add rank
    top1000_with_coords.insert(0, 'Rank', range(1, len(top1000_with_coords) + 1))
    
    # Ensure CensusTract is string and properly formatted (11 digits)
    top1000_with_coords['CensusTract'] = top1000_with_coords['CensusTract'].astype(str).str.zfill(11)
    
    # Merge with features for demand estimation
    if features_df is not None:
        top1000_with_coords = pd.merge(top1000_with_coords, features_df, on='CensusTract', how='left')
    
    # Create output DataFrame with required columns in correct order
    output = pd.DataFrame()
    
    # Required: tract_id (11-digit format as string)
    output['tract_id'] = top1000_with_coords['CensusTract'].astype(str).str.zfill(11)
    
    # Required: lat, lon (latitude, longitude) - Already verified to exist
    print(f"\nMapping coordinates to {len(output)} tracts with valid coordinates...")
    
    # Map coordinates to output (we know they all exist)
    lat_list = []
    lon_list = []
    
    for tract_id in output['tract_id']:
        lat, lon = tract_coords.get(tract_id, (None, None))
        if lat is not None and lon is not None:
            lat_list.append(f"{lat:.6f}")
            lon_list.append(f"{lon:.6f}")
        else:
            # This shouldn't happen since we filtered, but handle it
            print(f"  ⚠ Unexpected: tract {tract_id} missing coordinates after filtering")
            lat_list.append("")
            lon_list.append("")
    
    output['lat'] = lat_list
    output['lon'] = lon_list
    
    # Verify all have coordinates
    geocoded_count = sum(1 for lat, lon in zip(lat_list, lon_list) if lat and lon)
    print(f"✓ All {geocoded_count}/{len(output)} tracts have valid coordinates\n")
    
    # Required: risk_probability (0-1)
    output['risk_probability'] = top1000_with_coords['probability'].clip(0, 1).round(4)
    
    # Required: demand_mean, demand_std (estimated weekly demand)
    demand_mean, demand_std = estimate_demand(top1000_with_coords)
    output['demand_mean'] = demand_mean
    output['demand_std'] = demand_std
    
    # Required: svi_score (Social Vulnerability Index [0-1])
    print("\nCalculating Social Vulnerability Index (SVI) scores...")
    svi_scores = calculate_svi_score(top1000_with_coords)
    output['svi_score'] = svi_scores
    print(f"✓ Calculated SVI scores for {len(svi_scores)} tracts")
    print(f"  SVI range: {min(svi_scores):.2f} - {max(svi_scores):.2f}")
    print(f"  Average SVI: {sum(svi_scores)/len(svi_scores):.2f}")
    
    # Reorder columns to match example: tract_id, lat, lon, risk_probability, demand_mean, demand_std, svi_score
    output = output[['tract_id', 'lat', 'lon', 'risk_probability', 'demand_mean', 'demand_std', 'svi_score']]
    
    # Save required format CSV (no quotes, standard CSV format)
    csv_file = OUTPUT_DIR / "top1000_highest_risk_tracts.csv"
    # Convert tract_id to string explicitly before saving
    output['tract_id'] = output['tract_id'].astype(str)
    output.to_csv(csv_file, index=False, quoting=0)  # quoting=0 means QUOTE_MINIMAL (only when needed)
    print(f"✓ Saved CSV to: {csv_file}")
    print(f"  Columns: {list(output.columns)}")
    print(f"  Rows: {len(output)}")
    print(f"  Sample tract_id: {output['tract_id'].iloc[0]} (length: {len(output['tract_id'].iloc[0])})")
    
    # Also save detailed version with additional info (if State/County columns exist)
    detailed_file = OUTPUT_DIR / "top1000_detailed.csv"
    if 'State' in top1000_with_coords.columns and 'County' in top1000_with_coords.columns:
        detailed_output = top1000_with_coords[['CensusTract', 'State', 'County', 'probability', 'risk_level']].copy()
        detailed_output['tract_id'] = detailed_output['CensusTract'].astype(str).str.zfill(11)
        detailed_output = detailed_output[['tract_id', 'State', 'County', 'probability', 'risk_level']]
        detailed_output.to_csv(detailed_file, index=False)
        print(f"✓ Saved detailed CSV to: {detailed_file}")
    
    # Save as formatted text (using top1000_with_coords for detailed info)
    txt_file = OUTPUT_DIR / "top1000_highest_risk_tracts.txt"
    with open(txt_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TOP {len(output)} CENSUS TRACTS MOST LIKELY TO BECOME FOOD DESERTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Based on predictive modeling using demographic, socioeconomic, retail,\n")
        f.write("and transportation access indicators.\n")
        f.write("Note: Only tracts with valid geographic coordinates are included.\n\n")
        f.write("=" * 80 + "\n\n")
        
        # Write tract details (only if State/County columns exist)
        if 'State' in top1000_with_coords.columns and 'County' in top1000_with_coords.columns:
            for idx, row in top1000_with_coords.iterrows():
                f.write(f"Rank {row['Rank']:4d}: Census Tract {row['CensusTract']}\n")
                f.write(f"  Location: {row['County']}, {row['State']}\n")
                f.write(f"  Risk Probability: {row['probability']:.4f}\n")
                if 'risk_level' in row:
                    f.write(f"  Risk Level: {row['risk_level']}\n")
                if 'currently_low_access' in row and pd.notna(row['currently_low_access']):
                    status = "Yes" if row['currently_low_access'] == 1 else "No"
                    f.write(f"  Currently Low Access: {status}\n")
                f.write("\n")
        else:
            # If no State/County info, just write tract IDs
            for idx, row in top1000_with_coords.iterrows():
                f.write(f"Rank {row['Rank']:4d}: Census Tract {row['CensusTract']}\n")
                f.write(f"  Risk Probability: {row['probability']:.4f}\n")
                f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Tracts Analyzed: {len(df):,}\n")
        f.write(f"Top {len(output)} Risk Probability Range: {output['risk_probability'].min():.4f} - {output['risk_probability'].max():.4f}\n")
        f.write(f"Average Risk Probability (Top {len(output)}): {output['risk_probability'].mean():.4f}\n")
        f.write(f"Median Risk Probability (Top {len(output)}): {output['risk_probability'].median():.4f}\n\n")
        
        # State breakdown (if available)
        if 'State' in top1000_with_coords.columns:
            state_counts = top1000_with_coords['State'].value_counts()
            f.write(f"Top {len(output)} by State:\n")
            for state, count in state_counts.items():
                f.write(f"  {state}: {count} tracts\n")
            f.write("\n")
            
            # County breakdown (top 10, if available)
            if 'County' in top1000_with_coords.columns:
                county_counts = top1000_with_coords['County'].value_counts().head(10)
                f.write("Top 10 Counties:\n")
                for county, count in county_counts.items():
                    f.write(f"  {county}: {count} tracts\n")
    
    print(f"✓ Saved formatted text to: {txt_file}")
    
    # Save as markdown format
    md_file = OUTPUT_DIR / "top1000_highest_risk_tracts.md"
    with open(md_file, 'w') as f:
        f.write("# Top 1000 Census Tracts Most Likely to Become Food Deserts\n\n")
        f.write("Based on predictive modeling using demographic, socioeconomic, retail, and transportation access indicators.\n")
        f.write("**Note:** Only tracts with valid geographic coordinates are included.\n\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Tracts Analyzed:** {len(df):,}\n")
        f.write(f"- **Risk Probability Range (Top {len(output)}):** {output['risk_probability'].min()*100:.2f}% - {output['risk_probability'].max()*100:.2f}%\n")
        f.write(f"- **Average Risk Probability (Top {len(output)}):** {output['risk_probability'].mean()*100:.2f}%\n")
        f.write(f"- **Median Risk Probability (Top {len(output)}):** {output['risk_probability'].median()*100:.2f}%\n")
        f.write(f"- **Average Weekly Demand:** {output['demand_mean'].mean():.1f} households\n")
        f.write(f"- **Average SVI Score:** {output['svi_score'].mean():.2f}\n\n")
        
        # State breakdown
        if 'State' in top1000_with_coords.columns:
            f.write("## Top 1000 by State\n\n")
            f.write("| State | Number of Tracts |\n")
            f.write("|-------|------------------|\n")
            state_counts = top1000_with_coords['State'].value_counts().sort_values(ascending=False)
            for state, count in state_counts.items():
                f.write(f"| {state} | {count} |\n")
            f.write("\n")
        
        # County breakdown (top 10)
        if 'State' in top1000_with_coords.columns and 'County' in top1000_with_coords.columns:
            f.write("## Top 10 Counties in Top 1000\n\n")
            f.write("| County | State | Number of Tracts |\n")
            f.write("|--------|-------|------------------|\n")
            county_counts = top1000_with_coords.groupby(['County', 'State']).size().reset_index(name='count')
            county_counts = county_counts.sort_values('count', ascending=False).head(10)
            for _, row in county_counts.iterrows():
                f.write(f"| {row['County']} | {row['State']} | {row['count']} |\n")
            f.write("\n")
        
        # Complete list (top 100 shown, rest available in CSV)
        f.write(f"## Complete List (Top {len(output)})\n\n")
        f.write(f"*Note: Full list of {len(output)} tracts is available in the CSV file. Showing top 100 below.*\n\n")
        f.write("| Rank | Census Tract | State | County | Risk Probability | Risk Level | Currently Low Access |\n")
        f.write("|------|--------------|-------|--------|------------------|-----------|---------------------|\n")
        
        for idx, row in top1000_with_coords.head(100).iterrows():
            rank = row['Rank']
            tract_id = row['CensusTract']
            state = row.get('State', 'N/A') if 'State' in row else 'N/A'
            county = row.get('County', 'N/A') if 'County' in row else 'N/A'
            prob_pct = row['probability'] * 100
            risk_level = row.get('risk_level', 'High Risk') if 'risk_level' in row else 'High Risk'
            
            # Currently low access status
            if 'currently_low_access' in row and pd.notna(row['currently_low_access']):
                low_access = "Yes" if row['currently_low_access'] == 1 else "No"
            else:
                low_access = "N/A"
            
            f.write(f"| {rank} | {tract_id} | {state} | {county} | {prob_pct:.2f}% | {risk_level} | {low_access} |\n")
    
    print(f"✓ Saved markdown format to: {md_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Top 1000 Summary")
    print("=" * 60)
    print(f"Risk Probability Range: {output['risk_probability'].min():.4f} - {output['risk_probability'].max():.4f}")
    print(f"Average Risk Probability: {output['risk_probability'].mean():.4f}")
    print(f"Average Weekly Demand: {output['demand_mean'].mean():.1f} households")
    if 'State' in top1000_with_coords.columns:
        print(f"\nTop 10 States:")
        for state, count in top1000_with_coords['State'].value_counts().head(10).items():
            print(f"  {state}: {count} tracts")
    
    print("\n" + "=" * 60)
    print("Top 1000 list generation complete!")
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
    output = generate_top1000()
    print("\nFirst 10 tracts (required format):")
    print(output.head(10).to_string(index=False))

