"""
Generate a complete CSV file of all U.S. census tract coordinates from TIGER shapefiles.

This script downloads TIGER/Line shapefiles for all states and extracts tract centroids,
creating a comprehensive mapping file that can be used by generate_top1000.py.

Usage:
    python scripts/generate_all_tract_coordinates.py

Output:
    data/01_census_demographics/tract_coordinates.csv
"""

import pandas as pd
import geopandas as gpd
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import time
import os

BASE_DIR = Path(__file__).parent.parent
CENSUS_DIR = BASE_DIR / "data" / "01_census_demographics"
TIGER_CACHE_DIR = CENSUS_DIR / "tiger_cache"
OUTPUT_FILE = CENSUS_DIR / "tract_coordinates.csv"

TIGER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_tiger_shapefile(state_fips, year=2022, max_retries=3):
    """
    Download Census TIGER/Line shapefile for a state with retry logic.
    
    Args:
        state_fips: 2-digit state FIPS code
        year: Year of TIGER data
        max_retries: Maximum number of retry attempts
    
    Returns:
        Path to downloaded shapefile directory, or None if failed
    """
    state_dir = TIGER_CACHE_DIR / f"state_{state_fips}"
    shapefile_path = state_dir / f"tl_{year}_{state_fips}_tract.shp"
    
    # Check if already downloaded
    if shapefile_path.exists():
        return state_dir
    
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    zip_path = state_dir / f"tl_{year}_{state_fips}_tract.zip"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            print(f"  Downloading TIGER shapefile for state {state_fips} (attempt {attempt + 1}/{max_retries})...")
            
            # Check if partial download exists and resume
            resume_header = {}
            if zip_path.exists() and zip_path.stat().st_size > 0:
                resume_header['Range'] = f'bytes={zip_path.stat().st_size}-'
                print(f"    Resuming download from byte {zip_path.stat().st_size}")
            
            # Use longer timeout and allow redirects
            response = requests.get(
                url, 
                stream=True, 
                timeout=(30, 300),  # (connect timeout, read timeout) - 5 min read timeout
                headers=resume_header,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Get total size
            if 'content-range' in response.headers:
                # Resuming download
                total_size = int(response.headers['content-range'].split('/')[-1])
                mode = 'ab'  # Append mode for resume
                initial_pos = zip_path.stat().st_size
            else:
                total_size = int(response.headers.get('content-length', 0))
                mode = 'wb'  # Write mode for new download
                initial_pos = 0
            
            # Download with progress bar
            with open(zip_path, mode) as f, tqdm(
                total=total_size, 
                initial=initial_pos,
                unit='B', 
                unit_scale=True, 
                desc=f"    Downloading",
                unit_divisor=1024
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192 * 4):  # Larger chunk size
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify download completed
            if zip_path.exists() and zip_path.stat().st_size > 0:
                # Try to extract to verify zip is valid
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.testzip()  # Test zip integrity
                        zip_ref.extractall(state_dir)
                    
                    # Verify shapefile exists after extraction
                    if shapefile_path.exists():
                        # Remove zip to save space
                        zip_path.unlink()
                        print(f"    ✓ Downloaded and extracted TIGER shapefile for state {state_fips}")
                        return state_dir
                    else:
                        raise ValueError("Shapefile not found after extraction")
                except zipfile.BadZipFile:
                    # Zip file is corrupted, delete and retry
                    zip_path.unlink()
                    raise ValueError("Corrupted zip file")
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                print(f"    ⚠ Timeout/Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"    Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"    ⚠ Failed after {max_retries} attempts: {e}")
                # Clean up partial download
                if zip_path.exists():
                    zip_path.unlink()
                return None
        except Exception as e:
            print(f"    ⚠ Error downloading state {state_fips}: {e}")
            # Clean up partial download
            if zip_path.exists():
                zip_path.unlink()
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                print(f"    Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return None
    
    return None

def extract_tract_coordinates_from_state(state_fips, year=2022):
    """Extract tract coordinates from a state's TIGER shapefile."""
    state_dir = download_tiger_shapefile(state_fips, year)
    if state_dir is None:
        return []
    
    shapefile_path = state_dir / f"tl_{year}_{state_fips}_tract.shp"
    if not shapefile_path.exists():
        return []
    
    try:
        gdf = gpd.read_file(shapefile_path)
        
        # Extract GEOID
        if 'GEOID' in gdf.columns:
            gdf['TRACT_GEOID'] = gdf['GEOID'].astype(str).str.zfill(11)
        elif 'TRACTCE' in gdf.columns and 'STATEFP' in gdf.columns and 'COUNTYFP' in gdf.columns:
            gdf['TRACT_GEOID'] = (
                gdf['STATEFP'].astype(str).str.zfill(2) +
                gdf['COUNTYFP'].astype(str).str.zfill(3) +
                gdf['TRACTCE'].astype(str).str.zfill(6)
            )
        else:
            print(f"    ⚠ Could not find GEOID in state {state_fips}")
            return []
        
        # Calculate centroids
        # First project to a projected CRS for accurate centroid calculation
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
    
        # Project to Albers Equal Area (good for US) for accurate centroid calculation
        # This avoids the geographic CRS warning
        from pyproj import CRS
        if gdf.crs.to_string() == 'EPSG:4326':
            # Use US Albers Equal Area Conic for accurate centroids
            gdf_projected = gdf.to_crs(epsg=5070)  # US Albers
            centroids_projected = gdf_projected.geometry.centroid
            # Convert back to WGS84
            centroids_gdf = gpd.GeoDataFrame(geometry=centroids_projected, crs='epsg:5070')
            centroids = centroids_gdf.to_crs(epsg=4326).geometry
        else:
            # Already in a projected CRS, calculate centroid then convert
            gdf_projected = gdf.to_crs(epsg=5070)
            centroids_projected = gdf_projected.geometry.centroid
            centroids_gdf = gpd.GeoDataFrame(geometry=centroids_projected, crs='epsg:5070')
            centroids = centroids_gdf.to_crs(epsg=4326).geometry
        gdf['lat'] = centroids.y
        gdf['lon'] = centroids.x
        
        # Create list of coordinates
        coords = []
        for _, row in gdf.iterrows():
            coords.append({
                'tract_id': str(row['TRACT_GEOID']).zfill(11),
                'lat': float(row['lat']),
                'lon': float(row['lon'])
            })
        
        return coords
        
    except Exception as e:
        print(f"    ⚠ Error processing state {state_fips}: {e}")
        return []

def main():
    """Generate complete tract coordinates CSV file."""
    print("=" * 60)
    print("Generating Complete Tract Coordinates File")
    print("=" * 60)
    print(f"\nThis will download TIGER shapefiles for all 50 U.S. states")
    print(f"and extract tract centroids. This may take 30-60 minutes.\n")
    
    # Get FIPS codes for 50 U.S. states only (excluding DC and territories)
    # Standard FIPS codes: 01-02, 04-06, 08-13, 15-42, 44-49, 51, 53-56
    # Excluded: 03, 07, 11 (DC), 14, 43, 52, and territories (60, 66, 69, 72, 78)
    valid_state_fips = [
        '01', '02',  # Alabama, Alaska
        '04', '05', '06',  # Arizona, Arkansas, California
        '08', '09', '10',  # Colorado, Connecticut, Delaware
        '12', '13',  # Florida, Georgia
        '15', '16', '17', '18', '19',  # Hawaii, Idaho, Illinois, Indiana, Iowa
        '20', '21', '22',  # Kansas, Kentucky, Louisiana
        '23', '24', '25', '26', '27', '28', '29',  # Maine, Maryland, Massachusetts, Michigan, Minnesota, Mississippi, Missouri
        '30', '31', '32', '33', '34', '35',  # Montana, Nebraska, Nevada, New Hampshire, New Jersey, New Mexico
        '36', '37', '38', '39',  # New York, North Carolina, North Dakota, Ohio
        '40', '41', '42',  # Oklahoma, Oregon, Pennsylvania
        '44', '45', '46', '47', '48', '49',  # Rhode Island, South Carolina, South Dakota, Tennessee, Texas, Utah
        '50', '51',  # Vermont, Virginia
        '53', '54', '55', '56'  # Washington, West Virginia, Wisconsin, Wyoming
    ]
    state_fips = valid_state_fips
    
    all_coords = []
    failed_states = []
    
    print(f"Processing {len(state_fips)} U.S. states...\n")
    
    # Check for existing partial results
    existing_coords = []
    if OUTPUT_FILE.exists():
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            existing_coords = existing_df.to_dict('records')
            print(f"  Found existing coordinate file with {len(existing_coords)} tracts")
            print(f"  Will append new coordinates and update existing ones\n")
        except:
            pass
    
    # Create a dictionary for faster lookups and updates (preserves existing, updates with new)
    coords_dict = {c['tract_id']: c for c in existing_coords} if existing_coords else {}
    
    for state_fips_code in tqdm(state_fips, desc="Processing states"):
        coords = extract_tract_coordinates_from_state(state_fips_code)
        if coords:
            # Update dictionary with new coordinates (overwrites existing if same tract_id)
            for coord in coords:
                coords_dict[coord['tract_id']] = coord
            print(f"  State {state_fips_code}: {len(coords)} tracts")
        else:
            failed_states.append(state_fips_code)
        time.sleep(1)  # Be polite to the server (increased delay)
    
    # Convert dictionary back to list
    all_coords = list(coords_dict.values())
    
    if not all_coords:
        print("\n⚠ No coordinates extracted. Check network connection and try again.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_coords)
    
    # Ensure tract_id is string type for consistent sorting
    df['tract_id'] = df['tract_id'].astype(str).str.zfill(11)
    
    # Remove duplicates (keep first)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['tract_id'], keep='first')
    if len(df) < initial_count:
        print(f"\n  Removed {initial_count - len(df)} duplicate tract IDs")
    
    # Sort by tract_id (now all strings, will sort correctly)
    df = df.sort_values('tract_id').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n" + "=" * 60)
    print(f"✓ Successfully generated tract coordinates file")
    print(f"  File: {OUTPUT_FILE}")
    print(f"  Total tracts: {len(df):,}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / (1024*1024):.1f} MB")
    
    if failed_states:
        print(f"\n  ⚠ Failed to process {len(failed_states)} states: {failed_states}")
    
    print(f"\nThis file will now be used by generate_top1000.py for geocoding.")
    print("=" * 60)

if __name__ == "__main__":
    main()

