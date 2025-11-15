"""
Generate a complete CSV file of all U.S. census tract coordinates from TIGER shapefiles.

This script downloads TIGER/Line shapefiles for all states and extracts tract centroids,
creating a comprehensive mapping file that can be used by generate_top100.py.

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

BASE_DIR = Path(__file__).parent.parent
CENSUS_DIR = BASE_DIR / "data" / "01_census_demographics"
TIGER_CACHE_DIR = CENSUS_DIR / "tiger_cache"
OUTPUT_FILE = CENSUS_DIR / "tract_coordinates.csv"

TIGER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_tiger_shapefile(state_fips, year=2022):
    """Download Census TIGER/Line shapefile for a state."""
    state_dir = TIGER_CACHE_DIR / f"state_{state_fips}"
    shapefile_path = state_dir / f"tl_{year}_{state_fips}_tract.shp"
    
    if shapefile_path.exists():
        return state_dir
    
    print(f"  Downloading TIGER shapefile for state {state_fips}...")
    
    try:
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
        zip_path = state_dir / f"tl_{year}_{state_fips}_tract.zip"
        state_dir.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=f"    Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(state_dir)
        
        zip_path.unlink()
        return state_dir
        
    except Exception as e:
        print(f"    ⚠ Error downloading state {state_fips}: {e}")
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
    print(f"\nThis will download TIGER shapefiles for all 50 states + DC + territories")
    print(f"and extract tract centroids. This may take 30-60 minutes.\n")
    
    # Get all state FIPS codes (01-56)
    state_fips = [f"{i:02d}" for i in range(1, 57)]
    
    all_coords = []
    failed_states = []
    
    print(f"Processing {len(state_fips)} states/territories...\n")
    
    for state_fips_code in tqdm(state_fips, desc="Processing states"):
        coords = extract_tract_coordinates_from_state(state_fips_code)
        if coords:
            all_coords.extend(coords)
            print(f"  State {state_fips_code}: {len(coords)} tracts")
        else:
            failed_states.append(state_fips_code)
        time.sleep(0.5)  # Be polite to the server
    
    if not all_coords:
        print("\n⚠ No coordinates extracted. Check network connection and try again.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_coords)
    
    # Remove duplicates (keep first)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['tract_id'], keep='first')
    if len(df) < initial_count:
        print(f"\n  Removed {initial_count - len(df)} duplicate tract IDs")
    
    # Sort by tract_id
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
    
    print(f"\nThis file will now be used by generate_top100.py for geocoding.")
    print("=" * 60)

if __name__ == "__main__":
    main()

