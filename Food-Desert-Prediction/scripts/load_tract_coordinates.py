"""
Script to load tract coordinates from a CSV file.

This allows you to provide a complete mapping of tract IDs to lat/lon coordinates
if you have them from another source or want to supplement missing coordinates.

Usage:
1. Create a CSV file with columns: tract_id, lat, lon
2. Place it at: data/01_census_demographics/tract_coordinates.csv
3. Run: python scripts/generate_top100.py

The script will automatically use this file if it exists.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
COORDS_FILE = BASE_DIR / "data" / "01_census_demographics" / "tract_coordinates.csv"

def load_tract_coordinates_from_csv():
    """
    Load tract coordinates from CSV file.
    
    Expected CSV format:
    tract_id,lat,lon
    01001020100,32.361538,-86.279118
    01001020200,32.354234,-86.265432
    ...
    
    Returns:
        Dictionary mapping tract_id (string) to (lat, lon) tuple, or None if file doesn't exist
    """
    if not COORDS_FILE.exists():
        return None
    
    try:
        print(f"  Loading coordinates from: {COORDS_FILE}")
        df = pd.read_csv(COORDS_FILE)
        
        # Validate required columns
        required_cols = ['tract_id', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠ Missing required columns: {missing_cols}")
            print(f"  Required columns: {required_cols}")
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
        
        print(f"  ✓ Loaded {len(coords_dict)} tract coordinates from CSV")
        return coords_dict
        
    except Exception as e:
        print(f"  ⚠ Error loading coordinates from CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test loading
    coords = load_tract_coordinates_from_csv()
    if coords:
        print(f"\nSample coordinates:")
        for i, (tract_id, (lat, lon)) in enumerate(list(coords.items())[:5]):
            print(f"  {tract_id}: ({lat:.6f}, {lon:.6f})")
    else:
        print(f"\nNo coordinates file found at: {COORDS_FILE}")
        print(f"\nTo create the file:")
        print(f"1. Create a CSV with columns: tract_id, lat, lon")
        print(f"2. Place it at: {COORDS_FILE}")
        print(f"3. Format example:")
        print(f"   tract_id,lat,lon")
        print(f"   01001020100,32.361538,-86.279118")
        print(f"   01001020200,32.354234,-86.265432")

