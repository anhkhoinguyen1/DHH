# Tract Coordinates CSV File

## Purpose

This file allows you to provide a complete mapping of census tract IDs to latitude/longitude coordinates. This is useful if:

1. You have coordinates from another source
2. The TIGER/Line shapefile download is missing some tracts
3. You want to use more accurate coordinates (e.g., from a geocoding service)

## File Location

Place your CSV file at:
```
data/01_census_demographics/tract_coordinates.csv
```

## CSV Format

The file must have exactly these columns:

```csv
tract_id,lat,lon
01001020100,32.361538,-86.279118
01001020200,32.354234,-86.265432
01001020300,32.347891,-86.251234
...
```

### Column Requirements

- **tract_id**: 11-digit census tract FIPS code (can be with or without leading zeros)
- **lat**: Latitude in decimal degrees (WGS84, EPSG:4326)
- **lon**: Longitude in decimal degrees (WGS84, EPSG:4326)

### Example

```csv
tract_id,lat,lon
48471790200,30.619852,-95.425684
24015030506,39.123456,-76.789012
13215010501,32.505891,-84.908636
```

## How to Generate This File

### Option 1: From TIGER/Line Shapefiles (Recommended)

Run this script to generate coordinates for all tracts from TIGER shapefiles:

```bash
python scripts/generate_all_tract_coordinates.py
```

This will:
1. Download TIGER shapefiles for all states
2. Extract centroids for all tracts
3. Save to `data/01_census_demographics/tract_coordinates.csv`

### Option 2: From Census Geocoding API

You can use the Census Geocoding API to get coordinates for specific tracts:

```python
import requests
import pandas as pd

# Example: Get coordinates for a tract
tract_id = "01001020100"
state = tract_id[:2]
county = tract_id[2:5]
tract = tract_id[5:]

url = f"https://geocoding.geo.census.gov/geocoder/geographies/address"
params = {
    'street': '',
    'city': '',
    'state': state,
    'zip': '',
    'benchmark': 'Public_AR_Current',
    'vintage': 'Current_Current',
    'format': 'json'
}
# Note: Census Geocoding API requires an address, not just tract ID
# This method is more complex - use TIGER shapefiles instead
```

### Option 3: Manual Entry

If you have coordinates from another source (e.g., Google Maps, ArcGIS), create the CSV manually:

1. Create a CSV file with headers: `tract_id,lat,lon`
2. Add one row per tract with the coordinates
3. Save as `data/01_census_demographics/tract_coordinates.csv`

### Option 4: From Existing Data

If you already have coordinates in another file, you can convert it:

```python
import pandas as pd

# Load your existing file
df = pd.read_csv('your_file.csv')

# Extract and format
output = pd.DataFrame({
    'tract_id': df['tract_id'].astype(str).str.zfill(11),
    'lat': df['latitude'],
    'lon': df['longitude']
})

# Save
output.to_csv('data/01_census_demographics/tract_coordinates.csv', index=False)
```

## Usage

Once the file is in place, the `generate_top100.py` script will automatically:

1. Check for `tract_coordinates.csv` first
2. Use those coordinates if available
3. Fall back to TIGER shapefiles for any missing tracts
4. Merge both sources if needed

## File Size

For all ~72,000 U.S. census tracts, the file will be approximately:
- **Size**: ~2-3 MB (CSV format)
- **Rows**: ~72,000 (one per tract)

## Validation

The script will automatically:
- Validate required columns exist
- Remove rows with missing coordinates
- Pad tract IDs to 11 digits
- Handle coordinate format conversion

## Troubleshooting

### "Missing required columns" error
- Make sure your CSV has exactly: `tract_id,lat,lon` as the header row
- Check for typos in column names

### Coordinates not being used
- Verify file is at: `data/01_census_demographics/tract_coordinates.csv`
- Check file permissions (should be readable)
- Verify CSV format is correct (no extra columns, proper encoding)

### Missing tracts
- The script will supplement with TIGER data for missing tracts
- To get all tracts, use Option 1 to generate from TIGER shapefiles

