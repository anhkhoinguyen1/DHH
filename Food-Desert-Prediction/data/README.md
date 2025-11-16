# Data Directory Structure

This directory contains all data sources organized into 5 main categories for the Food Desert Prediction project.

## Directory Organization

### `01_census_demographics/`
**Purpose**: U.S. Census Bureau demographic and socioeconomic data

**Contents**:
- `census_acs_api_data.csv` - American Community Survey (ACS) 5-year estimates collected via API
  - Contains: Income, poverty, education, housing, transportation, population demographics
  - Source: U.S. Census Bureau API (https://api.census.gov)
  - Year: 2023 ACS 5-year estimates
  - Coverage: ~84,400 census tracts across all U.S. states
  
- `CensusACSTract.xlsx` - Alternative local ACS data file (if API unavailable)
- `CensusACSBlockGroup.xlsx`, `CensusACSMCD.xlsx`, etc. - Additional geographic levels
- `metadata/` - Data documentation and metadata
- `tiger_cache/` - Cached Census TIGER/Line shapefiles for tract geocoding

**Key Variables**:
- Median household income
- Poverty rates
- Education levels (bachelor's degree and higher)
- Housing characteristics (rent, home values, owner/renter status)
- Vehicle ownership
- Population demographics (age, race, ethnicity)

---

### `02_food_access/`
**Purpose**: USDA Food Access Research Atlas data

**Contents**:
- `FoodAccessResearchAtlasData2019.xlsx` - Current food access conditions (2019)
- `FoodAccessResearchAtlasData2015.xlsx` - Historical food access conditions (2015)
- `USDA Food Environment Atlas/` - Additional USDA documentation and data files

**Key Variables**:
- Low-income and low-access tract flags (LILATracts)
- Distance to nearest grocery store
- Low-access population counts
- SNAP recipient households
- Vehicle availability

**Source**: USDA Economic Research Service
- Website: https://www.ers.usda.gov/data-products/food-access-research-atlas/
- Direct Download: https://www.ers.usda.gov/webdocs/DataFiles/80591/

---

### `03_health_outcomes/`
**Purpose**: CDC PLACES health outcome data

**Contents**:
- `CDC_PLACES_Tract.csv` - Health outcomes at census tract level
  - Contains: Diabetes, obesity, hypertension, physical inactivity prevalence
  - Source: CDC PLACES (Place-Based Data for Better Health)
  - Year: 2023 release
  - Coverage: ~2.58 million records (all census tracts, multiple health measures)

**Key Variables**:
- Diabetes prevalence (%)
- Obesity prevalence (%)
- Hypertension prevalence (%)
- Physical inactivity (%)
- Other chronic disease indicators

**Source**: Centers for Disease Control and Prevention
- Website: https://www.cdc.gov/places/
- API Endpoint: https://data.cdc.gov/resource/cwsq-ngmh.json
- Direct Download: https://data.cdc.gov/api/views/cwsq-ngmh/rows.csv?accessType=DOWNLOAD

---

### `04_housing_economics/`
**Purpose**: Zillow housing market and affordability data

**Contents**:
- `Zillow/` - Directory containing Zillow Research data files:
  - `Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv` - Home Value Index (ZHVI)
  - `Metro_zori_uc_sfrcondomfr_sm_month.csv` - Rent Index (ZORI)
  - `Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv` - Income needed for homeownership
  - `Metro_new_renter_income_needed_uc_sfrcondomfr_sm_sa_month.csv` - Income needed for renting
  - `Metro_new_renter_affordability_uc_sfrcondomfr_sm_sa_month.csv` - Renter affordability (% income on rent)
  - `Metro_market_temp_index_uc_sfrcondo_month.csv` - Market temperature index

**Key Variables**:
- Home values (ZHVI)
- Rent prices (ZORI)
- Affordability metrics
- Market volatility indicators

**Source**: Zillow Research
- Website: https://www.zillow.com/research/data/
- Note: Data is at Metro/MSA level, aggregated to state level for tract matching

---

### `05_infrastructure_transportation/`
**Purpose**: Grocery store locations and public transit data

**Contents**:
- `grocery_stores_osm.csv` - Grocery store locations from OpenStreetMap
  - Contains: Store names, addresses, coordinates, store types
  - Source: OpenStreetMap Overpass API
  - Coverage: All U.S. grocery stores in OSM database
  
- `gtfs/` - General Transit Feed Specification (GTFS) data:
  - `NYC_MTA.zip` - New York City Metropolitan Transportation Authority
  - `LA_Metro.zip` - Los Angeles Metro
  - Additional transit agency feeds can be added

**Key Variables**:
- Grocery store locations (latitude, longitude)
- Store types (supermarket, convenience store, etc.)
- Transit stop locations
- Route coverage and frequency

**Sources**:
- OpenStreetMap: https://www.openstreetmap.org/
- Overpass API: https://wiki.openstreetmap.org/wiki/Overpass_API
- GTFS Feeds: https://transitfeeds.com/

---

## Additional Directories

### `processed/`
**Purpose**: Cleaned and merged datasets ready for analysis

**Contents**:
- `food_access_2019.csv` - Processed Food Access Atlas 2019 data
- `food_access_2015.csv` - Processed Food Access Atlas 2015 data (if available)
- `change_features_2015_2019.csv` - Calculated change metrics between years
- `target_variable.csv` - Target variable for model training
- `data_coverage_summary.csv` - Summary of data coverage by state (tract counts per state)
- `data_coverage_map.png` - **Geographic map visualization** showing census tract data coverage:
  - Map of the contiguous United States (excluding Alaska and Hawaii)
  - Census tracts with data are plotted as green points
  - Shows geographic distribution of data availability
  - Generated automatically during data processing
  - Uses tract coordinates from `01_census_demographics/tract_coordinates.csv`

### `features/`
**Purpose**: Engineered features for machine learning models

**Contents**:
- `modeling_features.csv` - Final feature set with all variables merged and engineered

### `predictions/`
**Purpose**: Model predictions and outputs

**Contents**:
- `predictions.csv` - Predictions for all census tracts
- `prediction_summary.csv` - Summary statistics
- `top1000_highest_risk_tracts.csv` - Top 1000 highest-risk tracts (final output)
- `top1000_highest_risk_tracts.txt` - Human-readable format
- `top1000_highest_risk_tracts.md` - Markdown format with tables

---

## Data Collection

To collect all data sources, run:
```bash
python scripts/collect_data.py
```

This script will:
1. Download Food Access Atlas data (2019, 2015)
2. Collect Census ACS data via API (requires API key)
3. Download CDC PLACES health outcomes data
4. Check for Zillow housing data files
5. Collect grocery store data from OpenStreetMap
6. Download GTFS transit feeds

**Note**: Some data sources require manual download:
- Zillow data must be downloaded from https://www.zillow.com/research/data/
- Place files in `04_housing_economics/Zillow/`

## Data Verification

To verify all data sources are collected correctly:
```bash
python scripts/verify_data_collection.py
```

## Data Sources Summary

| Category | Source | Update Frequency | Geographic Level |
|----------|--------|------------------|------------------|
| Census Demographics | U.S. Census Bureau | Annual | Census Tract |
| Food Access | USDA ERS | Annual (1-2 year lag) | Census Tract |
| Health Outcomes | CDC PLACES | Annual | Census Tract |
| Housing Economics | Zillow Research | Monthly | Metro/MSA (aggregated to state) |
| Infrastructure | OpenStreetMap, Transit Agencies | Varies | Point locations, Routes |

## Data Licensing

All data sources are publicly available and free to use:
- **USDA**: Public domain
- **U.S. Census**: Public domain
- **CDC**: Public domain
- **Zillow**: Free for research use (check terms)
- **OpenStreetMap**: Open Database License (ODbL)
- **GTFS**: Varies by agency (typically open data)

Always attribute data sources in publications and documentation.

---

**Last Updated**: 2025-01-27

