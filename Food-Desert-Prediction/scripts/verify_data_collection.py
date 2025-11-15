"""
Quick verification script to check if all data sources are collected successfully.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CENSUS_DIR = BASE_DIR / "data" / "01_census_demographics"
FOOD_ACCESS_DIR = BASE_DIR / "data" / "02_food_access"
HEALTH_DIR = BASE_DIR / "data" / "03_health_outcomes"
HOUSING_DIR = BASE_DIR / "data" / "04_housing_economics"
INFRASTRUCTURE_DIR = BASE_DIR / "data" / "05_infrastructure_transportation"

def verify_data_collection():
    """Verify all data sources are collected."""
    print("=" * 60)
    print("VERIFYING DATA COLLECTION")
    print("=" * 60)
    
    all_good = True
    
    # 1. Food Access Atlas 2019
    print("\n1. Food Access Atlas 2019:")
    fa_2019 = FOOD_ACCESS_DIR / "FoodAccessResearchAtlasData2019.xlsx"
    if fa_2019.exists() and fa_2019.stat().st_size > 1024 * 1024:
        size_mb = fa_2019.stat().st_size / (1024 * 1024)
        print(f"   ✓ Found ({size_mb:.1f} MB)")
        try:
            df = pd.read_excel(fa_2019, nrows=1)
            print(f"   ✓ Valid Excel file ({len(df.columns)} columns)")
        except:
            print(f"   ⚠ File exists but may be corrupted")
            all_good = False
    else:
        print(f"   ✗ Missing or too small")
        all_good = False
    
    # 2. Food Access Atlas 2015
    print("\n2. Food Access Atlas 2015:")
    fa_2015 = FOOD_ACCESS_DIR / "FoodAccessResearchAtlasData2015.xlsx"
    local_2015 = FOOD_ACCESS_DIR / "USDA Food Environment Atlas" / "2015 Food Access Research Atlas" / "FoodAccessResearchAtlasData2015.xlsx"
    
    found = False
    if fa_2015.exists() and fa_2015.stat().st_size > 1024 * 1024:
        size_mb = fa_2015.stat().st_size / (1024 * 1024)
        print(f"   ✓ Found in data/raw ({size_mb:.1f} MB)")
        found = True
    elif local_2015.exists() and local_2015.stat().st_size > 1024 * 1024:
        size_mb = local_2015.stat().st_size / (1024 * 1024)
        print(f"   ✓ Found in local folder ({size_mb:.1f} MB)")
        found = True
    
    if found:
        try:
            filepath = fa_2015 if fa_2015.exists() else local_2015
            df = pd.read_excel(filepath, nrows=1)
            print(f"   ✓ Valid Excel file ({len(df.columns)} columns)")
        except:
            print(f"   ⚠ File exists but may be corrupted")
            all_good = False
    else:
        print(f"   ✗ Missing")
        all_good = False
    
    # 3. Census ACS Data
    print("\n3. Census ACS Data:")
    acs_file = CENSUS_DIR / "census_acs_api_data.csv"
    if acs_file.exists() and acs_file.stat().st_size > 1024 * 1024:
        size_mb = acs_file.stat().st_size / (1024 * 1024)
        df = pd.read_csv(acs_file, nrows=100)
        print(f"   ✓ Found ({size_mb:.1f} MB)")
        print(f"   ✓ {len(df)} sample rows, {len(df.columns)} columns")
        if 'CensusTract' in df.columns:
            print(f"   ✓ Contains CensusTract column")
        else:
            print(f"   ⚠ Missing CensusTract column")
            all_good = False
    else:
        print(f"   ✗ Missing or too small")
        print(f"   ⚠ Run collect_data.py to collect via API")
        all_good = False
    
    # 4. CDC PLACES Data
    print("\n4. CDC PLACES Data:")
    cdc_file = HEALTH_DIR / "CDC_PLACES_Tract.csv"
    if cdc_file.exists():
        size_mb = cdc_file.stat().st_size / (1024 * 1024)
        if size_mb > 5000:
            print(f"   ⚠ File is very large ({size_mb:.1f} MB) - likely COVID data, not health outcomes")
            print(f"   ⚠ Should be < 500MB for health outcomes data")
            all_good = False
        elif size_mb > 100:
            print(f"   ✓ Found ({size_mb:.1f} MB)")
            try:
                df = pd.read_csv(cdc_file, nrows=100)
                print(f"   ✓ {len(df)} sample rows, {len(df.columns)} columns")
            except:
                print(f"   ⚠ File exists but may be corrupted")
                all_good = False
        else:
            print(f"   ⚠ File too small ({size_mb:.1f} MB) - may be incomplete")
            all_good = False
    else:
        print(f"   ✗ Missing")
        print(f"   ⚠ Run collect_data.py to collect via API")
        all_good = False
    
    # 5. Zillow Data
    print("\n5. Zillow Housing Data:")
    required_files = [
        "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "Metro_zori_uc_sfrcondomfr_sm_month.csv",
        "Metro_new_homeowner_income_needed_downpayment_0.20_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
        "Metro_new_renter_income_needed_uc_sfrcondomfr_sm_sa_month.csv",
        "Metro_new_renter_affordability_uc_sfrcondomfr_sm_sa_month.csv",
        "Metro_market_temp_index_uc_sfrcondo_month.csv"
    ]
    
    found_count = 0
    for filename in required_files:
        filepath = HOUSING_DIR / "Zillow" / filename
        if filepath.exists():
            found_count += 1
    
    print(f"   Found {found_count}/{len(required_files)} required files")
    if found_count == len(required_files):
        print(f"   ✓ All Zillow files present")
    else:
        print(f"   ⚠ Missing {len(required_files) - found_count} files")
        all_good = False
    
    # 6. Grocery Store Data
    print("\n6. Grocery Store Data (OpenStreetMap):")
    grocery_file = INFRASTRUCTURE_DIR / "grocery_stores_osm.csv"
    if grocery_file.exists() and grocery_file.stat().st_size > 1024:
        size_kb = grocery_file.stat().st_size / 1024
        df = pd.read_csv(grocery_file, nrows=100)
        print(f"   ✓ Found ({size_kb:.1f} KB)")
        print(f"   ✓ {len(df)} sample rows")
    else:
        print(f"   ✗ Missing or too small")
        print(f"   ⚠ Run collect_data.py to collect via API")
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✓ ALL DATA SOURCES VERIFIED")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python scripts/process_data.py")
    else:
        print("⚠ SOME DATA SOURCES MISSING")
        print("=" * 60)
        print("\nRun data collection:")
        print("  python scripts/collect_data.py")
        print("\nOr check manual download instructions above.")
    print("=" * 60)

if __name__ == "__main__":
    verify_data_collection()

