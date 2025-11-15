# Food Desert Prediction Project

## Overview

This project predicts which U.S. census tracts are at risk of becoming food deserts in the next decade. A food desert is defined as a low-income area where a significant number of residents have low access to grocery stores or supermarkets.

The program uses machine learning to analyze demographic shifts, socioeconomic changes, housing market trends, health outcomes, and infrastructure access to identify census tracts most likely to experience food access challenges.

## Research Question

**Can we predict which U.S. census tracts are at risk of becoming food deserts in the next decade, based on demographic shifts, retailer patterns, transit access, and socioeconomic change?**

## How It Works

### 1. Data Collection
The program integrates multiple publicly available data sources:
- **Census ACS**: Demographic and socioeconomic data (income, poverty, education, housing)
- **USDA Food Access Atlas**: Current and historical food access conditions
- **CDC PLACES**: Health outcomes (diabetes, obesity, hypertension)
- **Zillow Housing Data**: Housing market trends and affordability metrics
- **OpenStreetMap**: Grocery store locations
- **GTFS Transit Feeds**: Public transportation access

### 2. Data Processing
- Cleans and standardizes data from all sources
- Merges datasets by census tract identifier
- Calculates change metrics between time periods (2015-2019)
- Engineers features for machine learning:
  - Food access indicators
  - Socioeconomic risk factors
  - Housing affordability pressures
  - Health outcome correlations
  - Infrastructure accessibility

### 3. Model Training
Trains multiple machine learning models:
- **Logistic Regression**: Baseline model
- **Random Forest**: Ensemble method for feature importance
- **XGBoost**: Gradient boosting for high accuracy

Models are evaluated and the best-performing model is selected for predictions.

### 4. Prediction Generation
- Generates risk probabilities for all ~72,000 U.S. census tracts
- Classifies tracts into risk categories (low, medium, high)
- Identifies top 100 highest-risk tracts

### 5. Output
The final output is a CSV file containing the **top 100 census tracts most likely to become food deserts**, including:
- Tract identifiers and geographic information
- Risk probability scores
- Key risk factors
- Geographic coordinates for mapping

## Project Structure

```
Food-Desert-Prediction/
├── data/
│   ├── 01_census_demographics/    # Census ACS data, TIGER shapefiles
│   ├── 02_food_access/            # USDA Food Access Atlas
│   ├── 03_health_outcomes/        # CDC PLACES health data
│   ├── 04_housing_economics/       # Zillow housing data
│   ├── 05_infrastructure_transportation/  # Grocery stores, GTFS
│   ├── processed/                  # Cleaned datasets
│   ├── features/                   # Engineered features
│   └── predictions/                # Model outputs
├── scripts/
│   ├── collect_data.py            # Download and collect all data sources
│   ├── process_data.py             # Process and merge data, engineer features
│   ├── train_model.py              # Train machine learning models
│   ├── generate_predictions.py     # Generate predictions for all tracts
│   ├── generate_top100.py          # Create top 100 highest-risk tracts CSV
│   ├── run_full_pipeline.py       # Run complete pipeline end-to-end
│   └── verify_data_collection.py   # Verify all data sources are collected
├── api/
│   └── main.py                     # FastAPI REST API for querying predictions
├── frontend/
│   └── app.py                      # Flask web interface
├── models/                         # Saved trained models
├── docs/
│   ├── sources.txt                 # Data sources and citations
│   └── strategy.md                 # Modeling strategy and methodology
├── QUICKSTART.md                   # Quick start guide and instructions
├── README.md                       # This file - program overview
└── requirements.txt                # Python dependencies
```

## Key Features

### Comprehensive Data Integration
- **5 major data categories**: Demographics, food access, health, housing, infrastructure
- **Multiple time periods**: Historical data (2015) and current data (2019) for trend analysis
- **Geographic coverage**: All U.S. census tracts (~72,000 tracts)

### Machine Learning Pipeline
- **Feature engineering**: 50+ features from multiple data sources
- **Multiple algorithms**: Logistic Regression, Random Forest, XGBoost
- **Model evaluation**: Cross-validation and performance metrics
- **Feature importance**: Identifies key risk factors

### Actionable Output
- **Top 100 highest-risk tracts**: Prioritized list for intervention
- **Risk probabilities**: Quantified likelihood for each tract
- **Geographic coordinates**: Ready for mapping and visualization
- **Risk factors**: Identifies specific vulnerabilities per tract

## Methodology

### Target Variable
The model predicts whether a census tract will become a "low-income and low-access" (LILA) tract, defined as:
- **Low-income**: Poverty rate ≥ 20% OR median family income ≤ 80% of state/metro median
- **Low-access**: Significant number of residents live >1 mile (urban) or >10 miles (rural) from nearest grocery store

### Feature Categories
1. **Food Access Status** (25% weight): Current low-access indicators, distance metrics
2. **Socioeconomic Factors** (30% weight): Income, poverty, education, employment
3. **Demographics** (15% weight): Age, race/ethnicity, household composition
4. **Housing & Economics** (20% weight): Rent burden, housing costs, market volatility
5. **Infrastructure** (10% weight): Grocery store density, transit access

### Model Training
- Uses historical data (2015-2019) to train on actual transitions
- 2,519 tracts became low-access between 2015-2019 (positive class)
- Balanced training with appropriate class weights
- Cross-validation for robust performance estimation

## Use Cases

1. **Policy Planning**: Identify areas needing food access interventions
2. **Resource Allocation**: Prioritize funding for food assistance programs
3. **Research**: Study factors contributing to food desert formation
4. **Advocacy**: Support community food access initiatives
5. **Urban Planning**: Inform grocery store siting decisions

## Limitations

- Predictions are based on historical patterns and may not account for:
  - Future policy changes
  - Economic shocks or disasters
  - Rapid demographic shifts
  - New infrastructure development
- Data availability varies by geographic area
- Some features are aggregated at state/metro level (e.g., Zillow data)
- Model performance depends on quality and completeness of input data

## Technical Requirements

- Python 3.8+
- 8GB+ RAM recommended (for processing large datasets)
- Internet connection for data collection
- Census API key (free, available at https://api.census.gov/key.html)

## Getting Started

See **QUICKSTART.md** for detailed setup and usage instructions.

## Data Sources

See **docs/sources.txt** for complete documentation of all data sources, including:
- Source URLs and APIs
- Data collection methods
- Update frequencies
- Licensing information
- Citation requirements

## Citation

If you use this project or its predictions, please cite the data sources as documented in `docs/sources.txt`.

## License

This project uses publicly available data sources. All data sources are either public domain or have permissive licenses. See `docs/sources.txt` for specific licensing information for each data source.

---

**Last Updated**: 2025-01-27
