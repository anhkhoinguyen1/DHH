# Quick Start Guide

## Setup

1. **Create and activate virtual environment:**
```bash
cd Food-Desert-Prediction
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Census API key
# Get a free API key at: https://api.census.gov/key.html
# Replace YOUR_API_KEY_HERE with your actual API key
```

**Note:** The `.env` file is gitignored for security. Each user should create their own `.env` file with their API key.

## Data Collection

1. **Download all data sources:**
```bash
python scripts/collect_data.py
```

This will download and organize data into 5 main folders:
- `data/01_census_demographics/` - Census ACS data (via API)
- `data/02_food_access/` - USDA Food Access Research Atlas (2019, 2015)
- `data/03_health_outcomes/` - CDC PLACES health outcomes data
- `data/04_housing_economics/` - Zillow housing data (requires manual download)
- `data/05_infrastructure_transportation/` - Grocery stores (OpenStreetMap) and GTFS transit feeds

**Note:** 
- Census API requires a free API key (set in `.env` file or provided in script)
- Zillow data must be downloaded manually from https://www.zillow.com/research/data/
  - Place files in `data/04_housing_economics/Zillow/`
- See `data/README.md` for detailed data structure and `docs/sources.txt` for data source details

2. **Verify data collection:**
```bash
python scripts/verify_data_collection.py
```

## Data Processing

Process the raw data into features for modeling:
```bash
python scripts/process_data.py
```

This will:
- Load and clean Food Access Atlas data
- Calculate change features (if historical data available)
- Create target variable (tracts that became low-access)
- Engineer features for modeling

Outputs:
- `data/processed/food_access_2019.csv` - Processed Food Access Atlas 2019
- `data/processed/food_access_2015.csv` - Processed Food Access Atlas 2015 (if available)
- `data/processed/change_features_2015_2019.csv` - Change metrics between years (if available)
- `data/processed/target_variable.csv` - Target variable for model training (if available)
- `data/processed/data_coverage_map.png` - **Geographic map** showing census tract data coverage across the contiguous United States (excluding Alaska and Hawaii). Tracts with data are plotted as points on a map of the US.
- `data/processed/data_coverage_summary.csv` - Summary statistics of data coverage by state
- `data/features/modeling_features.csv` - Final feature set for modeling

## Model Training

Train predictive models:
```bash
python scripts/train_model.py
```

This will:
- Train Logistic Regression (baseline)
- Train Random Forest
- Train XGBoost
- Compare models and select best one
- Save models to `models/` directory

Outputs:
- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/xgboost.pkl`
- `models/best_model_name.pkl`
- `models/feature_names.pkl`
- `models/model_comparison.csv`

## Generate Predictions

Generate predictions for all tracts:
```bash
python scripts/generate_predictions.py
```

This will:
- Load the best trained model
- Generate predictions for all tracts
- Classify risk levels
- Save predictions to `data/predictions/predictions.csv`

Outputs:
- `data/predictions/predictions.csv`
- `data/predictions/prediction_summary.csv`

## Generate Top 1000 Highest Risk Tracts

Generate a prioritized list of the top 1000 highest-risk census tracts:
```bash
python scripts/generate_top1000.py
```

This will:
- Load predictions for all tracts
- Sort by risk probability (highest first)
- Extract top 1000 tracts
- Create formatted output files

Outputs:
- `data/predictions/top1000_highest_risk_tracts.csv` - **Final output CSV** with columns:
  - `tract_id`: 11-digit census tract identifier
  - `lat`, `lon`: Geographic coordinates
  - `risk_probability`: Predicted probability of becoming food desert (0-1)
  - `demand_mean`, `demand_std`: Estimated weekly grocery shopping demand
  - `svi_score`: Social Vulnerability Index (0-1, higher = more vulnerable)
- `data/predictions/top1000_highest_risk_tracts.txt` - Human-readable text format
- `data/predictions/top1000_highest_risk_tracts.md` - Markdown format with tables

## Start API

**Important**: Make sure your virtual environment is activated first!

Start the FastAPI server:
```bash
# Make sure you're in the project root and venv is activated
source venv/bin/activate  # If not already activated
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or from the project root:
```bash
source venv/bin/activate  # If not already activated
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Note**: If you get `ModuleNotFoundError: No module named 'fastapi'`, make sure:
1. Virtual environment is activated (`source venv/bin/activate`)
2. Dependencies are installed (`pip install -r requirements.txt`)

API will be available at: `http://localhost:8000`

**API Endpoints:**
- `GET /` - API information
- `GET /tract/{tract_id}` - Get prediction for specific tract
- `GET /state/{state}` - Get all predictions for a state
- `GET /county/{state}/{county}` - Get all predictions for a county
- `GET /summary` - Get summary statistics
- `GET /search` - Search tracts with filters
- `GET /health` - Health check

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Start Frontend

Start the Flask frontend:
```bash
cd frontend
python app.py
```

Frontend will be available at: `http://localhost:5000`

**Note:** Make sure the API is running first!

## Workflow Summary

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Collect data (or copy existing data to data/raw/)
python scripts/collect_data.py

# 3. Process data
python scripts/process_data.py

# 4. Train models
python scripts/train_model.py

# 5. Generate predictions
python scripts/generate_predictions.py

# 6. Generate top 1000 highest risk tracts (optional but recommended)
python scripts/generate_top1000.py

# 7. Start API (in one terminal)
cd api && uvicorn main:app --reload

# 8. Start Frontend (in another terminal)
cd frontend && python app.py
```

## Troubleshooting

### "File not found" errors
- Make sure you've run the data collection and processing scripts first
- Check that files are in the correct directories

### API connection errors
- Make sure the API is running on port 8000
- Check that predictions have been generated
- Verify `data/predictions/predictions.csv` exists

### Model training errors
- Ensure you have processed data with features
- Check that target variable exists (may need historical data)
- If no target variable, the script will create a dummy one for demonstration

### Missing dependencies
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt` again

## Next Steps

1. **Collect additional data:**
   - Census ACS data (requires API key)
   - Grocery store locations (Yelp/Google Places API)
   - GTFS transit feeds for your target area
   - Zillow housing data

2. **Improve model:**
   - Add more features
   - Tune hyperparameters
   - Try different algorithms
   - Add temporal features

3. **Validate results:**
   - Use the frontend to explore predictions
   - Compare with known food deserts
   - Get feedback from domain experts

4. **Deploy:**
   - Set up production API server
   - Deploy frontend to web server
   - Set up automated data updates

## Data Sources

See `data/README.md` for detailed data directory structure and organization.

See `docs/sources.txt` for complete documentation of all data sources, APIs, and download instructions.

## Strategy

See `docs/strategy.md` for detailed strategy on data weighting and probability calculation.

