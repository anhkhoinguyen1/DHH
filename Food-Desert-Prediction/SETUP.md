# Setup Instructions for New Users

This guide will help you set up the Food Desert Prediction project from scratch.

## Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Food-Desert-Prediction.git
cd Food-Desert-Prediction
```

## Step 2: Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Get Census API Key

1. Visit https://api.census.gov/key.html
2. Fill out the form to get a free API key
3. Copy your API key

## Step 4: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and replace YOUR_API_KEY_HERE with your actual API key
# You can use any text editor:
nano .env
# or
vim .env
# or
code .env  # if using VS Code
```

The `.env` file should look like:
```
CENSUS_API_KEY=your_actual_api_key_here
```

## Step 5: Collect Data

The project requires downloading data from multiple sources. Run:

```bash
python scripts/collect_data.py
```

This will:
- Download Food Access Atlas data (2019, 2015)
- Collect Census ACS data via API (requires API key)
- Download CDC PLACES health outcomes data
- Collect grocery store data from OpenStreetMap
- Download GTFS transit feeds

**Note:** Some data sources require manual download:
- **Zillow data**: Download from https://www.zillow.com/research/data/
  - Place files in `data/04_housing_economics/Zillow/`
  - Required files are listed in the collection script output

**Time:** Data collection may take 30-60 minutes depending on your internet connection.

## Step 6: Verify Data Collection

```bash
python scripts/verify_data_collection.py
```

This will check that all required data files are present.

## Step 7: Run the Pipeline

Once data is collected, run the full pipeline:

```bash
python scripts/run_full_pipeline.py
```

This will:
1. Process all data and engineer features
2. Train machine learning models
3. Generate predictions for all tracts
4. Create the top 100 highest-risk tracts CSV

**Output:** `data/predictions/top100_highest_risk_tracts.csv`

## Troubleshooting

### "ModuleNotFoundError" when running scripts
- Make sure virtual environment is activated: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### "API key not found" errors
- Check that `.env` file exists and contains `CENSUS_API_KEY=your_key`
- Verify the API key is correct at https://api.census.gov/data/key.html

### Data collection fails
- Check internet connection
- Some data sources may require manual download (instructions provided in script output)
- See `docs/sources.txt` for manual download links

### Out of memory errors
- Close other applications
- Process data in smaller batches (modify scripts if needed)
- Use a machine with more RAM (8GB+ recommended)

## Next Steps

- See `QUICKSTART.md` for detailed usage instructions
- See `README.md` for project overview
- See `docs/sources.txt` for data source documentation

