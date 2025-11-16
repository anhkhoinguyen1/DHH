"""
Full Pipeline Script for Food Desert Prediction

This script runs the complete pipeline:
1. Data collection (if needed)
2. Data processing
3. Model training (if model doesn't exist)
4. Prediction generation
5. Top 1000 highest risk tracts CSV output

The final output is: data/predictions/top1000_highest_risk_tracts.csv
"""

import sys
import subprocess
from pathlib import Path
import pandas as pd

# Set up paths
BASE_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
MODELS_DIR = BASE_DIR / "models"
FEATURES_DIR = BASE_DIR / "data" / "features"

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "=" * 60)
    print(f"{description}")
    print("=" * 60)
    
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"⚠ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=False,
            text=True
        )
        if result.returncode != 0:
            print(f"⚠ {description} failed with exit code {result.returncode}")
            return False
        return True
    except Exception as e:
        print(f"⚠ Error running {description}: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a required file exists."""
    if filepath.exists():
        print(f"✓ {description} exists")
        return True
    else:
        print(f"⚠ {description} not found: {filepath}")
        return False

def main():
    """Run the full pipeline."""
    print("=" * 60)
    print("Food Desert Prediction - Full Pipeline")
    print("=" * 60)
    print("\nThis will run the complete pipeline to generate:")
    print("  → data/predictions/top1000_highest_risk_tracts.csv")
    print("\nPipeline steps:")
    print("  1. Data collection (if needed)")
    print("  2. Data processing")
    print("  3. Model training (if model doesn't exist)")
    print("  4. Prediction generation")
    print("  5. Top 100 CSV generation")
    print("=" * 60)
    
    # Step 1: Data Collection (optional - skip if data already exists)
    print("\n[Step 1/5] Checking data collection...")
    food_access_file = BASE_DIR / "data" / "raw" / "FoodAccessResearchAtlasData2019.xlsx"
    if not food_access_file.exists():
        print("  Food Access data not found. Running data collection...")
        if not run_script("collect_data.py", "Data Collection"):
            print("  ⚠ Data collection had issues, but continuing...")
    else:
        print("  ✓ Food Access data already exists, skipping collection")
    
    # Step 2: Data Processing
    print("\n[Step 2/5] Processing data...")
    if not run_script("process_data.py", "Data Processing"):
        print("  ❌ Data processing failed. Cannot continue.")
        return False
    
    # Check if features were created
    features_file = FEATURES_DIR / "modeling_features.csv"
    if not check_file_exists(features_file, "Features file"):
        print("  ❌ Features file not created. Cannot continue.")
        return False
    
    # Step 3: Model Training (if model doesn't exist)
    print("\n[Step 3/5] Checking model...")
    model_file = MODELS_DIR / "random_forest.pkl"
    if not model_file.exists():
        print("  Model not found. Training model...")
        if not run_script("train_model.py", "Model Training"):
            print("  ⚠ Model training had issues, but continuing...")
    else:
        print("  ✓ Model already exists, skipping training")
    
    # Check if model exists (try different model types)
    model_exists = any([
        (MODELS_DIR / "random_forest.pkl").exists(),
        (MODELS_DIR / "xgboost.pkl").exists(),
        (MODELS_DIR / "logistic_regression.pkl").exists()
    ])
    
    if not model_exists:
        print("  ❌ No trained model found. Cannot generate predictions.")
        print("  Please run: python scripts/train_model.py")
        return False
    
    # Step 4: Generate Predictions
    print("\n[Step 4/5] Generating predictions...")
    if not run_script("generate_predictions.py", "Prediction Generation"):
        print("  ❌ Prediction generation failed. Cannot continue.")
        return False
    
    # Check if predictions were created
    predictions_file = PREDICTIONS_DIR / "predictions.csv"
    if not check_file_exists(predictions_file, "Predictions file"):
        print("  ❌ Predictions file not created. Cannot continue.")
        return False
    
    # Step 5: Generate Top 1000 CSV
    print("\n[Step 5/5] Generating Top 1000 highest risk tracts...")
    if not run_script("generate_top1000.py", "Top 1000 Generation"):
        print("  ❌ Top 1000 generation failed.")
        return False
    
    # Verify final output
    top1000_file = PREDICTIONS_DIR / "top1000_highest_risk_tracts.csv"
    if check_file_exists(top1000_file, "Top 1000 CSV file"):
        print("\n" + "=" * 60)
        print("✓ PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\n📊 Final Output:")
        print(f"   {top1000_file}")
        
        # Show summary
        try:
            df = pd.read_csv(top1000_file)
            print(f"\n📈 Summary:")
            print(f"   Total tracts in top 1000: {len(df)}")
            
            # Check if risk_probability column exists (new format) or probability_pct (old format)
            if 'risk_probability' in df.columns:
                risk_col = 'risk_probability'
                # Convert to percentage for display
                risk_min = df[risk_col].min() * 100
                risk_max = df[risk_col].max() * 100
                risk_mean = df[risk_col].mean() * 100
                print(f"   Risk probability range: {risk_min:.2f}% - {risk_max:.2f}%")
                print(f"   Average risk probability: {risk_mean:.2f}%")
            elif 'probability_pct' in df.columns:
                print(f"   Risk probability range: {df['probability_pct'].min():.2f}% - {df['probability_pct'].max():.2f}%")
                print(f"   Average risk probability: {df['probability_pct'].mean():.2f}%")
            
            # Check for SVI score
            if 'svi_score' in df.columns:
                print(f"   SVI score range: {df['svi_score'].min():.2f} - {df['svi_score'].max():.2f}")
                print(f"   Average SVI score: {df['svi_score'].mean():.2f}")
            
            # Show top 5 tracts (if we have tract_id)
            if 'tract_id' in df.columns:
                print(f"\n   Top 5 highest risk tracts:")
                top5 = df.head(5)
                for idx, (_, row) in enumerate(top5.iterrows(), 1):
                    tract_id = str(row['tract_id']).zfill(11)
                    risk = row.get('risk_probability', row.get('probability_pct', 0))
                    if isinstance(risk, float) and risk <= 1.0:
                        risk_display = f"{risk * 100:.2f}%"
                    else:
                        risk_display = f"{risk:.2f}%"
                    print(f"     Rank {idx}: Tract {tract_id} (Risk: {risk_display})")
            elif 'CensusTract' in df.columns and 'State' in df.columns:
                # Old format with State/County info
                print(f"\n   Top 5 states:")
                for state, count in df['State'].value_counts().head(5).items():
                    print(f"     {state}: {count} tracts")
                
                print(f"\n   Top 5 highest risk tracts:")
                top5 = df.head(5)
                for _, row in top5.iterrows():
                    prob = row.get('probability_pct', row.get('risk_probability', 0) * 100)
                    print(f"     Rank {row['Rank']}: Tract {row['CensusTract']}, {row['County']}, {row['State']} ({prob:.2f}%)")
        except Exception as e:
            print(f"   (Could not load summary: {e})")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        return True
    else:
        print("  ❌ Top 100 CSV file was not created.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

