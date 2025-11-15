"""
Generate Predictions Script for Food Desert Prediction Project

Generates predictions for all tracts using the trained model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up paths
BASE_DIR = Path(__file__).parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_model():
    """Load the best trained model."""
    print("Loading model...")
    
    # Check which model was selected as best
    best_model_name_file = MODELS_DIR / "best_model_name.pkl"
    if best_model_name_file.exists():
        best_model_name = joblib.load(best_model_name_file)
        print(f"  Using best model: {best_model_name}")
    else:
        # Default to Random Forest if available
        best_model_name = "random_forest"
        print(f"  Using default model: {best_model_name}")
    
    # Load model
    model_file = MODELS_DIR / f"{best_model_name.replace(' ', '_').lower()}.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}\nRun train_model.py first!")
    
    model = joblib.load(model_file)
    
    # Load scaler if it's logistic regression
    scaler = None
    if best_model_name.lower() == "logistic regression":
        scaler_file = MODELS_DIR / "logistic_scaler.pkl"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
    
    # Load feature names
    feature_names_file = MODELS_DIR / "feature_names.pkl"
    if feature_names_file.exists():
        feature_names = joblib.load(feature_names_file)
    else:
        feature_names = None
    
    return model, scaler, feature_names, best_model_name

def load_features():
    """Load features for prediction."""
    print("Loading features...")
    filepath = FEATURES_DIR / "modeling_features.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Features file not found: {filepath}\nRun process_data.py first!")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} tracts")
    return df

def prepare_features(df, feature_names):
    """Prepare features in the same format as training."""
    print("Preparing features...")
    
    # Identify feature columns
    exclude_cols = ['CensusTract', 'State', 'County', 'target']
    available_feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[available_feature_cols].copy()
    
    # Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    
    # Convert categorical to numeric
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Align with training features
    if feature_names:
        # Add missing columns (fill with 0)
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Select only training features in correct order
        X = X[feature_names]
    
    return X

def classify_risk(probability):
    """Classify risk level based on probability."""
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Moderate Risk"
    elif probability >= 0.20:
        return "Low Risk (Emerging)"
    else:
        return "Very Low Risk"

def main():
    """Main prediction function."""
    print("=" * 60)
    print("Food Desert Prediction - Generate Predictions")
    print("=" * 60)
    
    # Load model
    model, scaler, feature_names, model_name = load_model()
    
    # Load features
    df = load_features()
    identifiers = df[['CensusTract', 'State', 'County']].copy()
    
    # Prepare features
    X = prepare_features(df, feature_names)
    
    # Scale if needed
    if scaler is not None:
        print("Scaling features...")
        X = scaler.transform(X)
    
    # Generate predictions
    print("Generating predictions...")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    
    # Create results dataframe
    results = identifiers.copy()
    results['probability'] = probabilities
    results['prediction'] = predictions
    results['risk_level'] = results['probability'].apply(classify_risk)
    
    # Add current status if available
    if 'LILATracts_1And10' in df.columns:
        results['currently_low_access'] = df['LILATracts_1And10'].values
    
    # Sort by probability (highest risk first)
    results = results.sort_values('probability', ascending=False)
    
    # Save results
    output_file = OUTPUT_DIR / "predictions.csv"
    results.to_csv(output_file, index=False)
    print(f"✓ Saved predictions to: {output_file}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("Prediction Summary")
    print("="*60)
    print(f"Total tracts: {len(results)}")
    print(f"\nRisk Level Distribution:")
    print(results['risk_level'].value_counts().to_string())
    print(f"\nTop 10 Highest Risk Tracts:")
    print(results.head(10)[['CensusTract', 'State', 'County', 'probability', 'risk_level']].to_string(index=False))
    
    # Save summary
    summary = {
        'total_tracts': len(results),
        'high_risk_count': (results['risk_level'] == 'High Risk').sum(),
        'moderate_risk_count': (results['risk_level'] == 'Moderate Risk').sum(),
        'low_risk_count': (results['risk_level'] == 'Low Risk (Emerging)').sum(),
        'very_low_risk_count': (results['risk_level'] == 'Very Low Risk').sum(),
        'mean_probability': results['probability'].mean(),
        'median_probability': results['probability'].median(),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(OUTPUT_DIR / "prediction_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("Prediction generation complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()

