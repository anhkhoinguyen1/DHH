"""
Model Training Script for Food Desert Prediction Project

Trains predictive models to identify tracts at risk of becoming food deserts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    classification_report, confusion_matrix, f1_score
)
import xgboost as xgb
import lightgbm as lgb

# Set up paths
BASE_DIR = Path(__file__).parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    """Load processed features."""
    print("Loading features...")
    filepath = FEATURES_DIR / "modeling_features.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Features file not found: {filepath}\nRun process_data.py first!")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} tracts with {len(df.columns)} features")
    return df

def prepare_data(df):
    """Prepare data for modeling."""
    print("Preparing data...")
    
    # Identify feature columns (exclude identifiers and target)
    exclude_cols = ['CensusTract', 'State', 'County', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Check if target exists
    if 'target' not in df.columns:
        print("⚠ Warning: 'target' column not found. Creating dummy target for demonstration.")
        # Create a dummy target based on LILATracts_1And10 if available
        if 'LILATracts_1And10' in df.columns:
            df['target'] = df['LILATracts_1And10']
        else:
            df['target'] = 0
    
    # Remove rows with missing target
    df = df.dropna(subset=['target'])
    
    # Select features and handle missing values
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # Fill missing values with median for numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    # Fill missing values with mode for categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    
    # Convert categorical to numeric if needed
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"✓ Prepared {X.shape[0]} samples with {X.shape[1]} features")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, df[['CensusTract', 'State', 'County']]

def train_baseline_logistic(X_train, y_train, X_test, y_test):
    """Train baseline logistic regression model."""
    print("\n" + "="*60)
    print("Training Baseline: Logistic Regression")
    print("="*60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(scaler, MODELS_DIR / "logistic_scaler.pkl")
    
    return model, scaler, auc, f1

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest model."""
    print("\n" + "="*60)
    print("Training: Random Forest")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model and feature importance
    joblib.dump(model, MODELS_DIR / "random_forest.pkl")
    feature_importance.to_csv(MODELS_DIR / "random_forest_feature_importance.csv", index=False)
    
    return model, auc, f1, feature_importance

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    print("\n" + "="*60)
    print("Training: XGBoost")
    print("="*60)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) if (y_train==1).sum() > 0 else 1,
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model and feature importance
    joblib.dump(model, MODELS_DIR / "xgboost.pkl")
    feature_importance.to_csv(MODELS_DIR / "xgboost_feature_importance.csv", index=False)
    
    return model, auc, f1, feature_importance

def main():
    """Main training function."""
    print("=" * 60)
    print("Food Desert Prediction - Model Training")
    print("=" * 60)
    
    # Load data
    df = load_features()
    X, y, identifiers = prepare_data(df)
    
    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    
    # Train models
    results = {}
    
    # Baseline: Logistic Regression
    try:
        lr_model, lr_scaler, lr_auc, lr_f1 = train_baseline_logistic(X_train, y_train, X_test, y_test)
        results['Logistic Regression'] = {'AUC': lr_auc, 'F1': lr_f1}
    except Exception as e:
        print(f"⚠ Error training Logistic Regression: {e}")
    
    # Random Forest
    try:
        rf_model, rf_auc, rf_f1, rf_importance = train_random_forest(X_train, y_train, X_test, y_test)
        results['Random Forest'] = {'AUC': rf_auc, 'F1': rf_f1}
    except Exception as e:
        print(f"⚠ Error training Random Forest: {e}")
    
    # XGBoost
    try:
        xgb_model, xgb_auc, xgb_f1, xgb_importance = train_xgboost(X_train, y_train, X_test, y_test)
        results['XGBoost'] = {'AUC': xgb_auc, 'F1': xgb_f1}
    except Exception as e:
        print(f"⚠ Error training XGBoost: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    
    # Save summary
    summary_df.to_csv(MODELS_DIR / "model_comparison.csv")
    
    # Select best model (by AUC)
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['AUC'])
        print(f"\n✓ Best model: {best_model_name} (AUC: {results[best_model_name]['AUC']:.4f})")
        joblib.dump(best_model_name, MODELS_DIR / "best_model_name.pkl")
    
    print("\n" + "="*60)
    print("Model training complete!")
    print(f"Models saved to: {MODELS_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()

