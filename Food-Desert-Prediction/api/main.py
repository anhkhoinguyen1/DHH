"""
FastAPI Application for Food Desert Prediction

Provides REST API endpoints for frontend to query predictions.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import joblib
from pydantic import BaseModel

# Set up paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
FEATURES_DIR = BASE_DIR / "data" / "features"

# Initialize FastAPI app
app = FastAPI(
    title="Food Desert Prediction API",
    description="API for querying food desert risk predictions for U.S. census tracts",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load predictions (cache in memory)
predictions_df = None
model = None
scaler = None
feature_names = None

def load_predictions():
    """Load predictions from CSV file."""
    global predictions_df
    if predictions_df is None:
        predictions_file = PREDICTIONS_DIR / "predictions.csv"
        if predictions_file.exists():
            predictions_df = pd.read_csv(predictions_file)
            # Convert CensusTract to string for consistent lookup
            predictions_df['CensusTract'] = predictions_df['CensusTract'].astype(str)
        else:
            raise FileNotFoundError("Predictions file not found. Run generate_predictions.py first!")
    return predictions_df

def load_model():
    """Load trained model for on-the-fly predictions."""
    global model, scaler, feature_names
    if model is None:
        # Load best model
        best_model_name_file = MODELS_DIR / "best_model_name.pkl"
        if best_model_name_file.exists():
            best_model_name = joblib.load(best_model_name_file)
        else:
            best_model_name = "random_forest"
        
        model_file = MODELS_DIR / f"{best_model_name.replace(' ', '_').lower()}.pkl"
        if model_file.exists():
            model = joblib.load(model_file)
        
        # Load scaler if needed
        scaler_file = MODELS_DIR / "logistic_scaler.pkl"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
        
        # Load feature names
        feature_names_file = MODELS_DIR / "feature_names.pkl"
        if feature_names_file.exists():
            feature_names = joblib.load(feature_names_file)
    
    return model, scaler, feature_names

# Pydantic models for request/response
class TractPrediction(BaseModel):
    CensusTract: str
    State: str
    County: str
    probability: float
    risk_level: str
    prediction: int
    currently_low_access: Optional[int] = None

class PredictionResponse(BaseModel):
    tract: TractPrediction
    message: str

class SummaryResponse(BaseModel):
    total_tracts: int
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    very_low_risk_count: int
    mean_probability: float
    median_probability: float

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Food Desert Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/tract/{tract_id}": "Get prediction for a specific census tract",
            "/state/{state}": "Get all predictions for a state",
            "/county/{state}/{county}": "Get all predictions for a county",
            "/summary": "Get summary statistics",
            "/search": "Search tracts by various criteria"
        }
    }

@app.get("/tract/{tract_id}", response_model=PredictionResponse)
async def get_tract_prediction(tract_id: str):
    """Get food desert risk prediction for a specific census tract."""
    try:
        df = load_predictions()
        tract_id_str = str(tract_id)
        
        result = df[df['CensusTract'] == tract_id_str]
        
        if result.empty:
            raise HTTPException(status_code=404, detail=f"Tract {tract_id} not found")
        
        tract_data = result.iloc[0].to_dict()
        
        return PredictionResponse(
            tract=TractPrediction(**tract_data),
            message="Success"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state/{state}")
async def get_state_predictions(
    state: str,
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    min_probability: Optional[float] = Query(None, ge=0, le=1, description="Minimum probability threshold"),
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Maximum number of results")
):
    """Get all predictions for a state, with optional filtering."""
    try:
        df = load_predictions()
        
        # Filter by state
        results = df[df['State'].str.upper() == state.upper()]
        
        # Apply filters
        if risk_level:
            results = results[results['risk_level'] == risk_level]
        
        if min_probability is not None:
            results = results[results['probability'] >= min_probability]
        
        # Sort by probability (highest risk first)
        results = results.sort_values('probability', ascending=False)
        
        # Limit results
        results = results.head(limit)
        
        return {
            "state": state,
            "count": len(results),
            "tracts": results.to_dict(orient='records')
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/county/{state}/{county}")
async def get_county_predictions(
    state: str,
    county: str,
    risk_level: Optional[str] = Query(None, description="Filter by risk level")
):
    """Get all predictions for a county."""
    try:
        df = load_predictions()
        
        results = df[
            (df['State'].str.upper() == state.upper()) &
            (df['County'].str.upper().str.contains(county.upper()))
        ]
        
        if risk_level:
            results = results[results['risk_level'] == risk_level]
        
        results = results.sort_values('probability', ascending=False)
        
        return {
            "state": state,
            "county": county,
            "count": len(results),
            "tracts": results.to_dict(orient='records')
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary", response_model=SummaryResponse)
async def get_summary():
    """Get summary statistics for all predictions."""
    try:
        df = load_predictions()
        
        return SummaryResponse(
            total_tracts=len(df),
            high_risk_count=(df['risk_level'] == 'High Risk').sum(),
            moderate_risk_count=(df['risk_level'] == 'Moderate Risk').sum(),
            low_risk_count=(df['risk_level'] == 'Low Risk (Emerging)').sum(),
            very_low_risk_count=(df['risk_level'] == 'Very Low Risk').sum(),
            mean_probability=float(df['probability'].mean()),
            median_probability=float(df['probability'].median())
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_tracts(
    state: Optional[str] = None,
    county: Optional[str] = None,
    risk_level: Optional[str] = None,
    min_probability: Optional[float] = Query(None, ge=0, le=1),
    max_probability: Optional[float] = Query(None, ge=0, le=1),
    limit: int = Query(100, ge=1, le=1000)
):
    """Search tracts with multiple criteria."""
    try:
        df = load_predictions()
        
        # Apply filters
        if state:
            df = df[df['State'].str.upper().str.contains(state.upper())]
        
        if county:
            df = df[df['County'].str.upper().str.contains(county.upper())]
        
        if risk_level:
            df = df[df['risk_level'] == risk_level]
        
        if min_probability is not None:
            df = df[df['probability'] >= min_probability]
        
        if max_probability is not None:
            df = df[df['probability'] <= max_probability]
        
        # Sort by probability
        df = df.sort_values('probability', ascending=False)
        
        # Limit results
        df = df.head(limit)
        
        return {
            "count": len(df),
            "tracts": df.to_dict(orient='records')
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        load_predictions()
        return {"status": "healthy", "predictions_loaded": True}
    except:
        return {"status": "unhealthy", "predictions_loaded": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

