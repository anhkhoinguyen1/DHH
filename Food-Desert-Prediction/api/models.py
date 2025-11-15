"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel
from typing import Optional

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

