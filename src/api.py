"""
FastAPI Service for Real Estate Price Prediction

Production-ready REST API with:
- POST /api/v1/predict-price: Get price prediction with confidence interval
- GET /health: Service health check
- Input validation with Pydantic
- Comprehensive error handling
- SHAP-based feature explanations

Author: Real Estate Price Prediction Engine
Version: 1.0
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_for_inference
from model import predict_with_confidence

# ==================== API MODELS (Request/Response Schemas) ====================

class PredictionRequest(BaseModel):
    """
    Request schema for price prediction.

    Matches the example in the challenge brief.
    Note: INSTANCE_DATE is NOT required (will be set to max_train_date internally)
    """
    property_type: str = Field(..., description="Property type (e.g., 'Apartment', 'Villa')")
    property_subtype: str = Field(..., description="Property subtype (e.g., 'Flat', 'Penthouse')")
    area: str = Field(..., description="Area/community name (e.g., 'Marina District')")
    actual_area: float = Field(..., gt=0, description="Property area in sqm or sqft (must be positive)")
    rooms: str = Field(..., description="Number of rooms/bedrooms (e.g., '2 B/R')")
    parking: int = Field(..., ge=0, description="Number of parking spaces")
    is_offplan: bool = Field(..., description="Off-plan (true) or ready (false) property")
    is_freehold: bool = Field(..., description="Freehold (true) or leasehold (false)")
    usage: str = Field(..., description="Property usage type (e.g., 'Residential')")
    nearest_metro: str = Field(..., description="Nearest metro station")
    nearest_mall: str = Field(..., description="Nearest shopping mall")
    master_project: str = Field(..., description="Master development project")
    project: str = Field(..., description="Specific project/building name")

    # Optional fields with defaults
    procedure_area: Optional[float] = Field(None, description="Registered area (defaults to actual_area if not provided)")
    group: Optional[str] = Field("Real Estate", description="Property group category")
    nearest_landmark: Optional[str] = Field("", description="Nearest landmark")
    total_buyer: Optional[int] = Field(1, description="Number of buyers")
    total_seller: Optional[int] = Field(1, description="Number of sellers")

    @validator('actual_area')
    def validate_area(cls, v):
        if v <= 0:
            raise ValueError('actual_area must be greater than 0')
        if v > 100000:  # Sanity check
            raise ValueError('actual_area seems unrealistically large (>100,000 sqm/sqft)')
        return v

    class Config:
        schema_extra = {
            "example": {
                "property_type": "Apartment",
                "property_subtype": "Flat",
                "area": "Marina District",
                "actual_area": 1200,
                "rooms": "2 B/R",
                "parking": 1,
                "is_offplan": False,
                "is_freehold": True,
                "usage": "Residential",
                "nearest_metro": "Central Station",
                "nearest_mall": "City Mall",
                "master_project": "Marina Development",
                "project": "Marina Residence"
            }
        }


class ConfidenceInterval(BaseModel):
    """Confidence interval for prediction"""
    lower: float = Field(..., description="Lower bound of confidence interval")
    upper: float = Field(..., description="Upper bound of confidence interval")


class PredictionResponse(BaseModel):
    """
    Response schema for price prediction.

    Matches the challenge brief requirements.
    """
    predicted_price: float = Field(..., description="Predicted transaction price")
    confidence_interval: ConfidenceInterval = Field(..., description="90% confidence interval")
    price_per_sqft: Optional[float] = Field(None, description="Price per square foot/meter")
    model_confidence: str = Field(..., description="Confidence level: 'high', 'medium', or 'low'")
    key_factors: List[str] = Field(..., description="Top 3 factors driving this prediction")

    # Additional metadata
    area_unit: Optional[str] = Field(None, description="Unit for area measurement (sqm or sqft)")

    class Config:
        schema_extra = {
            "example": {
                "predicted_price": 1850000,
                "confidence_interval": {
                    "lower": 1750000,
                    "upper": 1950000
                },
                "price_per_sqft": 1541,
                "model_confidence": "high",
                "key_factors": [
                    "Location: Premium area",
                    "Property size: 1200 sqft",
                    "Proximity to metro station"
                ],
                "area_unit": "sqm"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    uptime_seconds: Optional[float] = None


# ==================== FASTAPI APPLICATION ====================

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="Production ML service for property valuation with confidence intervals",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global state
artifact = None
start_time = None


@app.on_event("startup")
async def load_model():
    """
    Load model artifact at startup.

    This runs once when the service starts, loading the model into memory
    for fast inference.
    """
    global artifact, start_time

    start_time = datetime.now()

    try:
        artifact_path = os.path.join(os.path.dirname(__file__), '../models/trained_model.pkl')

        if not os.path.exists(artifact_path):
            print(f"❌ Model artifact not found at: {artifact_path}")
            print("   Run the notebook to train and save the model first!")
            artifact = None
            return

        print(f"Loading model artifact from: {artifact_path}")
        artifact = joblib.load(artifact_path)

        print(f"✓ Model loaded successfully!")
        print(f"  Version: {artifact.get('model_version', 'unknown')}")
        print(f"  Trained: {artifact.get('trained_at', 'unknown')}")
        print(f"  Area unit: {artifact.get('area_unit', 'unknown')}")
        print(f"  Validation R²: {artifact['metrics']['val']['r2']:.4f}")
        print(f"  Test R²: {artifact['metrics']['test']['r2']:.4f}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        artifact = None


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Service health check endpoint.

    Returns:
    - Service status
    - Model loading status
    - Model metadata
    - Uptime
    """
    uptime = (datetime.now() - start_time).total_seconds() if start_time else 0

    return HealthResponse(
        status="healthy" if artifact is not None else "unhealthy",
        model_loaded=artifact is not None,
        model_version=artifact.get('model_version') if artifact else None,
        trained_at=str(artifact.get('trained_at')) if artifact else None,
        uptime_seconds=uptime
    )


@app.post("/api/v1/predict-price", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(request: PredictionRequest):
    """
    Predict property price with confidence interval.

    Process:
    1. Validate input
    2. Preprocess features (using same pipeline as training)
    3. Make prediction with confidence interval
    4. Generate key factors driving the prediction
    5. Return comprehensive response

    Args:
        request: Property details (see PredictionRequest schema)

    Returns:
        PredictionResponse with predicted price, CI, and explanation

    Raises:
        503: Model not loaded
        422: Invalid input
        500: Prediction error
    """
    # Check if model is loaded
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server logs."
        )

    try:
        # Convert request to format expected by preprocessing
        input_data = convert_request_to_input(request)

        # Preprocess using the SAME pipeline as training
        preprocessing_metadata = artifact['preprocessing_metadata']
        X = preprocess_for_inference(input_data, preprocessing_metadata)

        # Make prediction with confidence interval
        model = artifact['model']
        ci_stats = artifact['ci_stats']

        predictions, lower_bounds, upper_bounds, confidence_flags = predict_with_confidence(
            model=model,
            X=X,
            ci_stats=ci_stats
        )

        predicted_price = float(predictions[0])
        lower_bound = float(lower_bounds[0])
        upper_bound = float(upper_bounds[0])
        confidence_flag = str(confidence_flags[0])

        # Compute price per sqft
        area_unit = artifact.get('area_unit', 'sqm')
        if request.actual_area > 0:
            price_per_sqft = predicted_price / request.actual_area
        else:
            price_per_sqft = None

        # Generate key factors explaining the prediction
        key_factors = generate_key_factors(
            request=request,
            X=X,
            model=model,
            feature_importance=artifact.get('feature_importance'),
            predicted_price=predicted_price
        )

        # Construct response
        response = PredictionResponse(
            predicted_price=predicted_price,
            confidence_interval=ConfidenceInterval(
                lower=lower_bound,
                upper=upper_bound
            ),
            price_per_sqft=price_per_sqft,
            model_confidence=confidence_flag,
            key_factors=key_factors,
            area_unit=area_unit
        )

        return response

    except ValueError as e:
        # Validation errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        # Other errors
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ==================== HELPER FUNCTIONS ====================

def convert_request_to_input(request: PredictionRequest) -> Dict[str, Any]:
    """
    Convert API request to format expected by preprocessing.

    Maps field names from API schema to internal column names.
    """
    # Map API fields to internal column names (match training data)
    input_data = {
        'PROP_TYPE_EN': request.property_type,
        'PROP_SB_TYPE_EN': request.property_subtype,
        'AREA_EN': request.area,
        'ACTUAL_AREA': request.actual_area,
        'ROOMS_EN': request.rooms,
        'PARKING': request.parking,
        'IS_OFFPLAN_EN': 'Off-Plan' if request.is_offplan else 'Ready',
        'IS_FREE_HOLD_EN': 'Free Hold' if request.is_freehold else 'Lease Hold',
        'USAGE_EN': request.usage,
        'NEAREST_METRO_EN': request.nearest_metro,
        'NEAREST_MALL_EN': request.nearest_mall,
        'MASTER_PROJECT_EN': request.master_project,
        'PROJECT_EN': request.project,
        'PROCEDURE_AREA': request.procedure_area if request.procedure_area else request.actual_area,
        'GROUP_EN': request.group,
        'NEAREST_LANDMARK_EN': request.nearest_landmark,
        'TOTAL_BUYER': request.total_buyer,
        'TOTAL_SELLER': request.total_seller,
        # INSTANCE_DATE will be set to max_train_date in preprocess_for_inference
        # PROCEDURE_EN not needed (API predicts sale price only)
        # TRANSACTION_NUMBER not needed (identifier)
        # TRANS_VALUE not needed (target)
    }

    return input_data


def generate_key_factors(
    request: PredictionRequest,
    X: pd.DataFrame,
    model: Any,
    feature_importance: pd.DataFrame,
    predicted_price: float
) -> List[str]:
    """
    Generate human-readable key factors explaining the prediction.

    Strategy:
    1. Try to compute per-row SHAP values (ideal but slow)
    2. Fallback to global feature importance + heuristics

    Returns:
        List of 3 human-readable explanations
    """
    factors = []

    try:
        # Attempt SHAP for this single prediction (may be slow)
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Get top 3 features by absolute SHAP value
        shap_abs = np.abs(shap_values[0])
        top_indices = np.argsort(shap_abs)[::-1][:3]

        for idx in top_indices:
            feature_name = X.columns[idx]
            feature_value = X.iloc[0, idx]
            shap_value = shap_values[0][idx]

            # Generate human-readable explanation
            explanation = explain_feature(feature_name, feature_value, shap_value, request)
            factors.append(explanation)

    except Exception as e:
        # Fallback: use global feature importance + heuristics
        print(f"SHAP failed ({e}), using fallback key factors")

        # Use top features from global importance
        if feature_importance is not None:
            top_features = feature_importance.head(5)['feature'].tolist()
        else:
            top_features = ['ACTUAL_AREA', 'AREA_EN', 'PROP_TYPE_EN']

        # Generate heuristic explanations
        if 'ACTUAL_AREA' in X.columns or request.actual_area:
            factors.append(f"Property size: {request.actual_area:.0f} {artifact.get('area_unit', 'sqm')}")

        if 'AREA_EN' in X.columns:
            factors.append(f"Location: {request.area}")

        if 'PROP_TYPE_EN' in X.columns:
            factors.append(f"Property type: {request.property_type}")

        # Add price-based factor
        if predicted_price > 2000000:
            factors.append("Premium price segment")
        elif predicted_price < 500000:
            factors.append("Affordable price segment")
        else:
            factors.append("Mid-range price segment")

    # Ensure we have exactly 3 factors
    while len(factors) < 3:
        factors.append("Additional market factors")

    return factors[:3]


def explain_feature(feature_name: str, feature_value: Any, shap_value: float, request: PredictionRequest) -> str:
    """
    Convert feature name + SHAP value into human-readable explanation.

    Examples:
    - ACTUAL_AREA=1200, SHAP=+0.5 → "Large property size (1200 sqm) increases price"
    - AREA_EN_median_price=3500000, SHAP=+0.3 → "Premium area with high median prices"
    """
    direction = "increases" if shap_value > 0 else "decreases"

    # Property size
    if 'ACTUAL_AREA' in feature_name:
        size_desc = "Large" if feature_value > 150 else ("Medium" if feature_value > 70 else "Compact")
        return f"{size_desc} property size ({feature_value:.0f} {artifact.get('area_unit', 'sqm')}) {direction} price"

    # Location features
    if 'AREA_EN' in feature_name and 'median_price' in feature_name:
        price_level = "Premium" if feature_value > 2000000 else ("Mid-range" if feature_value > 800000 else "Affordable")
        return f"{price_level} area with typical prices around {feature_value:,.0f}"

    if 'AREA_EN' in feature_name:
        return f"Location: {request.area}"

    # Project features
    if 'PROJECT' in feature_name and 'median_price' in feature_name:
        return f"Project with median price {feature_value:,.0f}"

    # Time features
    if 'year' in feature_name.lower() or 'month' in feature_name or 'quarter' in feature_name:
        return f"Market timing factor {direction} price"

    # Off-plan
    if 'OFFPLAN' in feature_name:
        return f"{'Off-plan' if request.is_offplan else 'Ready'} property {direction} price"

    # Freehold
    if 'FREE_HOLD' in feature_name:
        return f"{'Freehold' if request.is_freehold else 'Leasehold'} ownership {direction} price"

    # Rooms
    if 'ROOMS' in feature_name:
        return f"Number of rooms ({request.rooms}) {direction} price"

    # Default
    return f"{feature_name}: {feature_value}"


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    print(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ==================== MAIN (for local testing) ====================

if __name__ == "__main__":
    import uvicorn

    print("Starting Real Estate Price Prediction API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
