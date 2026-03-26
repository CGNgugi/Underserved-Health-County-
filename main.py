"""
FastAPI Application for Kenya Healthcare Access Inequality API
=============================================================

This FastAPI application provides REST API endpoints for the 
Underserved County Score (UCS) analysis of Kenya's 47 counties.

Based on: UCS.ipynb - Integrating Machine Learning and Spatial Analytics 
to Predict and Map Healthcare Access Inequalities in Kenya

Author: Ngugi, Cynthia (138725)
Supervisor: Dr. Lorna Mutegi-Kamau
Institution: Strathmore University

Run with: uvicorn main:app --reload
"""

# ============================================================================
# IMPORTS
# ============================================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
import joblib
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & COLORS
# ============================================================================

C_RED    = "#d73027"
C_ORANGE = "#fc8d59"
C_YELLOW = "#ffffcc"
C_GREEN  = "#1a9850"
C_BLUE   = "#4575b4"
C_NAVY   = "#2c3e6b"
C_GREY   = "#f5f5f5"
C_PURPLE = "#7b2d8b"

DOMAIN_COLOURS = {
    "Healthcare Access Index":    C_RED,
    "Population Vulnerability Index": C_BLUE,
    "Immunization Coverage Index": C_GREEN,
    "Disease Burden Index":        C_PURPLE,
}

# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Kenya Healthcare Access Inequality API",
    description="""
    ## Overview
    This API provides access to the Underserved County Score (UCS) analysis 
    for all 47 Kenya counties. The UCS is a composite score that measures 
    healthcare access inequality across multiple dimensions.

    ## Domains
    - **Healthcare Access Index (HAI)**: ANC, skilled delivery, PNC, geo/financial barriers
    - **Population Vulnerability Index (PVI)**: Poverty, WASH deficits, demographic dependency
    - **Immunization Coverage Index (ICI)**: EPI antigens, full vaccination
    - **Disease Burden Index (DBI)**: Child nutrition, malaria/ARI/diarrhoea, anaemia

    ## Features
    - County-level UCS scores and rankings
    - Cluster analysis for county typologies
    - Anomaly detection for structurally underserved counties
    - ML model predictions for underserved county classification
    - SHAP explanations for model interpretability
    
    ## Models Available
    - xgboost (default)
    - xgboost_tuned
    - random_forest
    - gradient_boosting
    - logistic_regression
    
    ## Data Source
    KDHS 2020 & 2022 via DHS Programme API
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA PATHS
# ============================================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(DATA_DIR, "models")

# Data file paths
DATA_PATHS = {
    "ucs_scores": "county_ucs_final.csv",
    "clusters": "county_ucs_final.csv",
    "trends": "ucs_trend_2014_2022.csv",
    "typologies": "cluster_typologies.csv",
}

MODEL_PATHS = {
    "xgboost": "xgboost.pkl",
    "xgboost_tuned": "xgboost_tuned.pkl",
    "random_forest": "random_forest.pkl",
    "gradient_boosting": "gradient_boosting.pkl",
    "logistic_regression": "logistic_regression.pkl",
    "shap_explainer": "shap_explainer.pkl",
    "metadata": "model_metadata.json",
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_data() -> Dict[str, pd.DataFrame]:
    """Load all data files with multiple path search."""
    data = {}
    
    # Search paths for data files
    search_paths = [
        DATA_DIR,
        os.path.join(DATA_DIR, ".."),
        os.getcwd(),
    ]
    
    for key, filename in DATA_PATHS.items():
        found = False
        for base_path in search_paths:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                try:
                    data[key] = pd.read_csv(filepath, index_col=0)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if not found:
            # Also try the exact filename in current directory
            if os.path.exists(filename):
                try:
                    data[key] = pd.read_csv(filename, index_col=0)
                except:
                    data[key] = None
            else:
                data[key] = None
    
    return data


def load_models() -> Dict[Any, Any]:
    """Load all trained ML models from my_api/models with multiple path search."""
    models = {}
    
    # Search paths for model files
    search_paths = [
        MODEL_DIR,
        os.path.join(DATA_DIR, "models"),
        os.path.join(DATA_DIR, "..", "models"),
        os.path.join(os.getcwd(), "models"),
        "models",
    ]
    
    for key, filename in MODEL_PATHS.items():
        found = False
        for base_path in search_paths:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                try:
                    if filename.endswith('.pkl'):
                        models[key] = joblib.load(filepath)
                    elif filename.endswith('.json'):
                        with open(filepath, 'r') as f:
                            models[key] = json.load(f)
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {key} ({filename}): {e}")
        
        if not found:
            models[key] = None
    
    return models


# Load data and models at startup
DATA = load_data()
MODELS = load_models()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CountyBase(BaseModel):
    """Base model for county information."""
    county: str = Field(..., description="County name")
    ucs: float = Field(..., description="Underserved County Score (0-100)")


class CountyDetail(CountyBase):
    """Detailed county information with domain scores."""
    healthcare_access_index: Optional[float] = Field(None, description="Healthcare Access Index score")
    population_vulnerability_index: Optional[float] = Field(None, description="Population Vulnerability Index score")
    immunization_coverage_index: Optional[float] = Field(None, description="Immunization Coverage Index score")
    disease_burden_index: Optional[float] = Field(None, description="Disease Burden Index score")
    cluster: Optional[int] = Field(None, description="Cluster assignment")
    cluster_label: Optional[str] = Field(None, description="Cluster label")
    anomaly: Optional[str] = Field(None, description="Anomaly detection result")
    anomaly_prob: Optional[float] = Field(None, description="Anomaly probability")


class CountyRanking(BaseModel):
    """County ranking information."""
    rank: int = Field(..., description="Rank (1 = most underserved)")
    county: str = Field(..., description="County name")
    ucs: float = Field(..., description="Underserved County Score")
    cluster_label: Optional[str] = Field(None, description="Cluster label")
    anomaly: Optional[str] = Field(None, description="Anomaly status")


class ClusterInfo(BaseModel):
    """Cluster information."""
    cluster_id: int = Field(..., description="Cluster ID")
    label: str = Field(..., description="Cluster label/description")
    counties: List[str] = Field(..., description="Counties in this cluster")
    avg_ucs: float = Field(..., description="Average UCS for cluster")


class PredictionInput(BaseModel):
    """Input for ML prediction."""
    healthcare_access_index: float = Field(..., description="Healthcare Access Index")
    population_vulnerability_index: float = Field(..., description="Population Vulnerability Index")
    immunization_coverage_index: float = Field(..., description="Immunization Coverage Index")
    disease_burden_index: float = Field(..., description="Disease Burden Index")


class PredictionOutput(BaseModel):
    """ML prediction output."""
    prediction: str = Field(..., description="Prediction: 'Underserved' or 'Normal'")
    probability: float = Field(..., description="Probability of being underserved")
    model_used: str = Field(..., description="Model used for prediction")


class SHAPExplanation(BaseModel):
    """SHAP explanation for a county."""
    county: str = Field(..., description="County name")
    prediction: str = Field(..., description="Prediction result")
    shap_values: Dict[str, float] = Field(..., description="SHAP values for each feature")
    feature_values: Dict[str, float] = Field(..., description="Feature values")
    base_value: float = Field(..., description="Base value (expected value)")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/statistics", tags=["Statistics"])
async def get_statistics():
    """
    Get summary statistics for the UCS analysis.
    """
    df = DATA.get("ucs_scores")
    if df is None:
        raise HTTPException(status_code=404, detail="UCS scores data not found")
    
    # Calculate statistics
    stats = {
        "total_counties": len(df),
        "ucs_mean": float(df["UCS"] if "UCS" in df.columns else df["ucs"].mean()) if "ucs" in df.columns else None,
        "ucs_median": float(df["UCS"] if "UCS" in df.columns else df["ucs"].median()) if "ucs" in df.columns else None,
        "ucs_std": float(df["UCS"] if "UCS" in df.columns else df["ucs"].std()) if "ucs" in df.columns else None,
        "ucs_min": float(df["UCS"] if "UCS" in df.columns else df["ucs"].min()) if "ucs" in df.columns else None,
        "ucs_max": float(df["UCS"] if "UCS" in df.columns else df["ucs"].max()) if "ucs" in df.columns else None,
    }
    
    # Domain statistics
    domains = ["Healthcare Access Index", "Population Vulnerability Index", 
               "Immunization Coverage Index", "Disease Burden Index"]
    
    domain_stats = {}
    for domain in domains:
        if domain in df.columns:
            domain_stats[domain.replace(" Index", "")] = {
                "mean": float(df[domain].mean()),
                "median": float(df[domain].median()),
                "std": float(df[domain].std()),
                "min": float(df[domain].min()),
                "max": float(df[domain].max())
            }
    
    # Anomaly counts
    anomaly_counts = {}
    if "Anomaly" in df.columns:
        anomaly_counts = df["Anomaly"].value_counts().to_dict()
    
    # Cluster counts
    cluster_counts = {}
    if "Cluster" in df.columns:
        cluster_counts = df["Cluster"].value_counts().to_dict()
    
    return {
        "summary": stats,
        "domains": domain_stats,
        "anomaly_counts": anomaly_counts,
        "cluster_counts": cluster_counts
    }


@app.get("/data-quality", tags=["Data Quality"])
async def get_data_quality():
    """
    Get data quality report including outlier detection.
    
    - Outliers are flagged using Z-score > 3
    - Statistical outliers are retained as they represent genuine equity gaps
    - Values clipped to [0, 100] (physical bounds for percentages)
    """
    # Try to load raw pivot data
    pivot_path = os.path.join(DATA_DIR, "health_gap_raw.csv")
    if not os.path.exists(pivot_path):
        # Try alternative paths
        alt_paths = [
            os.path.join(DATA_DIR, "..", "health_gap_raw.csv"),
            "health_gap_raw.csv"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                pivot_path = p
                break
    
    try:
        # Load raw data
        df_raw = pd.read_csv(pivot_path)
        
        # Filter to county data
        data = df_raw[
            (df_raw['SurveyYearLabel'].isin([2020, 2022])) &
            (df_raw['CharacteristicLabel'].str.contains('Nairobi', na=False) |
             df_raw['CharacteristicLabel'].str.startswith('..', na=False))
        ].copy()
        data['County'] = data['CharacteristicLabel'].str.replace(r'^\.\.', '', regex=True).str.strip()
        
        # Pivot
        data_s = data.sort_values('SurveyYearLabel', ascending=True)
        pivot_df = data_s.pivot_table(index='County', columns='Indicator', 
                                      values='Value', aggfunc='last')
        
        # Step 1: Clip values to [0, 100]
        pivot_clipped = pivot_df.clip(lower=0, upper=100)
        clipped_count = (pivot_df != pivot_clipped).sum().sum()
        
        # Step 2: Flag statistical outliers (Z-score > 3)
        try:
            from scipy import stats
            z_scores = np.abs(stats.zscore(pivot_clipped.fillna(pivot_clipped.mean()), nan_policy='omit'))
            outlier_mask = z_scores > 3
            outlier_counts = outlier_mask.sum(axis=1)
            outlier_counties = outlier_counts[outlier_counts > 0].sort_values(ascending=False)
            
            # Get counties with outliers
            counties_with_outliers = {}
            for county in outlier_counties.index:
                outlier_indicators = pivot_clipped.loc[county][outlier_mask[outlier_counties.index.get_loc(county)]].to_dict()
                counties_with_outliers[county] = {
                    "count": int(outlier_counties[county]),
                    "indicators": {k: float(v) for k, v in outlier_indicators.items()}
                }
        except:
            counties_with_outliers = {}
            outlier_counties = pd.Series()
        
        # Step 3: Missingness summary
        missing_summary = pivot_clipped.isnull().sum()
        high_missing = missing_summary[missing_summary > 10].sort_values(ascending=False)
        total_missing = pivot_clipped.isnull().sum().sum()
        total_cells = pivot_clipped.size
        
        return {
            "data_cleaning": {
                "values_clipped_to_100": int(clipped_count),
                "clipping_note": "Values >100 or <0 clipped to physical bounds [0,100]"
            },
            "outlier_detection": {
                "method": "Z-score > 3",
                "counties_with_outliers": len(outlier_counties),
                "outlier_counties": counties_with_outliers,
                "note": "Statistical outliers retained as genuine equity signals"
            },
            "missingness": {
                "total_missing_cells": int(total_missing),
                "total_cells": int(total_cells),
                "missing_percentage": float(total_missing / total_cells * 100),
                "indicators_with_high_missing": {k: int(v) for k, v in high_missing.head(20).to_dict().items()}
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "note": "Raw data not available for data quality analysis"
        }


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "name": "Kenya Healthcare Access Inequality API",
        "version": "1.0.0",
        "description": "REST API for Underserved County Score (UCS) analysis of Kenya's 47 counties",
        "author": "Ngugi, Cynthia (138725)",
        "institution": "Strathmore University",
        "supervisor": "Dr. Lorna Mutegi-Kamau",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /statistics": "Summary statistics",
            "GET /counties": "Get all counties with UCS scores",
            "GET /counties/{county_name}": "Get county details",
            "GET /rankings": "County rankings",
            "GET /clusters": "Cluster analysis",
            "GET /anomalies": "Anomaly detection results",
            "POST /predict": "ML prediction endpoint",
            "GET /shap/{county_name}": "SHAP explanations",
            "GET /models": "Available ML models",
            "GET /trends": "UCS trends over time",
            "GET /typologies": "Cluster typology descriptions"
        },
        "domains": {
            "Healthcare Access Index": "ANC, skilled delivery, PNC, geo/financial barriers",
            "Population Vulnerability Index": "Poverty, WASH deficits, demographic dependency",
            "Immunization Coverage Index": "EPI antigens, full vaccination",
            "Disease Burden Index": "Child nutrition, malaria/ARI/diarrhoea, anaemia"
        },
        "models_available": [
            "xgboost",
            "xgboost_tuned", 
            "random_forest",
            "gradient_boosting",
            "logistic_regression"
        ],
        "data_source": "KDHS 2020 & 2022 via DHS Programme API"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint - shows API and data status.
    """
    # Check data status
    data_status = {}
    for key, df in DATA.items():
        if df is not None:
            data_status[key] = {
                "loaded": True,
                "rows": len(df),
                "columns": list(df.columns)[:10]  # First 10 columns
            }
        else:
            data_status[key] = {"loaded": False}
    
    # Check model status
    model_status = {}
    for key, model in MODELS.items():
        model_status[key] = {
            "loaded": model is not None,
            "type": type(model).__name__ if model is not None else None
        }
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "data_loaded": data_status,
        "models_loaded": model_status,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "counties": "/counties",
            "rankings": "/rankings",
            "clusters": "/clusters",
            "anomalies": "/anomalies",
            "predict": "/predict",
            "shap": "/shap/{county_name}",
            "models": "/models"
        }
    }


@app.get("/counties", response_model=List[CountyBase], tags=["Counties"])
async def get_all_counties(
    year: Optional[int] = Query(None, description="Filter by year (2014-2022)"),
    limit: Optional[int] = Query(None, description="Limit number of results"),
    sort_by: str = Query("ucs", description="Sort by: ucs, county"),
    ascending: bool = Query(False, description="Sort in ascending order")
):
    """
    Get all counties with UCS scores.
    
    - **year**: Filter by specific year
    - **limit**: Limit number of results
    - **sort_by**: Sort by 'ucs' or 'county'
    - **ascending**: Sort in ascending order
    """
    df = DATA.get("ucs_scores")
    if df is None:
        raise HTTPException(status_code=404, detail="UCS scores data not found")
    
    # Filter by year if specified
    if year and 'year' in df.columns:
        df = df[df['year'] == year]
    
    # Sort
    if sort_by == "ucs":
        df = df.sort_values(by="UCS" if "UCS" in df.columns else "ucs", ascending=ascending)
    elif sort_by == "county":
        df = df.sort_values(by=df.index.name or "county", ascending=ascending)
    
    # Limit
    if limit:
        df = df.head(limit)
    
    # Convert to response model
    result = []
    for county, row in df.iterrows():
        result.append(CountyBase(
            county=str(county),
            ucs=float(row.get('UCS', row.get('ucs', 0)))
        ))
    
    return result


@app.get("/counties/{county_name}", response_model=CountyDetail, tags=["Counties"])
async def get_county_detail(county_name: str):
    """
    Get detailed information for a specific county.
    
    Includes domain scores, cluster assignment, and anomaly detection results.
    """
    df = DATA.get("ucs_scores")
    if df is None:
        raise HTTPException(status_code=404, detail="UCS scores data not found")
    
    # Find county (case-insensitive)
    counties = df.index.str.lower().tolist()
    county_lower = county_name.lower()
    
    if county_lower not in counties:
        raise HTTPException(
            status_code=404, 
            detail=f"County '{county_name}' not found. Available counties: {df.index.tolist()}"
        )
    
    idx = counties.index(county_lower)
    county = df.index[idx]
    row = df.loc[county]
    
    return CountyDetail(
        county=str(county),
        ucs=float(row.get('UCS', row.get('ucs', 0))),
        healthcare_access_index=float(row.get('Healthcare Access Index')) if 'Healthcare Access Index' in row else None,
        population_vulnerability_index=float(row.get('Population Vulnerability Index')) if 'Population Vulnerability Index' in row else None,
        immunization_coverage_index=float(row.get('Immunization Coverage Index')) if 'Immunization Coverage Index' in row else None,
        disease_burden_index=float(row.get('Disease Burden Index')) if 'Disease Burden Index' in row else None,
        cluster=int(row.get('Cluster')) if 'Cluster' in row and pd.notna(row.get('Cluster')) else None,
        cluster_label=str(row.get('Cluster_label')) if 'Cluster_label' in row else None,
        anomaly=str(row.get('Anomaly')) if 'Anomaly' in row else None,
        anomaly_prob=float(row.get('Anomaly_prob')) if 'Anomaly_prob' in row and pd.notna(row.get('Anomaly_prob')) else None
    )


@app.get("/rankings", response_model=List[CountyRanking], tags=["Rankings"])
async def get_rankings(
    top_n: Optional[int] = Query(None, description="Get top N most underserved counties"),
    year: Optional[int] = Query(None, description="Filter by year")
):
    """
    Get county rankings by UCS score.
    
    - **top_n**: Get top N most underserved counties (1 = most underserved)
    - **year**: Filter by specific year
    """
    df = DATA.get("ucs_scores")
    if df is None:
        raise HTTPException(status_code=404, detail="UCS scores data not found")
    
    # Filter by year if specified
    if year and 'year' in df.columns:
        df = df[df['year'] == year]
    
    # Sort by UCS descending (higher = more underserved)
    df = df.sort_values(by='UCS' if 'UCS' in df.columns else 'ucs', ascending=False)
    
    # Limit
    if top_n:
        df = df.head(top_n)
    
    # Create rankings
    result = []
    for rank, (county, row) in enumerate(df.iterrows(), 1):
        result.append(CountyRanking(
            rank=rank,
            county=str(county),
            ucs=float(row.get('UCS', row.get('ucs', 0))),
            cluster_label=str(row.get('Cluster_label')) if 'Cluster_label' in row else None,
            anomaly=str(row.get('Anomaly')) if 'Anomaly' in row else None
        ))
    
    return result


@app.get("/clusters", response_model=List[ClusterInfo], tags=["Clusters"])
async def get_clusters():
    """
    Get cluster analysis results.
    
    Returns cluster assignments and descriptions for all counties.
    """
    clusters_df = DATA.get("clusters")
    typologies_df = DATA.get("typologies")
    
    if clusters_df is None:
        raise HTTPException(status_code=404, detail="Cluster data not found")
    
    # Get unique clusters
    if 'Cluster' in clusters_df.columns:
        clusters = clusters_df['Cluster'].unique()
    else:
        clusters = clusters_df.iloc[:, 0].unique() if len(clusters_df) > 0 else []
    
    result = []
    for cluster_id in sorted(clusters):
        cluster_counties = clusters_df[clusters_df['Cluster'] == cluster_id].index.tolist()
        
        # Get cluster label from typologies if available
        label = f"Cluster {cluster_id}"
        if typologies_df is not None:
            for _, row in typologies_df.iterrows():
                if str(row.get('Cluster')) == str(cluster_id):
                    label = row.get('Label', row.get('Description', f"Cluster {cluster_id}"))
                    break
        
        # Calculate average UCS
        cluster_ucs = clusters_df.loc[cluster_counties, 'ucs'].mean() if 'ucs' in clusters_df.columns else 0
        
        result.append(ClusterInfo(
            cluster_id=int(cluster_id),
            label=str(label),
            counties=[str(c) for c in cluster_counties],
            avg_ucs=float(cluster_ucs)
        ))
    
    return result


@app.get("/anomalies", response_model=List[CountyDetail], tags=["Anomalies"])
async def get_anomalies():
    """
    Get counties flagged as anomalous by Isolation Forest.
    
    These are counties that are structurally different from the typical 
    county pattern and may require special attention.
    """
    df = DATA.get("ucs_scores")
    if df is None:
        raise HTTPException(status_code=404, detail="UCS scores data not found")
    
    # Filter to anomaly counties
    if 'Anomaly' not in df.columns:
        raise HTTPException(status_code=404, detail="Anomaly data not available")
    
    anomalies_df = df[df['Anomaly'] == 'Anomaly']
    
    result = []
    for county, row in anomalies_df.iterrows():
        result.append(CountyDetail(
            county=str(county),
            ucs=float(row.get('UCS', row.get('ucs', 0))),
            healthcare_access_index=float(row.get('Healthcare Access Index')) if 'Healthcare Access Index' in row else None,
            population_vulnerability_index=float(row.get('Population Vulnerability Index')) if 'Population Vulnerability Index' in row else None,
            immunization_coverage_index=float(row.get('Immunization Coverage Index')) if 'Immunization Coverage Index' in row else None,
            disease_burden_index=float(row.get('Disease Burden Index')) if 'Disease Burden Index' in row else None,
            cluster=int(row.get('Cluster')) if 'Cluster' in row and pd.notna(row.get('Cluster')) else None,
            cluster_label=str(row.get('Cluster_label')) if 'Cluster_label' in row else None,
            anomaly=str(row.get('Anomaly')),
            anomaly_prob=float(row.get('Anomaly_prob')) if 'Anomaly_prob' in row and pd.notna(row.get('Anomaly_prob')) else None
        ))
    
    return result


@app.get("/models", tags=["ML"])
async def list_models():
    """
    List all available ML models and their status.
    """
    available_models = ["xgboost", "xgboost_tuned", "random_forest", "gradient_boosting", "logistic_regression"]
    metadata = MODELS.get("metadata")
    
    result = {}
    for model_name in available_models:
        model = MODELS.get(model_name)
        model_type = type(model).__name__ if model is not None else "N/A"
        
        # Get model info
        info = {
            "loaded": model is not None,
            "type": model_type,
        }
        
        # Add model-specific info
        if model is not None:
            if hasattr(model, 'feature_importances_'):
                info["has_feature_importances"] = True
            if hasattr(model, 'predict_proba'):
                info["supports_probability"] = True
            if hasattr(model, 'n_estimators'):
                info["n_estimators"] = model.n_estimators
            if hasattr(model, 'max_depth'):
                info["max_depth"] = model.max_depth
        
        result[model_name] = info
    
    return {
        "api_version": "1.0.0",
        "available_models": result,
        "metadata": metadata,
        "domain_columns": ["Healthcare Access Index", "Population Vulnerability Index", 
                          "Immunization Coverage Index", "Disease Burden Index"],
        "color_scheme": {
            "Healthcare Access Index": C_RED,
            "Population Vulnerability Index": C_BLUE,
            "Immunization Coverage Index": C_GREEN,
            "Disease Burden Index": C_PURPLE
        }
    }


@app.post("/predict", response_model=PredictionOutput, tags=["ML"])
async def predict_underserved(
    input_data: PredictionInput,
    model_name: str = Query("xgboost", description="Model to use: xgboost, xgboost_tuned, random_forest, gradient_boosting, logistic_regression")
):
    """
    Predict whether a county is underserved based on domain scores.
    
    - **input_data**: Domain scores for the county
    - **model_name**: Choose the ML model to use
    """
    available_models = ["xgboost", "xgboost_tuned", "random_forest", "gradient_boosting", "logistic_regression"]
    
    if model_name not in available_models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available: {', '.join(available_models)}"
        )
    
    model = MODELS.get(model_name)
    if model is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found. Available: xgboost, random_forest, gradient_boosting"
        )
    
    metadata = MODELS.get("metadata")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Model metadata not found")
    
    # Prepare input features — models expect 5 features as a named DataFrame
    ucs_val = (
        input_data.healthcare_access_index +
        input_data.population_vulnerability_index +
        input_data.immunization_coverage_index +
        input_data.disease_burden_index
    ) / 4.0

    X = pd.DataFrame([[
        input_data.healthcare_access_index,
        input_data.population_vulnerability_index,
        input_data.immunization_coverage_index,
        input_data.disease_burden_index,
        ucs_val
    ]], columns=[
        "Healthcare Access Index",
        "Population Vulnerability Index",
        "Immunization Coverage Index",
        "Disease Burden Index",
        "UCS"
    ])

    # Make prediction
    try:
        prediction = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0]
        result = PredictionOutput(
            prediction="Underserved" if prediction == 1 else "Normal",
            probability=float(prob[1]),
            model_used=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    return result


@app.get("/shap/{county_name}", response_model=SHAPExplanation, tags=["SHAP"])
async def get_shap_explanation(county_name: str):
    """
    Get SHAP explanation for a specific county.
    
    Shows which features contributed most to the model's prediction.
    """
    shap_explainer = MODELS.get("shap_explainer")
    model = MODELS.get("xgboost")
    metadata = MODELS.get("metadata")
    
    if shap_explainer is None:
        raise HTTPException(status_code=404, detail="SHAP explainer not found")
    
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get county data
    df = DATA.get("ucs_scores")
    if df is None:
        raise HTTPException(status_code=404, detail="UCS scores data not found")
    
    # Find county
    counties = df.index.str.lower().tolist()
    county_lower = county_name.lower()
    
    if county_lower not in counties:
        raise HTTPException(
            status_code=404, 
            detail=f"County '{county_name}' not found"
        )
    
    idx = counties.index(county_lower)
    county = df.index[idx]
    row = df.loc[county]
    
    # Prepare features
    feature_columns = metadata.get("feature_columns", [
        "Healthcare Access Index",
        "Population Vulnerability Index",
        "Immunization Coverage Index",
        "Disease Burden Index"
    ]) if metadata else [
        "Healthcare Access Index",
        "Population Vulnerability Index", 
        "Immunization Coverage Index",
        "Disease Burden Index"
    ]
    
    features = []
    feature_dict = {}
    for col in feature_columns:
        val = row.get(col, 0)
        features.append(float(val) if pd.notna(val) else 0.0)
        feature_dict[col] = features[-1]
    
    # Calculate SHAP values
    try:
        shap_values = shap_explainer.shap_values([features])
        
        # Create SHAP values dict
        shap_dict = {}
        for i, col in enumerate(feature_columns):
            shap_dict[col] = float(shap_values[0][i])
        
        # Get base value
        base_value = float(shap_explainer.expected_value) if hasattr(shap_explainer, 'expected_value') else 0.0
        
        # Make prediction
        prediction = model.predict([features])[0]
        prediction_str = "Underserved" if prediction == 1 else "Normal"
        
        return SHAPExplanation(
            county=str(county),
            prediction=prediction_str,
            shap_values=shap_dict,
            feature_values=feature_dict,
            base_value=base_value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP calculation error: {str(e)}")


@app.get("/trends", tags=["Trends"])
async def get_trends():
    """
    Get UCS trends over time (2014-2022).
    """
    df = DATA.get("trends")
    if df is None:
        raise HTTPException(status_code=404, detail="Trend data not found")
    
    # Convert to list of dicts
    result = []
    for county, row in df.iterrows():
        trend = {"county": str(county)}
        for year_col in df.columns:
            if year_col.isdigit() or (isinstance(year_col, int)):
                trend[str(year_col)] = float(row.get(year_col, 0))
        result.append(trend)
    
    return result


@app.get("/typologies", tags=["Typologies"])
async def get_typologies():
    """
    Get cluster typology descriptions.
    
    Returns the descriptive labels for each county cluster.
    """
    df = DATA.get("typologies")
    if df is None:
        raise HTTPException(status_code=404, detail="Typology data not found")
    
    result = []
    for _, row in df.iterrows():
        typology = {
            "cluster": int(row.get('Cluster', 0)),
            "label": str(row.get('Label', '')),
            "description": str(row.get('Description', '')),
            "characteristics": str(row.get('Characteristics', ''))
        }
        result.append(typology)
    
    return result


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return {
        "error": str(exc),
        "status_code": 500
    }