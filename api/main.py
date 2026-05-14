"""
main.py — FastAPI application
──────────────────────────────
REST API for real-time and batch loan propensity scoring.

Endpoints:
  GET  /health          — liveness probe (used by AWS ELB / k8s)
  POST /predict         — single account propensity score
  POST /predict/batch   — batch propensity scoring
  GET  /model/info      — current model version and metrics

Run locally:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import pickle
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger("api")
cfg    = load_config("config/config.yaml")

app = FastAPI(
    title="Loan Payment Propensity API",
    description="Predicts the probability of a loan account making a payment in the next 30 days.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts at startup ──────────────────────────
@app.on_event("startup")
def load_model():
    global predictor
    try:
        from src.prediction.prediction import PropensityPredictor
        predictor = PropensityPredictor()
        logger.info("Model loaded at API startup ✅")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None


# ── Request / Response schemas ────────────────────────────────
class LoanRecord(BaseModel):
    """Single loan account input — all raw features before transformation."""
    loan_id:             Optional[str] = Field(None, description="Unique loan ID")
    original_balance:    float  = Field(..., description="Original loan amount")
    current_balance:     float  = Field(..., description="Outstanding balance")
    last_pmt_amt:        float  = Field(0.0,  description="Last payment amount")
    last_pmt_date:       Optional[str] = None
    birthday:            str    = Field(..., description="Debtor birthday YYYY-MM-DD")
    status:              int    = Field(..., description="Disposition status code")
    lastNoticeSent:      str    = Field(..., description="Last notice date YYYY-MM-DD")
    state:               Optional[str] = "Unknown"
    creditor_name:       str    = Field(..., description="Original creditor bank")
    chargeoff_date:      str    = Field(..., description="Chargeoff/placement date YYYY-MM-DD")
    total_portal_visit:  int    = Field(0)
    times_dials:         int    = Field(0)
    times_connect:       int    = Field(0)
    times_contact:       int    = Field(0)
    times_rpc:           int    = Field(0)
    times_ptp:           int    = Field(0)
    times_up:            int    = Field(0)
    times_drop:          int    = Field(0)
    times_lm:            int    = Field(0)


class PredictionResponse(BaseModel):
    loan_id:              Optional[str]
    propensity_score:     float = Field(..., description="Raw probability 0–1")
    propensity_score_pct: float = Field(..., description="Propensity as percentage")
    risk_band:            str   = Field(..., description="Risk band label")
    predicted_label:      int   = Field(..., description="1=likely to pay, 0=unlikely")
    predicted_at:         str


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe — used by load balancers and k8s."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return current model version, algorithm, and training metrics."""
    try:
        with open(cfg.artifacts.metrics_path) as f:
            metrics = json.load(f)
        return {"status": "active", "metrics": metrics}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model metrics not found. Run training pipeline first.")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(record: LoanRecord):
    """
    Score a single loan account in real time.
    Returns propensity score (%), risk band, and predicted label.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    try:
        import pandas as pd
        # Convert Pydantic model → dict → DataFrame
        data = record.dict()
        data["Creditor name"] = data.pop("creditor_name")
        data["Loan Id"]       = data.pop("loan_id", None)

        # Parse date strings to Timestamps
        for date_col in ["birthday", "chargeoff_date", "lastNoticeSent", "last_pmt_date"]:
            if data.get(date_col):
                data[date_col] = pd.to_datetime(data[date_col])

        result = predictor.predict_single(data)
        result["loan_id"] = record.loan_id
        logger.info(f"Prediction: loan={record.loan_id}  score={result['propensity_score_pct']:.2f}%  band={result['risk_band']}")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(background_tasks: BackgroundTasks, file_path: str):
    """
    Trigger batch scoring for a full Excel file.
    Runs asynchronously in the background.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    def _run_batch():
        predictor.score_portfolio(file_path)
        logger.info(f"Batch scoring complete for: {file_path}")

    background_tasks.add_task(_run_batch)
    return {"status": "accepted", "message": f"Batch scoring started for {file_path}"}


# ── Run directly for local dev ────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)
