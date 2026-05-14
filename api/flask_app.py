"""
app.py — Flask Application
───────────────────────────
REST API for real-time and batch loan propensity scoring using Flask.

Endpoints:
  GET  /health          — liveness probe
  POST /predict         — single account propensity score
  POST /predict/batch   — batch scoring (JSON array)
  GET  /model/info      — current model version and metrics
  GET  /docs            — simple API documentation page

Run locally:
  python api/flask_app.py
  OR
  flask --app api/flask_app run --host 0.0.0.0 --port 5000
"""

import os
import sys
import json
import pickle
import traceback
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, render_template_string
from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger("flask_app")
cfg    = load_config("config/config.yaml")

app = Flask(__name__)

# ── Load model at startup ─────────────────────────────────────
predictor = None

def load_model():
    global predictor
    try:
        from src.prediction.prediction import PropensityPredictor
        predictor = PropensityPredictor()
        logger.info("✅ Model loaded at Flask startup")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        predictor = None

load_model()


# ── Helper: standard JSON response ───────────────────────────
def success_response(data, status=200):
    return jsonify({"status": "success", "data": data}), status

def error_response(message, status=400):
    return jsonify({"status": "error", "message": message}), status


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

# ── GET /health ───────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """
    Liveness probe — used by AWS ELB, Kubernetes, and CI/CD health checks.
    Returns 200 if service is running, 503 if model failed to load.
    """
    status_code = 200 if predictor is not None else 503
    return jsonify({
        "status":       "healthy" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "service":      "Loan Propensity Prediction API",
        "version":      "1.0.0",
        "timestamp":    datetime.utcnow().isoformat(),
    }), status_code


# ── GET /model/info ───────────────────────────────────────────
@app.route("/model/info", methods=["GET"])
def model_info():
    """Return current model metrics and algorithm info."""
    try:
        with open(cfg.artifacts.metrics_path) as f:
            metrics = json.load(f)
        return success_response({
            "algorithm":    metrics.get("model_name", "Random Forest (Tuned)"),
            "metrics":      metrics,
            "artifact_path":cfg.artifacts.model_path,
            "loaded_at":    datetime.utcnow().isoformat(),
        })
    except FileNotFoundError:
        return error_response("Model metrics not found. Run training pipeline first.", 404)


# ── POST /predict ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_single():
    """
    Score a single loan account in real time.

    Request body (JSON):
    {
        "loan_id":           "L001",
        "original_balance":  5000.0,
        "current_balance":   4800.0,
        "last_pmt_amt":      0.0,
        "last_pmt_date":     null,
        "birthday":          "1985-06-15",
        "status":            1,
        "lastNoticeSent":    "2026-01-15",
        "state":             "TX",
        "Creditor name":     "CAPITAL ONE",
        "chargeoff_date":    "2022-01-01",
        "total_portal_visit":0,
        "times_dials":       10,
        "times_connect":     3,
        "times_contact":     2,
        "times_rpc":         1,
        "times_ptp":         1,
        "times_up":          0,
        "times_drop":        2,
        "times_lm":          3
    }

    Response:
    {
        "status": "success",
        "data": {
            "loan_id":              "L001",
            "propensity_score":     0.7823,
            "propensity_score_pct": 78.23,
            "risk_band":            "Critical (>50%)",
            "predicted_label":      1,
            "predicted_at":         "2026-04-30T10:00:00"
        }
    }
    """
    if predictor is None:
        return error_response("Model not loaded. Check server logs.", 503)

    # Validate Content-Type
    if not request.is_json:
        return error_response("Request must be JSON (Content-Type: application/json)", 415)

    data = request.get_json()

    # Required fields validation
    required = ["original_balance", "current_balance", "birthday",
                "chargeoff_date", "lastNoticeSent", "status", "Creditor name"]
    missing = [f for f in required if f not in data or data[f] is None]
    if missing:
        return error_response(f"Missing required fields: {missing}", 400)

    try:
        import pandas as pd

        # Parse date strings → Timestamps
        for col in ["birthday", "chargeoff_date", "lastNoticeSent", "last_pmt_date"]:
            if data.get(col):
                data[col] = pd.to_datetime(data[col])
            else:
                data[col] = None

        # Fill optional fields with defaults
        data.setdefault("state",             "Unknown")
        data.setdefault("last_pmt_amt",      0.0)
        data.setdefault("total_portal_visit", 0)
        data.setdefault("times_dials",        0)
        data.setdefault("times_connect",      0)
        data.setdefault("times_contact",      0)
        data.setdefault("times_rpc",          0)
        data.setdefault("times_ptp",          0)
        data.setdefault("times_up",           0)
        data.setdefault("times_drop",         0)
        data.setdefault("times_lm",           0)

        result = predictor.predict_single(data)
        result["loan_id"] = data.get("loan_id", "UNKNOWN")

        logger.info(
            f"Prediction — loan={result['loan_id']} "
            f"score={result['propensity_score_pct']:.2f}% "
            f"band={result['risk_band']}"
        )
        return success_response(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        return error_response(f"Prediction failed: {str(e)}", 500)


# ── POST /predict/batch ───────────────────────────────────────
@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Score multiple loan accounts in one request.

    Request body (JSON array of loan records):
    [
        { "loan_id": "L001", "original_balance": 5000, ... },
        { "loan_id": "L002", "original_balance": 8000, ... }
    ]

    Response:
    {
        "status": "success",
        "data": {
            "total_scored":   2,
            "predictions": [
                { "loan_id": "L001", "propensity_score_pct": 78.23, ... },
                { "loan_id": "L002", "propensity_score_pct": 12.50, ... }
            ]
        }
    }
    """
    if predictor is None:
        return error_response("Model not loaded.", 503)

    if not request.is_json:
        return error_response("Request must be JSON", 415)

    records = request.get_json()

    if not isinstance(records, list):
        return error_response("Request body must be a JSON array of loan records", 400)

    if len(records) == 0:
        return error_response("Empty records array", 400)

    if len(records) > 10000:
        return error_response("Batch size limit is 10,000 records per request", 400)

    try:
        import pandas as pd

        df = pd.DataFrame(records)

        # Parse date columns
        for col in ["birthday", "chargeoff_date", "lastNoticeSent", "last_pmt_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Fill defaults for missing columns
        defaults = {
            "state": "Unknown", "last_pmt_amt": 0.0, "total_portal_visit": 0,
            "times_dials": 0, "times_connect": 0, "times_contact": 0,
            "times_rpc": 0, "times_ptp": 0, "times_up": 0,
            "times_drop": 0, "times_lm": 0,
        }
        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        scores_df = predictor.predict_batch(df, source="flask_batch")

        predictions = scores_df.to_dict(orient="records")
        logger.info(f"Batch prediction complete — {len(predictions)} records scored")

        return success_response({
            "total_scored": len(predictions),
            "predictions":  predictions,
            "scored_at":    datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error(f"Batch prediction error: {e}\n{traceback.format_exc()}")
        return error_response(f"Batch prediction failed: {str(e)}", 500)


# ── GET /docs ─────────────────────────────────────────────────
@app.route("/docs", methods=["GET"])
def api_docs():
    """Simple HTML documentation page for the API."""
    html = """
    <!DOCTYPE html><html><head><meta charset='utf-8'>
    <title>Loan Propensity API — Docs</title>
    <style>
      body{font-family:Arial,sans-serif;max-width:900px;margin:auto;padding:40px;background:#f8f9fa}
      h1{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}
      h2{color:#2980b9;margin-top:30px}
      .endpoint{background:white;border-left:4px solid #3498db;padding:15px 20px;
                margin:15px 0;border-radius:0 8px 8px 0;box-shadow:0 2px 6px #ddd}
      .method{display:inline-block;padding:3px 10px;border-radius:4px;
              font-weight:bold;color:white;margin-right:10px}
      .get{background:#27ae60}.post{background:#2980b9}
      pre{background:#2c3e50;color:#ecf0f1;padding:15px;border-radius:6px;overflow-x:auto}
      code{background:#ecf0f1;padding:2px 6px;border-radius:3px;color:#e74c3c}
    </style></head><body>
    <h1>💳 Loan Propensity Prediction API</h1>
    <p>Predicts the probability of a loan account making a payment in the next 30 days.</p>
    <p><strong>Base URL:</strong> <code>http://localhost:5000</code></p>

    <h2>Endpoints</h2>

    <div class='endpoint'>
      <span class='method get'>GET</span><strong>/health</strong>
      <p>Liveness probe — returns service status and model load state.</p>
    </div>

    <div class='endpoint'>
      <span class='method get'>GET</span><strong>/model/info</strong>
      <p>Returns current model algorithm and performance metrics.</p>
    </div>

    <div class='endpoint'>
      <span class='method post'>POST</span><strong>/predict</strong>
      <p>Score a single loan account. Returns propensity score (%), risk band, and predicted label.</p>
      <pre>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "loan_id":          "L001",
    "original_balance": 5000.0,
    "current_balance":  4800.0,
    "birthday":         "1985-06-15",
    "chargeoff_date":   "2022-01-01",
    "lastNoticeSent":   "2026-01-15",
    "status":           1,
    "Creditor name":    "CAPITAL ONE",
    "state":            "TX",
    "times_dials":      10,
    "times_ptp":        1
  }'</pre>
    </div>

    <div class='endpoint'>
      <span class='method post'>POST</span><strong>/predict/batch</strong>
      <p>Score multiple accounts in one request. Accepts a JSON array. Max 10,000 records.</p>
      <pre>curl -X POST http://localhost:5000/predict/batch \\
  -H "Content-Type: application/json" \\
  -d '[{"loan_id":"L001","original_balance":5000,...},
       {"loan_id":"L002","original_balance":8000,...}]'</pre>
    </div>

    </body></html>
    """
    return render_template_string(html)


# ── Error handlers ────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return error_response(f"Endpoint not found. Visit /docs for available endpoints.", 404)

@app.errorhandler(405)
def method_not_allowed(e):
    return error_response("Method not allowed for this endpoint.", 405)

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return error_response("Internal server error.", 500)


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Flask app on port 5000...")
    app.run(
        host=cfg.api.host,
        port=5000,
        debug=False,       # Set True only for local dev
    )
