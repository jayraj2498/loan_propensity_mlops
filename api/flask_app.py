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
@app.route("/", methods=["GET", "POST"])
def home():
    """Simple browser UI for single-record loan propensity scoring."""
    result = None
    error = None
    form_values = {
        "loan_id": "L001",
        "original_balance": "5000",
        "current_balance": "4800",
        "last_pmt_amt": "0",
        "birthday": "1985-06-15",
        "status": "1",
        "lastNoticeSent": "2026-01-15",
        "state": "TX",
        "creditor_name": "CAPITAL ONE",
        "chargeoff_date": "2022-01-01",
        "total_portal_visit": "0",
        "times_dials": "10",
        "times_connect": "3",
        "times_contact": "2",
        "times_rpc": "1",
        "times_ptp": "1",
        "times_up": "0",
        "times_drop": "2",
        "times_lm": "3",
    }

    if request.method == "POST":
        form_values.update({k: request.form.get(k, "").strip() for k in form_values})
        if predictor is None:
            error = "Model is not loaded, so predictions are currently unavailable."
        else:
            try:
                import pandas as pd

                record = {
                    "Loan Id": form_values["loan_id"] or "WEB001",
                    "original_balance": float(form_values["original_balance"] or 0),
                    "current_balance": float(form_values["current_balance"] or 0),
                    "last_pmt_amt": float(form_values["last_pmt_amt"] or 0),
                    "last_pmt_date": None,
                    "birthday": pd.to_datetime(form_values["birthday"]),
                    "status": int(form_values["status"] or 0),
                    "lastNoticeSent": pd.to_datetime(form_values["lastNoticeSent"]),
                    "state": form_values["state"] or "Unknown",
                    "Creditor name": form_values["creditor_name"] or "Unknown",
                    "chargeoff_date": pd.to_datetime(form_values["chargeoff_date"]),
                    "total_portal_visit": int(form_values["total_portal_visit"] or 0),
                    "times_dials": int(form_values["times_dials"] or 0),
                    "times_connect": int(form_values["times_connect"] or 0),
                    "times_contact": int(form_values["times_contact"] or 0),
                    "times_rpc": int(form_values["times_rpc"] or 0),
                    "times_ptp": int(form_values["times_ptp"] or 0),
                    "times_up": int(form_values["times_up"] or 0),
                    "times_drop": int(form_values["times_drop"] or 0),
                    "times_lm": int(form_values["times_lm"] or 0),
                }
                result = predictor.predict_single(record)
                result["loan_id"] = record["Loan Id"]
            except Exception as e:
                error = f"Could not score this record: {e}"

    html = f"""
    <!DOCTYPE html><html><head><meta charset='utf-8'>
    <title>Loan Propensity Flask App</title>
    <style>
      body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:32px;background:#f8f9fa;color:#2c3e50}}
      h1{{border-bottom:3px solid #3498db;padding-bottom:10px}}
      h2{{margin:0 0 14px 0;color:#1f5f8b}}
      .card{{background:white;border-radius:10px;padding:20px 24px;box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-top:20px}}
      .status{{display:inline-block;padding:6px 12px;border-radius:999px;background:{'#27ae60' if predictor else '#e67e22'};color:white;font-weight:bold}}
      a{{color:#2980b9;text-decoration:none;font-weight:bold}}
      ul{{line-height:1.9}}
      code{{background:#ecf0f1;padding:2px 6px;border-radius:4px}}
      .grid{{display:grid;grid-template-columns:repeat(3, minmax(0, 1fr));gap:14px}}
      label{{display:block;font-weight:bold;margin-bottom:6px}}
      input{{width:100%;padding:10px 12px;border:1px solid #cfd8dc;border-radius:8px;box-sizing:border-box}}
      .submit{{margin-top:18px;background:#2980b9;color:white;border:none;padding:12px 18px;border-radius:8px;font-size:16px;cursor:pointer}}
      .submit:hover{{background:#21658f}}
      .result{{background:#eef8f0;border-left:5px solid #27ae60}}
      .error{{background:#fff3f3;border-left:5px solid #c0392b}}
      .metric{{display:inline-block;min-width:180px;margin:10px 16px 0 0;padding:14px 16px;background:#f4f8fb;border-radius:8px}}
      .metric strong{{display:block;font-size:13px;color:#5b6b76;margin-bottom:6px}}
      .metric span{{font-size:24px;font-weight:bold;color:#1f5f8b}}
      @media (max-width: 900px) {{ .grid{{grid-template-columns:1fr;}} }}
    </style></head><body>
    <h1>Loan Propensity Prediction Flask App</h1>
    <div class="card">
      <p><span class="status">{'RUNNING' if predictor else 'DEGRADED'}</span></p>
      <p>This Flask service is up and listening on port <code>5000</code>. You can enter one loan record below and get the predicted value directly in the browser.</p>
      <p><b>Model loaded:</b> {'Yes' if predictor else 'No'} </p>
      <p><b>Current time:</b> {datetime.utcnow().isoformat()}</p>
    </div>

    <div class="card">
      <h2>Prediction Form</h2>
      <form method="post">
        <div class="grid">
          <div><label>Loan ID</label><input name="loan_id" value="{form_values['loan_id']}"></div>
          <div><label>Original Balance</label><input name="original_balance" type="number" step="0.01" value="{form_values['original_balance']}"></div>
          <div><label>Current Balance</label><input name="current_balance" type="number" step="0.01" value="{form_values['current_balance']}"></div>
          <div><label>Last Payment Amount</label><input name="last_pmt_amt" type="number" step="0.01" value="{form_values['last_pmt_amt']}"></div>
          <div><label>Birthday</label><input name="birthday" type="date" value="{form_values['birthday']}"></div>
          <div><label>Status Code</label><input name="status" type="number" value="{form_values['status']}"></div>
          <div><label>Last Notice Sent</label><input name="lastNoticeSent" type="date" value="{form_values['lastNoticeSent']}"></div>
          <div><label>State</label><input name="state" value="{form_values['state']}"></div>
          <div><label>Creditor Name</label><input name="creditor_name" value="{form_values['creditor_name']}"></div>
          <div><label>Chargeoff Date</label><input name="chargeoff_date" type="date" value="{form_values['chargeoff_date']}"></div>
          <div><label>Portal Visits</label><input name="total_portal_visit" type="number" value="{form_values['total_portal_visit']}"></div>
          <div><label>Times Dials</label><input name="times_dials" type="number" value="{form_values['times_dials']}"></div>
          <div><label>Times Connect</label><input name="times_connect" type="number" value="{form_values['times_connect']}"></div>
          <div><label>Times Contact</label><input name="times_contact" type="number" value="{form_values['times_contact']}"></div>
          <div><label>Times RPC</label><input name="times_rpc" type="number" value="{form_values['times_rpc']}"></div>
          <div><label>Times PTP</label><input name="times_ptp" type="number" value="{form_values['times_ptp']}"></div>
          <div><label>Times Urgent Pay</label><input name="times_up" type="number" value="{form_values['times_up']}"></div>
          <div><label>Times Drop</label><input name="times_drop" type="number" value="{form_values['times_drop']}"></div>
          <div><label>Times Left Message</label><input name="times_lm" type="number" value="{form_values['times_lm']}"></div>
        </div>
        <button class="submit" type="submit">Predict Value</button>
      </form>
    </div>

    {f'''
    <div class="card result">
      <h2>Prediction Result</h2>
      <div class="metric"><strong>Loan ID</strong><span>{result["loan_id"]}</span></div>
      <div class="metric"><strong>Propensity Score</strong><span>{result["propensity_score_pct"]:.2f}%</span></div>
      <div class="metric"><strong>Predicted Label</strong><span>{result["predicted_label"]}</span></div>
      <div class="metric"><strong>Risk Band</strong><span>{result["risk_band"]}</span></div>
    </div>
    ''' if result else ''}

    {f'''
    <div class="card error">
      <h2>Problem</h2>
      <p>{error}</p>
    </div>
    ''' if error else ''}

    <div class="card">
      <h2>API Endpoints</h2>
      <ul>
        <li><a href="/health">/health</a> - service and model status</li>
        <li><a href="/docs">/docs</a> - API usage documentation</li>
        <li><a href="/model/info">/model/info</a> - trained model metrics</li>
      </ul>
      <p><code>/predict</code> and <code>/predict/batch</code> still work as JSON API endpoints if you want programmatic access later.</p>
    </div>
    </body></html>
    """
    return render_template_string(html)


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
