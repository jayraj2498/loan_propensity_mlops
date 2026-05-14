"""
prediction.py
─────────────
Step 6 of the MLOps pipeline.
Loads fitted model + preprocessor → generates propensity scores
for new/unseen loan accounts → logs predictions to SQLite.

Used by both:
  - FastAPI (/predict endpoint) for single real-time inference
  - Batch scoring script for full portfolio scoring
"""

import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger(__name__)

# Risk band labels — mirrors collections team priority levels
RISK_BANDS = {
    (0, 1):   "Very Low (<1%)",
    (1, 5):   "Low (1-5%)",
    (5, 10):  "Medium (5-10%)",
    (10, 25): "High (10-25%)",
    (25, 50): "Very High (25-50%)",
    (50, 100):"Critical (>50%)",
}


def assign_risk_band(score_pct: float) -> str:
    for (lo, hi), label in RISK_BANDS.items():
        if lo <= score_pct < hi:
            return label
    return "Critical (>50%)"


class PropensityPredictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and preprocessor from disk (or S3 in production)."""
        with open(self.cfg.artifacts.model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(self.cfg.artifacts.preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)
        logger.info("Model and preprocessor loaded ✅")

    def _preprocess_input(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the same feature engineering used at training time."""
        today = pd.Timestamp(self.cfg.features.reference_date)

        # Impute missing
        df["last_pmt_date"] = df["last_pmt_date"].fillna(df["chargeoff_date"])
        df["state"] = df["state"].fillna("Unknown")

        # Date features
        df["age"]                    = (today - df["birthday"]).dt.days // 365
        df["days_since_chargeoff"]   = (today - df["chargeoff_date"]).dt.days
        df["days_since_last_notice"] = (today - df["lastNoticeSent"]).dt.days
        df["days_since_last_payment"]= (today - df["last_pmt_date"]).dt.days

        # Ratio features
        df["balance_ratio"] = df["current_balance"] / (df["original_balance"] + 1)
        df["contact_rate"]  = df["times_connect"]   / (df["times_dials"] + 1)
        df["rpc_rate"]      = df["times_rpc"]        / (df["times_connect"] + 1)
        df["ptp_rate"]      = df["times_ptp"]        / (df["times_contact"] + 1)

        # Drop raw date + ID columns
        drop_cols = (self.cfg.features.date_columns
                     + self.cfg.features.drop_columns
                     + ["Payment_Next30Days"])
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return self.preprocessor.transform(df)

    def predict_single(self, record: dict) -> dict:
        """
        Score a single loan account (used by FastAPI).
        Input: dict with raw loan features
        Output: dict with propensity_score_pct, risk_band, predicted_label
        """
        df = pd.DataFrame([record])
        X = self._preprocess_input(df)
        proba = float(self.model.predict_proba(X)[0, 1])
        pct   = round(proba * 100, 4)
        label = int(proba >= 0.5)

        return {
            "propensity_score":     proba,
            "propensity_score_pct": pct,
            "risk_band":            assign_risk_band(pct),
            "predicted_label":      label,
            "predicted_at":         datetime.utcnow().isoformat(),
        }

    def predict_batch(self, df: pd.DataFrame, source: str = "batch") -> pd.DataFrame:
        """
        Score entire portfolio (batch job).
        Returns original df with propensity scores appended.
        Logs all predictions to SQLite.
        """
        loan_ids = df["Loan Id"].tolist() if "Loan Id" in df.columns else [None] * len(df)

        X = self._preprocess_input(df.copy())
        probas = self.model.predict_proba(X)[:, 1]
        pcts   = (probas * 100).round(4)
        labels = (probas >= 0.5).astype(int)

        df_out = pd.DataFrame({
            "loan_id":            loan_ids,
            "propensity_score":   probas,
            "propensity_pct":     pcts,
            "predicted_label":    labels,
            "risk_band":          [assign_risk_band(p) for p in pcts],
            "request_source":     source,
            "predicted_at":       datetime.utcnow().isoformat(),
        })

        self._log_predictions(df_out)
        logger.info(f"Batch scoring complete — {len(df_out):,} accounts scored")
        return df_out

    def _log_predictions(self, df_preds: pd.DataFrame):
        """Write predictions to SQLite predictions table for monitoring."""
        try:
            conn = sqlite3.connect(self.cfg.data.sql_db_path)
            df_preds.to_sql("predictions", conn, if_exists="append", index=False)
            conn.close()
            logger.info(f"Predictions logged to DB ({len(df_preds)} rows)")
        except Exception as e:
            logger.error(f"Failed to log predictions: {e}")

    def score_portfolio(self, excel_path: str, output_path: str = "artifacts/propensity_scores.xlsx"):
        """Full portfolio scoring — reads Excel, scores, saves ranked output."""
        df = pd.read_excel(excel_path)
        logger.info(f"Portfolio loaded: {df.shape[0]:,} accounts")

        scores = self.predict_batch(df)

        # Merge back with key account info
        output = pd.concat([
            df[["Loan Id", "clnt_no", "current_balance", "state", "Creditor name",
                "status", "Payment_Next30Days"]].reset_index(drop=True),
            scores[["propensity_pct", "risk_band", "predicted_label"]].reset_index(drop=True)
        ], axis=1)

        output = output.sort_values("propensity_pct", ascending=False)
        output.to_excel(output_path, index=False)
        logger.info(f"Propensity scores saved to: {output_path}")
        return output


if __name__ == "__main__":
    predictor = PropensityPredictor()
    predictor.score_portfolio("data/raw/Loan_Data_Clean_cld.xlsx")
