"""
monitoring.py
─────────────
Step 7 of the MLOps pipeline.
Monitors model health in production:
  - Data drift detection using PSI (Population Stability Index)
  - Prediction volume and score distribution monitoring
  - Accuracy tracking when ground truth becomes available
  - Alerts when retraining is needed
"""

import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger(__name__)


class ModelMonitor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)

    # ── PSI: Population Stability Index ──────────────────────
    @staticmethod
    def compute_psi(expected: np.ndarray, actual: np.ndarray,
                    buckets: int = 10) -> float:
        """
        PSI measures how much a feature distribution has shifted.
        Interpretation:
          PSI < 0.1  → No significant change (stable)
          PSI 0.1–0.2 → Slight change (monitor)
          PSI > 0.2  → Major shift → trigger retraining
        """
        def _psi_buckets(arr, bins):
            counts, _ = np.histogram(arr, bins=bins)
            pct = counts / len(arr)
            return np.where(pct == 0, 0.0001, pct)  # avoid log(0)

        bins = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        bins[0] -= 1e-9; bins[-1] += 1e-9

        exp_pct = _psi_buckets(expected, bins)
        act_pct = _psi_buckets(actual, bins)

        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return round(float(psi), 4)

    # ── Check drift on all numeric features ──────────────────
    def check_feature_drift(self, baseline_df: pd.DataFrame,
                            current_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare baseline (training) distribution vs current (live) data.
        Returns drift report with PSI per feature.
        """
        threshold = self.cfg.monitoring.drift_threshold
        numeric_cols = baseline_df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c in current_df.columns]

        drift_results = []
        for col in numeric_cols:
            psi = self.compute_psi(
                baseline_df[col].dropna().values,
                current_df[col].dropna().values
            )
            drift_detected = psi > threshold
            drift_results.append({
                "feature_name":   col,
                "psi_score":      psi,
                "drift_detected": int(drift_detected),
                "status":         "🚨 DRIFT" if drift_detected else "✅ Stable",
                "checked_at":     datetime.utcnow().isoformat(),
            })
            if drift_detected:
                logger.warning(f"DRIFT DETECTED — {col}  PSI={psi:.4f} (threshold={threshold})")

        drift_df = pd.DataFrame(drift_results).sort_values("psi_score", ascending=False)
        self._save_drift_to_db(drift_df)
        logger.info(f"Drift check complete — {drift_df['drift_detected'].sum()} features drifted")
        return drift_df

    def _save_drift_to_db(self, drift_df: pd.DataFrame):
        try:
            conn = sqlite3.connect(self.cfg.data.sql_db_path)
            drift_df[["feature_name","psi_score","drift_detected","checked_at"]] \
                .to_sql("drift_monitoring", conn, if_exists="append", index=False)
            conn.close()
        except Exception as e:
            logger.error(f"Could not save drift results: {e}")

    # ── Monitor prediction volume + score stats ───────────────
    def prediction_health_check(self) -> dict:
        """
        Pull recent predictions from DB and compute health stats.
        Alert if score distribution has shifted significantly.
        """
        try:
            conn = sqlite3.connect(self.cfg.data.sql_db_path)
            df = pd.read_sql("SELECT * FROM predictions ORDER BY predicted_at DESC LIMIT 10000", conn)
            conn.close()
        except Exception as e:
            logger.warning(f"Could not read predictions table: {e}")
            return {}

        if len(df) == 0:
            logger.info("No predictions in DB yet")
            return {}

        health = {
            "total_predictions":    len(df),
            "avg_propensity_pct":   round(df["propensity_pct"].mean(), 4),
            "pct_predicted_payers": round((df["predicted_label"] == 1).mean() * 100, 2),
            "risk_band_counts":     df["risk_band"].value_counts().to_dict(),
            "checked_at":           datetime.utcnow().isoformat(),
        }

        logger.info(f"Prediction health: avg_propensity={health['avg_propensity_pct']}%  payers={health['pct_predicted_payers']}%")
        return health

    # ── Live accuracy (after ground truth arrives) ────────────
    def compute_live_accuracy(self) -> float | None:
        """
        Once actual payment labels are updated in the predictions table,
        compute live accuracy. If accuracy drops >5%, trigger retrain alert.
        """
        try:
            conn = sqlite3.connect(self.cfg.data.sql_db_path)
            df = pd.read_sql(
                "SELECT predicted_label, actual_label FROM predictions WHERE actual_label IS NOT NULL",
                conn
            )
            conn.close()
        except Exception as e:
            logger.warning(f"Could not compute live accuracy: {e}")
            return None

        if len(df) == 0:
            logger.info("No ground truth available yet for live accuracy")
            return None

        live_acc = (df["predicted_label"] == df["actual_label"]).mean()
        logger.info(f"Live accuracy: {live_acc*100:.2f}%")

        # Load training accuracy from metrics file
        with open(self.cfg.artifacts.metrics_path) as f:
            train_metrics = json.load(f)
        train_acc = train_metrics.get("accuracy", 1.0)

        drop = train_acc - live_acc
        if drop > self.cfg.monitoring.accuracy_drop_threshold:
            logger.warning(f"🚨 ACCURACY DROP ALERT: {drop*100:.2f}% drop — consider retraining!")

        return round(live_acc, 4)

    # ── Full monitoring run ───────────────────────────────────
    def run(self, baseline_df: pd.DataFrame = None, current_df: pd.DataFrame = None):
        logger.info("=" * 50)
        logger.info("STEP 7: MODEL MONITORING")
        logger.info("=" * 50)

        health = self.prediction_health_check()
        live_acc = self.compute_live_accuracy()

        if baseline_df is not None and current_df is not None:
            drift_report = self.check_feature_drift(baseline_df, current_df)
            logger.info("\nDrift Report:\n" + drift_report.to_string(index=False))

        summary = {
            "health": health,
            "live_accuracy": live_acc,
            "checked_at": datetime.utcnow().isoformat(),
        }
        logger.info(f"Monitoring complete: {summary}")
        return summary
