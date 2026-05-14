"""
tests/unit/test_pipeline.py
────────────────────────────
Unit tests for all pipeline modules.
Run: pytest tests/unit/ -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Create a minimal valid loan dataframe for testing."""
    today = pd.Timestamp("2026-04-30")
    return pd.DataFrame({
        "Loan Id":           ["L001", "L002", "L003"],
        "clnt_no":           ["C001", "C002", "C003"],
        "original_balance":  [5000.0, 10000.0, 3000.0],
        "current_balance":   [4800.0, 9500.0, 2800.0],
        "last_pmt_amt":      [0.0, 100.0, 0.0],
        "last_pmt_date":     [None, today - pd.Timedelta(days=30), None],
        "birthday":          [pd.Timestamp("1985-01-01"), pd.Timestamp("1990-06-15"), pd.Timestamp("1975-03-20")],
        "status":            [1, 2, 1],
        "lastNoticeSent":    [today - pd.Timedelta(days=10)] * 3,
        "state":             ["TX", "CA", None],
        "Creditor name":     ["CAPITAL ONE", "CHASE", "CITI"],
        "chargeoff_date":    [today - pd.Timedelta(days=365)] * 3,
        "total_portal_visit":[0, 1, 0],
        "times_dials":       [10, 20, 5],
        "times_connect":     [3, 8, 1],
        "times_contact":     [2, 5, 0],
        "times_rpc":         [1, 3, 0],
        "times_ptp":         [1, 2, 0],
        "times_up":          [0, 1, 0],
        "times_drop":        [2, 4, 1],
        "times_lm":          [3, 6, 2],
        "Payment_Next30Days":[1, 0, 0],
    })


# ── DataIngestion tests ───────────────────────────────────────
class TestDataIngestion:
    def test_schema_validation_passes(self, sample_df):
        from src.data_ingestion.data_ingestion import DataIngestion
        ingestion = DataIngestion.__new__(DataIngestion)
        assert ingestion.validate_schema(sample_df) is True

    def test_schema_validation_fails_missing_col(self, sample_df):
        from src.data_ingestion.data_ingestion import DataIngestion
        ingestion = DataIngestion.__new__(DataIngestion)
        df_bad = sample_df.drop(columns=["Payment_Next30Days"])
        with pytest.raises(ValueError, match="Missing columns"):
            ingestion.validate_schema(df_bad)


# ── DataTransformation tests ──────────────────────────────────
class TestDataTransformation:
    def test_missing_values_imputed(self, sample_df):
        from src.data_transformation.data_transformation import DataTransformation
        transformer = DataTransformation.__new__(DataTransformation)
        transformer.today = pd.Timestamp("2026-04-30")

        result = transformer.handle_missing_values(sample_df.copy())
        # last_pmt_date nulls should be filled
        assert result["last_pmt_date"].isnull().sum() == 0
        # state null should be 'Unknown'
        assert "Unknown" in result["state"].values

    def test_date_features_created(self, sample_df):
        from src.data_transformation.data_transformation import DataTransformation
        transformer = DataTransformation.__new__(DataTransformation)
        transformer.today = pd.Timestamp("2026-04-30")
        # Patch cfg.features.date_columns
        transformer.cfg = MagicMock()
        transformer.cfg.features.date_columns = ["birthday","chargeoff_date","lastNoticeSent","last_pmt_date"]
        transformer.cfg.features.drop_columns = ["Loan Id","clnt_no"]

        df = transformer.handle_missing_values(sample_df.copy())
        _, _ = transformer.drop_identifiers(df)
        result = transformer.extract_date_features(df)

        assert "age" in result.columns
        assert "days_since_chargeoff" in result.columns
        assert "days_since_last_notice" in result.columns
        assert "days_since_last_payment" in result.columns
        assert result["age"].between(0, 120).all()

    def test_ratio_features_created(self, sample_df):
        from src.data_transformation.data_transformation import DataTransformation
        transformer = DataTransformation.__new__(DataTransformation)
        transformer.today = pd.Timestamp("2026-04-30")
        result = transformer.engineer_ratio_features(sample_df.copy())

        assert "balance_ratio" in result.columns
        assert "contact_rate" in result.columns
        assert "rpc_rate" in result.columns
        assert "ptp_rate" in result.columns
        # Ratios should be non-negative
        assert (result["balance_ratio"] >= 0).all()
        assert (result["contact_rate"] >= 0).all()


# ── Monitoring tests ──────────────────────────────────────────
class TestMonitoring:
    def test_psi_stable(self):
        from src.monitoring.monitoring import ModelMonitor
        monitor = ModelMonitor.__new__(ModelMonitor)
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current  = np.random.normal(0, 1, 1000)   # same distribution
        psi = monitor.compute_psi(baseline, current)
        assert psi < 0.1, f"PSI should be low for same distribution, got {psi}"

    def test_psi_drift(self):
        from src.monitoring.monitoring import ModelMonitor
        monitor = ModelMonitor.__new__(ModelMonitor)
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current  = np.random.normal(3, 1, 1000)   # shifted distribution
        psi = monitor.compute_psi(baseline, current)
        assert psi > 0.2, f"PSI should be high for shifted distribution, got {psi}"

    def test_risk_band_assignment(self):
        from src.prediction.prediction import assign_risk_band
        assert assign_risk_band(0.5)   == "Very Low (<1%)"
        assert assign_risk_band(3.0)   == "Low (1-5%)"
        assert assign_risk_band(7.0)   == "Medium (5-10%)"
        assert assign_risk_band(15.0)  == "High (10-25%)"
        assert assign_risk_band(35.0)  == "Very High (25-50%)"
        assert assign_risk_band(75.0)  == "Critical (>50%)"


# ── Run tests ─────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
