"""
data_transformation.py
──────────────────────
Step 2 of the MLOps pipeline.
Cleans raw data → imputes missing values → extracts features from dates → engineers ratios.
Saves processed data to data/processed/ and feature store in SQLite.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger(__name__)


class DataTransformation:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        self.today = pd.Timestamp(self.cfg.features.reference_date)

    # ── Step 2a: Handle missing values ──────────────────────
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        last_pmt_date: 99.6% missing (accounts that never paid)
          → impute with chargeoff_date as conservative proxy
        state: 0.4% missing → fill with 'Unknown' category
        """
        before = df.isnull().sum().sum()

        df["last_pmt_date"] = df["last_pmt_date"].fillna(df["chargeoff_date"])
        df["state"] = df["state"].fillna("Unknown")

        after = df.isnull().sum().sum()
        logger.info(f"Missing values: {before} → {after} after imputation")
        return df

    # ── Step 2b: Drop identifier columns ────────────────────
    def drop_identifiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Loan Id and clnt_no are row identifiers with no predictive signal.
        Keeping them risks data leakage (model memorises IDs).
        Store them separately before dropping.
        """
        id_cols = self.cfg.features.drop_columns
        # Keep IDs for later joining with predictions output
        ids = df[id_cols].copy() if all(c in df.columns for c in id_cols) else None
        df = df.drop(columns=[c for c in id_cols if c in df.columns])
        logger.info(f"Dropped identifier columns: {id_cols}")
        return df, ids

    # ── Step 2c: Date feature extraction ────────────────────
    def extract_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw dates into interpretable numeric signals:
        - age: debtor's current age
        - days_since_chargeoff: older = harder to collect
        - days_since_last_notice: lower = more recent outreach
        - days_since_last_payment: key recency signal
        """
        df["age"] = (self.today - df["birthday"]).dt.days // 365
        df["days_since_chargeoff"] = (self.today - df["chargeoff_date"]).dt.days
        df["days_since_last_notice"] = (self.today - df["lastNoticeSent"]).dt.days
        df["days_since_last_payment"] = (self.today - df["last_pmt_date"]).dt.days

        # Drop original date columns — all info now in numeric features
        date_cols = self.cfg.features.date_columns
        df = df.drop(columns=[c for c in date_cols if c in df.columns])
        logger.info(f"Date features engineered: age, days_since_chargeoff, days_since_last_notice, days_since_last_payment")
        return df

    # ── Step 2d: Ratio feature engineering ──────────────────
    def engineer_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create efficiency ratios from collection funnel — these normalise
        for dial volume and capture collector effectiveness.
        +1 smoothing avoids division by zero.
        """
        df["balance_ratio"] = df["current_balance"] / (df["original_balance"] + 1)
        df["contact_rate"]  = df["times_connect"]   / (df["times_dials"]   + 1)
        df["rpc_rate"]      = df["times_rpc"]        / (df["times_connect"] + 1)
        df["ptp_rate"]      = df["times_ptp"]        / (df["times_contact"] + 1)
        logger.info("Ratio features created: balance_ratio, contact_rate, rpc_rate, ptp_rate")
        return df

    # ── Save processed data ──────────────────────────────────
    def save_processed(self, df: pd.DataFrame):
        os.makedirs("data/processed", exist_ok=True)
        path = self.cfg.data.processed_data_path
        df.to_csv(path, index=False)
        logger.info(f"Processed data saved to: {path} — shape: {df.shape}")

    def save_to_feature_store(self, df: pd.DataFrame):
        """Write engineered features to SQLite feature_store table."""
        conn = sqlite3.connect(self.cfg.data.sql_db_path)
        df_sql = df.rename(columns={
            "Creditor name": "creditor_name",
            "Payment_Next30Days": "payment_next_30days",
        })
        df_sql.to_sql("feature_store", conn, if_exists="replace", index=False)
        conn.close()
        logger.info("Feature store updated in SQLite")

    # ── Full transformation pipeline ────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("=" * 50)
        logger.info("STEP 2: DATA TRANSFORMATION STARTED")
        logger.info("=" * 50)

        df = self.handle_missing_values(df)
        df, _ = self.drop_identifiers(df)
        df = self.extract_date_features(df)
        df = self.engineer_ratio_features(df)
        self.save_processed(df)
        self.save_to_feature_store(df)

        logger.info(f"Transformation complete — final shape: {df.shape}")
        return df


if __name__ == "__main__":
    from src.data_ingestion.data_ingestion import DataIngestion
    raw_df = DataIngestion().run()
    transformer = DataTransformation()
    transformer.run(raw_df)
