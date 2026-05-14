"""
data_ingestion.py
─────────────────
Step 1 of the MLOps pipeline.
Reads raw Excel data → validates schema → saves to data/raw/ → stores in SQLite.

In production: replace read_excel with read from AWS S3 using boto3.
"""

import os
import sqlite3
import pandas as pd
from pathlib import Path
from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)

    # ── Validate that required columns are present ──────────
    REQUIRED_COLUMNS = [
        "Loan Id", "clnt_no", "original_balance", "current_balance",
        "last_pmt_amt", "last_pmt_date", "birthday", "status",
        "lastNoticeSent", "state", "Creditor name", "chargeoff_date",
        "total_portal_visit", "times_dials", "times_connect",
        "times_contact", "times_rpc", "times_ptp", "times_up",
        "times_drop", "times_lm", "Payment_Next30Days",
    ]

    def read_data(self) -> pd.DataFrame:
        """Read raw Excel file from configured path."""
        path = self.cfg.data.raw_data_path
        logger.info(f"Reading raw data from: {path}")
        df = pd.read_excel(path)
        logger.info(f"Raw data loaded — shape: {df.shape}")
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Check all required columns are present in dataset."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.error(f"Schema validation FAILED — missing columns: {missing}")
            raise ValueError(f"Missing columns: {missing}")
        logger.info("Schema validation PASSED ✅")
        return True

    def save_to_csv(self, df: pd.DataFrame) -> str:
        """Save raw data snapshot to data/raw/ for reproducibility."""
        os.makedirs("data/raw", exist_ok=True)
        out_path = "data/raw/loan_raw_snapshot.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Raw snapshot saved to: {out_path}")
        return out_path

    def save_to_sqlite(self, df: pd.DataFrame):
        """
        Write raw data to SQLite database (raw_loan_data table).
        In production: swap sqlite3 with psycopg2 + AWS RDS endpoint.
        """
        os.makedirs("data/external", exist_ok=True)
        conn = sqlite3.connect(self.cfg.data.sql_db_path)

        # Rename columns to match SQL schema
        df_sql = df.rename(columns={
            "Loan Id": "loan_id", "clnt_no": "clnt_no",
            "lastNoticeSent": "last_notice_sent",
            "Creditor name": "creditor_name",
            "Payment_Next30Days": "payment_next_30days",
        })

        df_sql.to_sql("raw_loan_data", conn, if_exists="replace", index=False)
        conn.close()
        logger.info(f"Data written to SQLite: {self.cfg.data.sql_db_path}")

    def run(self) -> pd.DataFrame:
        """Full ingestion pipeline: read → validate → save → store in DB."""
        logger.info("=" * 50)
        logger.info("STEP 1: DATA INGESTION STARTED")
        logger.info("=" * 50)

        df = self.read_data()
        self.validate_schema(df)
        self.save_to_csv(df)
        self.save_to_sqlite(df)

        logger.info(f"Data ingestion complete — {len(df):,} records ingested")
        return df


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()
