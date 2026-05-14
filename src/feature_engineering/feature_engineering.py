"""
feature_engineering.py
───────────────────────
Step 3 of the MLOps pipeline.
Builds the ColumnTransformer preprocessor, handles class imbalance,
splits data, and saves the fitted preprocessor as a pickle artifact.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import (OneHotEncoder, StandardScaler,
                                    OrdinalEncoder, PowerTransformer)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger(__name__)


class FeatureEngineering:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        self.preprocessor = None

    # ── Balance target classes ────────────────────────────────
    def balance_classes(self, df: pd.DataFrame) -> tuple:
        """
        Severe imbalance: 245:1 (non-payers : payers).
        Strategy: oversample minority to 15,000 + subsample majority to 15,000.
        This ensures the model learns payment patterns effectively.
        """
        target = self.cfg.features.target_column
        rs = self.cfg.resampling

        minority = df[df[target] == 1]
        majority = df[df[target] == 0]

        logger.info(f"Before balancing — Payers: {len(minority):,}  Non-payers: {len(majority):,}")

        minority_up   = resample(minority, replace=True,
                                 n_samples=rs.minority_target_size,
                                 random_state=rs.random_state)
        majority_down = majority.sample(n=rs.majority_target_size,
                                        random_state=rs.random_state)

        df_balanced = pd.concat([minority_up, majority_down]).sample(
            frac=1, random_state=rs.random_state
        )

        X = df_balanced.drop(columns=[target])
        y = df_balanced[target].values

        logger.info(f"After balancing — Total: {len(df_balanced):,}  (50/50 split)")
        return X, y

    # ── Build ColumnTransformer ───────────────────────────────
    def build_preprocessor(self) -> ColumnTransformer:
        """
        ColumnTransformer applies correct encoding/scaling to each feature group:
        ┌──────────────────────┬───────────────────────────┬────────────────────────────────────┐
        │ Feature group        │ Transformer               │ Reason                             │
        ├──────────────────────┼───────────────────────────┼────────────────────────────────────┤
        │ state                │ OneHotEncoder             │ Nominal; 59 unique categories      │
        │ Creditor name, status│ OrdinalEncoder            │ High cardinality / ordinal codes   │
        │ Skewed numerics      │ PowerTransformer (YJ)     │ Right-skewed; Yeo-Johnson handles  │
        │                      │                           │ zeros and negatives (unlike Box-Cox)│
        │ Remaining numerics   │ StandardScaler            │ Zero mean, unit variance           │
        └──────────────────────┴───────────────────────────┴────────────────────────────────────┘
        """
        cfg = self.cfg.features

        preprocessor = ColumnTransformer(transformers=[
            ("OneHotEncoder",
             OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             cfg.one_hot_columns),

            ("OrdinalEncoder",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             cfg.ordinal_columns),

            ("PowerTransformer",
             Pipeline([("pt", PowerTransformer(method="yeo-johnson"))]),
             cfg.power_transform_columns),

            ("StandardScaler",
             StandardScaler(),
             cfg.standard_scale_columns),
        ])

        logger.info("ColumnTransformer preprocessor built ✅")
        return preprocessor

    # ── Fit and transform ────────────────────────────────────
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> tuple:
        """Fit preprocessor on training data and transform."""
        self.preprocessor = self.build_preprocessor()
        X_processed = self.preprocessor.fit_transform(X)
        logger.info(f"Preprocessing complete — feature matrix shape: {X_processed.shape}")
        return X_processed, y

    # ── Train/test split ─────────────────────────────────────
    def split(self, X_processed: np.ndarray, y: np.ndarray) -> tuple:
        """Stratified 80/20 split — preserves class ratio in both sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y,
            test_size=self.cfg.model.test_size,
            random_state=self.cfg.model.random_state,
            stratify=y
        )
        logger.info(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
        return X_train, X_test, y_train, y_test

    # ── Save preprocessor artifact ───────────────────────────
    def save_preprocessor(self):
        os.makedirs("artifacts/models", exist_ok=True)
        path = self.cfg.artifacts.preprocessor_path
        with open(path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        logger.info(f"Preprocessor saved to: {path}")

    # ── Full pipeline ────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> tuple:
        logger.info("=" * 50)
        logger.info("STEP 3: FEATURE ENGINEERING STARTED")
        logger.info("=" * 50)

        X, y = self.balance_classes(df)
        X_processed, y = self.fit_transform(X, y)
        X_train, X_test, y_train, y_test = self.split(X_processed, y)
        self.save_preprocessor()

        logger.info("Feature engineering complete ✅")
        return X_train, X_test, y_train, y_test, self.preprocessor


if __name__ == "__main__":
    from src.data_ingestion.data_ingestion import DataIngestion
    from src.data_transformation.data_transformation import DataTransformation
    raw_df = DataIngestion().run()
    processed_df = DataTransformation().run(raw_df)
    fe = FeatureEngineering()
    fe.run(processed_df)
