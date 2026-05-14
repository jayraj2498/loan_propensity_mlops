"""
training_pipeline.py
─────────────────────
Master orchestrator for the full MLOps training pipeline.
Run this single file to execute all steps end-to-end:
  1. Data Ingestion
  2. Data Transformation
  3. Feature Engineering
  4. Model Training
  5. Model Evaluation

Usage:
  python pipeline/training_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logger import get_logger
from src.data_ingestion.data_ingestion import DataIngestion
from src.data_transformation.data_transformation import DataTransformation
from src.feature_engineering.feature_engineering import FeatureEngineering
from src.model_training.model_training import ModelTrainer
from src.model_evaluation.model_evaluation import ModelEvaluator

logger = get_logger("training_pipeline")


def run_pipeline():
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║   LOAN PROPENSITY — TRAINING PIPELINE        ║")
    logger.info("╚══════════════════════════════════════════════╝")

    try:
        # ── Step 1: Data Ingestion ──────────────────────────
        logger.info("\n▶ STEP 1 / 5 — Data Ingestion")
        ingestion = DataIngestion()
        raw_df = ingestion.run()

        # ── Step 2: Data Transformation ─────────────────────
        logger.info("\n▶ STEP 2 / 5 — Data Transformation")
        transformer = DataTransformation()
        processed_df = transformer.run(raw_df)

        # ── Step 3: Feature Engineering ─────────────────────
        logger.info("\n▶ STEP 3 / 5 — Feature Engineering")
        fe = FeatureEngineering()
        X_train, X_test, y_train, y_test, preprocessor = fe.run(processed_df)

        # Also get balanced full dataset for RandomizedSearchCV
        import pandas as pd
        processed_df2 = pd.read_csv("data/processed/loan_processed.csv")
        X_bal, y_bal = fe.balance_classes(processed_df2)
        X_bal_proc = preprocessor.transform(X_bal)

        # ── Step 4: Model Training ───────────────────────────
        logger.info("\n▶ STEP 4 / 5 — Model Training")
        trainer = ModelTrainer()
        best_model, metrics = trainer.run(X_train, X_test, y_train, y_test,
                                          X_bal_proc, y_bal)

        # ── Step 5: Model Evaluation ─────────────────────────
        logger.info("\n▶ STEP 5 / 5 — Model Evaluation")
        evaluator = ModelEvaluator()
        evaluator.run(best_model, preprocessor, X_test, y_test)

        logger.info("\n╔══════════════════════════════════════════════╗")
        logger.info("║   ✅ PIPELINE COMPLETED SUCCESSFULLY          ║")
        logger.info(f"║   Accuracy : {metrics['accuracy']*100:.2f}%                        ║")
        logger.info(f"║   ROC-AUC  : {metrics['roc_auc']:.4f}                        ║")
        logger.info("╚══════════════════════════════════════════════╝")

    except Exception as e:
        logger.error(f"Pipeline FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run_pipeline()
