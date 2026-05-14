"""
model_training.py
─────────────────
Step 4 of the MLOps pipeline.
Trains 6 base models → evaluates → runs RandomizedSearchCV on top 3
→ selects best model → saves artifact → logs to MLflow.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, average_precision_score)

from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger(__name__)


def evaluate_clf(y_true, y_pred, y_proba=None) -> dict:
    """Compute all classification metrics in one call."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba if y_proba is not None else y_pred), 4),
        "pr_auc":    round(average_precision_score(y_true, y_proba if y_proba is not None else y_pred), 4),
    }


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        self.best_model = None
        self.best_metrics = {}

    # ── Define all base models ────────────────────────────────
    def _get_base_models(self) -> dict:
        rs = self.cfg.model.random_state
        return {
            "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=rs, n_jobs=-1),
            "Decision Tree":       DecisionTreeClassifier(class_weight="balanced", random_state=rs),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=rs),
            "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=500, random_state=rs),
            "K-Neighbors":         KNeighborsClassifier(n_neighbors=5),
            "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=rs),
        }

    # ── Train and evaluate all base models ───────────────────
    def train_base_models(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        """Train all models and return a leaderboard DataFrame."""
        models = self._get_base_models()
        rows = []

        for name, model in models.items():
            logger.info(f"Training: {name}")
            model.fit(X_train, y_train)

            y_tr_pred  = model.predict(X_train)
            y_te_pred  = model.predict(X_test)
            y_te_proba = model.predict_proba(X_test)[:, 1]

            tr = evaluate_clf(y_train, y_tr_pred)
            te = evaluate_clf(y_test, y_te_pred, y_te_proba)

            overfit = tr["accuracy"] - te["accuracy"] > 0.05

            rows.append({
                "Model":         name,
                "Train Acc":     tr["accuracy"],
                "Test Acc":      te["accuracy"],
                "Test F1":       te["f1"],
                "Test Precision":te["precision"],
                "Test Recall":   te["recall"],
                "Test ROC-AUC":  te["roc_auc"],
                "Test PR-AUC":   te["pr_auc"],
                "Overfit":       "⚠️ Yes" if overfit else "✅ No",
            })

            logger.info(f"  TRAIN Acc:{tr['accuracy']:.4f}  TEST Acc:{te['accuracy']:.4f}  AUC:{te['roc_auc']:.4f}")

        leaderboard = pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False).reset_index(drop=True)
        logger.info("\n" + leaderboard.to_string(index=False))
        return leaderboard

    # ── Hyperparameter tuning ────────────────────────────────
    def tune_models(self, X_processed, y_balanced) -> dict:
        """
        RandomizedSearchCV on top 3 models — optimise ROC-AUC
        with 3-fold cross-validation.
        """
        rs = self.cfg.model.random_state
        cfg = self.cfg.model

        candidates = {
            "RF":  (RandomForestClassifier(random_state=rs),
                    {k: v for k, v in vars(cfg.rf_params).items()}),
            "GB":  (GradientBoostingClassifier(random_state=rs),
                    {k: v for k, v in vars(cfg.gb_params).items()}),
            "KNN": (KNeighborsClassifier(),
                    {k: v for k, v in vars(cfg.knn_params).items()}),
        }

        best_params = {}
        for name, (model, params) in candidates.items():
            logger.info(f"Tuning {name}...")
            # Convert None strings back to Python None for max_depth
            if "max_depth" in params:
                params["max_depth"] = [None if v is None else v for v in params["max_depth"]]

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=cfg.n_iter_random_search,
                cv=cfg.cv_folds,
                scoring=cfg.scoring_metric,
                n_jobs=-1,
                random_state=rs,
                verbose=0,
            )
            search.fit(X_processed, y_balanced)
            best_params[name] = search.best_params_
            logger.info(f"  Best CV {cfg.scoring_metric}: {search.best_score_:.4f}")
            logger.info(f"  Best params: {search.best_params_}")

        return best_params

    # ── Select and save best model ────────────────────────────
    def select_best_model(self, X_train, X_test, y_train, y_test,
                          best_params: dict) -> tuple:
        """
        Re-train tuned models on full train set and pick the best by ROC-AUC.
        Random Forest is typically best: high accuracy + feature importance.
        """
        rs = self.cfg.model.random_state
        tuned = {
            "Random Forest (Tuned)":     RandomForestClassifier(**best_params["RF"], random_state=rs),
            "Gradient Boosting (Tuned)": GradientBoostingClassifier(**best_params["GB"], random_state=rs),
            "KNN (Tuned)":               KNeighborsClassifier(**best_params["KNN"]),
        }

        best_auc = 0
        best_model = None
        best_name = ""

        for name, model in tuned.items():
            model.fit(X_train, y_train)
            y_te_pred  = model.predict(X_test)
            y_te_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_clf(y_test, y_te_pred, y_te_proba)

            logger.info(f"{name} — Acc:{metrics['accuracy']:.4f}  AUC:{metrics['roc_auc']:.4f}")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_model = model
                best_name = name
                self.best_metrics = metrics

        logger.info(f"\n🏆 Winner: {best_name}  ROC-AUC: {best_auc:.4f}")
        self.best_model = best_model
        return best_model, best_name, self.best_metrics

    # ── Save model artifact ───────────────────────────────────
    def save_model(self):
        os.makedirs("artifacts/models", exist_ok=True)
        path = self.cfg.artifacts.model_path
        with open(path, "wb") as f:
            pickle.dump(self.best_model, f)
        logger.info(f"Model saved to: {path}")

    # ── Save metrics ──────────────────────────────────────────
    def save_metrics(self, model_name: str):
        os.makedirs("artifacts/metrics", exist_ok=True)
        metrics_out = {"model_name": model_name, **self.best_metrics}
        path = self.cfg.artifacts.metrics_path
        with open(path, "w") as f:
            json.dump(metrics_out, f, indent=2)
        logger.info(f"Metrics saved to: {path}")

    # ── Log to MLflow ─────────────────────────────────────────
    def log_to_mlflow(self, model_name: str):
        """
        Log model + metrics to MLflow for experiment tracking.
        In production: set MLFLOW_TRACKING_URI to your MLflow server or AWS.
        """
        try:
            import mlflow
            import mlflow.sklearn
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            mlflow.set_experiment(self.cfg.mlflow.experiment_name)
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(self.best_model.get_params())
                for k, v in self.best_metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.sklearn.log_model(self.best_model, "model")
            logger.info("MLflow logging complete ✅")
        except ImportError:
            logger.warning("MLflow not installed — skipping experiment tracking")

    # ── Full pipeline ─────────────────────────────────────────
    def run(self, X_train, X_test, y_train, y_test, X_processed, y_balanced):
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL TRAINING STARTED")
        logger.info("=" * 50)

        self.train_base_models(X_train, X_test, y_train, y_test)
        best_params = self.tune_models(X_processed, y_balanced)
        best_model, best_name, metrics = self.select_best_model(
            X_train, X_test, y_train, y_test, best_params
        )
        self.save_model()
        self.save_metrics(best_name)
        self.log_to_mlflow(best_name)

        logger.info("Model training complete ✅")
        return best_model, metrics
