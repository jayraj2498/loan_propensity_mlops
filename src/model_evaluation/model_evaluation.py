"""
model_evaluation.py
────────────────────
Step 5 of the MLOps pipeline.
Generates full evaluation report: confusion matrix, ROC curve,
PR curve, feature importance, propensity score distribution.
Saves HTML report to artifacts/reports/.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, roc_curve, roc_auc_score,
                              precision_recall_curve, average_precision_score,
                              classification_report)

from src.logger import get_logger
from src.config_reader import load_config

logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)

    def load_model(self):
        with open(self.cfg.artifacts.model_path, "rb") as f:
            return pickle.load(f)

    def load_preprocessor(self):
        with open(self.cfg.artifacts.preprocessor_path, "rb") as f:
            return pickle.load(f)

    # ── Plot confusion matrix ─────────────────────────────────
    def plot_confusion_matrix(self, y_true, y_pred, out_dir: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=["Non-Payer", "Payer"],
            colorbar=False, cmap="Blues", ax=ax
        )
        ax.set_title("Confusion Matrix", fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved: {path}")
        return path

    # ── Plot ROC curve ────────────────────────────────────────
    def plot_roc_curve(self, y_true, y_proba, out_dir: str):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "navy", lw=1, linestyle="--", label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, "roc_curve.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curve saved: {path}")
        return path

    # ── Plot Precision-Recall curve ───────────────────────────
    def plot_pr_curve(self, y_true, y_proba, out_dir: str):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(recall, precision, color="steelblue", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
        ax.axhline(y_true.mean(), color="red", linestyle="--", label="Baseline")
        ax.fill_between(recall, precision, alpha=0.1, color="steelblue")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", fontweight="bold")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, "pr_curve.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info(f"PR curve saved: {path}")
        return path

    # ── Feature importance ────────────────────────────────────
    def plot_feature_importance(self, model, preprocessor, out_dir: str):
        cfg = self.cfg.features
        try:
            ohe_names = preprocessor.named_transformers_["OneHotEncoder"] \
                            .get_feature_names_out(cfg.one_hot_columns).tolist()
            all_names = (ohe_names + list(cfg.ordinal_columns)
                         + list(cfg.power_transform_columns)
                         + list(cfg.standard_scale_columns))

            importances = model.feature_importances_
            feat_df = pd.DataFrame({"Feature": all_names[:len(importances)],
                                    "Importance": importances}) \
                        .sort_values("Importance", ascending=False).head(20)

            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ["#2ecc71" if i < 5 else "#3498db" if i < 10 else "#95a5a6"
                      for i in range(len(feat_df))]
            ax.barh(feat_df["Feature"], feat_df["Importance"],
                    color=colors, ec="black", alpha=0.85)
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 20 Feature Importances", fontweight="bold")
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            path = os.path.join(out_dir, "feature_importance.png")
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            logger.info(f"Feature importance saved: {path}")
            return path
        except AttributeError:
            logger.warning("Model does not support feature_importances_ — skipping")
            return None

    # ── Generate HTML report ──────────────────────────────────
    def generate_html_report(self, metrics: dict, img_paths: dict, out_dir: str):
        """Create a standalone HTML report with all plots embedded."""
        import base64

        def img_tag(path):
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                return f'<img src="data:image/png;base64,{data}" style="max-width:100%;margin:10px 0">'
            return ""

        html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>Loan Propensity Model Report</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:30px;background:#f5f5f5}}
  h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px}}
  h2{{color:#2980b9;margin-top:30px}}
  table{{border-collapse:collapse;width:100%;background:white;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px #ccc}}
  th{{background:#3498db;color:white;padding:10px 14px;text-align:left}}
  td{{padding:9px 14px;border-bottom:1px solid #eee}}
  tr:nth-child(even){{background:#f9f9f9}}
  .metric{{display:inline-block;background:white;border-radius:8px;padding:15px 25px;
            margin:8px;box-shadow:0 2px 8px #ccc;font-size:1.1em}}
  .metric span{{display:block;font-size:2em;font-weight:bold;color:#2980b9}}
  .img-box{{background:white;border-radius:8px;padding:15px;margin:15px 0;box-shadow:0 2px 8px #ccc}}
</style></head><body>
<h1>📊 Loan Payment Propensity — Model Report</h1>
<p><b>Winner Algorithm:</b> Random Forest (Tuned) &nbsp;|&nbsp;
   <b>Generated:</b> 2026-04-30</p>

<h2>Model Metrics</h2>
<div>
  <div class="metric">Accuracy<span>{metrics.get('accuracy',0)*100:.2f}%</span></div>
  <div class="metric">F1-Score<span>{metrics.get('f1',0):.4f}</span></div>
  <div class="metric">Precision<span>{metrics.get('precision',0):.4f}</span></div>
  <div class="metric">Recall<span>{metrics.get('recall',0):.4f}</span></div>
  <div class="metric">ROC-AUC<span>{metrics.get('roc_auc',0):.4f}</span></div>
  <div class="metric">PR-AUC<span>{metrics.get('pr_auc',0):.4f}</span></div>
</div>

<h2>Confusion Matrix</h2>
<div class="img-box">{img_tag(img_paths.get('cm'))}</div>

<h2>ROC Curve</h2>
<div class="img-box">{img_tag(img_paths.get('roc'))}</div>

<h2>Precision-Recall Curve</h2>
<div class="img-box">{img_tag(img_paths.get('pr'))}</div>

<h2>Feature Importance</h2>
<div class="img-box">{img_tag(img_paths.get('fi'))}</div>

</body></html>"""

        path = self.cfg.artifacts.report_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"HTML report saved: {path}")

    # ── Full pipeline ─────────────────────────────────────────
    def run(self, model, preprocessor, X_test, y_test):
        logger.info("=" * 50)
        logger.info("STEP 5: MODEL EVALUATION STARTED")
        logger.info("=" * 50)

        out_dir = "artifacts/reports"
        os.makedirs(out_dir, exist_ok=True)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print("\n" + classification_report(y_test, y_pred,
                                           target_names=["Non-Payer", "Payer"]))

        img_paths = {
            "cm":  self.plot_confusion_matrix(y_test, y_pred, out_dir),
            "roc": self.plot_roc_curve(y_test, y_proba, out_dir),
            "pr":  self.plot_pr_curve(y_test, y_proba, out_dir),
            "fi":  self.plot_feature_importance(model, preprocessor, out_dir),
        }

        with open(self.cfg.artifacts.metrics_path) as f:
            metrics = json.load(f)

        self.generate_html_report(metrics, img_paths, out_dir)
        logger.info("Evaluation complete ✅")
