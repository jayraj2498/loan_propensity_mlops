# 🗺️ COMPLETE PROJECT WALKTHROUGH — FROM START TO END
# Loan Payment Propensity MLOps Project
# ============================================================
# READ THIS FIRST before touching any code!
# ============================================================

# ╔══════════════════════════════════════════════════════════╗
# ║  HOW TO READ THIS GUIDE                                  ║
# ║  Each section = 1 file                                   ║
# ║  Shows: what the file does + the code inside it         ║
# ║  + where it connects to next                             ║
# ╚══════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────
# PROJECT FOLDER STRUCTURE (created once, never changes)
# ─────────────────────────────────────────────────────────

loan-propensity-mlops/
│
├── config/
│   └── config.yaml           ← STEP 1: Read this FIRST
│
├── src/
│   ├── logger.py             ← STEP 2: Logging utility
│   ├── config_reader.py      ← STEP 3: Config loader
│   │
│   ├── data_ingestion/
│   │   └── data_ingestion.py ← STEP 4: Read Excel → Save to DB
│   │
│   ├── data_transformation/
│   │   └── data_transformation.py ← STEP 5: Clean + Engineer features
│   │
│   ├── feature_engineering/
│   │   └── feature_engineering.py ← STEP 6: Encode + Scale + Balance
│   │
│   ├── model_training/
│   │   └── model_training.py ← STEP 7: Train + Tune + Save model
│   │
│   ├── model_evaluation/
│   │   └── model_evaluation.py ← STEP 8: Plots + HTML report
│   │
│   ├── prediction/
│   │   └── prediction.py     ← STEP 9: Score new accounts
│   │
│   └── monitoring/
│       └── monitoring.py     ← STEP 10: Watch model in production
│
├── pipeline/
│   └── training_pipeline.py  ← STEP 11: Runs STEPS 4-8 all at once
│
├── api/
│   └── flask_app.py          ← STEP 12: REST API (Flask)
│
├── streamlit_app/
│   └── app.py                ← STEP 13: Web dashboard
│
├── sql/
│   └── schema_and_queries.sql ← STEP 14: Database queries
│
├── tests/
│   └── unit/test_pipeline.py ← STEP 15: Test your code
│
├── Dockerfile                ← STEP 16: Package into container
├── docker-compose.yml        ← STEP 17: Run all services
│
└── infrastructure/
    └── terraform/main.tf     ← STEP 18: Deploy to AWS


# ══════════════════════════════════════════════════════════
# STEP 1 — config/config.yaml
# ══════════════════════════════════════════════════════════
# WHAT:  The brain of the project. Controls everything.
#        Every other file reads settings from here.
#        Change paths, model params, AWS settings — all here.
# WHY:   No hardcoded values anywhere. One place to change things.
# NEXT:  config_reader.py reads this file for all other modules.

# KEY SETTINGS INSIDE config.yaml:
# ---------------------------------
# data:
#   raw_data_path: "data/raw/Loan_Data_Clean_cld.xlsx"   ← where your Excel file goes
#   processed_data_path: "data/processed/loan_processed.csv"
#
# features:
#   target_column: "Payment_Next30Days"                  ← what we are predicting
#   drop_columns: ["Loan Id", "clnt_no"]                 ← IDs, not useful for ML
#
# model:
#   random_state: 42
#   test_size: 0.2                                       ← 80% train, 20% test
#
# aws:
#   region: "ap-south-1"                                 ← Mumbai region
#   s3_bucket: "loan-propensity-mlops"


# ══════════════════════════════════════════════════════════
# STEP 2 — src/logger.py
# ══════════════════════════════════════════════════════════
# WHAT:  Creates a logger that every file uses.
#        Writes messages to console AND saves to logs/ folder.
# WHY:   Instead of print(), we use logger so we have a record
#        of everything that happened.
# USED BY: Every single file in src/ imports this.

# CODE INSIDE logger.py:
# ----------------------
# import logging
#
# def get_logger(name):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#     # Console handler — shows in terminal
#     ch = logging.StreamHandler()
#     # File handler — saves to logs/2026-04-30.log
#     fh = logging.FileHandler(f"logs/{today}.log")
#     logger.addHandler(ch)
#     logger.addHandler(fh)
#     return logger
#
# HOW OTHER FILES USE IT:
# from src.logger import get_logger
# logger = get_logger(__name__)
# logger.info("Data loaded successfully")   ← appears in terminal + log file


# ══════════════════════════════════════════════════════════
# STEP 3 — src/config_reader.py
# ══════════════════════════════════════════════════════════
# WHAT:  Reads config.yaml and lets you access values with dots.
# WHY:   Makes config easy to use anywhere in the project.
# USED BY: Every module in src/ imports this.

# CODE INSIDE config_reader.py:
# -----------------------------
# import yaml
#
# def load_config(config_path="config/config.yaml"):
#     with open(config_path) as f:
#         raw = yaml.safe_load(f)       ← reads the YAML file
#     return _dict_to_ns(raw)           ← converts to dot-access object
#
# HOW OTHER FILES USE IT:
# cfg = load_config()
# print(cfg.data.raw_data_path)        → "data/raw/Loan_Data_Clean_cld.xlsx"
# print(cfg.model.random_state)        → 42
# print(cfg.aws.region)                → "ap-south-1"


# ══════════════════════════════════════════════════════════
# STEP 4 — src/data_ingestion/data_ingestion.py
# ══════════════════════════════════════════════════════════
# WHAT:  First step of pipeline.
#        Reads your Excel file → checks it has correct columns
#        → saves a CSV copy → stores in SQLite database
# WHY:   We never work directly on original data.
#        We always save a copy first (reproducibility).
# INPUT:  data/raw/Loan_Data_Clean_cld.xlsx
# OUTPUT: data/raw/loan_raw_snapshot.csv
#         data/external/loan_database.db  (SQLite)
# NEXT:   data_transformation.py reads the raw dataframe

# CODE INSIDE data_ingestion.py:
# --------------------------------
# class DataIngestion:
#
#     def read_data(self):
#         df = pd.read_excel("data/raw/Loan_Data_Clean_cld.xlsx")
#         return df                      ← returns 154,849 rows × 22 columns
#
#     def validate_schema(self, df):
#         # Checks all 22 required columns exist
#         missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
#         if missing:
#             raise ValueError(f"Missing: {missing}")   ← stops if data is wrong
#         return True
#
#     def save_to_csv(self, df):
#         df.to_csv("data/raw/loan_raw_snapshot.csv")   ← saves backup copy
#
#     def save_to_sqlite(self, df):
#         conn = sqlite3.connect("data/external/loan_database.db")
#         df.to_sql("raw_loan_data", conn)              ← stores in database
#
#     def run(self):                     ← CALL THIS to run step 4
#         df = self.read_data()
#         self.validate_schema(df)
#         self.save_to_csv(df)
#         self.save_to_sqlite(df)
#         return df                      ← passes df to next step

# HOW TO RUN:
# python src/data_ingestion/data_ingestion.py


# ══════════════════════════════════════════════════════════
# STEP 5 — src/data_transformation/data_transformation.py
# ══════════════════════════════════════════════════════════
# WHAT:  Cleans the raw data + creates new useful features.
# WHY:   Raw data has missing values, date columns, and no
#        ratio features. ML models need clean numeric data.
# INPUT:  Raw dataframe from Step 4
# OUTPUT: data/processed/loan_processed.csv
# NEXT:   feature_engineering.py reads this processed CSV

# TRANSFORMATIONS DONE:
# ─────────────────────
# 1. MISSING VALUES
#    last_pmt_date (99.6% missing) → fill with chargeoff_date
#    state (682 missing)           → fill with "Unknown"
#
# 2. DROP IDENTIFIERS
#    Loan Id, clnt_no → dropped (no predictive value)
#
# 3. DATE FEATURES (extracts numbers from dates)
#    birthday       → age = (today - birthday) / 365
#    chargeoff_date → days_since_chargeoff = today - chargeoff_date
#    lastNoticeSent → days_since_last_notice = today - lastNoticeSent
#    last_pmt_date  → days_since_last_payment = today - last_pmt_date
#
# 4. RATIO FEATURES (captures collection efficiency)
#    balance_ratio = current_balance / (original_balance + 1)
#    contact_rate  = times_connect / (times_dials + 1)
#    rpc_rate      = times_rpc / (times_connect + 1)
#    ptp_rate      = times_ptp / (times_contact + 1)

# CODE INSIDE data_transformation.py:
# ------------------------------------
# class DataTransformation:
#
#     def handle_missing_values(self, df):
#         df["last_pmt_date"] = df["last_pmt_date"].fillna(df["chargeoff_date"])
#         df["state"] = df["state"].fillna("Unknown")
#         return df
#
#     def extract_date_features(self, df):
#         today = pd.Timestamp("2026-04-30")
#         df["age"] = (today - df["birthday"]).dt.days // 365
#         df["days_since_chargeoff"] = (today - df["chargeoff_date"]).dt.days
#         df["days_since_last_notice"] = (today - df["lastNoticeSent"]).dt.days
#         df["days_since_last_payment"] = (today - df["last_pmt_date"]).dt.days
#         df.drop(["birthday","chargeoff_date","lastNoticeSent","last_pmt_date"], axis=1)
#         return df
#
#     def engineer_ratio_features(self, df):
#         df["balance_ratio"] = df["current_balance"] / (df["original_balance"] + 1)
#         df["contact_rate"]  = df["times_connect"] / (df["times_dials"] + 1)
#         df["rpc_rate"]      = df["times_rpc"] / (df["times_connect"] + 1)
#         df["ptp_rate"]      = df["times_ptp"] / (df["times_contact"] + 1)
#         return df
#
#     def run(self, df):                 ← CALL THIS to run step 5
#         df = self.handle_missing_values(df)
#         df = self.extract_date_features(df)
#         df = self.engineer_ratio_features(df)
#         df.to_csv("data/processed/loan_processed.csv")
#         return df                      ← passes cleaned df to next step

# HOW TO RUN:
# python src/data_transformation/data_transformation.py


# ══════════════════════════════════════════════════════════
# STEP 6 — src/feature_engineering/feature_engineering.py
# ══════════════════════════════════════════════════════════
# WHAT:  Encodes categories + scales numbers + fixes imbalance
#        + splits data into train and test sets
# WHY:   ML models only understand numbers.
#        Categories must be encoded. Numbers must be scaled.
#        245:1 imbalance means model won't learn minority class.
# INPUT:  Processed dataframe from Step 5
# OUTPUT: X_train, X_test, y_train, y_test (numpy arrays)
#         artifacts/models/preprocessor.pkl (saved encoder)
# NEXT:   model_training.py uses X_train, X_test to train models

# ENCODING STRATEGY:
# ──────────────────
# state           → OneHotEncoder      (59 states → 59 binary columns)
# Creditor name   → OrdinalEncoder     (234 creditors → 1 numeric column)
# status          → OrdinalEncoder     (status codes → numeric)
# skewed numbers  → PowerTransformer   (removes right skew from balances, counts)
# other numbers   → StandardScaler     (mean=0, std=1)

# CLASS IMBALANCE FIX:
# ────────────────────
# Original:  629 payers  vs  154,220 non-payers  (245:1 ratio)
# After fix: 15,000 payers  vs  15,000 non-payers (1:1 ratio)
# Method: Oversample minority (629 → 15,000 copies)
#         Undersample majority (154,220 → 15,000 random sample)

# CODE INSIDE feature_engineering.py:
# ------------------------------------
# class FeatureEngineering:
#
#     def balance_classes(self, df):
#         minority = df[df["Payment_Next30Days"] == 1]   ← 629 payers
#         majority = df[df["Payment_Next30Days"] == 0]   ← 154,220 non-payers
#         minority_up = resample(minority, n_samples=15000)  ← oversample
#         majority_down = majority.sample(n=15000)           ← undersample
#         df_balanced = pd.concat([minority_up, majority_down])
#         return X, y
#
#     def build_preprocessor(self):
#         preprocessor = ColumnTransformer([
#             ("OneHotEncoder",  OneHotEncoder(), ["state"]),
#             ("OrdinalEncoder", OrdinalEncoder(), ["Creditor name", "status"]),
#             ("PowerTransform", PowerTransformer(), skewed_columns),
#             ("StandardScaler", StandardScaler(), other_columns),
#         ])
#         return preprocessor
#
#     def run(self, df):                 ← CALL THIS to run step 6
#         X, y = self.balance_classes(df)
#         preprocessor = self.build_preprocessor()
#         X_processed = preprocessor.fit_transform(X)    ← encodes + scales
#         X_train, X_test, y_train, y_test = train_test_split(X_processed, y)
#         pickle.dump(preprocessor, open("artifacts/models/preprocessor.pkl","wb"))
#         return X_train, X_test, y_train, y_test, preprocessor

# HOW TO RUN:
# python src/feature_engineering/feature_engineering.py


# ══════════════════════════════════════════════════════════
# STEP 7 — src/model_training/model_training.py
# ══════════════════════════════════════════════════════════
# WHAT:  Trains 6 ML models → finds best hyperparameters
#        → selects winner → saves model → logs to MLflow
# WHY:   We try multiple algorithms to find the best one.
#        RandomizedSearchCV finds optimal hyperparameters.
# INPUT:  X_train, X_test, y_train, y_test from Step 6
# OUTPUT: artifacts/models/final_model.pkl
#         artifacts/metrics/model_metrics.json
# NEXT:   model_evaluation.py loads this model to make plots

# 6 MODELS TRAINED:
# ─────────────────
# 1. Random Forest      ← trees voting together
# 2. Decision Tree      ← single tree
# 3. Gradient Boosting  ← trees learning from mistakes
# 4. Logistic Regression ← linear model
# 5. K-Neighbors        ← similarity-based
# 6. AdaBoost           ← boosting weak learners

# HYPERPARAMETER TUNING (top 3 models):
# ──────────────────────────────────────
# RandomizedSearchCV tries 30 random combinations
# 3-fold cross validation on each combination
# Picks the combo with best ROC-AUC score

# CODE INSIDE model_training.py:
# --------------------------------
# class ModelTrainer:
#
#     def train_base_models(self, X_train, X_test, y_train, y_test):
#         models = {
#             "Random Forest": RandomForestClassifier(class_weight="balanced"),
#             "Gradient Boosting": GradientBoostingClassifier(),
#             "Logistic Regression": LogisticRegression(),
#             ... 3 more models
#         }
#         for name, model in models.items():
#             model.fit(X_train, y_train)          ← trains on 80% data
#             y_pred = model.predict(X_test)       ← tests on 20% data
#             accuracy = accuracy_score(y_test, y_pred)
#             # prints accuracy, F1, AUC for each model
#
#     def tune_models(self, X, y):
#         # RandomizedSearchCV for RF, GB, KNN
#         search = RandomizedSearchCV(model, params, n_iter=30, cv=3)
#         search.fit(X, y)
#         return search.best_params_              ← best hyperparameters
#
#     def select_best_model(self, ...):
#         # Retrains with best params
#         # Picks model with highest ROC-AUC
#         pickle.dump(best_model, open("artifacts/models/final_model.pkl","wb"))
#         return best_model                       ← Random Forest wins!
#
#     def run(self, X_train, X_test, y_train, y_test, X_processed, y_balanced):
#         self.train_base_models(...)             ← compare all 6
#         best_params = self.tune_models(...)     ← tune top 3
#         best_model, metrics = self.select_best_model(...)  ← pick winner
#         self.save_model()                       ← saves pickle
#         self.log_to_mlflow()                    ← experiment tracking
#         return best_model, metrics

# HOW TO RUN:
# python src/model_training/model_training.py


# ══════════════════════════════════════════════════════════
# STEP 8 — src/model_evaluation/model_evaluation.py
# ══════════════════════════════════════════════════════════
# WHAT:  Loads saved model → generates all evaluation charts
#        → creates HTML report with all results
# WHY:   Visual proof of model performance.
#        The HTML report is what you show to business team.
# INPUT:  artifacts/models/final_model.pkl (from Step 7)
#         X_test, y_test arrays
# OUTPUT: artifacts/reports/confusion_matrix.png
#         artifacts/reports/roc_curve.png
#         artifacts/reports/pr_curve.png
#         artifacts/reports/feature_importance.png
#         artifacts/reports/model_report.html  ← open this in browser!
# NEXT:   prediction.py loads same model to score new accounts

# CODE INSIDE model_evaluation.py:
# ---------------------------------
# class ModelEvaluator:
#
#     def plot_confusion_matrix(self, y_true, y_pred):
#         # Shows TP, TN, FP, FN in a grid
#         ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
#         plt.savefig("artifacts/reports/confusion_matrix.png")
#
#     def plot_roc_curve(self, y_true, y_proba):
#         # Higher AUC = better model (1.0 = perfect)
#         fpr, tpr = roc_curve(y_true, y_proba)
#         plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
#         plt.savefig("artifacts/reports/roc_curve.png")
#
#     def plot_feature_importance(self, model):
#         # Shows which features matter most
#         importances = model.feature_importances_
#         # days_since_last_payment → most important feature!
#         plt.barh(feature_names, importances)
#         plt.savefig("artifacts/reports/feature_importance.png")
#
#     def generate_html_report(self, metrics, img_paths):
#         # Creates a beautiful HTML page with all plots + metrics
#         # Open artifacts/reports/model_report.html in Chrome!
#
#     def run(self, model, preprocessor, X_test, y_test):
#         y_pred  = model.predict(X_test)
#         y_proba = model.predict_proba(X_test)[:, 1]
#         self.plot_confusion_matrix(y_test, y_pred)
#         self.plot_roc_curve(y_test, y_proba)
#         self.plot_feature_importance(model)
#         self.generate_html_report(...)

# HOW TO RUN:
# python src/model_evaluation/model_evaluation.py


# ══════════════════════════════════════════════════════════
# STEP 9 — src/prediction/prediction.py
# ══════════════════════════════════════════════════════════
# WHAT:  Loads trained model + preprocessor → scores new accounts
#        → assigns risk bands → saves Excel with propensity scores
# WHY:   This is the ACTUAL OUTPUT of the project.
#        Collections team uses this ranked list every day.
# INPUT:  New loan Excel file (any new data)
#         artifacts/models/final_model.pkl
#         artifacts/models/preprocessor.pkl
# OUTPUT: artifacts/propensity_scores.xlsx
#         (154,849 accounts ranked by % likelihood to pay)
# ALSO USED BY: flask_app.py calls predict_single() for API

# RISK BANDS ASSIGNED:
# ────────────────────
# 0–1%    → Very Low     (ignore, unlikely to pay)
# 1–5%    → Low
# 5–10%   → Medium
# 10–25%  → High
# 25–50%  → Very High
# 50–100% → Critical     (call these first!)

# CODE INSIDE prediction.py:
# ---------------------------
# class PropensityPredictor:
#
#     def __init__(self):
#         self.model = pickle.load(open("artifacts/models/final_model.pkl","rb"))
#         self.preprocessor = pickle.load(open("artifacts/models/preprocessor.pkl","rb"))
#
#     def predict_single(self, record):
#         # Used by Flask API for real-time scoring
#         df = pd.DataFrame([record])
#         X = self._preprocess_input(df)           ← same transformations as training
#         proba = self.model.predict_proba(X)[0,1] ← probability 0-1
#         pct = proba * 100                        ← convert to percentage
#         return {"propensity_score_pct": pct, "risk_band": assign_risk_band(pct)}
#
#     def predict_batch(self, df):
#         # Used for scoring all 154,849 accounts at once
#         X = self._preprocess_input(df)
#         probas = self.model.predict_proba(X)[:,1]
#         pcts = probas * 100
#         return results_dataframe
#
#     def score_portfolio(self, excel_path):
#         df = pd.read_excel(excel_path)           ← reads all accounts
#         scores = self.predict_batch(df)          ← scores them
#         output.to_excel("artifacts/propensity_scores.xlsx")  ← saves ranked list

# HOW TO RUN:
# python src/prediction/prediction.py


# ══════════════════════════════════════════════════════════
# STEP 10 — src/monitoring/monitoring.py
# ══════════════════════════════════════════════════════════
# WHAT:  Monitors model health in production
#        Detects if data has changed (drift)
#        Tracks live accuracy over time
# WHY:   Models degrade over time as data patterns change.
#        We need alerts when model needs retraining.
# RUNS:  As a scheduled daily job in production
# NEXT:  If drift detected → triggers retraining pipeline

# PSI (Population Stability Index):
# ──────────────────────────────────
# Compares training data distribution vs live data
# PSI < 0.10  → Stable, no action needed
# PSI 0.1-0.2 → Monitor closely
# PSI > 0.20  → ALERT! Data has changed → retrain model

# CODE INSIDE monitoring.py:
# ---------------------------
# class ModelMonitor:
#
#     def compute_psi(self, expected, actual):
#         # Measures how much a feature distribution has shifted
#         psi = sum((actual% - expected%) * log(actual%/expected%))
#         return psi
#
#     def check_feature_drift(self, baseline_df, current_df):
#         for each feature:
#             psi = self.compute_psi(baseline[feature], current[feature])
#             if psi > 0.10:
#                 logger.warning(f"DRIFT DETECTED in {feature}!")
#
#     def compute_live_accuracy(self):
#         # After real payments come in, check if model was right
#         live_acc = actual_vs_predicted accuracy
#         if live_acc drops more than 5%:
#             logger.warning("RETRAIN MODEL!")

# HOW TO RUN:
# python src/monitoring/monitoring.py


# ══════════════════════════════════════════════════════════
# STEP 11 — pipeline/training_pipeline.py
# ══════════════════════════════════════════════════════════
# WHAT:  The MASTER file that runs Steps 4→5→6→7→8 in order.
#        One command runs the entire ML pipeline.
# WHY:   Instead of running 5 separate files, run just this one.
# INPUT:  data/raw/Loan_Data_Clean_cld.xlsx
# OUTPUT: All artifacts (model, preprocessor, metrics, plots, report)

# CODE INSIDE training_pipeline.py:
# -----------------------------------
# def run_pipeline():
#
#     # STEP 4: Ingest data
#     ingestion = DataIngestion()
#     raw_df = ingestion.run()                     ← reads Excel, saves to DB
#
#     # STEP 5: Transform data
#     transformer = DataTransformation()
#     processed_df = transformer.run(raw_df)       ← clean + engineer features
#
#     # STEP 6: Feature engineering
#     fe = FeatureEngineering()
#     X_train, X_test, y_train, y_test, preprocessor = fe.run(processed_df)
#
#     # STEP 7: Train model
#     trainer = ModelTrainer()
#     best_model, metrics = trainer.run(X_train, X_test, y_train, y_test, ...)
#
#     # STEP 8: Evaluate model
#     evaluator = ModelEvaluator()
#     evaluator.run(best_model, preprocessor, X_test, y_test)
#
#     print("Pipeline complete! Accuracy:", metrics["accuracy"])

# HOW TO RUN:
# python pipeline/training_pipeline.py


# ══════════════════════════════════════════════════════════
# STEP 12 — api/flask_app.py
# ══════════════════════════════════════════════════════════
# WHAT:  Serves the trained model as a REST API using Flask.
#        Other systems can call /predict to get a score.
# WHY:   Makes the model available to any application.
#        Collections software, CRM, mobile apps can all call this.
# INPUT:  JSON request with loan account details
# OUTPUT: JSON response with propensity score + risk band
# RUNS ON: http://localhost:5000

# ENDPOINTS:
# ──────────
# GET  /health        → checks if API is running
# GET  /docs          → shows API documentation in browser
# GET  /model/info    → shows model accuracy metrics
# POST /predict       → scores 1 account, returns propensity %
# POST /predict/batch → scores many accounts at once

# CODE INSIDE flask_app.py:
# --------------------------
# app = Flask(__name__)
#
# @app.route("/health")
# def health():
#     return {"status": "healthy", "model_loaded": True}
#
# @app.route("/predict", methods=["POST"])
# def predict_single():
#     data = request.get_json()          ← receives loan details as JSON
#     result = predictor.predict_single(data)  ← calls prediction.py
#     return jsonify(result)             ← returns score as JSON
#     # Example response:
#     # {"propensity_score_pct": 78.23, "risk_band": "Critical (>50%)"}
#
# @app.route("/predict/batch", methods=["POST"])
# def predict_batch():
#     records = request.get_json()       ← list of loan accounts
#     scores = predictor.predict_batch(records)
#     return jsonify(scores)             ← list of scores

# HOW TO RUN:
# python api/flask_app.py
# Then open: http://localhost:5000/docs


# ══════════════════════════════════════════════════════════
# STEP 13 — streamlit_app/app.py
# ══════════════════════════════════════════════════════════
# WHAT:  Interactive web dashboard — no coding needed to use it.
#        Upload Excel, see scores, explore model performance.
# WHY:   Business users (not data scientists) need a simple UI.
#        They upload a file and see a ranked list of accounts.
# PAGES:
#   🏠 Home          → project overview + architecture
#   🔍 Single Score  → type in one account, get score instantly
#   📊 Batch Score   → upload Excel → download scored file
#   📈 Performance   → confusion matrix, ROC curve, PR curve
#   🔬 Features      → feature importance chart

# HOW TO RUN:
# streamlit run streamlit_app/app.py
# Then browser opens: http://localhost:8501


# ══════════════════════════════════════════════════════════
# STEP 14 — sql/schema_and_queries.sql
# ══════════════════════════════════════════════════════════
# WHAT:  Database tables + analytical queries for business reporting.
# WHY:   SQL lets non-Python users query predictions and insights.
# TABLES CREATED:
#   raw_loan_data    ← original Excel data
#   feature_store    ← engineered features
#   predictions      ← every prediction the API ever made
#   model_registry   ← tracks model versions
#   drift_monitoring ← PSI scores over time

# USEFUL QUERIES:
# ───────────────
# Top 20 priority accounts for collectors today:
# SELECT loan_id, propensity_pct, risk_band, current_balance
# FROM predictions
# WHERE predicted_label = 1
# ORDER BY propensity_pct DESC LIMIT 20;
#
# Model live accuracy after payments come in:
# SELECT COUNT(*) total,
#        SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END) * 100 / COUNT(*) accuracy
# FROM predictions WHERE actual_label IS NOT NULL;


# ══════════════════════════════════════════════════════════
# STEP 15 — tests/unit/test_pipeline.py
# ══════════════════════════════════════════════════════════
# WHAT:  Automated tests that verify your code works correctly.
# WHY:   Before deploying, we verify nothing is broken.
#        CI/CD runs these tests automatically on every git push.
# TESTS INCLUDED:
#   test_schema_validation_passes    ← correct data passes check
#   test_schema_validation_fails     ← wrong data raises error
#   test_missing_values_imputed      ← nulls are filled correctly
#   test_date_features_created       ← age, days_since work
#   test_ratio_features_created      ← balance_ratio etc work
#   test_psi_stable                  ← PSI low for same distribution
#   test_psi_drift                   ← PSI high for different distribution
#   test_risk_band_assignment        ← correct bands assigned

# HOW TO RUN:
# pytest tests/unit/ -v


# ══════════════════════════════════════════════════════════
# STEP 16+17 — Dockerfile + docker-compose.yml
# ══════════════════════════════════════════════════════════
# WHAT:  Packages the entire app into a Docker container.
#        Docker makes it run the same on any computer or server.
# WHY:   "Works on my machine" problem is solved by Docker.
#        AWS EC2 runs the same container as your laptop.

# docker-compose.yml runs 3 services together:
#   loan-propensity-flask-api  → http://localhost:5000
#   loan-propensity-dashboard  → http://localhost:8501
#   loan-mlflow                → http://localhost:5000 (experiment tracking)

# HOW TO RUN:
# docker-compose up --build


# ══════════════════════════════════════════════════════════
# STEP 18 — infrastructure/terraform/main.tf
# ══════════════════════════════════════════════════════════
# WHAT:  Creates AWS infrastructure automatically using code.
# WHY:   Instead of clicking in AWS console, we write code
#        that creates EC2, ECR, S3 in one command.
# CREATES:
#   AWS S3 bucket    ← stores data + model artifacts
#   AWS ECR          ← stores Docker images
#   AWS EC2 (t3.medium) ← runs the Flask API in production
#   Security groups  ← allows traffic on port 5000

# HOW TO RUN:
# cd infrastructure/terraform
# terraform init
# terraform apply


# ══════════════════════════════════════════════════════════
# .github/workflows/ci-cd.yml
# ══════════════════════════════════════════════════════════
# WHAT:  Automates deployment on every git push.
# WHY:   No manual deployment steps needed.
#        Push code → tests run → Docker builds → deploys to AWS.
# FLOW:
#   git push main
#       ↓
#   GitHub Actions starts
#       ↓
#   Job 1: Run pytest tests  (if fails → stops here)
#       ↓
#   Job 2: Build Docker image → push to AWS ECR
#       ↓
#   Job 3: SSH into EC2 → pull new image → restart container
#       ↓
#   Health check: curl http://ec2-ip:5000/health


# ══════════════════════════════════════════════════════════
# COMPLETE DATA FLOW SUMMARY
# ══════════════════════════════════════════════════════════

# Loan_Data_Clean_cld.xlsx
#         │
#         │ data_ingestion.py reads it
#         ▼
# raw dataframe (154,849 rows × 22 cols)
#         │
#         │ data_transformation.py cleans it
#         ▼
# processed dataframe (154,849 rows × 26 cols)
# [+age, +days_since_*, +balance_ratio, +contact_rate, +rpc_rate, +ptp_rate]
#         │
#         │ feature_engineering.py encodes + scales + balances
#         ▼
# X_train (24,000 rows), X_test (6,000 rows)  ← balanced 50/50
# preprocessor.pkl saved
#         │
#         │ model_training.py trains 6 models, tunes top 3
#         ▼
# final_model.pkl (Random Forest, 100% accuracy)
# model_metrics.json saved
#         │
#         │ model_evaluation.py generates plots
#         ▼
# confusion_matrix.png, roc_curve.png, feature_importance.png
# model_report.html  ← open in browser!
#         │
#         │ prediction.py scores all accounts
#         ▼
# propensity_scores.xlsx
# (154,849 accounts ranked by % chance of payment)
# Risk bands: Very Low → Low → Medium → High → Very High → Critical
#         │
#         │ flask_app.py serves as REST API
#         ▼
# POST /predict → returns propensity score in real time
#         │
#         │ Docker packages everything
#         ▼
# Container image pushed to AWS ECR
#         │
#         │ Terraform creates EC2 on AWS
#         ▼
# API live on AWS: http://your-ec2-ip:5000/predict
#         │
#         │ monitoring.py watches production
#         ▼
# Drift alerts + accuracy tracking + retrain trigger


# ══════════════════════════════════════════════════════════
# QUICK COMMAND REFERENCE
# ══════════════════════════════════════════════════════════

# SETUP (do once):
# pip install -r requirements.txt
# copy Excel file to data/raw/

# RUN STEP BY STEP (to learn):
# python src/data_ingestion/data_ingestion.py
# python src/data_transformation/data_transformation.py
# python src/feature_engineering/feature_engineering.py
# python src/model_training/model_training.py
# python src/model_evaluation/model_evaluation.py

# RUN ALL AT ONCE (after learning):
# python pipeline/training_pipeline.py

# SCORE ACCOUNTS:
# python src/prediction/prediction.py

# START API:
# python api/flask_app.py          → http://localhost:5000

# START DASHBOARD:
# streamlit run streamlit_app/app.py  → http://localhost:8501

# RUN WITH DOCKER:
# docker-compose up --build

# RUN TESTS:
# pytest tests/unit/ -v

# DEPLOY TO AWS:
# cd infrastructure/terraform && terraform apply
