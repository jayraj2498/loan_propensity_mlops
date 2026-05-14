# 💳 Loan Payment Propensity — End-to-End MLOps Project

[![CI/CD](https://github.com/your-username/loan-propensity-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-username/loan-propensity-mlops/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-ECR%20%2B%20EC2-orange.svg)](https://aws.amazon.com)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-blueviolet.svg)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Predict which loan accounts will make a payment in the next 30 days** — a full production-grade MLOps pipeline from raw data to cloud-deployed REST API, built the way it's done in financial services companies.

---

## 📋 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Directory Structure](#-directory-structure)
- [ML Pipeline Steps](#-ml-pipeline-steps)
- [Statistical Methods](#-statistical-methods)
- [Model Results](#-model-results)
- [API Documentation](#-api-documentation)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Docker Setup](#-docker-setup)
- [AWS Deployment](#-aws-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Monitoring](#-monitoring)
- [SQL Layer](#-sql-layer)
- [Quick Start](#-quick-start)
- [Tech Stack](#-tech-stack)

---

## 🏢 Business Problem

Collections teams at financial institutions face a critical prioritisation challenge: with thousands of delinquent loan accounts, **who do you call first?**

This project builds a **propensity scoring model** that predicts the probability (0–100%) that a loan account will make a payment in the next 30 days. By ranking accounts from highest to lowest propensity, collectors can:

- **Focus outreach** on accounts most likely to respond
- **Improve recovery rates** without increasing headcount
- **Reduce unnecessary contacts** on low-propensity accounts
- **Prioritise urgent-pay accounts** (times_up, times_ptp signals)

### Target Variable
`Payment_Next30Days = 1` if the account made at least one payment in the next 30 days, else `0`.

### Key Challenge
Severe class imbalance: only **~0.4%** of 154,849 accounts are positive class (629 payers). This makes it a realistic, hard industry problem — not a toy dataset.

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| `Loan Id` | Unique loan identifier |
| `original_balance` | Original loan amount |
| `current_balance` | Current outstanding balance |
| `last_pmt_amt` | Last payment amount made |
| `birthday` | Debtor date of birth (→ age) |
| `chargeoff_date` | Date loan was placed as delinquent |
| `lastNoticeSent` | Date of last collection notice |
| `state` | US state of debtor |
| `Creditor name` | Original lending bank |
| `status` | Disposition code from last call attempt |
| `times_dials` | Total dial attempts on account |
| `times_connect` | Times a connection was made |
| `times_contact` | Times debtor personally answered |
| `times_rpc` | Right Party Contact count |
| `times_ptp` | Promise-to-Pay count |
| `times_up` | Urgent payment commitment count |
| `total_portal_visit` | Payment link clicks via SMS/email |
| `Payment_Next30Days` | **Target variable** |

**Size:** 154,849 rows × 22 columns

---

## 🏗️ Project Architecture

```
Raw Excel Data (S3 / Local)
         │
         ▼
┌─────────────────────┐
│  1. Data Ingestion  │ ── Schema validation → SQLite/RDS
└─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  2. Data Transformation  │ ── Imputation → Date Features → Ratio Features
└──────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  3. Feature Engineering      │ ── ColumnTransformer → Oversampling → Split
└──────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  4. Model Training                                      │
│     Base models: RF, DT, GBM, LR, KNN, AdaBoost        │
│     Tuning: RandomizedSearchCV (ROC-AUC, 3-fold CV)     │
│     Tracking: MLflow experiment logging                  │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  5. Model Evaluation         │ ── CM + ROC + PR Curve + HTML Report
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  6. FastAPI REST API         │ ── /predict (real-time) + /predict/batch
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  7. Docker Container         │ ── Multi-stage build, non-root user
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  8. AWS Deployment                   │
│     ECR (image registry)             │
│     EC2 (API hosting, t3.medium)     │
│     S3 (data + model artifacts)      │
│     Terraform (IaC provisioning)     │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  9. GitHub Actions CI/CD             │
│     Test → Build → Push → Deploy     │
│     Auto-retrain workflow (manual)   │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  10. Monitoring                      │
│     PSI drift detection per feature  │
│     Live accuracy tracking           │
│     Prediction volume dashboard      │
└──────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
loan-propensity-mlops/
│
├── 📁 config/
│   └── config.yaml                  # Central config — all paths, params, AWS settings
│
├── 📁 data/
│   ├── raw/                         # Original Excel snapshot
│   ├── processed/                   # Cleaned, feature-engineered CSV
│   └── external/                    # SQLite database (local dev)
│
├── 📁 src/
│   ├── logger.py                    # Centralised logging
│   ├── config_reader.py             # YAML config loader
│   ├── data_ingestion/
│   │   └── data_ingestion.py        # Read → validate → save → DB
│   ├── data_transformation/
│   │   └── data_transformation.py   # Impute → dates → ratios
│   ├── feature_engineering/
│   │   └── feature_engineering.py   # ColumnTransformer + balancing
│   ├── model_training/
│   │   └── model_training.py        # Train → tune → evaluate → MLflow
│   ├── model_evaluation/
│   │   └── model_evaluation.py      # Plots + HTML report
│   ├── prediction/
│   │   └── prediction.py            # Single + batch scoring
│   └── monitoring/
│       └── monitoring.py            # PSI drift + accuracy tracking
│
├── 📁 pipeline/
│   └── training_pipeline.py         # ▶ Master orchestrator (run this!)
│
├── 📁 api/
│   └── main.py                      # FastAPI app — /health /predict /predict/batch
│
├── 📁 streamlit_app/
│   └── app.py                       # Interactive dashboard
│
├── 📁 notebooks/
│   └── Loan_Payment_Propensity_CaseStudy.ipynb
│
├── 📁 sql/
│   └── schema_and_queries.sql       # DB schema + analytical queries
│
├── 📁 artifacts/
│   ├── models/                      # Pickled model + preprocessor
│   ├── metrics/                     # model_metrics.json
│   └── reports/                     # Confusion matrix, ROC, HTML report
│
├── 📁 infrastructure/
│   ├── terraform/main.tf            # AWS IaC — EC2, ECR, S3
│   └── docker/Dockerfile.streamlit
│
├── 📁 tests/
│   └── unit/test_pipeline.py        # pytest unit tests
│
├── 📁 .github/workflows/
│   └── ci-cd.yml                    # GitHub Actions CI/CD
│
├── 📁 mlflow_tracking/              # Local MLflow experiment logs
├── 📁 logs/                         # Daily rotating log files
│
├── Dockerfile                       # Multi-stage FastAPI Docker image
├── docker-compose.yml               # Full stack (API + Streamlit + MLflow)
├── requirements.txt
└── README.md
```

---

## 🔄 ML Pipeline Steps

### Step 1 — Data Ingestion
- Reads raw Excel file from `data/raw/`
- Validates schema: 22 required columns checked
- Saves raw snapshot to `data/raw/loan_raw_snapshot.csv`
- Writes to SQLite `raw_loan_data` table (or AWS RDS in prod)

### Step 2 — Data Transformation
| Issue | Solution |
|-------|----------|
| `last_pmt_date` — 99.6% missing | Imputed with `chargeoff_date` |
| `state` — 682 nulls | Filled with `'Unknown'` |
| Raw dates | Extracted: `age`, `days_since_chargeoff`, `days_since_last_notice`, `days_since_last_payment` |
| Raw counts | Engineered: `balance_ratio`, `contact_rate`, `rpc_rate`, `ptp_rate` |

### Step 3 — Feature Engineering
- **ColumnTransformer** with 4 transformer groups:
  - `OneHotEncoder` → `state` (nominal, 59 categories)
  - `OrdinalEncoder` → `Creditor name`, `status` (high cardinality)
  - `PowerTransformer (Yeo-Johnson)` → 11 skewed features
  - `StandardScaler` → remaining numeric features
- **Class balancing**: Oversample minority (629 → 15,000) + subsample majority (154,220 → 15,000)
- **Stratified 80/20 train-test split**

### Step 4 — Model Training
- 6 base models trained: Random Forest, Decision Tree, Gradient Boosting, Logistic Regression, KNN, AdaBoost
- Top 3 tuned with `RandomizedSearchCV` (30 iterations, 3-fold CV, scoring=ROC-AUC)
- All experiments tracked in MLflow

### Step 5 — Model Evaluation
Generates: Confusion Matrix, ROC Curve, Precision-Recall Curve, Feature Importance, HTML report

---

## 📐 Statistical Methods

### Feature Selection — 2 Methods

**1. Mann-Whitney U Test (Numerical Features)**
- Tests whether distributions differ significantly between payers and non-payers
- H₀: Feature distribution is the same across classes
- p < 0.05 → Feature is retained

**2. Chi-Square Test (Categorical Features)**
- Tests independence between categorical feature and target
- H₀: Feature is independent of `Payment_Next30Days`
- p < 0.05 → Feature is associated with target

**3. PSI (Population Stability Index) — for monitoring**
- PSI < 0.10 → Stable
- PSI 0.10–0.20 → Minor shift
- PSI > 0.20 → Major drift → trigger retraining

---

## 🏆 Model Results

| Model | Test Accuracy | F1 | Precision | Recall | ROC-AUC |
|-------|-------------|-----|-----------|--------|---------|
| **Random Forest (Tuned)** ⭐ | **100%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Gradient Boosting (Tuned) | 99.8% | 0.9980 | 0.9981 | 0.9979 | 0.9999 |
| KNN (Tuned) | 99.5% | 0.9950 | 0.9952 | 0.9948 | 0.9998 |
| Random Forest (Base) | 99.7% | 0.9970 | 0.9971 | 0.9969 | 0.9998 |
| Gradient Boosting (Base) | 99.2% | 0.9920 | 0.9920 | 0.9920 | 0.9997 |
| AdaBoost | 97.1% | 0.9710 | 0.9711 | 0.9710 | 0.9980 |
| Logistic Regression | 88.4% | 0.8830 | 0.8834 | 0.8826 | 0.9540 |

> Results on balanced test set. Winner: **Random Forest (Tuned)** — highest ROC-AUC + built-in interpretability

### Why Random Forest Won
- ✅ Highest accuracy and ROC-AUC
- ✅ Feature importance for business explainability
- ✅ No significant overfitting (train ≈ test)
- ✅ `class_weight='balanced'` handles residual imbalance
- ✅ Fast at inference — critical for real-time API

### Top Features Driving Propensity
1. `days_since_last_payment` — recent payers are most likely to pay again
2. `balance_ratio` — proportion of original balance still outstanding
3. `ptp_rate` — promise-to-pay rate per contact
4. `rpc_rate` — right party contact rate
5. `total_portal_visit` — digital engagement strongly predicts payment

---

## 🚀 API Documentation

The Flask app runs at `http://localhost:5000`. Full docs at `/docs`.

### Run Flask API

```bash
# Development
python api/flask_app.py

# Production (gunicorn — already set in Dockerfile)
gunicorn api.flask_app:app --bind 0.0.0.0:5000 --workers 2
```

### Endpoints

#### `GET /health`
```json
{"status": "healthy", "model_loaded": true, "timestamp": "2026-04-30T10:00:00"}
```

#### `POST /predict`
Score a single loan account.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_id":          "L001",
    "original_balance": 5000.0,
    "current_balance":  4800.0,
    "birthday":         "1985-06-15",
    "chargeoff_date":   "2022-01-01",
    "lastNoticeSent":   "2026-01-15",
    "status":           1,
    "Creditor name":    "CAPITAL ONE",
    "state":            "TX",
    "times_dials":      10,
    "times_ptp":        1
  }'
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "loan_id":              "L001",
    "propensity_score":     0.7823,
    "propensity_score_pct": 78.23,
    "risk_band":            "Critical (>50%)",
    "predicted_label":      1,
    "predicted_at":         "2026-04-30T10:00:00"
  }
}
```

#### `POST /predict/batch`
Score up to 10,000 accounts in one call. Accepts a JSON array.

#### `GET /model/info`
Returns current model algorithm and training metrics.

#### `GET /docs`
Interactive HTML documentation page.

---

## 📊 Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py
```

Available at `http://localhost:8501`

Pages:
- 🏠 **Home** — Architecture overview and KPI cards
- 🔍 **Single Account Scoring** — Interactive form with gauge chart
- 📊 **Batch Portfolio Scoring** — Upload Excel, download ranked scores
- 📈 **Model Performance** — All evaluation charts
- 🔬 **Feature Importance** — Explainability dashboard

---

## 🐳 Docker Setup

### Quick Start
```bash
# Build and run the full stack
docker-compose up --build

# Services:
#   FastAPI    → http://localhost:8000
#   Streamlit  → http://localhost:8501
#   MLflow     → http://localhost:5000
```

### API only
```bash
docker build -t loan-propensity-api .
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts loan-propensity-api
```

---

## ☁️ AWS Deployment

### Prerequisites
- AWS CLI configured (`aws configure`)
- Terraform installed
- EC2 key pair created

### 1. Provision infrastructure with Terraform
```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
# Outputs: EC2 IP, ECR URL, S3 bucket name
```

### 2. Push image to ECR
```bash
# Login to ECR
aws ecr get-login-password --region ap-south-1 \
  | docker login --username AWS \
    --password-stdin <your-account-id>.dkr.ecr.ap-south-1.amazonaws.com

# Build and push
docker build -t loan-propensity-api .
docker tag loan-propensity-api:latest \
  <your-ecr-url>/loan-propensity-api:latest
docker push <your-ecr-url>/loan-propensity-api:latest
```

### 3. Deploy on EC2
```bash
ssh -i your-key.pem ec2-user@<ec2-ip>
docker pull <your-ecr-url>/loan-propensity-api:latest
docker run -d --name loan-api -p 8000:8000 \
  -v ~/artifacts:/app/artifacts \
  <your-ecr-url>/loan-propensity-api:latest
```

### Required GitHub Secrets
| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `EC2_HOST` | Public IP of EC2 instance |
| `EC2_SSH_KEY` | Private SSH key for EC2 |
| `ECR_REGISTRY` | ECR registry URL |

---

## ⚙️ CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push to `main`:

```
Push to main
    │
    ▼
[Job 1] 🧪 Test
    - Install dependencies
    - Run pytest with coverage
    - Upload coverage to Codecov
    │
    ▼
[Job 2] 🐳 Build & Push
    - Configure AWS credentials
    - Login to ECR
    - Build Docker image (tagged with git SHA)
    - Push to ECR
    │
    ▼
[Job 3] 🚀 Deploy
    - SSH into EC2
    - Pull latest image
    - Stop old container
    - Start new container
    - Health check
```

Manual trigger available for model retraining (`workflow_dispatch`).

---

## 📡 Monitoring

### Data Drift (PSI)
```python
from src.monitoring.monitoring import ModelMonitor
monitor = ModelMonitor()
drift_report = monitor.check_feature_drift(baseline_df, current_df)
```

### Prediction Health
```python
health = monitor.prediction_health_check()
# Returns: total predictions, avg propensity, risk band distribution
```

### Live Accuracy
Once ground truth payment data arrives, live accuracy is computed automatically:
```python
live_acc = monitor.compute_live_accuracy()
# Triggers alert if accuracy drops >5% from training accuracy
```

All predictions are logged to the `predictions` SQL table for monitoring.

---

## 🗃️ SQL Layer

```sql
-- Top priority accounts for collectors today
SELECT loan_id, propensity_pct, risk_band
FROM predictions
WHERE predicted_label = 1 AND actual_label IS NULL
ORDER BY propensity_pct DESC
LIMIT 50;

-- Model live accuracy
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN predicted_label = actual_label THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS live_accuracy
FROM predictions
WHERE actual_label IS NOT NULL;
```

See `sql/schema_and_queries.sql` for full schema and 5 analytical queries.

---

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-username/loan-propensity-mlops.git
cd loan-propensity-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your data file
cp Loan_Data_Clean_cld.xlsx data/raw/

# 5. Run the full training pipeline
python pipeline/training_pipeline.py

# 6. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Start the dashboard
streamlit run streamlit_app/app.py

# 8. Run tests
pytest tests/unit/ -v
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| ML Framework | scikit-learn 1.4 |
| Data | pandas, numpy, scipy |
| API | FastAPI + uvicorn |
| Dashboard | Streamlit |
| Experiment Tracking | MLflow |
| Database | SQLite (dev) / AWS RDS PostgreSQL (prod) |
| Containerisation | Docker + docker-compose |
| Cloud | AWS EC2, ECR, S3 |
| Infrastructure as Code | Terraform |
| CI/CD | GitHub Actions |
| Monitoring | PSI drift detection, custom SQL monitoring |
| Testing | pytest + pytest-cov |
| Visualisation | matplotlib, seaborn |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built as a professional end-to-end MLOps project demonstrating real-world financial services ML engineering practices.

---

*⭐ Star this repo if you found it useful!*
