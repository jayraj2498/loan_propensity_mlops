-- ============================================================
-- Loan Propensity MLOps — SQL Layer
-- Database: SQLite (local dev) / AWS RDS PostgreSQL (production)
-- ============================================================

-- ── 1. RAW LOAN DATA TABLE ──────────────────────────────────
-- Stores raw ingested loan records before any transformation
CREATE TABLE IF NOT EXISTS raw_loan_data (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id             TEXT NOT NULL UNIQUE,
    clnt_no             TEXT,
    original_balance    REAL,
    current_balance     REAL,
    last_pmt_amt        REAL,
    last_pmt_date       DATE,
    birthday            DATE,
    status              INTEGER,
    last_notice_sent    DATE,
    state               TEXT,
    creditor_name       TEXT,
    chargeoff_date      DATE,
    total_portal_visit  INTEGER,
    times_dials         INTEGER,
    times_connect       INTEGER,
    times_contact       INTEGER,
    times_rpc           INTEGER,
    times_ptp           INTEGER,
    times_up            INTEGER,
    times_drop          INTEGER,
    times_lm            INTEGER,
    payment_next_30days INTEGER,          -- target label
    ingested_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── 2. FEATURE STORE TABLE ──────────────────────────────────
-- Stores engineered features ready for model training/scoring
CREATE TABLE IF NOT EXISTS feature_store (
    id                        INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id                   TEXT NOT NULL UNIQUE,
    age                       INTEGER,
    days_since_chargeoff      INTEGER,
    days_since_last_notice    INTEGER,
    days_since_last_payment   INTEGER,
    balance_ratio             REAL,
    contact_rate              REAL,
    rpc_rate                  REAL,
    ptp_rate                  REAL,
    original_balance          REAL,
    current_balance           REAL,
    times_dials               INTEGER,
    times_connect             INTEGER,
    times_contact             INTEGER,
    times_rpc                 INTEGER,
    times_ptp                 INTEGER,
    times_up                  INTEGER,
    times_drop                INTEGER,
    times_lm                  INTEGER,
    total_portal_visit        INTEGER,
    last_pmt_amt              REAL,
    state                     TEXT,
    creditor_name             TEXT,
    status                    INTEGER,
    payment_next_30days       INTEGER,
    created_at                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── 3. PREDICTIONS TABLE ─────────────────────────────────────
-- Every prediction made by the API is logged here for monitoring
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    loan_id             TEXT,
    model_version       TEXT,
    propensity_score    REAL,          -- probability 0–1
    propensity_pct      REAL,          -- score * 100
    risk_band           TEXT,          -- Very Low / Low / Medium / High / Critical
    predicted_label     INTEGER,       -- 0 or 1
    actual_label        INTEGER,       -- filled when ground truth arrives
    request_source      TEXT,          -- "api" | "batch" | "streamlit"
    predicted_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── 4. MODEL REGISTRY TABLE ──────────────────────────────────
-- Tracks every trained model version — manual model registry
CREATE TABLE IF NOT EXISTS model_registry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version   TEXT NOT NULL UNIQUE,
    algorithm       TEXT,
    accuracy        REAL,
    f1_score        REAL,
    roc_auc         REAL,
    pr_auc          REAL,
    precision_val   REAL,
    recall_val      REAL,
    train_rows      INTEGER,
    test_rows       INTEGER,
    artifact_path   TEXT,              -- S3 path or local path
    is_active       INTEGER DEFAULT 0, -- 1 = currently deployed
    trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── 5. DATA DRIFT MONITORING TABLE ───────────────────────────
-- Stores PSI scores for each feature — triggers retraining alert
CREATE TABLE IF NOT EXISTS drift_monitoring (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name    TEXT,
    psi_score       REAL,
    drift_detected  INTEGER,           -- 0 = stable, 1 = drift
    baseline_date   DATE,
    current_date    DATE,
    checked_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- ANALYTICAL QUERIES
-- ============================================================

-- Q1: Daily prediction volume and average propensity score
-- SELECT DATE(predicted_at) AS date,
--        COUNT(*) AS total_predictions,
--        AVG(propensity_pct) AS avg_propensity,
--        SUM(predicted_label) AS predicted_payers
-- FROM predictions
-- GROUP BY DATE(predicted_at)
-- ORDER BY date DESC;

-- Q2: Top 20 highest propensity accounts not yet paid
-- SELECT p.loan_id, p.propensity_pct, p.risk_band, r.state,
--        r.current_balance, r.times_ptp, r.total_portal_visit
-- FROM predictions p
-- JOIN raw_loan_data r ON p.loan_id = r.loan_id
-- WHERE p.actual_label IS NULL
--   AND p.predicted_label = 1
-- ORDER BY p.propensity_pct DESC
-- LIMIT 20;

-- Q3: Model accuracy over time (after ground truth arrives)
-- SELECT model_version,
--        COUNT(*) AS total,
--        SUM(CASE WHEN predicted_label = actual_label THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS live_accuracy
-- FROM predictions
-- WHERE actual_label IS NOT NULL
-- GROUP BY model_version
-- ORDER BY model_version DESC;

-- Q4: Feature drift summary — features with PSI > 0.1
-- SELECT feature_name, psi_score, checked_at
-- FROM drift_monitoring
-- WHERE drift_detected = 1
-- ORDER BY psi_score DESC;

-- Q5: State-wise propensity analysis
-- SELECT r.state,
--        COUNT(*) AS accounts,
--        AVG(p.propensity_pct) AS avg_propensity,
--        SUM(p.predicted_label) AS predicted_payers,
--        AVG(r.current_balance) AS avg_balance
-- FROM predictions p
-- JOIN raw_loan_data r ON p.loan_id = r.loan_id
-- GROUP BY r.state
-- ORDER BY avg_propensity DESC;
