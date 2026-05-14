"""
app.py — Streamlit Dashboard
─────────────────────────────
Interactive web app for:
  1. Single account propensity scoring
  2. Batch portfolio scoring with Excel upload
  3. Model performance metrics dashboard
  4. Feature importance visualisation
  5. Risk band distribution charts

Run:  streamlit run streamlit_app/app.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Propensity Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS styling ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card{background:white;padding:20px;border-radius:10px;
                 box-shadow:0 2px 8px rgba(0,0,0,0.1);text-align:center}
    .metric-value{font-size:2.2em;font-weight:bold;color:#2980b9}
    .metric-label{color:#7f8c8d;font-size:0.9em}
    .risk-critical{background:#e74c3c;color:white;padding:4px 10px;border-radius:5px}
    .risk-high{background:#e67e22;color:white;padding:4px 10px;border-radius:5px}
    .risk-medium{background:#f39c12;color:white;padding:4px 10px;border-radius:5px}
    .risk-low{background:#27ae60;color:white;padding:4px 10px;border-radius:5px}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open("artifacts/models/final_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("artifacts/models/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_metrics():
    try:
        with open("artifacts/metrics/model_metrics.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

model, preprocessor = load_artifacts()
metrics = load_metrics()

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank.png", width=80)
st.sidebar.title("💳 Loan Propensity")
st.sidebar.markdown("**MLOps Dashboard**")
page = st.sidebar.radio("Navigation", [
    "🏠 Home / Overview",
    "🔍 Single Account Scoring",
    "📊 Batch Portfolio Scoring",
    "📈 Model Performance",
    "🔬 Feature Importance",
])
st.sidebar.markdown("---")
model_loaded = model is not None
st.sidebar.markdown(f"**Model Status:** {'✅ Loaded' if model_loaded else '❌ Not Loaded'}")
if metrics:
    st.sidebar.markdown(f"**Accuracy:** {metrics.get('accuracy',0)*100:.2f}%")
    st.sidebar.markdown(f"**ROC-AUC:** {metrics.get('roc_auc',0):.4f}")


# ── Risk band helper ──────────────────────────────────────────
def get_risk_band(pct):
    if pct < 1:   return "Very Low (<1%)",    "#27ae60"
    if pct < 5:   return "Low (1-5%)",        "#2ecc71"
    if pct < 10:  return "Medium (5-10%)",    "#f39c12"
    if pct < 25:  return "High (10-25%)",     "#e67e22"
    if pct < 50:  return "Very High (25-50%)","#e74c3c"
    return "Critical (>50%)",                  "#c0392b"

def preprocess_input(df, preprocessor):
    """Apply same feature engineering as training."""
    today = pd.Timestamp("2026-04-30")
    df = df.copy()
    df["last_pmt_date"] = df["last_pmt_date"].fillna(df["chargeoff_date"])
    df["state"] = df["state"].fillna("Unknown")
    df["age"]                     = (today - df["birthday"]).dt.days // 365
    df["days_since_chargeoff"]    = (today - df["chargeoff_date"]).dt.days
    df["days_since_last_notice"]  = (today - df["lastNoticeSent"]).dt.days
    df["days_since_last_payment"] = (today - df["last_pmt_date"]).dt.days
    df["balance_ratio"] = df["current_balance"] / (df["original_balance"] + 1)
    df["contact_rate"]  = df["times_connect"]   / (df["times_dials"] + 1)
    df["rpc_rate"]      = df["times_rpc"]        / (df["times_connect"] + 1)
    df["ptp_rate"]      = df["times_ptp"]        / (df["times_contact"] + 1)
    drop = ["Loan Id","clnt_no","birthday","chargeoff_date","lastNoticeSent",
            "last_pmt_date","Payment_Next30Days"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return preprocessor.transform(df)


# ══════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home / Overview":
    st.title("💳 Loan Payment Propensity — MLOps Dashboard")
    st.markdown("Predict which loan accounts are likely to make a payment in the next 30 days.")
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("Accuracy",   f"{metrics.get('accuracy',0)*100:.2f}%"),
        ("F1 Score",   f"{metrics.get('f1',0):.4f}"),
        ("ROC-AUC",    f"{metrics.get('roc_auc',0):.4f}"),
        ("Precision",  f"{metrics.get('precision',0):.4f}"),
        ("Recall",     f"{metrics.get('recall',0):.4f}"),
    ]
    for col, (label, value) in zip([col1,col2,col3,col4,col5], kpis):
        col.metric(label, value)

    st.markdown("---")
    st.subheader("📐 Project Architecture")
    st.markdown("""
    ```
    Raw Excel Data
         │
         ▼
    [1] Data Ingestion ──────────── SQLite / AWS RDS
         │
         ▼
    [2] Data Transformation ──────── Imputation + Date Features + Ratios
         │
         ▼
    [3] Feature Engineering ─────── ColumnTransformer + Oversampling
         │
         ▼
    [4] Model Training ──────────── RF + GB + KNN + MLflow Tracking
         │
         ▼
    [5] Model Evaluation ────────── Confusion Matrix + ROC + PR Curve
         │
         ▼
    [6] FastAPI ─────────────────── REST API (/predict)
         │
         ▼
    [7] Docker ──────────────────── Containerised
         │
         ▼
    [8] AWS ECR + EC2 ────────────── Cloud Deployment
         │
         ▼
    [9] GitHub Actions CI/CD ─────── Auto-deploy on push
         │
         ▼
    [10] Monitoring ─────────────── PSI Drift + Accuracy Tracking
    ```
    """)


# ══════════════════════════════════════════════════════════════
# PAGE 2: SINGLE ACCOUNT SCORING
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Single Account Scoring":
    st.title("🔍 Single Account Propensity Score")
    st.markdown("Enter loan account details to get a real-time propensity score.")

    if not model_loaded:
        st.error("⚠️ Model not loaded. Please run the training pipeline first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("💰 Loan Details")
        original_balance = st.number_input("Original Balance ($)", 0.0, 1e7, 5000.0)
        current_balance  = st.number_input("Current Balance ($)", 0.0, 1e7, 4800.0)
        last_pmt_amt     = st.number_input("Last Payment Amount ($)", 0.0, 1e6, 0.0)
        creditor_name    = st.text_input("Creditor Name", "CAPITAL ONE")
        status           = st.number_input("Status Code", 0, 999, 1)

    with col2:
        st.subheader("👤 Demographics")
        birthday       = st.date_input("Date of Birth", pd.Timestamp("1985-06-15"))
        state          = st.selectbox("State", ["TX","CA","FL","NY","IL","OH","PA","Unknown"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        chargeoff_date = st.date_input("Chargeoff Date", pd.Timestamp("2022-01-01"))
        last_notice    = st.date_input("Last Notice Sent", pd.Timestamp("2026-01-15"))

    with col3:
        st.subheader("📞 Collection Activity")
        times_dials   = st.number_input("Times Dialled", 0, 1000, 10)
        times_connect = st.number_input("Times Connected", 0, 1000, 3)
        times_contact = st.number_input("Times Contacted", 0, 500, 1)
        times_rpc     = st.number_input("Times RPC", 0, 500, 1)
        times_ptp     = st.number_input("Times PTP", 0, 100, 1)
        times_up      = st.number_input("Times Urgent Pay", 0, 100, 0)
        times_drop    = st.number_input("Times Drop", 0, 500, 2)
        times_lm      = st.number_input("Times Left Message", 0, 500, 3)
        portal_visits = st.number_input("Portal Visits", 0, 100, 0)

    if st.button("🎯 Calculate Propensity Score", type="primary"):
        record = pd.DataFrame([{
            "Loan Id": "TEST001", "clnt_no": "C001",
            "original_balance": original_balance, "current_balance": current_balance,
            "last_pmt_amt": last_pmt_amt, "last_pmt_date": None,
            "birthday": pd.Timestamp(birthday), "status": status,
            "lastNoticeSent": pd.Timestamp(last_notice), "state": state,
            "Creditor name": creditor_name, "chargeoff_date": pd.Timestamp(chargeoff_date),
            "total_portal_visit": portal_visits, "times_dials": times_dials,
            "times_connect": times_connect, "times_contact": times_contact,
            "times_rpc": times_rpc, "times_ptp": times_ptp,
            "times_up": times_up, "times_drop": times_drop, "times_lm": times_lm,
            "Payment_Next30Days": 0,
        }])

        X = preprocess_input(record, preprocessor)
        proba = float(model.predict_proba(X)[0, 1])
        pct   = round(proba * 100, 2)
        band, color = get_risk_band(pct)

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        r1, r2, r3 = st.columns(3)
        r1.metric("Propensity Score", f"{pct:.2f}%")
        r2.metric("Risk Band", band)
        r3.metric("Predicted Label", "✅ Likely to Pay" if proba >= 0.5 else "❌ Unlikely to Pay")

        # Gauge chart
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.barh(["Score"], [pct], color=color, height=0.5)
        ax.barh(["Score"], [100 - pct], left=[pct], color="#ecf0f1", height=0.5)
        ax.set_xlim(0, 100); ax.set_xlabel("Propensity %")
        ax.set_title(f"Propensity: {pct:.2f}%", fontweight="bold")
        st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════
# PAGE 3: BATCH SCORING
# ══════════════════════════════════════════════════════════════
elif page == "📊 Batch Portfolio Scoring":
    st.title("📊 Batch Portfolio Scoring")
    st.markdown("Upload an Excel file with loan accounts to score the entire portfolio.")

    if not model_loaded:
        st.error("⚠️ Model not loaded. Please run the training pipeline first.")
        st.stop()

    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ File loaded: {df.shape[0]:,} accounts")
        st.dataframe(df.head())

        if st.button("🚀 Score Portfolio", type="primary"):
            with st.spinner("Scoring all accounts..."):
                for col in ["birthday","chargeoff_date","lastNoticeSent","last_pmt_date"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")

                X = preprocess_input(df.copy(), preprocessor)
                probas = model.predict_proba(X)[:, 1]
                pcts   = (probas * 100).round(4)

                df["Propensity_Score_%"] = pcts
                df["Risk_Band"] = [get_risk_band(p)[0] for p in pcts]
                df["Predicted_Label"] = (probas >= 0.5).astype(int)
                df_out = df.sort_values("Propensity_Score_%", ascending=False)

            st.success(f"✅ Scoring complete!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Accounts", f"{len(df_out):,}")
            c2.metric("Avg Propensity", f"{pcts.mean():.2f}%")
            c3.metric("Predicted Payers", f"{(probas>=0.5).sum():,}")
            c4.metric("Critical (>50%)", f"{(pcts>50).sum():,}")

            st.subheader("Top 20 Priority Accounts")
            st.dataframe(df_out[["Loan Id","current_balance","state",
                                  "Propensity_Score_%","Risk_Band"]].head(20))

            # Risk band distribution
            band_counts = df_out["Risk_Band"].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(band_counts.index, band_counts.values, color="#3498db", ec="black")
            ax.set_title("Risk Band Distribution", fontweight="bold")
            ax.set_ylabel("Account Count")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            # Download button
            csv = df_out.to_csv(index=False)
            st.download_button("⬇️ Download Scored File", csv,
                                "propensity_scores.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    if metrics:
        st.subheader("Evaluation Metrics — Random Forest (Tuned)")
        cols = st.columns(6)
        for col, (k, label) in zip(cols, [("accuracy","Accuracy"),("f1","F1"),
                                           ("precision","Precision"),("recall","Recall"),
                                           ("roc_auc","ROC-AUC"),("pr_auc","PR-AUC")]):
            val = metrics.get(k, 0)
            col.metric(label, f"{val*100:.2f}%" if k=="accuracy" else f"{val:.4f}")

    st.markdown("---")
    # Show saved plots if they exist
    for title, path in [("Confusion Matrix", "artifacts/reports/confusion_matrix.png"),
                         ("ROC Curve", "artifacts/reports/roc_curve.png"),
                         ("Precision-Recall Curve", "artifacts/reports/pr_curve.png")]:
        if os.path.exists(path):
            st.subheader(title)
            st.image(path)
        else:
            st.info(f"Run training pipeline to generate {title}")


# ══════════════════════════════════════════════════════════════
# PAGE 5: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Feature Importance":
    st.title("🔬 Feature Importance & Interpretability")

    fi_path = "artifacts/reports/feature_importance.png"
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Top 20 Features — Random Forest")
        st.markdown("""
        **How to read this chart:**
        - 🟢 **Green bars** = Top 5 most important features
        - 🔵 **Blue bars** = Features 6-10
        - ⚪ **Grey bars** = Features 11-20

        **Key drivers of payment propensity:**
        1. `days_since_last_payment` — recent payers are most likely to pay again
        2. `balance_ratio` — how much of the loan remains outstanding
        3. `ptp_rate` — accounts that promised to pay have higher follow-through
        4. `rpc_rate` — right party contacts indicate genuine engagement
        5. `total_portal_visit` — digital engagement strongly predicts payment
        """)
    else:
        st.info("Run the training pipeline to generate feature importance.")
