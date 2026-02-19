import streamlit as st
import pandas as pd
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from pathlib import Path
from agentic_auditor import generate_compliance_memo

# --- Configuration ---
st.set_page_config(page_title="N26 Compliance Cockpit", layout="wide", initial_sidebar_state="expanded")

# Theme tweak
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    h1, h2, h3 {
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgb.pkl"

# --- Caching Resources ---
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    # Load a sample for speed if full dataset is too large, but for this demo using full
    df = pd.read_csv(DATA_PATH)
    # Preprocess
    X = df.drop('Class', axis=1)
    y = df['Class']
    return model, X, y

try:
    model, X, y = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("🛡️ Compliance Cockpit")
st.sidebar.markdown("---")
st.sidebar.header("Audit Controls")

# Transaction Selector
transaction_id = st.sidebar.number_input(
    "Transaction ID", 
    min_value=0, 
    max_value=len(X)-1, 
    value=541,
    help="Select a transaction to audit."
)

run_audit = st.sidebar.button("Run Compliance Audit", type="primary")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Agentic Auditor**\n"
    "Powered by XGBoost, SHAP, and Gemini Pro.\n"
    "Compliant with BaFin MaRisk & EU AI Act."
)

# --- Main Logic ---

if run_audit:
    # Get Data
    row = X.iloc[[transaction_id]]
    
    # 1. Model Verdict
    probs = model.predict_proba(row)
    fraud_prob = probs[0][1]
    is_fraud = fraud_prob > 0.5
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Verdict")
        st.metric(
            label="Fraud Probability", 
            value=f"{fraud_prob:.2%}", 
            delta="High Risk" if is_fraud else "Low Risk",
            delta_color="inverse"
        )
        if is_fraud:
            st.error("🚨 FLAGGED FOR REVIEW")
        else:
            st.success("✅ TRANSACTION CLEARED")

    with col2:
        st.subheader("XAI: Feature Contribution (SHAP)")
        # SHAP Force Plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row)
        
        # Visualize
        try:
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0], row, matplotlib=True, show=False), height=200)
        except:
             # Fallback if st_shap not avail (it's a custom component usually), use matplotlib
             fig, ax = plt.subplots(figsize=(10, 3))
             shap.force_plot(explainer.expected_value, shap_values[0], row, matplotlib=True, show=False, ax=ax)
             st.pyplot(fig, bbox_inches='tight')

    st.markdown("---")
    
    # 2. Agentic Memo
    st.subheader("🤖 Agentic Compliance Memo")
    
    with st.spinner("Agent is analyzing regulations and SHAP scores..."):
        # Ensure API Key is in env
        if "GOOGLE_API_KEY" not in os.environ:
             # Try to get from st.secrets or user input (for now assume env set in terminal)
             pass
             
        memo = generate_compliance_memo(transaction_id)
        
        st.text_area("Generated Audit Trail", value=memo, height=400)
        
    st.markdown("---")
    st.caption(f"Audit generated for Transaction {transaction_id}. System ID: N26-XAI-v1.0")

else:
    st.title("Ready to Audit")
    st.write("Select a transaction ID in the sidebar and click **Run Compliance Audit**.")
    
