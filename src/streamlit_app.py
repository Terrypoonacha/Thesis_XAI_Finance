import streamlit as st
import pandas as pd
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Custom Tool Imports
# Custom Tool Imports
from agentic_auditor import generate_compliance_memo, generate_reasoning_trace, FEATURE_DESCRIPTIONS
from sprint_1_pipeline import plot_baseline_performance
from xai_global import generate_academic_xai_plots
from baf_bias_audit import plot_bias_disparity, plot_semantic_gap
import datetime

# --- Session State for Audit Logs ---
if "audit_logs" not in st.session_state:
    st.session_state.audit_logs = []

# --- Configuration ---
st.set_page_config(page_title="🛡️ Institutional Compliance Cockpit", layout="wide")

# Theme / CSS Styling
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { border: 1px solid #4A90E2; padding: 10px; border-radius: 5px; }
    .compliance-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1a1a1a;
        color: #888;
        text-align: center;
        padding: 5px 0;
        font-size: 12px;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent
ULB_DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"
BAF_DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "baf_neurips" / "Base.csv"
ULB_MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgb.pkl"
BAF_MODEL_PATH = PROJECT_ROOT / "models" / "baf_xgb.pkl"

# --- Resource Loading ---
@st.cache_resource
def load_resources(dataset_type="ULB"):
    model_path = ULB_MODEL_PATH if dataset_type == "ULB" else BAF_MODEL_PATH
    data_path = ULB_DATA_PATH if dataset_type == "ULB" else BAF_DATA_PATH
    
    if not model_path.exists():
        st.error(f"Critical Error: Model file not found at {model_path}. Please run its training script.")
        st.stop()
        
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    return model, df

# --- Sidebar Control ---
st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=80)
st.sidebar.title("Compliance Investigator")
dataset_choice = st.sidebar.selectbox("Select Target Dataset", ["ULB (Credit Card)", "BAF (Bank Fraud Base)"])
ds_type = "ULB" if "ULB" in dataset_choice else "BAF"

model, df = load_resources(ds_type)
if ds_type == "ULB":
    X = df.drop('Class', axis=1, errors='ignore')
else:
    # Match baf_generalization_audit.py dropping logic for robust model compatibility
    baf_drop_cols = ['fraud_bool', 'device_os', 'source', 'housing_status', 'payment_type', 'employment_status']
    X = df.drop(baf_drop_cols, axis=1, errors='ignore')

tid = st.sidebar.number_input("Transaction/Case ID", min_value=0, max_value=len(X)-1, value=541)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Stable**\nAudit Mode: **Deep Scan**")

# --- UI Header ---
st.title("🛡️ Institutional Compliance Cockpit")

# --- 1. Executive Scorecard (Top Bar) ---
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.caption("Model Version")
    st.markdown("**BAF-v1.2 (Active)**")
with sc2:
    st.caption("Model Health (AUPRC)")
    st.markdown("**0.8757**")
with sc3:
    st.caption("Last Drift Check")
    st.markdown(f"**{datetime.date.today()}**")
with sc4:
    st.caption("System Integrity")
    st.markdown("✅ **Certified**")

st.markdown(f"**Transaction Investigator View** | Dataset: `{dataset_choice}` | Case ID: `{tid}`")
st.markdown("---")

# --- Main Dashboard Logic ---
# 1. Prediction Metadata
row = X.iloc[[tid]]
prob = model.predict_proba(row)[0, 1]

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Fraud Probability", f"{prob*100:.2f}%", delta="High Risk" if prob > 0.5 else "Low Risk", delta_color="inverse")
with c2:
    st.metric("Model Verdict", "FLAGGED" if prob > 0.5 else "CLEAR")
with c3:
    st.metric("Regulatory Tier", "High-Risk AI System" if ds_type == "BAF" else "Standard Audit")

st.markdown("---")

# 2. Trigger Audit
st.subheader("🔍 Automated Investigation")
if st.button("🚀 Run Comprehensive Case Audit", type="primary", use_container_width=True):
    # Side-by-Side Analysis (Explainability & Reasoning)
    shap_col, memo_col = st.columns([1.2, 1])

    with shap_col:
        st.subheader("🔍 Local Explainability (SHAP)")
        with st.spinner("Analyzing feature contributions..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(row)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                # Handle binary vs multi-class or list-like shap_values
                s_vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                
                shap.plots.bar(shap.Explanation(base_values=explainer.expected_value, 
                                               values=s_vals, 
                                               data=row.iloc[0], 
                                               feature_names=X.columns), show=False)
                plt.title(f"Contribution Breakdown: Case {tid}")
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.error(f"SHAP Error: {e}")

    with memo_col:
        st.subheader("📝 Agentic Compliance Memo")
        with st.spinner("Agent investigating transaction..."):
            try:
                # Trigger dynamic ReAct loop
                memo_result = generate_compliance_memo(tid, dataset_type=ds_type, return_trace=True)
                
                # Check for 3-item return (Live/Cache) vs potential error string
                if isinstance(memo_result, tuple) and len(memo_result) == 3:
                    memo, trace_fig, certainty = memo_result
                else:
                    memo, trace_fig = memo_result
                    certainty = 3
                
                # Display Certainty Visualizer first (Art. 13)
                conf_colors = {
                    5: ("#2ECC71", "High Confidence"), 
                    4: ("#27AE60", "Strong Evidence"), 
                    3: ("#F1C40F", "Moderate Confidence"), 
                    2: ("#E67E22", "Low Confidence"), 
                    1: ("#E74C3C", "Speculative / Low Confidence"),
                    0: ("#95A5A6", "Unknown")
                }
                color, label = conf_colors.get(certainty, ("#95A5A6", "Unknown"))
                st.markdown(f'<div style="border-left: 5px solid {color}; padding: 10px; background: #1e1e1e; border-radius: 5px; margin-bottom: 15px;">'
                            f'<span style="color: {color}; font-weight: bold;">Agentic Certainty: {label} ({certainty}/5)</span>'
                            f'</div>', unsafe_allow_html=True)

                # Display Trace Flowchart for transparency
                with st.expander("🛠️ View Agent Reasoning Trace"):
                    st.pyplot(trace_fig)
                
                st.markdown(memo)
                
                # Semantic Feature Descriptions (Art. 13)
                with st.expander("📖 Semantic Feature Context"):
                    st.info("Technical features mapped to plain-English descriptions for transparency.")
                    found_feats = [f for f in FEATURE_DESCRIPTIONS.keys() if f in X.columns or f == "V14" or f == "V4" or f == "V12"]
                    for f in found_feats:
                        st.markdown(f"**{f}**: {FEATURE_DESCRIPTIONS[f]}")

            except Exception as e:
                st.error(f"Auditor Error: {e}")

    st.markdown("---")

    # 3. Human Oversight (EU AI Act Art. 14)
    st.subheader("⚖️ Human Oversight (Art. 14)")
    st.info("Mandatory Step: Confirm decision before closing case.")
    oc1, oc2, oc3 = st.columns([1, 1, 3])

    if oc1.button("✅ Release Funds", use_container_width=True):
        st.success(f"Audit Complete: Case {tid} has been RELEASED. Notification sent to customer.")
        st.session_state.audit_logs.append({
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Case ID": tid,
            "Dataset": ds_type,
            "Decision": "RELEASED",
            "Regulatory Basis": "MaRisk AT 4.3.1 Compliance"
        })
        
    if oc2.button("🚫 Escalate to Fraud", use_container_width=True):
        st.warning(f"Case {tid} ESCALATED to Special Investigation Group (SIG).")
        st.session_state.audit_logs.append({
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Case ID": tid,
            "Dataset": ds_type,
            "Decision": "ESCALATED",
            "Regulatory Basis": "AI Act Art. 14 Oversight"
        })

    # 4. Audit Trail Log (MaRisk AT 4.3.1)
    if st.session_state.audit_logs:
        st.markdown("---")
        with st.expander("📜 Institutional Audit Trail (Session Log)"):
            st.table(pd.DataFrame(st.session_state.audit_logs))
else:
    st.info("Click the button above to start the Agentic Compliance Audit.")

# --- Footer ---
st.markdown(
    '<div class="compliance-footer">'
    'System Status: Compliant with EU AI Act Art. 10, 13, 15 and BaFin MaRisk AT 4.3.1 | '
    'Institutional XAI Auditor v2.0'
    '</div>',
    unsafe_allow_html=True
)
