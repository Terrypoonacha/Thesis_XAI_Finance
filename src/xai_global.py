import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from pathlib import Path

# --- Configuration ---
plt.rcParams.update({'font.size': 12})
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgb.pkl"
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"

# Regulatory Tagging
FEATURE_RELEVANCE = {
    "V14": "Identity - High",
    "V4": "Transactional - High",
    "V10": "Identity - Medium",
    "V12": "Transactional - High",
    "V17": "Identity - Medium",
    "V11": "Transactional - Low",
    "V16": "Transactional - High",
    "V7": "Behavioral - High"
}

def generate_academic_xai_plots():
    """Generates SHAP Summary and Regulatory Relevance bar charts."""
    print("Loading model and data for Academic XAI...")
    if not MODEL_PATH.exists():
        return None
    
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1)
    X_sample = X.sample(n=min(5000, len(X)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    
    # 1. SHAP Summary Plot
    plt.sca(ax1)
    shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
    ax1.set_title("Global Feature Importance (SHAP TreeExplainer)", fontsize=18, fontweight='bold')
    
    # 2. Regulatory Relevance Bar Chart
    top_indices = pd.Series(shap_values[0] if isinstance(shap_values, list) else shap_values[0]).abs().sort_values(ascending=False).index[:10]
    top_features = X.columns[top_indices]
    top_scores = pd.Series(shap_values[0] if isinstance(shap_values, list) else shap_values[0]).abs().sort_values(ascending=False).values[:10]
    
    relevance_tags = [FEATURE_RELEVANCE.get(f, "Behavioral - General") for f in top_features]
    
    plot_df = pd.DataFrame({
        'Feature': top_features,
        'Impact': top_scores,
        'Regulatory Category': relevance_tags
    })
    
    sns.barplot(x='Impact', y='Feature', hue='Regulatory Category', data=plot_df, palette='viridis', ax=ax2)
    ax2.set_title("Feature Impact vs. Regulatory Relevance (BaFin MaRisk AT 4.3.2)", fontsize=18, fontweight='bold')
    ax2.set_xlabel("Mean Absolute SHAP Value (Global Impact)")
    
    plt.suptitle("Sprint 2: Global Interpretability & Compliance Alignment", fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

if __name__ == "__main__":
    generate_academic_xai_plots()
    plt.show()
