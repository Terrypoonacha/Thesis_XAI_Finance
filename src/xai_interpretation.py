import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgb.pkl"
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"

FEATURE_DESCRIPTIONS = {
    "V14": "Transaction behavior factor often associated with identity spoofing or account takeover in fraud clusters.",
    "V4": "Feature capturing transaction velocity and frequency anomalies.",
    "V10": "Indicator of card-present vs. card-not-present behavioral anomalies.",
    "V12": "Time-based correlation factor linked to rapid-fire transaction sequences."
}

def generate_global_shap_plot():
    """Generates a SHAP summary plot and provides regulatory context."""
    print("Loading model and data for XAI...")
    if not MODEL_PATH.exists():
        return None, "Error: Model file not found. Please run Training first."
    
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1)
    
    # Subsample for speed
    X_sample = X.sample(n=min(5000, len(X)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Create plot
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
    plt.title("Global Feature Importance (SHAP)", fontsize=16)
    
    # Generate textual description for Compliance
    print("\n--- Compliance Interpretation (BaFin MaRisk AT 4.3.2) ---")
    top_features = X.columns[pd.Series(shap_values[0]).abs().sort_values(ascending=False).index[:4]]
    
    context_text = "Analysis of top driving features for transparency:\n"
    for feat in top_features:
        desc = FEATURE_DESCRIPTIONS.get(feat, "General behavioral feature used for fraud detection.")
        context_text += f"- {feat}: {desc}\n"
        
    print(context_text)
    return fig, context_text

if __name__ == "__main__":
    generate_global_shap_plot()
    plt.show()
