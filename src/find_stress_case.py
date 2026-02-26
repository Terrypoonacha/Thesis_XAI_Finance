import joblib
import pandas as pd
import shap
import xgboost as xgb
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgb.pkl"

def find_edge_case():
    print("Loading Data and Model...")
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    
    X = df.drop('Class', axis=1)
    
    print("Predicting Probabilities...")
    probs = model.predict_proba(X)
    fraud_probs = probs[:, 1]
    
    # Filter for low prob (5% - 20%)
    mask = (fraud_probs >= 0.05) & (fraud_probs <= 0.20)
    candidates = X[mask].index.tolist()
    
    print(f"Found {len(candidates)} candidates with prob in [0.05, 0.20]")
    
    if not candidates:
        print("No candidates found in that range. Widening search to [0.01, 0.30]")
        mask = (fraud_probs >= 0.01) & (fraud_probs <= 0.30)
        candidates = X[mask].index.tolist()
        
    # Calculate SHAP for candidates
    explainer = shap.TreeExplainer(model)
    
    for idx in candidates[:50]: # Sample first 50
        row = X.iloc[[idx]]
        shap_vals = explainer.shap_values(row)
        
        # Check SHAP for V14 (or similar top feature)
        # In our model, V14 is often index 14
        v14_shap = shap_vals[0][14] # assuming index 14 is V14 based on data scan
        
        if abs(v14_shap) > 1.0: # High individual attribution
            print(f"\nMatch Found! Transaction ID: {idx}")
            print(f"Fraud Probability: {fraud_probs[idx]:.4%}")
            print(f"V14 SHAP value: {v14_shap:.4f}")
            print(f"Feature Value for V14: {row.iloc[0]['V14']:.4f}")
            
            # Verify its actual class
            print(f"Actual Class: {df.iloc[idx]['Class']}")
            return idx
            
    print("No specific edge case matching criteria found in sampled candidates.")
    return None

if __name__ == "__main__":
    find_edge_case()
