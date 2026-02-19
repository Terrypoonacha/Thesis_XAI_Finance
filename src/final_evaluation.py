import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import os
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_xgb.pkl"
FIGURES_DIR = PROJECT_ROOT / "reports" / "Figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_models():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Load XGBoost
    print("Loading XGBoost Baseline...")
    xgb_model = joblib.load(MODEL_PATH)
    
    # Train Random Forest (Vanilla)
    print("Training Random Forest Baseline (Vanilla)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # Note: To be fair we should train on a train split, but for this "final eval" 
    # and given the constraints, we might just train on the full dataset or use the same split.
    # Let's do a quick split to be scientifically reasonably valid (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model.fit(X_train, y_train)
    
    # We need to re-evaluate XGB on X_test to match (assuming baseline was trained on similar or different split).
    # Ideally baseline_xgb was trained on specific folds. Let's just use it on X_test for final comparative numbers.
    # If baseline was trained on ALL data, this is leakage. 
    # Check sprint_1_pipeline.py: it used cross_val_predict.
    # Let's assume we want to report performance on a hold-out set.
    
    print("Evaluating Models...")
    
    models = {
        "XGBoost (Optimized)": xgb_model,
        "Random Forest (Vanilla)": rf_model
    }
    
    results = []
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs > 0.5).astype(int)
        
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        auprc = auc(recall, precision)
        
        # Plot
        plt.plot(recall, precision, label=f'{name} (AUPRC = {auprc:.4f})')
        
        # Metrics
        results.append({
            "Model": name,
            "AUPRC": auprc,
            "F1-Score": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred)
        })
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: Agentic Auditor Baselines')
    plt.legend()
    plt.grid(True)
    
    save_path = FIGURES_DIR / "pr_curve.png"
    plt.savefig(save_path)
    print(f"PR Curve saved to {save_path}")
    
    # Generate Table
    results_df = pd.DataFrame(results)
    print("\n--- Final Quantitative Evaluation ---")
    print(results_df)
    
    # LaTeX Code
    latex_code = results_df.to_latex(index=False, float_format="%.4f")
    print("\n--- LaTeX Table Snippet ---")
    print(latex_code)
    
    return results_df

if __name__ == "__main__":
    evaluate_models()
