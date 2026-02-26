import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
import numpy as np

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "Figures"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

def load_data():
    """Locates and loads the dataset."""
    print(f"Searching for dataset at: {DATA_PATH}")
    if not DATA_PATH.exists():
        # Fallback search
        print("Path not found. Searching recursively...")
        found_files = list(PROJECT_ROOT.parent.rglob("creditcard.csv"))
        if not found_files:
            raise FileNotFoundError("creditcard.csv not found in project directory.")
        data_path = found_files[0]
        print(f"Dataset found at: {data_path}")
    else:
        data_path = DATA_PATH
    
    try:
        df = pd.read_csv(data_path, on_bad_lines='skip', low_memory=False)
        print("Data Loaded Successfully.")
        print(df.info())
        print(df.head())
        return df
    except TypeError:
        # Fallback for older pandas versions
        df = pd.read_csv(data_path, error_bad_lines=False, low_memory=False)
        print("Data Loaded Successfully (fallback mode).")
        print(df.info())
        print(df.head())
        return df

def perform_eda(df):
    """Calculates imbalance and generates countplot."""
    counts = df['Class'].value_counts()
    imbalance_ratio = counts[1] / counts[0]
    print(f"Class Imbalance Ratio (Fraud/Non-Fraud): {imbalance_ratio:.5f}")
    print(f"Fraud cases: {counts[1]}, Non-fraud cases: {counts[0]}")

    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='Class', data=df, palette='viridis')
    plt.title('Class Distribution (0: No Fraud, 1: Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.yscale('log') # Log scale to see the minority class better
    
    # Add counts
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
                    
    save_path = FIGURES_DIR / "class_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved class distribution plot to {save_path}")
    plt.close()

def train_and_evaluate(df):
    """Trains XGBoost and evaluates with PRC/AUPRC."""
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

    print(f"XGBoost version: {xgb.__version__}")
    print(f"SHAP version: {shap.__version__}")

    # Explicitly set base_score to avoid SHAP parsing issue with newer XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss',
        base_score=0.5 
    )

    auprc_scores = []
    
    print("Starting Training (5-Fold CV)...")
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train, y_train)
        
        y_probs = model.predict_proba(X_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_val, y_probs)
        auprc = auc(recall, precision)
        auprc_scores.append(auprc)
        print(f"Fold {fold+1} AUPRC: {auprc:.4f}")

    mean_auprc = np.mean(auprc_scores)
    print(f"Mean AUPRC: {mean_auprc:.4f}")

    # Train on full dataset for final model and SHAP
    print("Retraining on full dataset...")
    model.fit(X, y)
    
    model_path = MODELS_DIR / "baseline_xgb.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    
    return model, X

def plot_baseline_performance(model, df):
    """Generates PR Curve, Confusion Matrix, and Performance Table inline using a Test Set."""
    plt.rcParams.update({'font.size': 12})
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 1. Stratified Train/Test Split (80/20) for visual validation
    print("Splitting data for visual validation (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Retrain/Fit models on X_train for the walkthrough
    print("Training XGBoost on training split...")
    model.fit(X_train, y_train) 
    
    print("Training Random Forest with balanced weights...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    models = {'XGBoost': model, 'Random Forest': rf}
    metrics_data = []
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')

    colors = {'XGBoost': '#4A90E2', 'Random Forest': '#50E3C2'}
    
    for name, m in models.items():
        # Evaluate on the hold-out TEST SET
        y_probs = m.predict_proba(X_test)[:, 1]
        y_preds = m.predict(X_test)
        
        # 3. Metrics Calculation (Academic Proof)
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        f1 = f1_score(y_test, y_preds)
        
        metrics_data.append([name, f"{pr_auc:.4f}", f"{f1:.4f}"])
        
        ax1.plot(recall, precision, label=f'{name} (Test AUPRC = {pr_auc:.4f})', color=colors[name], lw=3)
    
    ax1.set_title('PR Curve Comparison (Hold-out Test Set)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix (XGBoost on Test Set)
    y_preds_xgb = model.predict(X_test)
    cm = confusion_matrix(y_test, y_preds_xgb)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
    ax2.set_title('XGBoost Confusion Matrix (Test Set)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    # 5. Model Performance Table
    table_data = [['Model (Evaluated on Test Set)', 'AUPRC', 'F1-Score']] + metrics_data
    table = ax3.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.5)
    
    # Style the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#444444')

    plt.suptitle('Sprint 1: Robust Baseline Performance Audit', fontsize=22, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig

def explain_model(model, X):
    """Generates SHAP summary plot."""
    print("Generating SHAP explanation...")
    # Use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(model)
    
    # SHAP values can be expensive, sample if necessary but creditcard.csv is medium sized.
    # For speed in this sprint, let's use a sample for the plot if X is huge, 
    # but exact calculation is better. 280k rows is manageable for TreeExplainer usually 
    # but might take a minute. Let's use 10% sample for speed in this interactive session context
    # or just run it. The user wants "High-Performance", let's use sklearn subsample or just first N rows.
    # To be safe and fast for the prototype:
    X_sample = X.sample(n=min(10000, len(X)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
    save_path = FIGURES_DIR / "shap_global_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved SHAP summary plot to {save_path}")
    plt.close()

def main():
    try:
        df = load_data()
        perform_eda(df)
        model, X = train_and_evaluate(df)
        
        # New: Direct visualization of baseline performance for Academic Walkthrough
        print("\n--- Generating Direct Visual Audit (Sprint 1) ---")
        fig = plot_baseline_performance(model, df)
        plt.show()
        
        explain_model(model, X)
        print("\nSprint 1 Pipeline Completed Successfully.")
    except Exception:
        print("Pipeline Failed With Traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
