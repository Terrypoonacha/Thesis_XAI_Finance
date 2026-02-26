import pandas as pd
from pathlib import Path

# Config
PROJECT_ROOT = Path("c:/Users/terry/OneDrive/Desktop/Thesis_XAI_Finance/Thesis_XAI_Finance")
ULB_DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"
BAF_DIR = PROJECT_ROOT.parent / "data" / "raw" / "baf_neurips"

def profile_baf():
    print("--- Step 1: Comparative Data Profiling ---")
    
    # 1. ULB Dataset Columns
    print(f"Loading ULB dataset columns from: {ULB_DATA_PATH.name}")
    ulb_sample = pd.read_csv(ULB_DATA_PATH, nrows=5)
    ulb_cols = set(ulb_sample.columns)
    
    # 2. BAF Dataset Columns
    base_path = BAF_DIR / "Base.csv"
    variant_ii_path = BAF_DIR / "Variant II.csv"
    
    print(f"Loading BAF Base dataset columns from: {base_path.name}")
    baf_base_sample = pd.read_csv(base_path, nrows=5)
    baf_cols = set(baf_base_sample.columns)
    
    print(f"Loading BAF Variant II dataset columns from: {variant_ii_path.name}")
    baf_variant_sample = pd.read_csv(variant_ii_path, nrows=5)
    
    # 3. Identify Semantic Gap
    semantic_gap = baf_cols - ulb_cols
    print(f"\nSemantic Gap (Features in BAF missing in ULB):")
    for feat in sorted(semantic_gap):
        print(f"- {feat}")
        
    # 4. Fraud Prevalence Calculation
    print("\nCalculating Fraud Prevalence (FRAUD_FRAUD)...")
    
    # Use chunking for large files
    def get_fraud_rate(path):
        fraud_counts = 0
        total_counts = 0
        for chunk in pd.read_csv(path, chunksize=100000, usecols=['fraud_bool']):
            fraud_counts += chunk['fraud_bool'].sum()
            total_counts += len(chunk)
        return fraud_counts, total_counts, (fraud_counts / total_counts) * 100

    base_fraud, base_total, base_rate = get_fraud_rate(base_path)
    print(f"Base.csv: {base_fraud} fraud cases out of {base_total} ({base_rate:.4f}%)")
    
    variant_fraud, variant_total, variant_rate = get_fraud_rate(variant_ii_path)
    print(f"Variant_II.csv: {variant_fraud} fraud cases out of {variant_total} ({variant_rate:.4f}%)")
    
    # 5. Output Preview for Step 2
    print("\nPreview of Variant_II.csv (First 5 records):")
    print(baf_variant_sample.head())

if __name__ == "__main__":
    profile_baf()
