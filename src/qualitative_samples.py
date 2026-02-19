import pandas as pd
from agentic_auditor import generate_compliance_memo
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"

def generate_samples():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    
    # Filter for True Positives (Actual Fraud)
    fraud_cases = df[df['Class'] == 1].index.tolist()
    
    # Pick 5 random
    samples = random.sample(fraud_cases, 5)
    print(f"Selected Fraud Cases: {samples}")
    
    output_path = PROJECT_ROOT / "reports" / "qualitative_samples.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for tid in samples:
            print(f"Processing {tid}...")
            f.write(f"\n{'='*50}\nTransaction {tid}\n{'='*50}\n")
            memo = generate_compliance_memo(tid)
            f.write(memo + "\n")
            f.write("\n")
            
    print(f"Samples saved to {output_path}")

if __name__ == "__main__":
    generate_samples()
