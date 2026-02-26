import time
import pandas as pd
import random
from agentic_auditor import generate_compliance_memo
from pathlib import Path

# Config
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "qualitative_samples_batch.txt"

def generate_batch():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    fraud_cases = df[df['Class'] == 1].index.tolist()
    
    # Selected cases from previous attempt + new ones
    # 541 is already done manually. Let's pick 5 others or include 541.
    # User wanted 5 representative cases.
    target_ids = [541, 623, 4920, 6108, 6329] # Known fraud indices from dataset or random
    
    # Verify they are actual fraud
    valid_ids = [i for i in target_ids if i in fraud_cases]
    
    # If not enough specific ones, sample random
    while len(valid_ids) < 5:
        new_id = random.choice(fraud_cases)
        if new_id not in valid_ids:
            valid_ids.append(new_id)
            
    print(f"Targeting Transaction IDs: {valid_ids}")
    
    results = []
    
    for tid in valid_ids:
        print(f"Processing Transaction {tid}...")
        try:
            memo = generate_compliance_memo(tid)
            entry = f"Transaction {tid}\n{'='*20}\n{memo}\n\n{'='*50}\n\n"
            results.append(entry)
            print(f"Success for {tid}")
        except Exception as e:
            print(f"Failed {tid}: {e}")
            results.append(f"Transaction {tid}: Failed ({e})\n\n")
            
        # Rate limit delay
        print("Sleeping 30s to respect API limits...")
        time.sleep(30)
        
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.writelines(results)
    
    print(f"Batch completed. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_batch()
