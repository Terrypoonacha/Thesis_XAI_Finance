import pandas as pd
from pathlib import Path

# Config
PROJECT_ROOT = Path("c:/Users/terry/OneDrive/Desktop/Thesis_XAI_Finance/Thesis_XAI_Finance")
ULB_DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"
BAF_DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "baf_neurips" / "Variant II.csv"

def find_baf_case():
    print("--- BAF Profiling & Step 2 Case Selection ---")
    
    # 1. Feature Gap
    ulb_cols = set(pd.read_csv(ULB_DATA_PATH, nrows=1).columns)
    baf_cols = set(pd.read_csv(BAF_DATA_PATH, nrows=1).columns)
    gap = sorted(list(baf_cols - ulb_cols))
    
    print(f"Features in BAF but not in ULB: {', '.join(gap)}")
    
    # 2. Find Case for Step 2
    # Load in chunks to find a fraudulent transaction with customer_age > 60
    print("\nSearching for fraud case where customer_age > 60...")
    found_cases = []
    
    # Note: BAF features are often lower-case. Let's check 'customer_age' and 'fraud_bool'
    for chunk in pd.read_csv(BAF_DATA_PATH, chunksize=50000):
        # The BAF NeurIPS dataset has 'customer_age' in 10-year bins (0, 10, 20, ...)
        # or sometimes it's an integer. Let's check the type.
        mask = (chunk['fraud_bool'] == 1) & (chunk['customer_age'] > 60)
        matches = chunk[mask]
        if not matches.empty:
            found_cases.append(matches.iloc[0])
            if len(found_cases) >= 1:
                break
                
    if found_cases:
        case = found_cases[0]
        print(f"\nStep 2 Case Found!")
        print(f"Index: {case.name}") # This is the index in the chunk, might need global index
        print(f"Customer Age: {case['customer_age']}")
        print(f"Fraud Bool: {case['fraud_bool']}")
        print(f"Employment Status: {case['employment_status']}")
        print(f"Housing Status: {case['housing_status']}")
        print("\nFull Feature Values for Case:")
        print(case.to_dict())
    else:
        print("\nNo such case found in Variant II.csv.")

if __name__ == "__main__":
    find_baf_case()
