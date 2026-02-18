import pandas as pd
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"

def debug_data():
    print(f"Loading data from: {DATA_PATH}")
    try:
        # Load with low_memory=False to ensure mixed types are not silenced
        df = pd.read_csv(DATA_PATH, low_memory=False)
        print("Data loaded via pandas.")
        print(df.info())

        # Check for rows where 'Time' is not numeric (likely repeated headers)
        non_numeric_time = df[pd.to_numeric(df['Time'], errors='coerce').isna()]
        if not non_numeric_time.empty:
            print("Found non-numeric rows in Time column:")
            print(non_numeric_time)
            
        # Check if 'V17' exists as a value in any column
        for col in df.columns:
             # Convert to string to safely check for 'V17'
             if df[col].astype(str).str.contains('V17').any():
                 print(f"Column {col} contains 'V17' value.")
                 bad_rows = df[df[col].astype(str) == 'V17']
                 print(bad_rows)

    except Exception as e:
        print(f"Error loading data: {e}")

if __name__ == "__main__":
    debug_data()
