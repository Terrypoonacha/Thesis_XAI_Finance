import pandas as pd
from pathlib import Path
import traceback

DATA_PATH = Path("../data/raw/creditcard.csv")

def debug():
    print(f"Reading {DATA_PATH.resolve()}")
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip', low_memory=False)
        print("Success!")
        print(df.info())
        if df['V17'].dtype == 'object':
             print("V17 is object!")
             print(df[pd.to_numeric(df['V17'], errors='coerce').isna()]['V17'])
        else:
             print(f"V17 dtype: {df['V17'].dtype}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    debug()
