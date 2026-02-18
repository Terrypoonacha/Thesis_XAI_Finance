from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT.parent / "data" / "raw" / "creditcard.csv"

def inspect_csv():
    print(f"Reading from: {DATA_PATH}")
    try:
        with open(DATA_PATH, 'r') as f:
            for i in range(5):
                line = f.readline()
                print(f"Line {i+1}: {line.strip()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_csv()
