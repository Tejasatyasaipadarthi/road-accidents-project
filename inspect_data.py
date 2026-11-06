import pandas as pd
from pathlib import Path

DATA_FILE = Path("../data/accidents.csv")

if not DATA_FILE.exists():
    print(f"ERROR: Dataset not found at {DATA_FILE}")
    print("Check the filename and place your CSV file inside the data folder.")
    raise SystemExit(1)

try:
    df = pd.read_csv(DATA_FILE)

    print("\n========== DATASET INFORMATION ==========")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\n========== COLUMN NAMES ==========")
    for column in df.columns:
        print(f"- {column}")

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head().to_string())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum().to_string())

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())

    print("\n========== POSSIBLE SEVERITY COLUMNS ==========")
    for column in df.columns:
        if any(word in column.lower() for word in
               ["severity", "casualty", "fatal", "injury", "accident_severity"]):
            print(f"\nColumn: {column}")
            print(df[column].value_counts(dropna=False).head(15).to_string())

except Exception as error:
    print(f"ERROR while reading dataset: {error}")