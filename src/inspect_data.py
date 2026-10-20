import pandas as pd
from pathlib import Path

# Locate the project root and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Online_Retail_Cleaned.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nBasic statistics:")
print(df.describe(include="all").T)