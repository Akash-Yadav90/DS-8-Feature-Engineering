import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Online_Retail_Cleaned.csv"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

print("Loading retail data...")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Clean data types
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)

df["CustomerID"] = pd.to_numeric(
    df["CustomerID"], errors="coerce"
)

df["Revenue"] = pd.to_numeric(
    df["Revenue"], errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"], errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"], errors="coerce"
)

# Remove invalid records
df = df.dropna(
    subset=[
        "InvoiceNo",
        "InvoiceDate",
        "CustomerID",
        "Revenue",
        "Quantity",
        "UnitPrice",
    ]
).copy()

df = df[
    (~df["InvoiceNo"].astype(str).str.startswith("C"))
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
    & (df["Revenue"] > 0)
].copy()

df["CustomerID"] = df["CustomerID"].astype(int)

# Aggregate line items into customer invoices
invoices = (
    df.groupby(["CustomerID", "InvoiceNo"], as_index=False)
    .agg(
        InvoiceDate=("InvoiceDate", "min"),
        InvoiceRevenue=("Revenue", "sum"),
        Items=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        Country=("Country", "first"),
    )
)

# Sort chronologically for each customer
invoices = invoices.sort_values(
    ["CustomerID", "InvoiceDate", "InvoiceNo"]
).reset_index(drop=True)

customer_groups = invoices.groupby("CustomerID", sort=False)

# Feature 1: Previous invoice count
invoices["PreviousInvoiceCount"] = (
    customer_groups.cumcount()
)

# Feature 2: Previous invoice revenue
invoices["PreviousRevenue"] = (
    customer_groups["InvoiceRevenue"].shift(1)
)

# Feature 3: Previous average invoice revenue
invoices["PreviousAverageRevenue"] = (
    invoices.groupby("CustomerID")["InvoiceRevenue"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

# Feature 4: Days since previous invoice
previous_dates = customer_groups["InvoiceDate"].shift(1)

invoices["DaysSincePreviousInvoice"] = (
    invoices["InvoiceDate"] - previous_dates
).dt.total_seconds() / 86400

# Feature 5: Calendar features
invoices["Month"] = invoices["InvoiceDate"].dt.month
invoices["DayOfWeek"] = invoices["InvoiceDate"].dt.dayofweek
invoices["Hour"] = invoices["InvoiceDate"].dt.hour

# Feature 6: Cyclical month encoding
invoices["MonthSin"] = np.sin(
    2 * np.pi * invoices["Month"] / 12
)

invoices["MonthCos"] = np.cos(
    2 * np.pi * invoices["Month"] / 12
)

# Feature 7: Cyclical weekday encoding
invoices["WeekdaySin"] = np.sin(
    2 * np.pi * invoices["DayOfWeek"] / 7
)

invoices["WeekdayCos"] = np.cos(
    2 * np.pi * invoices["DayOfWeek"] / 7
)

# Feature 8: Interaction between historical revenue and purchase frequency
invoices["RevenueFrequencyInteraction"] = (
    invoices["PreviousAverageRevenue"]
    * invoices["PreviousInvoiceCount"]
)

# Feature 9: Historical average items per invoice
invoices["PreviousAverageItems"] = (
    invoices.groupby("CustomerID")["Items"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

# Save engineered dataset
output_path = REPORT_DIR / "engineered_features.csv"

invoices.to_csv(output_path, index=False)

print("\nFeature engineering completed.")
print("Rows:", len(invoices))
print("Columns:", len(invoices.columns))
print("\nNew features:")
print([
    "PreviousInvoiceCount",
    "PreviousRevenue",
    "PreviousAverageRevenue",
    "DaysSincePreviousInvoice",
    "Month",
    "DayOfWeek",
    "Hour",
    "MonthSin",
    "MonthCos",
    "WeekdaySin",
    "WeekdayCos",
    "RevenueFrequencyInteraction",
    "PreviousAverageItems",
])

print("\nSaved to:", output_path)