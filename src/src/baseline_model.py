import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "Online_Retail_Cleaned.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

print("Loading retail dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Clean and prepare data
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")

df = df.dropna(
    subset=["InvoiceNo", "InvoiceDate", "CustomerID", "Revenue"]
).copy()

# Keep positive-revenue transactions for this prediction task.
# Exclude cancellation invoices and invalid line quantities/prices.
df = df[
    (~df["InvoiceNo"].astype(str).str.startswith("C"))
    & (df["Quantity"] > 0)
    & (df["UnitPrice"] > 0)
    & (df["Revenue"] > 0)
].copy()

df["CustomerID"] = df["CustomerID"].astype(int)

# Aggregate line items into one row per customer invoice.
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

invoices = invoices.sort_values(
    ["CustomerID", "InvoiceDate", "InvoiceNo"]
).reset_index(drop=True)

# Build historical features using only earlier invoices.
# Shift(1) ensures the current invoice is not included in its history.
grouped = invoices.groupby("CustomerID", sort=False)

invoices["PreviousInvoiceCount"] = grouped.cumcount()
invoices["PreviousRevenue"] = grouped["InvoiceRevenue"].shift(1)

invoices["PreviousAverageRevenue"] = (
    invoices.groupby("CustomerID")["InvoiceRevenue"]
    .transform(lambda s: s.shift(1).expanding().mean())
)

invoices["PreviousItems"] = grouped["Items"].shift(1)

invoices["PreviousInvoiceDate"] = grouped["InvoiceDate"].shift(1)

invoices["DaysSincePreviousInvoice"] = (
    invoices["InvoiceDate"] - invoices["PreviousInvoiceDate"]
).dt.total_seconds() / 86400

# The first invoice per customer has no previous purchase history.
# Exclude it because the prediction explicitly depends on prior history.
invoices = invoices[
    invoices["PreviousInvoiceCount"] > 0
].copy()

# Baseline features: simple historical information only.
numeric_features = [
    "PreviousInvoiceCount",
    "PreviousRevenue",
    "PreviousAverageRevenue",
    "PreviousItems",
    "DaysSincePreviousInvoice",
]

categorical_features = ["Country"]

# Chronological split: train on earlier invoices, test on later invoices.
invoices = invoices.sort_values("InvoiceDate").reset_index(drop=True)

split_index = int(len(invoices) * 0.80)
train = invoices.iloc[:split_index].copy()
test = invoices.iloc[split_index:].copy()

X_train = train[numeric_features + categorical_features]
y_train = train["InvoiceRevenue"]

X_test = test[numeric_features + categorical_features]
y_test = test["InvoiceRevenue"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_features,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

print(f"Training invoices: {len(train):,}")
print(f"Testing invoices: {len(test):,}")
print("Training baseline model...")

model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== BASELINE RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

results = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2],
})

results.to_csv(REPORT_DIR / "baseline_metrics.csv", index=False)

print("\nBaseline metrics saved to reports/baseline_metrics.csv")