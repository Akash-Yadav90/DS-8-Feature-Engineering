# ============================================
# IMPORTS
# ============================================

from pathlib import Path

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# PROJECT CONFIGURATION
# ============================================

# File location:
# DS-8 Feature Engineering/src/src/src/improved_model.py

# Move up three levels to the project root.
ROOT = Path(__file__).resolve().parents[3]

# Input dataset created by feature_engineering.py
DATA_PATH = ROOT / "reports" / "engineered_features.csv"

# Folder for evaluation reports
REPORT_DIR = ROOT / "reports"

# Folder for saved machine learning models
MODEL_DIR = ROOT / "models"


# ============================================
# CREATE OUTPUT DIRECTORIES
# ============================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# VERIFY PATHS
# ============================================

print("=" * 50)
print("DS-8 FEATURE ENGINEERING PROJECT")
print("=" * 50)

print("Project root:", ROOT)
print("Dataset path:", DATA_PATH)
print("Reports folder:", REPORT_DIR)
print("Models folder:", MODEL_DIR)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Engineered dataset not found: {DATA_PATH}\n"
        "Run feature_engineering.py first."
    )

print("\nConfiguration completed successfully.")
# ============================================
# LOAD ENGINEERED DATASET
# ============================================

print("\nLoading engineered dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================
# PREPARE FEATURES AND TARGET
# ============================================

TARGET = "InvoiceRevenue"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

# Exclude identifiers, dates, and the target from predictors.
exclude_columns = [
    TARGET,
    "InvoiceNo",
    "CustomerID",
    "InvoiceDate",
    "Description",
    "StockCode",
    "Country"
]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

if not feature_columns:
    raise ValueError("No numeric feature columns found.")

X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
y = pd.to_numeric(df[TARGET], errors="coerce")

# Remove invalid target values.
valid = y.notna() & (y > 0)

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# Sort chronologically when invoice dates are available.
if "InvoiceDate" in df.columns:
    dates = pd.to_datetime(
        df.loc[valid, "InvoiceDate"],
        errors="coerce"
    ).reset_index(drop=True)

    order = dates.sort_values(kind="stable").index

    X = X.loc[order].reset_index(drop=True)
    y = y.loc[order].reset_index(drop=True)

print("Features:", feature_columns)
print("Valid records:", len(X))


# ============================================
# SPLIT INTO TRAINING AND TESTING DATA
# ============================================

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================
# BUILD AND TRAIN RANDOM FOREST
# ============================================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ============================================
# EVALUATE THE MODEL
# ============================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# ============================================
# SAVE EVALUATION METRICS
# ============================================

metrics = pd.DataFrame([{
    "Model": "Random Forest - Feature Engineered",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

metrics.to_csv(
    REPORT_DIR / "improved_model_metrics.csv",
    index=False
)


# ============================================
# SAVE FEATURE IMPORTANCE
# ============================================

regressor = model.named_steps["regressor"]

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": regressor.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 10 Important Features:")
print(importance.head(10).to_string(index=False))


# ============================================
# SAVE TRAINED MODEL
# ============================================

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns
    },
    MODEL_DIR / "improved_model.joblib"
)

print("\nAll output files saved successfully!")
print("Metrics:", REPORT_DIR / "improved_model_metrics.csv")
print("Feature importance:", REPORT_DIR / "feature_importance.csv")
print("Trained model:", MODEL_DIR / "improved_model.joblib")