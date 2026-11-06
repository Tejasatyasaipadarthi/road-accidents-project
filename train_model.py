
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "accidents.csv"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Check that accidents.csv is inside this project's data folder."
    )

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))

# Find a likely severity target column
target_candidates = [
    "accident_severity",
    "accident_severity_c",
    "severity",
    "accident severity",
    "accident_severity_level",
    "severity_level"
]

column_lookup = {column.lower(): column for column in df.columns}
target_column = next(
    (column_lookup[name] for name in target_candidates
     if name in column_lookup),
    None
)

if target_column is None:
    print("\nCould not automatically identify the severity column.")
    print("Available columns:", list(df.columns))
    raise ValueError(
        "Please identify the correct target column from the output."
    )

print("\nTarget column:", target_column)

# Remove rows where the target is missing
df = df.dropna(subset=[target_column]).copy()

# Remove duplicate records
df = df.drop_duplicates().copy()

# Remove columns that could cause leakage or do not help prediction
drop_candidates = [
    "id",
    "accident_id",
    "case_id",
    "record_id"
]

drop_columns = [
    column for column in df.columns
    if column.lower() in drop_candidates
]

X = df.drop(columns=[target_column] + drop_columns)
y = df[target_column].astype(str).str.strip()

# Remove empty targets
valid_target = y.ne("") & y.ne("nan")
X = X.loc[valid_target].copy()
y = y.loc[valid_target].copy()

# Remove columns with no useful values
useful_columns = [
    column for column in X.columns
    if X[column].notna().any() and X[column].nunique(dropna=True) > 1
]
X = X[useful_columns]

if X.shape[1] == 0:
    raise ValueError("No useful feature columns were found.")

if y.nunique() < 2:
    raise ValueError(
        "The severity target has fewer than two classes. "
        "A classifier needs at least two severity classes."
    )

print("\nSeverity class counts:")
print(y.value_counts())

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = [
    column for column in X.columns
    if column not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# Use stratification only when every class has enough examples
class_counts = y.value_counts()
stratify_target = (
    y if len(class_counts) > 1 and class_counts.min() >= 2
    else None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=stratify_target
)

# Build and train the complete pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=None,
    n_jobs=-1
))
])

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

# Save model and metadata
joblib.dump(model, MODEL_DIR / "random_forest_model.pkl")

metadata = {
    "target_column": target_column,
    "feature_columns": X.columns.tolist(),
    "classes": [str(value) for value in model.classes_],
    "accuracy": float(accuracy),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test))
}

with open(OUTPUT_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

with open(OUTPUT_DIR / "classification_report.json", "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nModel saved to:", MODEL_DIR / "random_forest_model.pkl")
print("Metadata saved to:", OUTPUT_DIR / "model_metadata.json")
print("\nTraining completed.")
