from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place WA_Fn-UseC_-HR-Employee-Attrition.csv in the data folder."
    )

df = pd.read_csv(DATA_PATH)
if "Attrition" not in df.columns:
    raise ValueError("Dataset must contain an 'Attrition' target column.")

df = df.dropna(subset=["Attrition"]).copy()
y = df["Attrition"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
if y.isna().any():
    raise ValueError("Attrition must contain only Yes/No values.")

# Remove identifiers and constant columns where present.
drop_cols = [c for c in ["Attrition", "EmployeeNumber", "EmployeeCount",
                         "Over18", "StandardHours"] if c in df.columns]
X = df.drop(columns=drop_cols)

# Remove any other columns with only one unique non-null value.
constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
X = X.drop(columns=constant_cols)

numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

results = []
fitted = {}

for name, classifier in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1_Score": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba)
    }
    results.append(metrics)
    fitted[name] = pipe

    print(f"\n===== {name} =====")
    print(pd.Series(metrics).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["No", "Yes"], zero_division=0))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, pred))

comparison = pd.DataFrame(results).sort_values(
    by=["Recall", "F1_Score", "ROC_AUC"], ascending=False
)
comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
best_name = comparison.iloc[0]["Model"]
best_model = fitted[best_name]

joblib.dump(best_model, MODEL_DIR / "best_attrition_model.joblib")
metadata = {
    "selected_model": best_name,
    "features": X.columns.tolist(),
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "positive_class": "Yes",
    "note": "Predicted probability is a risk estimate, not certainty."
}
(MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

# Score held-out records for the watch-list output.
test_output = X_test.copy()
test_output.insert(0, "Actual_Attrition", y_test.map({0: "No", 1: "Yes"}))
test_output["Predicted_Attrition_Probability"] = best_model.predict_proba(X_test)[:, 1]
test_output["Predicted_Risk"] = np.where(
    test_output["Predicted_Attrition_Probability"] >= 0.50, "Higher", "Lower"
)
test_output.sort_values(
    "Predicted_Attrition_Probability", ascending=False
).to_csv(OUT_DIR / "at_risk_employees.csv", index=False)

# Department rates are descriptive, not causal.
if "Department" in df.columns:
    dept = df[["Department", "Attrition"]].copy()
    dept["Attrition_Flag"] = dept["Attrition"].astype(str).str.lower().eq("yes").astype(int)
    dept_summary = dept.groupby("Department").agg(
        Employees=("Attrition_Flag", "size"),
        Attrition_Count=("Attrition_Flag", "sum"),
        Attrition_Rate=("Attrition_Flag", "mean")
    ).reset_index()
    dept_summary["Attrition_Rate"] *= 100
    dept_summary.to_csv(OUT_DIR / "attrition_by_department.csv", index=False)

# Permutation importance measures how much test performance changes when a feature is shuffled.
perm = permutation_importance(
    best_model, X_test, y_test, n_repeats=8,
    random_state=42, scoring="recall", n_jobs=-1
)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance_Mean": perm.importances_mean,
    "Importance_Std": perm.importances_std
}).sort_values("Importance_Mean", ascending=False)
importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)

print("\n===== MODEL SELECTION =====")
print(comparison.to_string(index=False))
print(f"\nSelected model: {best_name}")
print(f"Saved model: {(MODEL_DIR / 'best_attrition_model.joblib').resolve()}")
print(f"All outputs saved in: {OUT_DIR.resolve()}")
print("\nReminder: do not use predictions as the sole basis for employment decisions.")
