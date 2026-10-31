from pathlib import Path
import json
import joblib
import pandas as pd

MODEL_PATH = Path("models/best_attrition_model.joblib")
META_PATH = Path("models/model_metadata.json")
SAMPLE_PATH = Path("sample_employee.json")

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError("Train the model first by running: python train.py")
if not SAMPLE_PATH.exists():
    raise FileNotFoundError("sample_employee.json is missing.")

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text())
features = metadata["features"]

sample = json.loads(SAMPLE_PATH.read_text())
missing = [feature for feature in features if feature not in sample]
if missing:
    raise ValueError(
        "Your sample JSON is missing these model input fields: " + ", ".join(missing)
    )

employee = pd.DataFrame([[sample[f] for f in features]], columns=features)
probability = float(model.predict_proba(employee)[0, 1])
prediction = "Yes" if probability >= 0.50 else "No"

print("\n===== EMPLOYEE ATTRITION PREDICTION =====")
print(f"Predicted attrition: {prediction}")
print(f"Estimated probability of attrition: {probability:.1%}")
print("Interpretation: this is a model estimate, not certainty.")
