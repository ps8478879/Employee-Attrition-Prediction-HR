# Employee Attrition Prediction for HR

A beginner-friendly machine-learning project that predicts employee attrition (Yes/No), compares Logistic Regression and Random Forest, and creates HR analysis outputs.

## 1. Setup (Windows PowerShell)

```powershell
cd "$HOME\Desktop\Employee_Attrition_Prediction_HR"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 2. Dataset

Use the IBM HR Analytics Employee Attrition dataset commonly named:
`WA_Fn-UseC_-HR-Employee-Attrition.csv`

Place the CSV file inside `data` and keep that exact filename. The dataset commonly contains a target column named `Attrition` with values `Yes` and `No`.

This project does not include a dataset. Check the dataset's license and source terms before using or redistributing it.

## 3. Run the project

First inspect the data and generate charts:
```powershell
python eda.py
```

Then train and evaluate both models:
```powershell
python train.py
```

To score a new employee record, edit `sample_employee.json` using the exact feature names and valid values from your dataset, then run:
```powershell
python predict.py
```

## 4. Output files

- `outputs/attrition_by_department.csv` — attrition rate by department.
- `outputs/feature_importance.csv` — permutation feature importance on the held-out test set.
- `outputs/at_risk_employees.csv` — employees ranked by predicted probability of attrition.
- `outputs/model_comparison.csv` — evaluation metrics for both models.
- `outputs/eda_*.png` — exploratory charts.
- `models/best_attrition_model.joblib` — complete preprocessing + trained model pipeline.
- `models/model_metadata.json` — selected model and input feature names.

## 5. Important interpretation

A prediction is a statistical risk estimate, not proof that an employee will resign. Use results to offer supportive conversations and investigate workplace conditions. Do not use this model as the sole basis for firing, promotion, compensation, or other consequential employment decisions. Review fairness, privacy, data quality, and local employment requirements before real deployment.
