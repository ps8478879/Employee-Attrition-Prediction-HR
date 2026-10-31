from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Download/place the HR attrition CSV in the data folder."
    )

df = pd.read_csv(DATA_PATH)
if "Attrition" not in df.columns:
    raise ValueError("Expected target column 'Attrition' was not found.")

print("\n--- Dataset shape ---")
print(df.shape)
print("\n--- First five rows ---")
print(df.head())
print("\n--- Missing values ---")
print(df.isnull().sum().sort_values(ascending=False).head(20))
print("\n--- Attrition counts ---")
print(df["Attrition"].value_counts(dropna=False))
print("\n--- Attrition percentage ---")
print((df["Attrition"].value_counts(normalize=True) * 100).round(2))

sns.set_theme(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Count")
plt.tight_layout()
plt.savefig(OUT / "eda_attrition_count.png", dpi=150)
plt.close()

if "Department" in df.columns:
    dept_rate = df.groupby("Department")["Attrition"].apply(
        lambda x: (x.astype(str).str.lower() == "yes").mean() * 100
    ).sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    dept_rate.plot(kind="bar")
    plt.ylabel("Attrition rate (%)")
    plt.title("Attrition Rate by Department")
    plt.tight_layout()
    plt.savefig(OUT / "eda_attrition_by_department.png", dpi=150)
    plt.close()

for col in ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany"]:
    if col in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x="Attrition", y=col)
        plt.title(f"{col} by Attrition")
        plt.tight_layout()
        plt.savefig(OUT / f"eda_{col.lower()}_by_attrition.png", dpi=150)
        plt.close()

if "OverTime" in df.columns:
    overtime_rate = df.groupby("OverTime")["Attrition"].apply(
        lambda x: (x.astype(str).str.lower() == "yes").mean() * 100
    )
    plt.figure(figsize=(6, 4))
    overtime_rate.plot(kind="bar")
    plt.ylabel("Attrition rate (%)")
    plt.title("Attrition Rate by Overtime")
    plt.tight_layout()
    plt.savefig(OUT / "eda_overtime_attrition.png", dpi=150)
    plt.close()

print(f"\nEDA complete. Charts saved in: {OUT.resolve()}")
