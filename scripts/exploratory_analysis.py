"""Exploratory data analysis and chart generation for the Telco churn dataset."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font="DejaVu Sans")
PAL = ["#1D7874", "#F18F01"]

df = pd.read_csv(DATA_PATH)
out = {"shape": list(df.shape)}

# TotalCharges has blanks for tenure 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
out["missing_totalcharges"] = int(df["TotalCharges"].isna().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)

out["churn_counts"] = df["Churn"].value_counts().to_dict()
out["churn_rate"] = round(float((df["Churn"] == "Yes").mean() * 100), 2)
out["duplicates"] = int(df["customerID"].duplicated().sum())

# Churn distribution
fig, ax = plt.subplots(figsize=(5, 4))
vc = df["Churn"].value_counts()
ax.bar(vc.index, vc.values, color=PAL)
for i, v in enumerate(vc.values):
    ax.text(i, v + 50, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=11)
ax.set_title("Churn distribution (n = 7,043)", fontsize=13)
ax.set_ylabel("Customers")
ax.set_ylim(0, 6000)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_churn_distribution.png", dpi=200)
plt.close()

# Churn rate by contract, internet service, payment method
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
    r = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100).sort_values()
    ax.barh(r.index, r.values, color="#0B3C5D")
    for i, v in enumerate(r.values):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_title(f"Churn rate by {col}", fontsize=12)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Churn rate (%)")
    ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_churn_by_segment.png", dpi=200)
plt.close()

out["churn_by_contract"] = df.groupby("Contract")["Churn"].apply(lambda s: round((s == "Yes").mean() * 100, 1)).to_dict()
out["churn_by_internet"] = df.groupby("InternetService")["Churn"].apply(lambda s: round((s == "Yes").mean() * 100, 1)).to_dict()
out["churn_by_payment"] = df.groupby("PaymentMethod")["Churn"].apply(lambda s: round((s == "Yes").mean() * 100, 1)).to_dict()

# Tenure and monthly charges distributions by churn
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col in zip(axes, ["tenure", "MonthlyCharges"]):
    for lab, c in zip(["No", "Yes"], PAL):
        ax.hist(df.loc[df["Churn"] == lab, col], bins=30, alpha=0.65, color=c, label=f"Churn = {lab}")
    ax.set_title(f"{col} by churn status", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Customers")
    ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "05_tenure_and_charges_distribution.png", dpi=200)
plt.close()

out["median_tenure_churn"] = float(df.loc[df.Churn == "Yes", "tenure"].median())
out["median_tenure_stay"] = float(df.loc[df.Churn == "No", "tenure"].median())
out["mean_monthly_churn"] = round(float(df.loc[df.Churn == "Yes", "MonthlyCharges"].mean()), 2)
out["mean_monthly_stay"] = round(float(df.loc[df.Churn == "No", "MonthlyCharges"].mean()), 2)

# Correlation heatmap of numeric + encoded binary features
enc = df.copy()
enc["Churn"] = (enc["Churn"] == "Yes").astype(int)
for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    enc[c] = (enc[c] == "Yes").astype(int)
enc["gender"] = (enc["gender"] == "Male").astype(int)
num = enc[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Partner", "Dependents",
           "PhoneService", "PaperlessBilling", "gender", "Churn"]]
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation matrix (numeric and binary features)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_feature_correlation_heatmap.png", dpi=200)
plt.close()

out["corr_with_churn"] = num.corr()["Churn"].drop("Churn").round(3).sort_values().to_dict()
out["columns"] = list(df.columns)
out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

with open(OUT_DIR / "eda_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
