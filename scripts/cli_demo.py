"""Lightweight CLI demo of the tuned XGBoost churn-prediction pipeline."""

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Feature engineering
df["AvgMonthlySpend"] = np.where(
    df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
)
svc = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
df["NumAddOnServices"] = (df[svc] == "Yes").sum(axis=1)
df["TenureGroup"] = pd.cut(
    df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]
).astype(str)

X = df.drop(columns=["Churn"])
y = df["Churn"]

num_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AvgMonthlySpend",
    "NumAddOnServices",
    "SeniorCitizen",
]
cat_cols = [c for c in X.columns if c not in num_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
    ]
)

pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline([("pre", preprocess), ("clf", model)])
pipeline.fit(X_train, y_train)

proba_test = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba_test)

prec, rec, thr = precision_recall_curve(y_test, proba_test)
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
best_thr = float(thr[int(np.argmax(f1))])

y_pred = (proba_test >= best_thr).astype(int)
conf = {
    "TP": int(((y_pred == 1) & (y_test == 1)).sum()),
    "FP": int(((y_pred == 1) & (y_test == 0)).sum()),
    "FN": int(((y_pred == 0) & (y_test == 1)).sum()),
    "TN": int(((y_pred == 0) & (y_test == 0)).sum()),
}

importance = model.get_booster().get_score(importance_type="weight")
top_raw = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:3]
feat_names = pipeline.named_steps["pre"].get_feature_names_out()
top_features = [
    feat_names[int(f[1:])].split("__")[-1] if f.startswith("f") and f[1:].isdigit() else f
    for f, _ in top_raw
]

out = {
    "test_auc": round(float(auc), 4),
    "best_thr": round(float(best_thr), 4),
    "confusion": conf,
    "top_features": top_features,
}

with open(OUT_DIR / "cli_demo_metrics.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"Test AUC: {out['test_auc']}")
print(f"Best threshold (max F1): {out['best_thr']}")
print(f"Confusion @ {out['best_thr']}: TP={conf['TP']}, FN={conf['FN']}, FP={conf['FP']}, TN={conf['TN']}")
print("Top 3 features (XGBoost split weight): " + ", ".join(top_features))

# Decile lift chart
df_res = pd.DataFrame({"p": proba_test, "y": y_test.values}).sort_values("p", ascending=False).reset_index(drop=True)
n = len(df_res)
df_res["decile"] = np.arange(n) * 10 // n + 1
base_rate = df_res.y.mean()
summary = df_res.groupby("decile").y.agg(["count", "sum", "mean"]).reset_index()
summary["lift"] = summary["mean"] / base_rate
summary["cum_capture"] = summary["sum"].cumsum() / df_res.y.sum()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(summary["decile"], summary["cum_capture"] * 100, marker="o", label="Tuned XGBoost")
ax.plot([0, 10], [0, 100], "k--", lw=1, label="Random targeting")
ax.set_xlabel("Decile (10% buckets)")
ax.set_ylabel("Churners captured (%)")
ax.set_title("Cumulative gains (CLI demo)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt_path = OUT_DIR / "cli_demo_gains.png"
plt.savefig(plt_path, dpi=150)
print(f"Gains chart saved to {plt_path}")
