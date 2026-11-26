"""Compute decile lift, cumulative gains curve, and subgroup fairness metrics for tuned XGBoost."""

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RS = 42
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumAddOnServices"] = (df[svc] == "Yes").sum(axis=1)
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)

X = df.drop(columns=["Churn"])
y = df["Churn"]
num = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumAddOnServices", "SeniorCitizen"]
cat = [c for c in X.columns if c not in num]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat),
])
pw = float((ytr == 0).sum() / (ytr == 1).sum())

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=RS),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                            class_weight="balanced_subsample", random_state=RS, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                    subsample=0.9, random_state=RS),
    "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9,
                             colsample_bytree=0.8, scale_pos_weight=pw, eval_metric="logloss",
                             random_state=RS, n_jobs=-1),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
out = {"folds": {}, "train_auc": {}, "test_auc": {}}
for n, c in models.items():
    p = Pipeline([("pre", pre), ("clf", c)])
    s = cross_val_score(p, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
    out["folds"][n] = [round(float(v), 4) for v in s]
    p.fit(Xtr, ytr)
    out["train_auc"][n] = round(float(roc_auc_score(ytr, p.predict_proba(Xtr)[:, 1])), 4)
    out["test_auc"][n] = round(float(roc_auc_score(yte, p.predict_proba(Xte)[:, 1])), 4)

# Tuned XGBoost
t = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(n_estimators=300, learning_rate=0.02, max_depth=3, subsample=0.9,
                          colsample_bytree=0.8, scale_pos_weight=pw, eval_metric="logloss",
                          random_state=RS, n_jobs=-1)),
])
t.fit(Xtr, ytr)
pt = t.predict_proba(Xte)[:, 1]
out["tuned_test_auc"] = round(float(roc_auc_score(yte, pt)), 4)
out["tuned_train_auc"] = round(float(roc_auc_score(ytr, t.predict_proba(Xtr)[:, 1])), 4)

pr, rc, th_ = precision_recall_curve(yte, pt)
f1s = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-9)
best_thr = float(th_[int(np.argmax(f1s))])
out["best_thr"] = round(best_thr, 4)

rows = []
for th in [0.3, 0.4, 0.5, best_thr, 0.7, 0.8]:
    yh = (pt >= th).astype(int)
    rows.append({
        "thr": round(float(th), 4),
        "flagged": int(yh.sum()),
        "tp": int(((yh == 1) & (yte == 1)).sum()),
        "acc": round(accuracy_score(yte, yh), 4),
        "prec": round(precision_score(yte, yh), 4),
        "rec": round(recall_score(yte, yh), 4),
        "f1": round(f1_score(yte, yh), 4),
    })
out["thr_table"] = rows

# Decile lift
d = pd.DataFrame({"p": pt, "y": yte.values}).sort_values("p", ascending=False).reset_index(drop=True)
d["dec"] = np.arange(len(d)) * 10 // len(d) + 1
base = d.y.mean()
g = d.groupby("dec").y.agg(["count", "sum", "mean"])
g["lift"] = g["mean"] / base
g["cum_capture"] = g["sum"].cumsum() / d.y.sum()
out["base_rate"] = round(float(base), 4)
out["deciles"] = [
    {"dec": int(i), "n": int(r["count"]), "churners": int(r["sum"]),
     "rate": round(float(r["mean"]), 4), "lift": round(float(r["lift"]), 2),
     "cum_capture": round(float(r["cum_capture"]), 4)}
    for i, r in g.iterrows()
]

# Cumulative gains figure
fig, ax = plt.subplots(figsize=(6, 4.2))
x = np.arange(0, 11) * 10
yc = [0] + [v * 100 for v in g["cum_capture"]]
ax.plot(x, yc, marker="o", color="#0B3C5D", label="Tuned XGBoost")
ax.plot([0, 100], [0, 100], "k--", lw=1, label="Random targeting")
ax.set_xlabel("Customers contacted (% of test set, highest risk first)")
ax.set_ylabel("Churners captured (%)")
ax.set_title("Cumulative gains chart (test set)", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "12_cumulative_gains_curve.png", dpi=200)
plt.close()

# Subgroup fairness evaluation
yb = (pt >= best_thr).astype(int)
grp = []
for col in ["gender", "SeniorCitizen", "Partner"]:
    for v in sorted(Xte[col].unique(), key=str):
        m = (Xte[col] == v).values
        grp.append({
            "group": f"{col}={v}",
            "n": int(m.sum()),
            "churn_rate": round(float(yte[m].mean()), 4),
            "flag_rate": round(float(yb[m].mean()), 4),
            "recall": round(float(recall_score(yte[m], yb[m])), 4),
            "precision": round(float(precision_score(yte[m], yb[m])), 4),
            "auc": round(float(roc_auc_score(yte[m], pt[m])), 4),
        })
out["groups"] = grp

with open(OUT_DIR / "lift_and_fairness_summary.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
