"""Benchmark 5 classifiers, tune the top model with 5-fold CV, optimize threshold, and export evaluation figures."""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
RS = 42

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Feature engineering
df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumAddOnServices"] = (df[svc_cols] == "Yes").sum(axis=1)
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)

X = df.drop(columns=["Churn"])
y = df["Churn"]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumAddOnServices", "SeniorCitizen"]
cat_cols = [c for c in X.columns if c not in num_cols]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
])

pos_w = float((y_tr == 0).sum() / (y_tr == 1).sum())

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=RS),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                            class_weight="balanced_subsample", random_state=RS, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                    subsample=0.9, random_state=RS),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9,
                                      colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="logloss",
                                      random_state=RS, n_jobs=-1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
results = {}
probs = {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    t0 = time.time()
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    yhat = (p >= 0.5).astype(int)
    results[name] = {
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        "accuracy": round(float(accuracy_score(y_te, yhat)), 4),
        "precision": round(float(precision_score(y_te, yhat)), 4),
        "recall": round(float(recall_score(y_te, yhat)), 4),
        "f1": round(float(f1_score(y_te, yhat)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, p)), 4),
        "train_time_s": round(time.time() - t0, 1),
    }
    probs[name] = p
    print(name, results[name])

best = max(results, key=lambda k: results[k]["roc_auc"])
print("BEST:", best)

if best == "XGBoost":
    grid = {"clf__max_depth": [3, 4, 5], "clf__learning_rate": [0.02, 0.05], "clf__n_estimators": [300, 500]}
elif best == "Gradient Boosting":
    grid = {"clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.03, 0.05, 0.1], "clf__n_estimators": [200, 300]}
elif best == "Random Forest":
    grid = {"clf__max_depth": [8, 10, 14], "clf__min_samples_leaf": [3, 5, 10]}
else:
    grid = {"clf__C": [0.1, 0.5, 1, 5]}

gs = GridSearchCV(Pipeline([("pre", pre), ("clf", models[best])]), grid, cv=cv, scoring="roc_auc", n_jobs=-1)
gs.fit(X_tr, y_tr)
tuned = gs.best_estimator_
p_t = tuned.predict_proba(X_te)[:, 1]
tuned_res = {
    "best_params": {k.replace("clf__", ""): (float(v) if isinstance(v, (np.floating, float)) else int(v))
                    for k, v in gs.best_params_.items()},
    "cv_auc": round(float(gs.best_score_), 4),
    "roc_auc": round(float(roc_auc_score(y_te, p_t)), 4),
}

prec, rec, thr = precision_recall_curve(y_te, p_t)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
best_thr = float(thr[int(np.argmax(f1s))])
yhat_t = (p_t >= best_thr).astype(int)
yhat_05 = (p_t >= 0.5).astype(int)
tuned_res.update({
    "threshold": round(best_thr, 3),
    "at_0.5": {"accuracy": round(float(accuracy_score(y_te, yhat_05)), 4),
               "precision": round(float(precision_score(y_te, yhat_05)), 4),
               "recall": round(float(recall_score(y_te, yhat_05)), 4),
               "f1": round(float(f1_score(y_te, yhat_05)), 4)},
    "at_best_thr": {"accuracy": round(float(accuracy_score(y_te, yhat_t)), 4),
                    "precision": round(float(precision_score(y_te, yhat_t)), 4),
                    "recall": round(float(recall_score(y_te, yhat_t)), 4),
                    "f1": round(float(f1_score(y_te, yhat_t)), 4)},
    "confusion_best_thr": confusion_matrix(y_te, yhat_t).tolist(),
    "confusion_0.5": confusion_matrix(y_te, yhat_05).tolist(),
})
print(json.dumps(tuned_res, indent=2))
print(classification_report(y_te, yhat_t))

# Model comparison bar chart
names = list(results)
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
fig, ax = plt.subplots(figsize=(11, 4.5))
w = 0.15
x = np.arange(len(names))
cols = ["#0B3C5D", "#1D7874", "#F18F01", "#C0392B", "#5C6B7A"]
for i, m in enumerate(metrics):
    vals = [results[n][m] for n in names]
    ax.bar(x + (i - 2) * w, vals, w,
           label=m.replace("_", "-").upper() if m == "roc_auc" else m.capitalize(), color=cols[i])
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score (test set)")
ax.set_title("Model comparison on held-out test set (threshold = 0.5)", fontsize=12)
ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.tight_layout()
plt.savefig(FIG_DIR / "07_model_comparison.png", dpi=200)
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(5.5, 5))
for (n, p), c in zip(probs.items(), cols):
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, label=f"{n} (AUC = {results[n]['roc_auc']:.3f})", color=c, lw=1.8)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves (test set)", fontsize=12)
ax.legend(fontsize=8.5, loc="lower right")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_roc_curves.png", dpi=200)
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, cm, t in zip(axes, [tuned_res["confusion_0.5"], tuned_res["confusion_best_thr"]], [0.5, best_thr]):
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Actual: No", "Actual: Yes"],
                annot_kws={"size": 13})
    ax.set_title(f"{best} (tuned), threshold = {t:.2f}", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "09_confusion_matrices.png", dpi=200)
plt.close()

# Precision-recall vs threshold
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thr, prec[:-1], label="Precision", color="#0B3C5D")
ax.plot(thr, rec[:-1], label="Recall", color="#F18F01")
ax.plot(thr, f1s, label="F1", color="#1D7874")
ax.axvline(best_thr, color="gray", ls="--", lw=1, label=f"Best F1 threshold = {best_thr:.2f}")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_title(f"Precision / Recall / F1 vs threshold ({best})", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "10_threshold_tradeoff.png", dpi=200)
plt.close()

# Permutation feature importance
pi = permutation_importance(tuned, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=RS, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
top = imp.head(12)
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.barh(top.index[::-1], top.values[::-1], color="#0B3C5D")
ax.set_xlabel("Mean drop in ROC-AUC when feature is shuffled")
ax.set_title(f"Permutation feature importance ({best}, test set)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "11_permutation_feature_importance.png", dpi=200)
plt.close()

out = {
    "has_xgb": HAS_XGB,
    "n_train": int(len(X_tr)),
    "n_test": int(len(X_te)),
    "n_features_raw": int(X.shape[1]),
    "n_features_encoded": int(tuned.named_steps["pre"].transform(X_tr[:5]).shape[1]),
    "pos_weight": round(pos_w, 3),
    "results": results,
    "best": best,
    "tuned": tuned_res,
    "importance_top12": {k: round(float(v), 4) for k, v in top.items()},
    "num_cols": num_cols,
    "cat_cols": cat_cols,
}
with open(OUT_DIR / "model_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE")
