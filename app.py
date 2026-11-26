"""
Customer Churn & Retention Engine - Interactive Web Application

Runs a local or hosted web app on top of the ML pipeline:
  * Feature engineering, preprocessing, and 80/20 stratified train/test split
  * 5 models compared with 5-fold stratified cross-validation
  * GridSearchCV tuning on the best model, F1-optimal threshold, permutation importance
  * Per-customer XGBoost driver attribution (pred_contribs) and what-if retention levers

Start locally with: python app.py (or start_app.bat on Windows).
The first run trains and caches artifacts in cache/model_cache.joblib.
"""

import os
import sys
import json
import time
import random
import threading
import warnings
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier, DMatrix

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "telco_churn.csv")
UI = os.path.join(BASE, "web", "index.html")
CACHE_DIR = os.path.join(BASE, "cache")
CACHE = os.path.join(CACHE_DIR, "model_cache.joblib")
CACHE_VERSION = 3
RS = 42

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumAddOnServices", "SeniorCitizen"]
RAW_FIELDS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
              "InternetService"] + SERVICES + ["Contract", "PaperlessBilling", "PaymentMethod",
                                               "MonthlyCharges", "TotalCharges"]

LABELS = {
    "gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Has partner", "Dependents": "Has dependents",
    "tenure": "Tenure (months)", "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet service", "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection", "TechSupport": "Tech support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "Contract": "Contract", "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
    "AvgMonthlySpend": "Avg monthly spend", "NumAddOnServices": "Add-on services", "TenureGroup": "Tenure group",
}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data and features
# ---------------------------------------------------------------------------
def engineer(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1),
                                     df["MonthlyCharges"])
    df["NumAddOnServices"] = (df[SERVICES] == "Yes").sum(axis=1)
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                               labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    return df


def load_data():
    raw = pd.read_csv(DATA)
    df = engineer(raw)
    ids = df["customerID"]
    df = df.drop(columns=["customerID"])
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return raw, X, y, ids


def make_pre(cat_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
    ])


def r4(v):
    return round(float(v), 4)


# ---------------------------------------------------------------------------
# Training (first run only, then cached)
# ---------------------------------------------------------------------------
def train():
    t_all = time.time()
    raw, X, y, ids = load_data()
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    pos_w = float((y_tr == 0).sum() / (y_tr == 1).sum())

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=RS),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced",
                                                random_state=RS),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced_subsample", random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                                        subsample=0.9, random_state=RS),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9,
                                 colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="logloss",
                                 random_state=RS, n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    grid_fpr = np.linspace(0, 1, 101)
    results = []
    for i, (name, clf) in enumerate(models.items(), 1):
        log(f"  [{i}/5] {name} ...")
        t0 = time.time()
        pipe = Pipeline([("pre", make_pre(cat_cols)), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        yhat = (p >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_te, p)
        results.append({
            "name": name,
            "cv_auc_mean": r4(cv_auc.mean()), "cv_auc_std": r4(cv_auc.std()),
            "accuracy": r4(accuracy_score(y_te, yhat)), "precision": r4(precision_score(y_te, yhat)),
            "recall": r4(recall_score(y_te, yhat)), "f1": r4(f1_score(y_te, yhat)),
            "roc_auc": r4(roc_auc_score(y_te, p)),
            "roc": [r4(v) for v in np.interp(grid_fpr, fpr, tpr)],
            "train_time_s": round(time.time() - t0, 1),
        })
    best = max(results, key=lambda r: r["roc_auc"])["name"]
    log(f"  Best by test ROC-AUC: {best}. Tuning with grid search ...")

    if best == "XGBoost":
        grid = {"clf__max_depth": [3, 4, 5], "clf__learning_rate": [0.02, 0.05], "clf__n_estimators": [300, 500]}
    elif best == "Gradient Boosting":
        grid = {"clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.03, 0.05, 0.1], "clf__n_estimators": [200, 300]}
    elif best == "Random Forest":
        grid = {"clf__max_depth": [8, 10, 14], "clf__min_samples_leaf": [3, 5, 10]}
    else:
        grid = {"clf__C": [0.1, 0.5, 1, 5]}
    gs = GridSearchCV(Pipeline([("pre", make_pre(cat_cols)), ("clf", models[best])]), grid, cv=cv,
                      scoring="roc_auc", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    tuned = gs.best_estimator_
    p_t = tuned.predict_proba(X_te)[:, 1]

    prec, rec, thr = precision_recall_curve(y_te, p_t)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_thr = float(thr[int(np.argmax(f1s))])
    yhat_t = (p_t >= best_thr).astype(int)
    yhat_05 = (p_t >= 0.5).astype(int)

    def metrics(yh):
        return {"accuracy": r4(accuracy_score(y_te, yh)), "precision": r4(precision_score(y_te, yh)),
                "recall": r4(recall_score(y_te, yh)), "f1": r4(f1_score(y_te, yh))}

    fpr, tpr, _ = roc_curve(y_te, p_t)
    tuned_res = {
        "model": best,
        "best_params": {k.replace("clf__", ""): (float(v) if isinstance(v, float) else int(v))
                        for k, v in gs.best_params_.items()},
        "cv_auc": r4(gs.best_score_), "roc_auc": r4(roc_auc_score(y_te, p_t)),
        "roc": [r4(v) for v in np.interp(grid_fpr, fpr, tpr)],
        "threshold": round(best_thr, 4),
        "at_best_thr": metrics(yhat_t), "at_0.5": metrics(yhat_05),
        "confusion_best_thr": confusion_matrix(y_te, yhat_t).tolist(),
    }

    log("  Permutation importance ...")
    pi = permutation_importance(tuned, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=RS, n_jobs=-1)
    imp = pd.DataFrame({"f": X.columns, "m": pi.importances_mean, "s": pi.importances_std}) \
        .sort_values("m", ascending=False).head(12)
    importance = [{"feature": r.f, "label": LABELS.get(r.f, r.f), "value": r4(r.m), "std": r4(r.s)}
                  for r in imp.itertuples()]

    # Cumulative gains on the test set
    order = np.argsort(-p_t)
    ys = y_te.values[order]
    n = len(ys)
    gains = [{"pct": 0, "capture": 0.0}]
    for d in range(1, 11):
        k = int(round(n * d / 10))
        gains.append({"pct": d * 10, "capture": r4(ys[:k].sum() / ys.sum())})

    # Segment churn rates (whole dataset)
    full = X.copy()
    full["Churn"] = y

    def seg(col, order_list=None):
        g = full.groupby(col)["Churn"].agg(["mean", "count"])
        keys = order_list or list(g.sort_values("mean", ascending=False).index)
        return [{"label": str(k), "rate": r4(g.loc[k, "mean"]), "n": int(g.loc[k, "count"])} for k in keys]

    segments = {
        "Contract": seg("Contract", ["Month-to-month", "One year", "Two year"]),
        "TenureGroup": seg("TenureGroup", ["0-12", "13-24", "25-48", "49-72"]),
        "InternetService": seg("InternetService", ["Fiber optic", "DSL", "No"]),
        "PaymentMethod": seg("PaymentMethod"),
    }

    pre = tuned.named_steps["pre"]
    summary = {
        "dataset": {"n": int(len(X)), "churn_rate": r4(y.mean()), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                    "n_features_raw": len(RAW_FIELDS), "n_features_model": int(X.shape[1]),
                    "n_features_encoded": int(len(pre.get_feature_names_out())), "pos_weight": round(pos_w, 3)},
        "models": results, "best": best, "tuned": tuned_res, "importance": importance, "gains": gains,
        "segments": segments,
        "test": {"p": [r4(v) for v in p_t], "y": [int(v) for v in y_te.values]},
        "train_seconds": round(time.time() - t_all, 1),
    }
    test_rows = raw.loc[X_te.index, ["customerID"] + RAW_FIELDS + ["Churn"]].copy()
    test_rows["p"] = p_t
    return {"version": CACHE_VERSION, "data_mtime": os.path.getmtime(DATA), "model": tuned,
            "summary": summary, "test_rows": test_rows}


def get_state():
    if os.path.exists(CACHE):
        try:
            st = joblib.load(CACHE)
            if st.get("version") == CACHE_VERSION and (st.get("pinned") or st.get("data_mtime") == os.path.getmtime(DATA)):
                log("Loaded trained models from cache/model_cache.joblib")
                return st
        except Exception as e:
            log(f"Cache could not be used ({e}); retraining.")
    log("First run: training all models (about 1 to 2 minutes) ...")
    st = train()
    os.makedirs(CACHE_DIR, exist_ok=True)
    joblib.dump(st, CACHE)
    log(f"Training done in {st['summary']['train_seconds']} s. Saved to cache/model_cache.joblib")
    return st


# ---------------------------------------------------------------------------
# Prediction with per-customer drivers
# ---------------------------------------------------------------------------
class Predictor:
    def __init__(self, state):
        self.model = state["model"]
        self.summary = state["summary"]
        self.thr = state["summary"]["tuned"]["threshold"]
        self.test_rows = state["test_rows"]
        self.pre = self.model.named_steps["pre"]
        self.clf = self.model.named_steps["clf"]
        self.is_xgb = isinstance(self.clf, XGBClassifier)
        names = list(self.pre.get_feature_names_out())
        cat_cols = list(self.pre.transformers_[1][2])
        self.enc_to_raw = []
        for nm in names:
            kind, rest = nm.split("__", 1)
            if kind == "num":
                self.enc_to_raw.append(rest)
            else:
                match = max((c for c in cat_cols if rest == c or rest.startswith(c + "_")), key=len)
                self.enc_to_raw.append(match)
        tr = self.test_rows
        top_idx = tr.sort_values("p", ascending=False).head(25)
        self.at_risk = []
        for _, r in top_idx.iterrows():
            rec = {k: r[k] for k in RAW_FIELDS}
            res = self.predict(rec, levers=False)
            self.at_risk.append({
                "id": r["customerID"], "p": r4(r["p"]), "tenure": int(r["tenure"]), "contract": r["Contract"],
                "internet": r["InternetService"], "monthly": float(r["MonthlyCharges"]),
                "driver": res["drivers"][0]["text"] if res["drivers"] else "",
                "actual": r["Churn"],
            })

    @staticmethod
    def normalize(d):
        rec = {}
        for k in RAW_FIELDS:
            rec[k] = d.get(k)
        sc = rec["SeniorCitizen"]
        rec["SeniorCitizen"] = 1 if str(sc) in ("1", "Yes", "True", "true") else 0
        rec["tenure"] = int(float(rec["tenure"] or 0))
        rec["MonthlyCharges"] = float(rec["MonthlyCharges"] or 0)
        tc = rec.get("TotalCharges")
        try:
            tc = float(tc)
        except (TypeError, ValueError):
            tc = None
        if tc is None:
            tc = round(rec["tenure"] * rec["MonthlyCharges"], 2)
        rec["TotalCharges"] = tc
        if rec["InternetService"] == "No":
            for s in SERVICES:
                rec[s] = "No internet service"
        else:
            for s in SERVICES:
                if rec[s] not in ("Yes", "No"):
                    rec[s] = "No"
        if rec["PhoneService"] == "No":
            rec["MultipleLines"] = "No phone service"
        elif rec["MultipleLines"] not in ("Yes", "No"):
            rec["MultipleLines"] = "No"
        return rec

    def _frame(self, rec):
        return engineer(pd.DataFrame([rec]))

    def prob(self, rec):
        return float(self.model.predict_proba(self._frame(rec))[0, 1])

    def band(self, p):
        if p >= self.thr:
            return "High"
        if p >= self.thr * 0.6:
            return "Medium"
        return "Low"

    def drivers(self, rec):
        if not self.is_xgb:
            return []
        Xt = self.pre.transform(self._frame(rec))
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        contrib = self.clf.get_booster().predict(DMatrix(Xt), pred_contribs=True)[0]
        agg = {}
        for j, raw in enumerate(self.enc_to_raw):
            agg[raw] = agg.get(raw, 0.0) + float(contrib[j])
        fr = self._frame(rec).iloc[0]
        out = []
        for raw, v in agg.items():
            val = fr[raw]
            if raw in ("MonthlyCharges", "TotalCharges", "AvgMonthlySpend"):
                vtxt = f"{float(val):,.2f}"
            elif raw == "SeniorCitizen":
                vtxt = "Yes" if int(val) == 1 else "No"
            else:
                vtxt = str(val)
            text = f"Tenure: {vtxt} months" if raw == "tenure" else f"{LABELS.get(raw, raw)}: {vtxt}"
            out.append({"feature": raw, "label": LABELS.get(raw, raw), "value": vtxt, "impact": r4(v), "text": text})
        out.sort(key=lambda d: -abs(d["impact"]))
        return out

    def levers(self, rec, p0):
        cands = []
        if rec["Contract"] == "Month-to-month":
            cands.append(("Move to a one-year contract", {"Contract": "One year"}))
        if rec["Contract"] != "Two year":
            cands.append(("Move to a two-year contract", {"Contract": "Two year"}))
        if rec["InternetService"] != "No":
            if rec["TechSupport"] == "No":
                cands.append(("Add tech support", {"TechSupport": "Yes"}))
            if rec["OnlineSecurity"] == "No":
                cands.append(("Add online security", {"OnlineSecurity": "Yes"}))
        if rec["PaymentMethod"] in ("Electronic check", "Mailed check"):
            cands.append(("Switch to automatic card payment", {"PaymentMethod": "Credit card (automatic)"}))
        out = []
        for label, change in cands:
            r2 = dict(rec)
            r2.update(change)
            p1 = self.prob(r2)
            if p1 < p0 - 0.005:
                out.append({"label": label, "p": r4(p1), "delta": r4(p1 - p0), "band": self.band(p1)})
        out.sort(key=lambda d: d["delta"])
        return out[:4]

    def predict(self, d, levers=True):
        rec = self.normalize(d)
        p = self.prob(rec)
        res = {"p": r4(p), "band": self.band(p), "threshold": self.thr, "flagged": p >= self.thr,
               "record": rec, "drivers": self.drivers(rec)[:8]}
        if levers:
            res["levers"] = self.levers(rec, p)
        return res

    def sample(self, kind):
        tr = self.test_rows
        if kind == "high":
            pool = tr[tr["p"] >= max(self.thr, 0.8)]
        elif kind == "low":
            pool = tr[tr["p"] <= 0.1]
        else:
            pool = tr
        if len(pool) == 0:
            pool = tr
        r = pool.iloc[random.randrange(len(pool))]
        return self.row_payload(r)

    def by_id(self, cid):
        m = self.test_rows[self.test_rows["customerID"] == cid]
        return self.row_payload(m.iloc[0]) if len(m) else None

    @staticmethod
    def row_payload(r):
        rec = {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in RAW_FIELDS}
        rec["SeniorCitizen"] = "Yes" if int(rec["SeniorCitizen"]) == 1 else "No"
        return {"id": r["customerID"], "record": rec, "actual": r["Churn"]}


# ---------------------------------------------------------------------------
# Web server
# ---------------------------------------------------------------------------
PRED = None


def to_json(obj):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(str(type(o)))
    return json.dumps(obj, default=conv).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if u.path in ("/", "/index.html"):
                with open(UI, "rb") as f:
                    self._send(200, f.read(), "text/html")
            elif u.path == "/api/summary":
                s = dict(PRED.summary)
                s["at_risk"] = PRED.at_risk
                self._send(200, to_json(s))
            elif u.path == "/api/sample":
                q = parse_qs(u.query)
                if "id" in q:
                    res = PRED.by_id(q["id"][0])
                    if res is None:
                        return self._send(404, to_json({"error": "not found"}))
                else:
                    res = PRED.sample(q.get("kind", ["random"])[0])
                self._send(200, to_json(res))
            else:
                self._send(404, to_json({"error": "not found"}))
        except Exception as e:
            self._send(500, to_json({"error": str(e)}))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/predict":
                self._send(200, to_json(PRED.predict(body)))
            else:
                self._send(404, to_json({"error": "not found"}))
        except Exception as e:
            self._send(500, to_json({"error": str(e)}))


def main():
    global PRED
    os.chdir(BASE)
    print("=" * 60)
    print(" Customer Churn & Retention Engine")
    print("=" * 60)
    state = get_state()
    PRED = Predictor(state)

    env_port = os.environ.get("PORT")
    host = os.environ.get("HOST", "0.0.0.0" if env_port else "127.0.0.1")

    server = None
    if env_port:
        port = int(env_port)
        server = ThreadingHTTPServer((host, port), Handler)
    else:
        for port in range(8765, 8785):
            try:
                server = ThreadingHTTPServer((host, port), Handler)
                break
            except OSError:
                continue

    if server is None:
        print("No free port found between 8765 and 8784.")
        sys.exit(1)

    url = f"http://{host}:{port}"
    print()
    print(f" App is running at {url}")
    if not env_port:
        print(" Your browser should open automatically. If not, paste the link above.")
    print(" Press Ctrl+C (or close this window) to stop the app.")
    print()
    if "--no-browser" not in sys.argv and not env_port:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
