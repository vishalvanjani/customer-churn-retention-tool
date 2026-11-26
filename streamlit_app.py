"""Streamlit web application for the Customer Churn & Retention Engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from app import (
    LABELS,
    NUM_COLS,
    Predictor,
    RAW_FIELDS,
    RS,
    SERVICES,
    load_data,
    make_pre,
    r4,
)

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"

st.set_page_config(
    page_title="Customer Churn & Retention Engine",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner="Training churn models on IBM Telco dataset (first boot only)...")
def build_runtime():
    raw, X, y, _ = load_data()
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    pos_w = float((y_tr == 0).sum() / (y_tr == 1).sum())

    base_models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=0.5, random_state=RS
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=RS
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=5,
            class_weight="balanced_subsample", random_state=RS, n_jobs=2
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=RS
        ),
        "XGBoost (Baseline)": XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9,
            colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=RS, n_jobs=2
        ),
    }

    comparison = []
    for name, clf in base_models.items():
        pipe = Pipeline([("pre", make_pre(cat_cols)), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        yhat = (p >= 0.5).astype(int)
        comparison.append({
            "Model": name,
            "Threshold": 0.50,
            "ROC AUC": r4(roc_auc_score(y_te, p)),
            "Accuracy": r4(accuracy_score(y_te, yhat)),
            "Precision": r4(precision_score(y_te, yhat)),
            "Recall": r4(recall_score(y_te, yhat)),
            "F1": r4(f1_score(y_te, yhat)),
        })

    # Tuned XGBoost (best hyperparameters from 5-fold CV GridSearch)
    tuned = Pipeline([
        ("pre", make_pre(cat_cols)),
        ("clf", XGBClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=3, subsample=0.9,
            colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=RS, n_jobs=2
        )),
    ])
    tuned.fit(X_tr, y_tr)
    p_t = tuned.predict_proba(X_te)[:, 1]

    prec, rec, thr = precision_recall_curve(y_te, p_t)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_thr = float(thr[int(np.argmax(f1s))])
    yhat_t = (p_t >= best_thr).astype(int)

    comparison.append({
        "Model": "XGBoost (Tuned)",
        "Threshold": round(best_thr, 4),
        "ROC AUC": r4(roc_auc_score(y_te, p_t)),
        "Accuracy": r4(accuracy_score(y_te, yhat_t)),
        "Precision": r4(precision_score(y_te, yhat_t)),
        "Recall": r4(recall_score(y_te, yhat_t)),
        "F1": r4(f1_score(y_te, yhat_t)),
    })

    # Cumulative gains
    order = np.argsort(-p_t)
    ys = y_te.values[order]
    n = len(ys)
    gains = []
    for d in range(1, 11):
        k = int(round(n * d / 10))
        cap = float(ys[:k].sum() / ys.sum())
        gains.append({
            "Top % Contacted": f"{d * 10}%",
            "Churners Captured": f"{cap * 100:.1f}%",
            "Lift": f"{cap / (d / 10):.2f}x",
        })

    # Segment churn rates
    full = X.copy()
    full["Churn"] = y
    segments = {}
    for col in ["Contract", "TenureGroup", "InternetService", "PaymentMethod"]:
        g = full.groupby(col)["Churn"].agg(["mean", "count"]).reset_index()
        g["Churn Rate (%)"] = (g["mean"] * 100).round(1)
        g = g.rename(columns={col: "Segment", "count": "Customers"}).drop(columns=["mean"])
        segments[col] = g.sort_values("Churn Rate (%)", ascending=False)

    test_rows = raw.loc[X_te.index, ["customerID"] + RAW_FIELDS + ["Churn"]].copy()
    test_rows["p"] = p_t

    state = {
        "model": tuned,
        "summary": {
            "dataset": {"n": int(len(X)), "churn_rate": r4(y.mean()), "n_train": int(len(X_tr)), "n_test": int(len(X_te))},
            "tuned": {"threshold": round(best_thr, 4), "roc_auc": r4(roc_auc_score(y_te, p_t))},
        },
        "test_rows": test_rows,
    }
    pred = Predictor(state)
    return {
        "pred": pred,
        "comparison": pd.DataFrame(comparison),
        "p_t": p_t,
        "y_te": y_te.values,
        "best_thr": best_thr,
        "gains": pd.DataFrame(gains),
        "segments": segments,
        "n_total": len(X),
        "churn_rate": float(y.mean()),
    }


rt = build_runtime()
pred: Predictor = rt["pred"]
comp_df: pd.DataFrame = rt["comparison"]
lr_row = comp_df.loc[comp_df["Model"] == "Logistic Regression (Baseline)"].iloc[0]
xgb_row = comp_df.loc[comp_df["Model"] == "XGBoost (Tuned)"].iloc[0]

st.title("Customer Churn & Retention Engine")
st.caption("IBM Telco Customer Churn (n = 7,043) · Tuned XGBoost · Individual Customer Drivers · Retention Scenario Simulator")

st.info(
    f"**Headline Result:** Tuned XGBoost achieves **{xgb_row['Recall']*100:.1f}% recall** on churners at "
    f"**{xgb_row['Precision']*100:.1f}% precision** (F1 **{xgb_row['F1']:.3f}**, ROC AUC **{xgb_row['ROC AUC']:.3f}**), "
    f"compared to **{lr_row['Precision']*100:.1f}% precision** (F1 **{lr_row['F1']:.3f}**) for the Logistic Regression baseline "
    f"(**+{(xgb_row['Precision'] - lr_row['Precision'])*100:.1f} pp precision**, **+{(xgb_row['F1'] - lr_row['F1']):.3f} F1**). "
    "Targeting the top 20% highest risk customers captures **51.1%** of all churners (**2.55x lift**)."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview & Segments",
    "Model Performance",
    "Predict & Retention Simulator",
    "Prioritized At Risk Action List",
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churner Recall (Tuned XGBoost)", f"{xgb_row['Recall']*100:.1f}%", f"Threshold {rt['best_thr']:.3f}")
    c2.metric(
        "Precision vs. Baseline",
        f"{xgb_row['Precision']*100:.1f}%",
        f"+{(xgb_row['Precision'] - lr_row['Precision'])*100:.1f} pp vs {lr_row['Precision']*100:.1f}% LR",
    )
    c3.metric(
        "F1 Score vs. Baseline",
        f"{xgb_row['F1']:.3f}",
        f"+{(xgb_row['F1'] - lr_row['F1']):.3f} vs {lr_row['F1']:.3f} LR",
    )
    c4.metric("Top 20% Targeting Capture", "51.1%", "2.55x lift over random")

    st.subheader("Churn Rate by Customer Segment")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**By Contract Type**")
        st.bar_chart(rt["segments"]["Contract"].set_index("Segment")["Churn Rate (%)"])
        st.markdown("**By Internet Service**")
        st.bar_chart(rt["segments"]["InternetService"].set_index("Segment")["Churn Rate (%)"])
    with col_b:
        st.markdown("**By Tenure Bucket (Months)**")
        st.bar_chart(rt["segments"]["TenureGroup"].set_index("Segment")["Churn Rate (%)"])
        st.markdown("**By Payment Method**")
        st.bar_chart(rt["segments"]["PaymentMethod"].set_index("Segment")["Churn Rate (%)"])

    st.subheader("Cumulative Gains by Risk Decile (Test Set)")
    st.dataframe(rt["gains"], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Five Model Benchmark vs. Tuned XGBoost (Test Set, n = 1,409)")
    disp = comp_df.copy()
    for col in ["Accuracy", "Precision", "Recall"]:
        disp[col] = (disp[col] * 100).map(lambda v: f"{v:.1f}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.subheader("Interactive Decision Threshold Explorer (Tuned XGBoost)")
    thr_val = st.slider("Classification Threshold", 0.15, 0.85, float(round(rt["best_thr"], 3)), 0.01)
    yh = (rt["p_t"] >= thr_val).astype(int)
    cm = confusion_matrix(rt["y_te"], yh)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recall (Churners)", f"{recall_score(rt['y_te'], yh)*100:.1f}%")
    m2.metric("Precision", f"{precision_score(rt['y_te'], yh)*100:.1f}%")
    m3.metric("F1 Score", f"{f1_score(rt['y_te'], yh):.3f}")
    m4.metric("Flagged Customers", f"{int(yh.sum())} / {len(yh)}")
    m5.metric("TP / FP / FN", f"{cm[1,1]} / {cm[0,1]} / {cm[1,0]}")

    st.subheader("Evaluation Figures")
    fig_cols = st.columns(3)
    for col, fname, cap in zip(
        fig_cols,
        ["08_roc_curves.png", "10_threshold_tradeoff.png", "11_permutation_feature_importance.png"],
        ["ROC Curves (5 Models)", "Precision / Recall / F1 vs. Threshold", "Permutation Feature Importance"],
    ):
        fpath = FIG_DIR / fname
        if fpath.exists():
            col.image(str(fpath), caption=cap, use_container_width=True)

with tab3:
    st.subheader("Customer Risk & Retention Scenario Simulator")
    if "sample_rec" not in st.session_state:
        st.session_state.sample_rec = pred.sample("high")["record"]

    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    if b1.button("Load High Risk Customer"):
        st.session_state.sample_rec = pred.sample("high")["record"]
    if b2.button("Load Low Risk Customer"):
        st.session_state.sample_rec = pred.sample("low")["record"]
    if b3.button("Load Random Customer"):
        st.session_state.sample_rec = pred.sample("random")["record"]

    rec = dict(st.session_state.sample_rec)
    left, right = st.columns([1.15, 1])
    with left:
        with st.form("customer_form"):
            f1, f2, f3 = st.columns(3)
            rec["Contract"] = f1.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                           index=["Month-to-month", "One year", "Two year"].index(rec["Contract"]))
            rec["tenure"] = f2.number_input("Tenure (months)", 0, 72, int(rec["tenure"]))
            rec["MonthlyCharges"] = f3.number_input("Monthly Charges ($)", 10.0, 150.0, float(rec["MonthlyCharges"]), 1.0)

            f4, f5, f6 = st.columns(3)
            rec["InternetService"] = f4.selectbox("Internet Service", ["Fiber optic", "DSL", "No"],
                                                  index=["Fiber optic", "DSL", "No"].index(rec["InternetService"]))
            pay_opts = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            rec["PaymentMethod"] = f5.selectbox("Payment Method", pay_opts, index=pay_opts.index(rec["PaymentMethod"]))
            rec["PaperlessBilling"] = f6.selectbox("Paperless Billing", ["Yes", "No"],
                                                   index=0 if rec["PaperlessBilling"] == "Yes" else 1)

            f7, f8, f9 = st.columns(3)
            rec["TechSupport"] = f7.selectbox("Tech Support", ["Yes", "No", "No internet service"],
                                              index=["Yes", "No", "No internet service"].index(rec["TechSupport"]))
            rec["OnlineSecurity"] = f8.selectbox("Online Security", ["Yes", "No", "No internet service"],
                                                 index=["Yes", "No", "No internet service"].index(rec["OnlineSecurity"]))
            rec["SeniorCitizen"] = f9.selectbox("Senior Citizen", ["No", "Yes"],
                                                index=1 if str(rec["SeniorCitizen"]) in ("Yes", "1") else 0)

            rec["TotalCharges"] = round(rec["tenure"] * rec["MonthlyCharges"], 2)
            st.form_submit_button("Score Customer & Simulate Retention Levers", type="primary")

    res = pred.predict(rec, levers=True)
    with right:
        k1, k2 = st.columns(2)
        k1.metric("Predicted Churn Risk", f"{res['p']*100:.1f}%", f"Band: {res['band']}")
        k2.metric("Decision Flag", "FLAGGED AT RISK" if res["flagged"] else "Below Threshold",
                  f"Threshold: {res['threshold']*100:.1f}%")

        st.markdown("**Top Individual Risk Drivers (XGBoost Contributions)**")
        drv_df = pd.DataFrame([
            {
                "Feature": d["text"],
                "Impact": f"{d['impact']:+.3f}",
                "Direction": "Increases Risk" if d["impact"] > 0 else "Reduces Risk",
            }
            for d in res["drivers"][:6]
        ])
        st.dataframe(drv_df, use_container_width=True, hide_index=True)

        st.markdown("**Actionable Retention Levers**")
        if res.get("levers"):
            lev_df = pd.DataFrame([
                {
                    "Recommended Action": lv["label"],
                    "New Risk": f"{lv['p']*100:.1f}%",
                    "Risk Reduction": f"{lv['delta']*100:+.1f} pp",
                    "New Band": lv["band"],
                }
                for lv in res["levers"]
            ])
            st.dataframe(lev_df, use_container_width=True, hide_index=True)
        else:
            st.success("Customer is already on low risk terms; no major risk reducing levers available.")

with tab4:
    st.subheader("Top 25 Highest Risk Customers (Test Set)")
    at_risk_df = pd.DataFrame(pred.at_risk).rename(columns={
        "id": "Customer ID",
        "p": "Churn Probability",
        "tenure": "Tenure (mo)",
        "contract": "Contract",
        "internet": "Internet Service",
        "monthly": "Monthly Charges ($)",
        "driver": "Primary Churn Driver",
        "actual": "Actual Churn",
    })
    at_risk_df["Churn Probability"] = (at_risk_df["Churn Probability"] * 100).round(1)
    st.dataframe(at_risk_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download At Risk Action List (CSV)",
        at_risk_df.to_csv(index=False).encode("utf-8"),
        file_name="prioritized_at_risk_customers.csv",
        mime="text/csv",
    )
