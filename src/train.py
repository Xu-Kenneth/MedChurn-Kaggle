import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.features import FEATURE_COLS


def temporal_split(df: pd.DataFrame, train_frac: float = 0.75):
    n = int(len(df) * train_frac)
    return df.iloc[:n].copy(), df.iloc[n:].copy()


def train_and_evaluate(df: pd.DataFrame):
    train, test = temporal_split(df)

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["no_show"]
    X_test  = test[FEATURE_COLS].fillna(0)
    y_test  = test["no_show"]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=neg / pos,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc_roc":       round(float(roc_auc_score(y_test, probs)), 4),
        "auprc":         round(float(average_precision_score(y_test, probs)), 4),
        "f1":            round(float(f1_score(y_test, preds)), 4),
        "train_rows":    int(len(train)),
        "test_rows":     int(len(test)),
        "no_show_rate":  round(float(y_train.mean()), 4),
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_importance = (
        pd.DataFrame({
            "feature": FEATURE_COLS,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    scored = test[["PatientId", "AppointmentID", "AppointmentDay"]].copy()
    scored["no_show_probability"] = probs.round(4)
    scored["risk_bucket"] = pd.cut(
        probs,
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
    )

    return model, metrics, shap_importance, scored
