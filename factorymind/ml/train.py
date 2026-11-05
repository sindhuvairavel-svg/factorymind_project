"""Trains and compares models, calibrates the best, and saves everything the API needs."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURES, HORIZON, SENSORS, add_features, make_targets

ROOT = Path(__file__).resolve().parent.parent
GAP = pd.Timedelta(days=HORIZON)


def candidates(pos_weight):
    m = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5,
                                               class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8,
                                     scale_pos_weight=pos_weight, eval_metric="aucpr", n_jobs=-1, random_state=0)
    except ImportError:
        print("xgboost not installed, skipping")
    return m


def best_threshold(y, p):
    if y.sum() == 0:
        return 0.5
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    return float(th[np.argmax(f1[:-1])])


def main():
    tel = pd.read_csv(ROOT / "data/PdM_telemetry.csv", parse_dates=["datetime"])
    fl = pd.read_csv(ROOT / "data/PdM_failures.csv", parse_dates=["datetime"])
    df = make_targets(add_features(tel), fl)

    # time-based split: past for training, future for testing, with a gap so labels never peek across
    cut = df.datetime.quantile(0.8)
    train, test = df[df.datetime < cut - GAP], df[df.datetime >= cut]
    cal_cut = train.datetime.quantile(0.85)
    fit, cal = train[train.datetime < cal_cut - GAP], train[train.datetime >= cal_cut]
    pos_w = (fit.fail_7d == 0).sum() / max(1, (fit.fail_7d == 1).sum())
    print(f"train {len(fit)} | calib {len(cal)} | test {len(test)} | failure rate {fit.fail_7d.mean():.3f}")

    metrics, fitted = {}, {}
    for name, model in candidates(pos_w).items():
        model.fit(fit[FEATURES], fit.fail_7d)
        thr = best_threshold(cal.fail_7d, model.predict_proba(cal[FEATURES])[:, 1])
        p = model.predict_proba(test[FEATURES])[:, 1]
        yhat = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(test.fail_7d, yhat, labels=[0, 1]).ravel()
        metrics[name] = dict(recall=recall_score(test.fail_7d, yhat), precision=precision_score(test.fail_7d, yhat, zero_division=0),
                             f1=f1_score(test.fail_7d, yhat), pr_auc=average_precision_score(test.fail_7d, p),
                             false_negatives=int(fn), false_positives=int(fp), threshold=thr)
        fitted[name] = model
        print(name, {k: round(v, 3) for k, v in metrics[name].items()})

    best = max(metrics, key=lambda k: metrics[k]["pr_auc"])
    clf = fitted[best]
    calib = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(cal[FEATURES])[:, 1], cal.fail_7d)

    near = train[train.days_to_fail <= 30]
    rul = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=0).fit(near[FEATURES], near.days_to_fail)
    pos = train[train.fail_7d == 1]
    ftype = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0).fit(pos[FEATURES], pos.next_comp)
    iso = IsolationForest(contamination=0.01, random_state=0).fit(train[SENSORS].sample(min(len(train), 50000), random_state=0))

    bundle = dict(clf=clf, name=best, calib=calib, rul=rul, ftype=ftype, iso=iso, features=FEATURES,
                  mu=train[FEATURES].mean().values, sd=train[FEATURES].std().values + 1e-9,
                  ref={s: np.quantile(train[s], np.linspace(0, 1, 11)).tolist() for s in SENSORS})
    (ROOT / "artifacts").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "artifacts/model.joblib")
    (ROOT / "artifacts/metrics.json").write_text(json.dumps({"best": best, "models": metrics}, indent=2))
    print("best model:", best)

    try:  # optional experiment tracking
        import mlflow
        with mlflow.start_run(run_name=best):
            mlflow.log_params({"horizon_days": HORIZON, "best_model": best})
            for n, m in metrics.items():
                mlflow.log_metrics({f"{n}_{k}": float(v) for k, v in m.items()})
    except ImportError:
        pass


if __name__ == "__main__":
    main()
