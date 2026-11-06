import joblib
import numpy as np
import pandas as pd

from ml.features import add_features
from .db import ROOT, SENSORS

PARTS = {"comp1": "Electrical unit (voltage)", "comp2": "Rotor (rotation)",
         "comp3": "Pressure line", "comp4": "Bearing (vibration)"}


class Predictor:
    def __init__(self, path=ROOT / "artifacts" / "model.joblib"):
        b = joblib.load(path)
        self.b, self.clf, self.cal = b, b["clf"], b["calib"]
        self.rul, self.ftype, self.iso = b["rul"], b["ftype"], b["iso"]
        self.explainer = None
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.clf)
        except Exception:
            pass  # logistic regression or shap missing: fall back to a simple attribution

    def _factors(self, x):
        v = None
        if self.explainer is not None:
            try:
                sv = self.explainer.shap_values(x)
                if isinstance(sv, list):
                    sv = sv[-1]
                sv = np.asarray(sv)
                v = (sv[:, :, -1] if sv.ndim == 3 else sv)[0]
            except Exception:
                v = None
        if v is None:
            imp = getattr(self.clf, "feature_importances_", np.ones(len(self.b["features"])))
            v = imp * (x.iloc[0].values - self.b["mu"]) / self.b["sd"]
        return {s: float(sum(val for f, val in zip(self.b["features"], v) if f.startswith(s))) for s in SENSORS}

    def predict(self, df: pd.DataFrame) -> dict:
        """df: recent hourly readings, oldest first, columns volt/rotate/pressure/vibration."""
        d = df[SENSORS].reset_index(drop=True).astype(float)
        d["machineID"] = 0
        x = add_features(d).iloc[[-1]][self.b["features"]]
        risk = float(self.cal.transform(self.clf.predict_proba(x)[:, 1])[0])
        part = str(self.ftype.predict(x)[0])
        return {
            "risk": round(risk, 4),
            "health": int(round(100 * (1 - risk))),
            "status": "critical" if risk >= 0.6 else "warning" if risk >= 0.3 else "healthy",
            "rul_days": round(float(np.clip(self.rul.predict(x)[0], 0.5, 30)), 1) if risk >= 0.2 else None,
            "part": part, "part_name": PARTS.get(part, part),
            "factors": self._factors(x),
            "anomaly": bool(self.iso.predict(d[SENSORS].tail(1))[0] == -1),
            "horizon_days": 7,
        }

    def risk_series(self, df: pd.DataFrame, k: int = 24) -> list:
        """Calibrated risk at each of the last k hours (feature windows use only past readings)."""
        d = df[SENSORS].reset_index(drop=True).astype(float)
        d["machineID"] = 0
        x = add_features(d)[self.b["features"]].iloc[-k:]
        r = self.cal.transform(self.clf.predict_proba(x)[:, 1])
        return [round(float(v), 4) for v in r]

    def drift(self, df: pd.DataFrame) -> dict:
        """Population Stability Index of live data vs training data. Above 0.25 means retrain."""
        out = {}
        for s in SENSORS:
            edges = self.b["ref"][s][1:-1]
            a = np.bincount(np.digitize(df[s].values, edges), minlength=10) / max(len(df), 1)
            a, e = np.clip(a, 1e-4, None), np.full(10, 0.1)
            out[s] = round(float(((a - e) * np.log(a / e)).sum()), 3)
        return out
