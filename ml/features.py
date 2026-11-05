"""Rolling-window features and the 7-day failure target (no future leakage)."""
import numpy as np
import pandas as pd

SENSORS = ["volt", "rotate", "pressure", "vibration"]
WINDOWS = (3, 12, 24)
HORIZON = 7
FEATURES = (
    SENSORS
    + [f"{s}_{a}_{w}h" for s in SENSORS for w in WINDOWS for a in ("mean", "std")]
    + [f"{s}_trend_24h" for s in SENSORS]
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly rows per machineID. Every feature uses only the past."""
    out = df.copy()
    if "datetime" in out:
        out = out.sort_values(["machineID", "datetime"])
    out = out.reset_index(drop=True)
    g = out.groupby("machineID")
    for s in SENSORS:
        for w in WINDOWS:
            r = g[s].rolling(w, min_periods=1)
            out[f"{s}_mean_{w}h"] = r.mean().reset_index(level=0, drop=True)
            out[f"{s}_std_{w}h"] = r.std().reset_index(level=0, drop=True).fillna(0)
        out[f"{s}_trend_24h"] = out[s] - out[f"{s}_mean_24h"]
    return out


def make_targets(df: pd.DataFrame, failures: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Adds days_to_fail, next_comp and fail_7d, built only from failure timestamps."""
    df = df.copy()
    days = np.full(len(df), np.nan)
    comp = np.full(len(df), "", dtype=object)
    T = df["datetime"].values.astype("datetime64[ns]").astype("int64")
    for m, idx in df.groupby("machineID").indices.items():
        f = failures[failures.machineID == m].sort_values("datetime")
        if f.empty:
            continue
        ft = f["datetime"].values.astype("datetime64[ns]").astype("int64")
        fc = f["failure"].values
        t = T[idx]
        pos = np.searchsorted(ft, t, side="right")  # first failure strictly after t
        ok = pos < len(ft)
        p = np.minimum(pos, len(ft) - 1)
        days[idx] = np.where(ok, (ft[p] - t) / 86400e9, np.nan)
        comp[idx] = np.where(ok, fc[p], "")
    df["days_to_fail"] = days
    df["next_comp"] = comp
    df["fail_7d"] = (df["days_to_fail"] <= horizon).astype(int)
    return df
