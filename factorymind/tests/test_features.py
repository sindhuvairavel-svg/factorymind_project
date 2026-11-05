import numpy as np
import pandas as pd
from ml.features import FEATURES, SENSORS, add_features, make_targets


def frame(n=300):
    rng = np.random.default_rng(0)
    d = {s: rng.normal(100, 5, n) for s in SENSORS}
    return pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=n, freq="h"), "machineID": 1, **d})


def test_feature_columns_exist():
    out = add_features(frame())
    assert set(FEATURES) <= set(out.columns)
    assert not out[FEATURES].isna().any().any()


def test_features_use_only_the_past():
    a = frame()
    b = a.copy()
    b.loc[200:, "vibration"] += 1000  # change the future
    fa, fb = add_features(a), add_features(b)
    cols = [c for c in FEATURES if c.startswith("vibration")]
    assert fa.loc[:199, cols].equals(fb.loc[:199, cols])


def test_target_marks_only_the_week_before_failure():
    df = frame()
    fl = pd.DataFrame({"datetime": [df.datetime[250]], "machineID": [1], "failure": ["comp2"]})
    out = make_targets(df, fl)
    assert out.fail_7d[250 - 168: 250].eq(1).all()   # inside the 7 days before
    assert out.fail_7d[:250 - 169].eq(0).all()        # earlier
    assert out.fail_7d[250:].eq(0).all()              # after the failure
    assert (out.next_comp[:250] == "comp2").all()
