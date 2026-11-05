"""Creates data in the same format as the Azure PdM dataset, so the pipeline runs without Kaggle.
To use the real data, put PdM_telemetry.csv and PdM_failures.csv in data/ and skip this step."""
import numpy as np
import pandas as pd
from pathlib import Path

MU = {"volt": 170, "rotate": 446, "pressure": 100, "vibration": 40}
SD = {"volt": 15, "rotate": 50, "pressure": 10, "vibration": 5}
SIGN = {"comp1": ("volt", 1), "comp2": ("rotate", -1), "comp3": ("pressure", 1), "comp4": ("vibration", 1)}
PENDING = {12: "comp4", 7: "comp3", 23: "comp2"}  # machines that are degrading right now (for the demo)


def generate(n_machines=40, days=120, seed=7):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    tel, fails = [], []
    for m in range(1, n_machines + 1):
        d = {s: rng.normal(MU[s], SD[s], len(t)) for s in MU}
        for _ in range(rng.poisson(1.2)):
            i = int(rng.integers(200, len(t) - 1))
            comp = str(rng.choice(list(SIGN)))
            s, sg = SIGN[comp]
            lo = max(0, i - 168)
            d[s][lo:i] += sg * 3 * SD[s] * np.linspace(0, 1, i - lo) ** 2
            fails.append((t[i], m, comp))
        if m in PENDING:
            s, sg = SIGN[PENDING[m]]
            d[s][-120:] += sg * 3 * SD[s] * np.linspace(0.55, 1, 120) ** 2
        tel.append(pd.DataFrame({"datetime": t, "machineID": m, **d}))
    return pd.concat(tel), pd.DataFrame(fails, columns=["datetime", "machineID", "failure"])


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    tel, fl = generate()
    tel.round({s: 2 for s in MU}).to_csv(out / "PdM_telemetry.csv", index=False)
    fl.to_csv(out / "PdM_failures.csv", index=False)
    print(f"telemetry rows: {len(tel)}, failures: {len(fl)}")
