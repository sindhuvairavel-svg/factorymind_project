import json
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .service import Predictor, SENSORS

S = {}
rng = np.random.default_rng()
MU = {"volt": 170, "rotate": 446, "pressure": 100, "vibration": 40}
SD = {"volt": 15, "rotate": 50, "pressure": 10, "vibration": 5}


@asynccontextmanager
async def lifespan(app):
    db.init()
    db.seed(db.ROOT / "data" / "PdM_telemetry.csv")
    S["p"] = Predictor()
    yield


app = FastAPI(title="FactoryMind API", lifespan=lifespan)


class Reading(BaseModel):
    volt: float = Field(ge=0, le=500)
    rotate: float = Field(ge=0, le=2000)
    pressure: float = Field(ge=0, le=300)
    vibration: float = Field(ge=0, le=200)


class PredictIn(BaseModel):
    readings: List[Reading] = Field(min_items=1, max_items=48)


def name(m):
    return f"MACH-{m:03d}"


def run(m, df, save=False):
    r = S["p"].predict(df)
    r.update(machine_id=m, name=name(m))
    if save:
        db.save_prediction(m, r)
    return r


@app.get("/api/health")
def health():
    return {"status": "ok", "model": S["p"].b["name"]}


@app.get("/api/fleet")
def fleet():
    rows = [run(m, db.recent(m)) for m in db.machines()]
    return sorted(rows, key=lambda r: -r["risk"])


@app.get("/api/machines/{m}")
def machine(m: int):
    df = db.recent(m)
    if df.empty:
        raise HTTPException(404, "Unknown machine")
    r = run(m, df, save=True)
    r["readings"] = df.round(1).to_dict("records")
    return r


@app.post("/api/machines/{m}/simulate")
def simulate(m: int, vibration_boost: float = 0.0):
    """Adds one simulated hourly reading. vibration_boost pushes vibration up (fault demo)."""
    df = db.recent(m, 1)
    if df.empty:
        raise HTTPException(404, "Unknown machine")
    last = df.iloc[-1]
    new = {s: float(0.8 * last[s] + 0.2 * MU[s] + rng.normal(0, SD[s] * 0.3)) for s in SENSORS}
    new["vibration"] += vibration_boost * SD["vibration"] * 0.4
    db.add_reading(m, new)
    return machine(m)


@app.post("/api/predict")
def predict(p: PredictIn):
    """Manual entry or CSV upload. Send up to 48 readings, oldest first."""
    return S["p"].predict(pd.DataFrame([r.dict() for r in p.readings]))


@app.get("/api/machines/{m}/history")
def history(m: int):
    return db.history(m)


@app.get("/api/drift")
def drift():
    return S["p"].drift(db.recent_all())


@app.get("/api/models")
def models():
    return json.loads((db.ROOT / "artifacts" / "metrics.json").read_text())


app.mount("/", StaticFiles(directory=db.ROOT / "frontend", html=True), name="ui")