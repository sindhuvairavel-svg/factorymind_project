import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "factorymind.db"
SENSORS = ["volt", "rotate", "pressure", "vibration"]


def conn():
    DB.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB)


def init():
    with conn() as c:
        c.execute("create table if not exists readings(id integer primary key autoincrement, machine_id int, ts text, volt real, rotate real, pressure real, vibration real)")
        c.execute("create table if not exists predictions(id integer primary key autoincrement, machine_id int, ts text default current_timestamp, risk real, rul real, part text, factors text)")


def seed(csv, hours=48):
    """Loads the latest readings of every machine on first start."""
    with conn() as c:
        if c.execute("select count(*) from readings").fetchone()[0]:
            return
        df = pd.read_csv(csv, parse_dates=["datetime"]).sort_values(["machineID", "datetime"])
        df = df.groupby("machineID").tail(hours).rename(columns={"machineID": "machine_id", "datetime": "ts"})
        df["ts"] = df["ts"].astype(str)
        df[["machine_id", "ts", *SENSORS]].to_sql("readings", c, if_exists="append", index=False)


def machines():
    with conn() as c:
        return [r[0] for r in c.execute("select distinct machine_id from readings order by 1")]


def recent(m, n=24):
    with conn() as c:
        return pd.read_sql("select volt,rotate,pressure,vibration from (select * from readings where machine_id=? order by id desc limit ?) order by id",
                           c, params=(m, n))


def recent_all():
    with conn() as c:
        return pd.read_sql("select machine_id,volt,rotate,pressure,vibration from readings", c)


def add_reading(m, v):
    with conn() as c:
        c.execute("insert into readings(machine_id,ts,volt,rotate,pressure,vibration) values(?,datetime('now'),?,?,?,?)",
                  (m, v["volt"], v["rotate"], v["pressure"], v["vibration"]))


def save_prediction(m, r):
    with conn() as c:
        c.execute("insert into predictions(machine_id,risk,rul,part,factors) values(?,?,?,?,?)",
                  (m, r["risk"], r["rul_days"], r["part"], json.dumps(r["factors"])))


def history(m, n=50):
    with conn() as c:
        return pd.read_sql("select ts,risk,rul,part from predictions where machine_id=? order by id desc limit ?", c, params=(m, n)).to_dict("records")
