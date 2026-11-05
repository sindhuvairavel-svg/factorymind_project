import pytest
from fastapi.testclient import TestClient
from app.db import ROOT

pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "model.joblib").exists(),
                                reason="run python -m ml.simulate_data and python -m ml.train first")


@pytest.fixture(scope="module")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_fleet_sorted_by_risk(client):
    rows = client.get("/api/fleet").json()
    assert len(rows) > 0
    assert [r["risk"] for r in rows] == sorted((r["risk"] for r in rows), reverse=True)


def test_predict_ok(client):
    body = {"readings": [{"volt": 170, "rotate": 446, "pressure": 100, "vibration": 40}]}
    r = client.post("/api/predict", json=body).json()
    assert 0 <= r["risk"] <= 1 and r["status"] in ("healthy", "warning", "critical")


def test_predict_rejects_bad_values(client):
    body = {"readings": [{"volt": -5, "rotate": 446, "pressure": 100, "vibration": 40}]}
    assert client.post("/api/predict", json=body).status_code == 422


def test_unknown_machine(client):
    assert client.get("/api/machines/9999").status_code == 404
