# FactoryMind

ML-based predictive maintenance: rolling-window features, model comparison (Logistic Regression, Random Forest, XGBoost),
calibrated 7-day failure risk, SHAP factors, RUL, failing-part prediction, anomaly and drift detection,
FastAPI backend with SQLite, and a dashboard.

## Run
```bash
pip install -r requirements.txt
python -m ml.simulate_data      # demo data (or put the Azure PdM CSVs in data/)
python -m ml.train              # trains, compares, calibrates, saves artifacts/
uvicorn app.main:app --reload   # open http://localhost:8000  (API docs: /docs)
pytest                          # tests
```
Docker: `docker compose up --build` then open http://localhost:8000

## Real data
Download the Microsoft Azure Predictive Maintenance dataset from Kaggle and copy `PdM_telemetry.csv` and
`PdM_failures.csv` into `data/`, then run `python -m ml.train`. Check the dataset licence first.
The data is synthetic, not real factory production data.

## Layout
- `ml/` data simulation, features and target (no future leakage), training
- `app/` FastAPI API, SQLite storage, prediction service
- `frontend/` dashboard (plain HTML and JS, served by FastAPI)
- `tests/` unit and API tests, `.github/workflows/ci.yml` runs them

## API
`GET /api/fleet`, `GET /api/machines/{id}`, `POST /api/machines/{id}/simulate`, `POST /api/predict`,
`GET /api/machines/{id}/history`, `GET /api/drift`, `GET /api/models`

## Next steps
Auto-retraining when PSI > 0.25, MLflow (pip install mlflow, train.py logs automatically), the 3D digital twin from the HTML demo,
survival analysis, a RAG assistant, role-based login.

## Python 3.8 (Windows 7)
Use `pip install -r requirements-py38.txt` instead of requirements.txt. If shap fails to install, skip it:
the API falls back to a simple attribution. xgboost is also optional, training skips it if missing.
