FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# builds demo data and a trained model into the image. Replace data/ with the Azure CSVs for real training.
RUN python -m ml.simulate_data && python -m ml.train
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
