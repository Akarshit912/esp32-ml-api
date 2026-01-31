from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pickle
import pandas as pd
from datetime import datetime
import os

# -----------------------------
# CONFIG
# -----------------------------
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not WEATHER_API_KEY:
    raise RuntimeError("WEATHER_API_KEY environment variable not set")

MODEL_PATH = "rfc_pipeline.pkl"

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="ESP32 ML Prediction API")

# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class SensorInput(BaseModel):
    temperature: float
    humidity: float
    lat: float
    lon: float

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# WEATHER FETCH
# -----------------------------
def fetch_weather(lat: float, lon: float):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    )

    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise RuntimeError(f"Weather API error: {e}")

    rain = data.get("rain", {}).get("1h", 0.0)
    wind_speed = data["wind"]["speed"]
    month = datetime.utcnow().month

    return rain, month, wind_speed

# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(input: SensorInput):
    try:
        rain, month, wind_speed = fetch_weather(input.lat, input.lon)

        # IMPORTANT: use DataFrame with correct feature names
        features = pd.DataFrame([{
            "temperature": input.temperature,
            "humidity": input.humidity,
            "rain": rain,
            "month": month,
            "wind_speed": wind_speed
        }])

        prediction = int(model.predict(features)[0])

        return {
            "prediction": prediction,
            "rain": rain,
            "month": month,
            "wind_speed": wind_speed
        }

    except Exception as e:
        # Expose exact error instead of silent 500
        raise HTTPException(status_code=500, detail=str(e))
