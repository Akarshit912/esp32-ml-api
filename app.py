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
        f"https://api.openweathermap.org/data/2.5/weather"
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

    return rain, wind_speed, month

# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(input: SensorInput):
    try:
        # Fetch live weather data
        rain, wind_speed, month = fetch_weather(input.lat, input.lon)

        # Build DataFrame with exact feature names expected by the model
        features = pd.DataFrame([{
            "Temperature": input.temperature,
            "RH": input.humidity,
            "Rain": rain,
            "Ws": wind_speed,
            "month": month   # required by trained model
        }])

        # Make prediction
        prediction = int(model.predict(features)[0])

        # Return prediction + some useful metadata
        return {
            "prediction": prediction,
            "rain": rain,
            "wind_speed": wind_speed,
            "month": month
        }

    except Exception as e:
        # Return explicit error if anything fails
        raise HTTPException(status_code=500, detail=str(e))
