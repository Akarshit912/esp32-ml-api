from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import requests
from datetime import datetime
import os

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

model = joblib.load("rfc_pipeline.pkl")

app = FastAPI(title="ESP32 Weather ML API")

class SensorInput(BaseModel):
    temperature: float
    humidity: float
    lat: float
    lon: float

@app.get("/health")
def health():
    return {"status": "ok"}

def fetch_weather(lat, lon):
    try:
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()

        rain = data.get("rain", {}).get("1h", 0.0)
        wind_speed = data["wind"]["speed"]
        month = datetime.utcnow().month

        return rain, month, wind_speed
    except:
        raise HTTPException(status_code=500, detail="Weather API error")

@app.post("/predict")
def predict(input: SensorInput):
    rain, month, wind_speed = fetch_weather(input.lat, input.lon)

    import pandas as pd

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
        "blink": prediction == 0
    }
