import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("addiction_model.pkl")

class AddictionInput(BaseModel):
    age: float
    daily_screen_time_hours: float
    social_media_hours: float
    gaming_hours: float
    work_study_hours: float
    sleep_hours: float
    notifications_per_day: float
    app_opens_per_day: float
    weekend_screen_time: float
    gender: str
    stress_level: str
    academic_work_impact: str

@app.post("/predict")
def predict(data: AddictionInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].max()
    return {"prediction": int(prediction), "confidence": float(probability)}