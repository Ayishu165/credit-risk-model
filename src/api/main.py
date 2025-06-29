# src/api/main.py

from fastapi import FastAPI
from src.api.pydantic_models import PredictionInput, PredictionOutput
import pandas as pd
import mlflow.sklearn

# Initialize FastAPI app
app = FastAPI()

# Load the best model from MLflow registry
model_uri = "models:/best_model/Production"
model = mlflow.sklearn.load_model(model_uri)

@app.get("/")
def root():
    return {"message": "Credit Risk Model API is running!"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    """
    Accept input features, return risk probability.
    """
    input_df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(input_df)[:, 1][0]  # Get probability of class 1 (high risk)
    return PredictionOutput(risk_score=round(prob, 4))
