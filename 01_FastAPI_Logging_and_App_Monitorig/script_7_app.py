import os
import pathlib
import pandas as pd
import mlflow
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel, Field
from typing import Literal
 
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting survival on the Titanic using a pre-trained MLflow model.",
    version="1.0.0"
)
 
class HealthResponse(BaseModel):
    status: str
    version: str
    service: str
    dependencies: str
 
@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "version": "1.0.0",
        "service": "Titanic Predictor API",
        "dependencies": "MLflow, Scikit-learn"
    }
 
class Passenger(BaseModel):
    Pclass: Literal[1, 2, 3] = Field(..., description="Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)")
    Sex: Literal["male", "female"] = Field(..., description="Sex of the passenger")
    Age: float = Field(..., gt=0, lt=100, description="Age must be between 0 and 100")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Fare paid must be non-negative")
    Embarked: Literal["C", "Q", "S"] = Field(..., description="Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)")
 
    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 1,
                "Sex": "female",
                "Age": 24.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 75.0,
                "Embarked": "C"
            }
        }
 
class PassengerBatch(RootModel[List[Passenger]]):
    class Config:
        json_schema_extra = {
            "example": [
                {
                    "Pclass": 1,
                    "Sex": "female",
                    "Age": 24.0,
                    "SibSp": 0,
                    "Parch": 0,
                    "Fare": 75.0,
                    "Embarked": "C"
                },
                {
                    "Pclass": 3,
                    "Sex": "male",
                    "Age": 22.0,
                    "SibSp": 1,
                    "Parch": 0,
                    "Fare": 7.25,
                    "Embarked": "S"
                },
                {
                    "Pclass": 2,
                    "Sex": "female",
                    "Age": 30.0,
                    "SibSp": 1,
                    "Parch": 1,
                    "Fare": 26.0,
                    "Embarked": "Q"
                }
            ]
        }
 
EXPERIMENT_ID = "697090457175885887"
RUN_ID = "m-1ff51c40a6f641068eae8145fd9de0b3"
MODEL_ARTIFACT_PATH = f"models/{RUN_ID}/artifacts"
model = None
 
@app.on_event("startup")
def load_model():
    global model
    try:
        current_dir = pathlib.Path(__file__).resolve().parent
        model_path = current_dir.parent / "mlruns" / EXPERIMENT_ID / MODEL_ARTIFACT_PATH
        model_uri = model_path.as_uri()
        print(f"Loading model from: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
 
@app.post("/predict_single")
def predict_survival(passenger: Passenger):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check path.")
 
    try:
        input_df = pd.DataFrame([passenger.dict()])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        return {
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "prediction": int(prediction),
            "survival_status": "Survived" if prediction == 1 else "Not Survived",
            "probability": {
                "Not Survived": probabilities[0],
                "Survived": probabilities[1]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
 
@app.post("/predict_batch")
def predict_survival_batch(passengers: PassengerBatch):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check path.")
 
    try:
        input_df = pd.DataFrame([p.dict() for p in passengers.root])
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
 
        results = [
            {
                "experiment_id": EXPERIMENT_ID,
                "run_id": RUN_ID,
                "passenger_index": i,
                "prediction": int(pred),
                "survival_status": "Survived" if pred == 1 else "Not Survived",
                "probability": {
                    "Not Survived": prob[0],
                    "Survived": prob[1]
                }
            }
            for i, (pred, prob) in enumerate(zip(predictions, probabilities))
        ]
 
        return {"batch_predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
 
@app.get("/")
def root():
    return {"message": "Titanic MLflow API is running"}
 
from fastapi import FastAPI, Request
from prometheus_client import Counter, make_asgi_app
import time
 
# Create FastAPI app
app = FastAPI()
 
# Define Prometheus counter metric
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests received, labeled by method and endpoint.",
    ["method", "endpoint"]
)
 
# Middleware to increment the counter
@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    return response
 
# Simple test route
@app.get("/")
def read_root():
    return {"message": "Hello, Prometheus is tracking requests!"}
 
# Mount Prometheus metrics at /metrics
metric_app = make_asgi_app()
app.mount("/metrics", metric_app)
 