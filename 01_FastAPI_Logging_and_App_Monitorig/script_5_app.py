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
 
@app.get("/")
def root():
    return {"message": "Titanic MLflow API is running"}
 
 