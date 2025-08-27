import os
import pathlib
import time
import json
import pandas as pd
import mlflow
from typing import List
from fastapi import FastAPI, Query, HTTPException, Request
from pydantic import BaseModel, RootModel, Field
from typing import Literal
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from datetime import datetime
import random
 
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="API for predicting survival on the Titanic using a pre-trained MLflow model.",
    version="1.0.0"
)
 
metric_app = make_asgi_app()
app.mount("/metrics", metric_app)
# Standard Metrics
 
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests received, labeled by method, endpoint, and HTTP status code.",
    ["method", "endpoint", "status_code"]
)
 
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds by endpoint.",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)  
)
 
http_errors_total = Counter(
    "http_errors_total",
    "Total number of HTTP error responses (status code >= 500), labeled by method and endpoint.",
    ["method", "endpoint"]
)
 
cpu_usage_percent = Gauge(
    "cpu_usage_percent",
    "Current CPU usage percentage for the application process."
)
 
# Custom metrics
 
titanic_predictions_total = Counter(
    "titanic_predictions_total",
    "Total number of predictions made by the model"
)
titanic_predictions_output = Counter(
    "titanic_predictions_output",
    "Number of predictions made by predicted class", ["predicted_class"]
)
prediction_type_total = Counter(
    "prediction_type_total",
    "Counts of prediction requests by type", ["type"]
)
 
# NEW: model inference latency
model_inference_seconds = Histogram(
    "model_inference_seconds",
    "Model inference time in seconds by request type.",
    ["type"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
)
 
@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    method = request.method
    # Use route template to avoid label cardinality explosion
    endpoint = getattr(request.scope.get("route"), "path", request.url.path)
 
    try:
        response = await call_next(request)
        status_code = response.status_code
        #return response
    except Exception:
        http_errors_total.labels(method=method, endpoint=endpoint).inc()
        status_code = 500
        raise
    finally:
        duration = time.perf_counter() - start
        http_request_duration_seconds.labels(endpoint=endpoint).observe(duration)
        http_requests_total.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()
       
        if status_code >= 500:
            http_errors_total.labels(method=method, endpoint=endpoint).inc()
    return response
 
 
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
 
EXPERIMENT_ID = "697090457175885887"
RUN_ID = "m-1ff51c40a6f641068eae8145fd9de0b3"
MODEL_ARTIFACT_PATH = f"models/{RUN_ID}/artifacts"
model = None
PREDICTION_LOG_PATH = None
 
# Optional: Set a custom path for logging predictions uncomment to use
# CUSTOM_LOG_PATH = pathlib.Path("/your/custom/path/inference_logs.jsonl")
 
@app.on_event("startup")
def load_model():
 
    global model, PREDICTION_LOG_PATH
    try:
        current_dir = pathlib.Path(__file__).resolve().parent
        model_path = current_dir.parent / "mlruns" / EXPERIMENT_ID / MODEL_ARTIFACT_PATH
        model_uri = model_path.as_uri()
        model = mlflow.sklearn.load_model(model_uri)
 
        # Use timestamped log filename each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        PREDICTION_LOG_PATH = (
            CUSTOM_LOG_PATH if "CUSTOM_LOG_PATH" in globals() and CUSTOM_LOG_PATH is not None
            else model_path / f"simulation_logs_{timestamp}.jsonl"
        )
        # Ensure directory exists
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Reset the log file each startup
        with open(PREDICTION_LOG_PATH, "w") as f:
            pass
 
        print("Model loaded successfully.")
        print(f"Prediction log path: {PREDICTION_LOG_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
 
def get_next_execution_number(log_path: pathlib.Path) -> int:
    if not log_path.exists():
        return 1
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            if not lines:
                return 1
            last_line = lines[-1]
            last_log = json.loads(last_line)
            return last_log.get("execute", 0) + 1
    except Exception:
        return 1
 
@app.post("/predict_single")
def predict_survival(passenger: Passenger):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
 
    try:
        input_df = pd.DataFrame([passenger.dict()])
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
 
        titanic_predictions_total.inc()
        titanic_predictions_output.labels(predicted_class=str(prediction)).inc()
        prediction_type_total.labels(type="single").inc()
 
        execution_number = get_next_execution_number(PREDICTION_LOG_PATH)
        log_entry = {
            "execute": execution_number,
            "execution_time": datetime.now().isoformat(),  # store datetime
            "experiment_id": EXPERIMENT_ID,
            "run_id": RUN_ID,
            "prediction": int(prediction),
            "probability": {
                "Not Survived": probabilities[0],
                "Survived": probabilities[1]
            }
        }
 
        with open(PREDICTION_LOG_PATH, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
 
        return {
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
        raise HTTPException(status_code=503, detail="Model not loaded.")
 
    try:
        input_df = pd.DataFrame([p.dict() for p in passengers.root])
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
 
        for i, pred in enumerate(predictions):
            titanic_predictions_total.inc()
            titanic_predictions_output.labels(predicted_class=str(pred)).inc()
 
        prediction_type_total.labels(type="batch").inc()
 
        execution_number = get_next_execution_number(PREDICTION_LOG_PATH)
 
        with open(PREDICTION_LOG_PATH, "a") as f:
            for pred, prob in zip(predictions, probabilities):
                log_entry = {
                    "execute": execution_number,
                    "execution_time": datetime.now().isoformat(),  # store datetime
                    "experiment_id": EXPERIMENT_ID,
                    "run_id": RUN_ID,
                    "prediction": int(pred),
                    "probability": {
                        "Not Survived": prob[0],
                        "Survived": prob[1]
                    }
                }
                json.dump(log_entry, f)
                f.write("\n")
                execution_number += 1
 
        results = [
            {
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
 
@app.post("/simulate_predictions")
def simulate_predictions():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
 
    try:
        execution_number = get_next_execution_number(PREDICTION_LOG_PATH)
        results = []
 
        for _ in range(10):
            passenger = {
                "Pclass": random.choice([1, 2, 3]),
                "Sex": random.choice(["male", "female"]),
                "Age": round(random.uniform(1, 80), 1),
                "SibSp": random.randint(0, 3),
                "Parch": random.randint(0, 3),
                "Fare": round(random.uniform(10, 250), 2),
                "Embarked": random.choice(["C", "Q", "S"])
            }
 
            df = pd.DataFrame([passenger])
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
 
            titanic_predictions_total.inc()
            titanic_predictions_output.labels(predicted_class=str(prediction)).inc()
            prediction_type_total.labels(type="simulation").inc()
 
            log_entry = {
                "execute": execution_number,
                "execution_time": datetime.now().isoformat(),  # store datetime
                "experiment_id": EXPERIMENT_ID,
                "run_id": RUN_ID,
                "prediction": int(prediction),
                "probability": {
                    "Not Survived": probabilities[0],
                    "Survived": probabilities[1]
                }
            }
 
            with open(PREDICTION_LOG_PATH, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
 
            results.append({
                "execute": execution_number,
                "execution_time": datetime.now().isoformat(),  # return datetime too
                "passenger": passenger,
                "prediction": int(prediction),
                "survival_status": "Survived" if prediction == 1 else "Not Survived",
                "probability": {
                    "Not Survived": probabilities[0],
                    "Survived": probabilities[1]
                }
            })
 
            execution_number += 1
 
        return {"simulated_predictions": results}
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")
 
@app.post("/input_simulate_predictions")
def input_simulate_predictions(
    num_executions: int = Query(10, ge=1, le=100, description="How many simulations to run")
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
 
    try:
        execution_number = get_next_execution_number(PREDICTION_LOG_PATH)
        results = []
 
        for _ in range(num_executions):
            passenger = {
                "Pclass": random.choice([1, 2, 3]),
                "Sex": random.choice(["male", "female"]),
                "Age": round(random.uniform(1, 80), 1),
                "SibSp": random.randint(0, 3),
                "Parch": random.randint(0, 3),
                "Fare": round(random.uniform(10, 250), 2),
                "Embarked": random.choice(["C", "Q", "S"])
            }
 
            df = pd.DataFrame([passenger])
            prediction = model.predict(df)[0]
            probabilities = model.predict_proba(df)[0]
 
            titanic_predictions_total.inc()
            titanic_predictions_output.labels(predicted_class=str(prediction)).inc()
            prediction_type_total.labels(type="simulation").inc()
 
            log_entry = {
                "execute": execution_number,
                "execution_time": datetime.now().isoformat(),
                "experiment_id": EXPERIMENT_ID,
                "run_id": RUN_ID,
                "prediction": int(prediction),
                "probability": {
                    "Not Survived": float(probabilities[0]),
                    "Survived": float(probabilities[1])
                }
            }
 
            with open(PREDICTION_LOG_PATH, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
 
            results.append({
                "execute": execution_number,
                "execution_time": datetime.now().isoformat(),
                "passenger": passenger,
                "prediction": int(prediction),
                "survival_status": "Survived" if prediction == 1 else "Not Survived",
                "probability": {
                    "Not Survived": float(probabilities[0]),
                    "Survived": float(probabilities[1])
                }
            })
 
            execution_number += 1
 
        return {
            "requested_executions": num_executions,
            "simulated_predictions": results
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")
 
@app.get("/")
def root():
    return {"message": "Titanic MLflow API is running"}
 
 