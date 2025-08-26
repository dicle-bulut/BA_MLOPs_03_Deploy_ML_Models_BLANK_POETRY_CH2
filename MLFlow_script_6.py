import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os
import pathlib

# --- Configuration ---
N_SPLITS_CV = 5 # Number of folds for cross-validation

# --- 1. Load Data ---
# You need train.csv and test.csv from Kaggle Titanic competition

DATA_PATH = pathlib.Path(__file__).resolve().parent
TRAIN_FILE = DATA_PATH / "data" / "train.csv"
TEST_FILE = DATA_PATH / "data" / "train.csv"


if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
    print(f"Required files ({TRAIN_FILE}, {TEST_FILE}) not found.")
    print("Please download them from Kaggle Titanic competition:")
    print("https://www.kaggle.com/c/titanic/data")
    exit()

df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE) # This is the "unseen" Kaggle test set

# Store PassengerId for potential use, though not used for submission anymore
test_passenger_ids = df_test['PassengerId']

# Drop 'PassengerId', 'Name', 'Ticket', 'Cabin' from both train and test data for this example
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_train_processed = df_train.drop(columns=[col for col in cols_to_drop if col in df_train.columns])
df_test_processed = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns])


# Define target and features
target = 'Survived'
X_full = df_train_processed.drop(target, axis=1)
y_full = df_train_processed[target]

# Define feature types for preprocessing
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# --- 2. Define Preprocessing Pipeline (with Cleaning Steps) ---

# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# --- Set MLflow Experiment ---
mlflow.set_experiment("Titanic CV & Model Logging") # Renamed experiment slightly
print("Starting Titanic Classification with CV and Model Logging...")

# Helper function to run CV, evaluate, and log to MLflow
def train_evaluate_log_model(model_name, classifier, X_full, y_full, X_test_kaggle, test_passenger_ids, model_params):
    with mlflow.start_run(run_name=f"{model_name}_CV_Run"): # Renamed run name slightly
        print(f"\n--- Training {model_name} with Cross-Validation ---")

        # Log parameters specific to this run
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_splits_cv", N_SPLITS_CV)
        mlflow.log_param("cleaning_steps", "Age Impute Mean, Embarked Impute Mode")
        mlflow.log_param("preprocessing_steps", "StandardScaler, OneHotEncoder")
        mlflow.log_param("numerical_features", str(numerical_features))
        mlflow.log_param("categorical_features", str(categorical_features))
        mlflow.log_params(model_params)

        # Create the full pipeline including preprocessing and classifier
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', classifier)])

        # --- Perform Cross-Validation ---
        cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_results = cross_validate(full_pipeline, X_full, y_full, cv=cv, scoring=scoring, return_train_score=False)

        # Log mean and standard deviation of CV metrics
        print(f"  {model_name} CV Results:")
        for metric in scoring:
            mean_metric = np.mean(cv_results[f'test_{metric}'])
            std_metric = np.std(cv_results[f'test_{metric}'])
            mlflow.log_metric(f"cv_mean_{metric}", mean_metric)
            mlflow.log_metric(f"cv_std_{metric}", std_metric)
            print(f"    Mean Test {metric}: {mean_metric:.4f} (Std: {std_metric:.4f})")

        # --- Train the final model on the full training data ---
        # This model will be saved to MLflow artifacts
        print(f"  Fitting final {model_name} pipeline on full training data for logging...")
        full_pipeline.fit(X_full, y_full)

        # Log the final trained pipeline
        mlflow.sklearn.log_model(full_pipeline, "model")
        print(f"  Final {model_name} pipeline logged to: {mlflow.active_run().info.artifact_uri}/model")

        # Note: Predictions on X_test_kaggle and submission file saving are removed.


# --- Define Models and their Parameters ---
models_to_compare = [
    {
        "name": "Logistic Regression",
        "classifier": LogisticRegression(C=0.1, solver='liblinear', max_iter=1000, random_state=42),
        "params": {"C": 0.1, "solver": "liblinear", "max_iter": 1000}
    },
    {
        "name": "Random Forest Classifier",
        "classifier": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "params": {"n_estimators": 100, "max_depth": 8}
    },
    {
        "name": "Gradient Boosting Classifier",
        "classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    }
]

# --- Run the full pipeline for each model ---
for model_info in models_to_compare:
    train_evaluate_log_model(
        model_info["name"],
        model_info["classifier"],
        X_full,
        y_full,
        df_test_processed, # X_test_kaggle is still passed, but not used for predictions in this version
        test_passenger_ids, # test_passenger_ids is still passed, but not used in this version
        model_info["params"]
    )

print("\nAll Titanic models processed with CV and models logged to MLflow.")
print("To view results, run 'mlflow ui' in your terminal and navigate to the 'Titanic CV & Model Logging' experiment.")

# --- Example of Loading and Using a Pipeline for Inference ---
try:
    last_run_id = mlflow.last_active_run().info.run_id
    loaded_pipeline = mlflow.sklearn.load_model(f"runs:/{last_run_id}/model")

    print(f"\n--- Demonstrating loaded model prediction from last run ({last_run_id[:8]}) ---")
    print(f"Loaded pipeline from MLflow URI: runs:/{last_run_id}/model")

    new_passenger = pd.DataFrame([{
        'Pclass': 3, 'Sex': 'male', 'Age': 28.0, 'SibSp': 0,
        'Parch': 0, 'Fare': 10.0, 'Embarked': 'S'
    }])

    predicted_survival = loaded_pipeline.predict(new_passenger)[0]
    print(f"New passenger data:\n{new_passenger}")
    print(f"Predicted Survival (0=No, 1=Yes): {predicted_survival}")

except Exception as e:
    print(f"\nCould not demonstrate model loading: {e}")
    print("Ensure you have run the script successfully and an MLflow run exists.")