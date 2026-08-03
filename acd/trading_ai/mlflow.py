import mlflow
import pickle
import numpy as np
from config import MODEL_PATH, SCALER_PATH, FEATURE_META_PATH

# Point your script to your local Docker container
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Stock_LSTM_Experiments")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("epochs", 100)
    mlflow.log_param("learning_rate", 0.001)
    
    # Log metrics 
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("auc", 0.89)
    
    # Track files (Ensure these files exist locally before logging)
    # mlflow.log_artifact("lstm_model.keras")
    # mlflow.log_artifact("baseline_rf.pkl")

print("Successfully sent metrics and models to the Dockerized MLflow server!")