import pandas as pd
import mlflow
import json
import os
from joblib import dump
import pickle

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score

mlflow.enable_system_metrics_logging()


# Set our tracking server uri for logging


# Set path to inputs
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
train_data_file = 'train.csv'
train_data_path = os.path.join(PROCESSED_DATA_DIR, train_data_file)

# Read data
df = pd.read_csv(train_data_path, sep=",")

# Split data into dependent and independent variables
# Drop useless variables
X_train = df.drop(['target'], axis='columns')
y_train = df['target']


# Define the model hyperparameters
params = {
    "max_iter": 1000
}



# Model 
logit_model = LogisticRegression(max_iter=10000)
logit_model = logit_model.fit(X_train, y_train)

# Cross validation
cv = StratifiedKFold(n_splits=3) 
val_logit = cross_val_score(logit_model, X_train, y_train, cv=cv).mean()


# Set path for the input (test data)
PROCESSED_DATA_DIR = os.environ["PROCESSED_DATA_DIR"]
test_data_file = 'test.csv'
test_data_path = os.path.join(PROCESSED_DATA_DIR, test_data_file)


# Load data
df = pd.read_csv(test_data_path, sep=",")


# Split data into dependent and independent variables
# Drop useless variables
X_test = df.drop(['target'], axis='columns')
y_test = df['target']

# Predict
logit_predictions = logit_model.predict(X_test)

# Compute test accuracy
test_logit = accuracy_score(y_test,logit_predictions)


# Start an MLflow run

mlflow.set_tracking_uri(uri="http://host.docker.internal:8070")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", test_logit)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for Heart disease data")

    # Infer the model signature
    signature = infer_signature(X_train, logit_model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=logit_model,
        artifact_path="Heart Disease",
        signature=signature,
        input_example=X_train,
    )
    
    print(mlflow.MlflowClient().get_run(run.info.run_id).data)
    print("Training Job Completed")
