import mlflow
import pandas as pd

mlflow.set_tracking_uri(uri="http://host.docker.internal:8070")
# Load your saved model from MLflow
model_uri = "models:/heart_disease/1"  # Replace <registered_model_name> with your model name
model = mlflow.pyfunc.load_model(model_uri)

# Define a sample input (e.g., from your test dataset)
# Ensure the input format is the same as what your model expects
sample_input = pd.DataFrame({
    "age": [57],
    "sex": [1],
    "cp": [0],
    "trestbps": [110],
    "chol": [201],
    "fbs": [0],
    "restecg": [1],
    "thalach": [126],
    "exang": [1],
    "oldpeak": [1.5],
    "slope": [1],
    "ca": [0],
    "thal": [1]
})

# Make a prediction using the loaded model
prediction = model.predict(sample_input)

# Print the prediction result
print("Prediction for the sample input:", prediction)
print("Test Completed")
