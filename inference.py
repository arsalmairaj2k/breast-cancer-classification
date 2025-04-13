# inference.py
import joblib
from huggingface_hub import hf_hub_download
import numpy as np

# Download models and scaler
REPO_ID = "arsalmairaj2k/breast-cancer-classification-models"
scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.joblib")
es_model_path = hf_hub_download(repo_id=REPO_ID, filename="es_model.joblib")

# Load the scaler and model
scaler_model = joblib.load(scaler_path)
es_model = joblib.load(es_model_path)

# Get user input
print("Enter 30 features for breast cancer prediction (separated by commas):")
features = input().strip().split(',')
features = [float(x) for x in features]

if len(features) != 30:
    print("Error: Please provide exactly 30 numerical features.")
else:
    # Convert to numpy array and reshape
    features = np.array(features).reshape(1, -1)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Make prediction
    prediction = es_model.predict(features_scaled)[0]
    probabilities = es_model.predict_proba(features_scaled)[0]

    # Display result
    print("Prediction:", "Malignant" if prediction == 0 else "Benign")
    print("Probability (Malignant):", f"{probabilities[0] * 100:.2f}%")
    print("Probability (Benign):", f"{probabilities[1] * 100:.2f}%")