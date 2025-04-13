from flask import Flask, request, jsonify, render_template
import joblib
from huggingface_hub import hf_hub_download
import numpy as np

app = Flask(__name__)

print("Starting Flask app...")

# Download models and scaler from Hugging Face
REPO_ID = "arsalmairaj2k/breast-cancer-classification-models"
print("Downloading scaler.joblib from Hugging Face...")
try:
    scaler_path = hf_hub_download(repo_id=REPO_ID, filename="scaler.joblib")
    print(f"Scaler downloaded to: {scaler_path}")
except Exception as e:
    print(f"Error downloading scaler.joblib: {e}")
    raise

print("Downloading es_model.joblib from Hugging Face...")
try:
    es_model_path = hf_hub_download(repo_id=REPO_ID, filename="es_model.joblib")
    print(f"ES model downloaded to: {es_model_path}")
except Exception as e:
    print(f"Error downloading es_model.joblib: {e}")
    raise

# Load the scaler and model
print("Loading scaler...")
try:
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")
    raise

print("Loading ES model...")
try:
    es_model = joblib.load(es_model_path)
    print("ES model loaded successfully.")
except Exception as e:
    print(f"Error loading ES model: {e}")
    raise

@app.route('/')
def home():
    print("Rendering index.html...")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request...")
    try:
        # Get the input data from the request
        data = request.get_json()
        features = data.get('features')

        # Validate input
        if not features or len(features) != 30:
            return jsonify({"error": "Please provide exactly 30 numerical features."}), 400

        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)

        # Scale the features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = es_model.predict(features_scaled)[0]
        probabilities = es_model.predict_proba(features_scaled)[0]

        # Prepare response
        result = {
            "prediction": "Malignant" if prediction == 0 else "Benign",
            "probabilities": {
                "Malignant": float(probabilities[0]),
                "Benign": float(probabilities[1])
            }
        }

        print("Prediction successful:", result)
        return jsonify(result), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000...")
    app.run(debug=True)