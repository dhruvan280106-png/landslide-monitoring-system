
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("landslide_model.pkl")

@app.route("/")
def home():
    return "Landslide Monitoring ML API is running!"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    # Get sensor values
    soil_moisture = data["soil_moisture"]
    rain_level = data["rain_level"]
    tilt = data["tilt"]
    vibration = data["vibration"]

    # Arrange values exactly like training data
    features = np.array([[
        soil_moisture,
        rain_level,
        tilt,
        vibration
    ]])

    # ML prediction
    prediction = model.predict(features)[0]

    return jsonify({
        "prediction": str(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
