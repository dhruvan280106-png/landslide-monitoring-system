from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Machine Learning model
model = joblib.load("landslide_model.pkl")


# Home page - Landslide Monitoring Dashboard
@app.route("/")
def home():
    return render_template("index.html")


# API endpoint - receives data from ESP32/Wokwi
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive JSON data from ESP32
        data = request.get_json()

        # Get sensor values
        soil_moisture = float(data["soil_moisture"])
        rain_level = float(data["rain_level"])
        tilt = float(data["tilt"])
        vibration = float(data["vibration"])

        # Arrange features exactly like the ML training data
        features = np.array([[
            soil_moisture,
            rain_level,
            tilt,
            vibration
        ]])

        # Make prediction using trained ML model
        prediction = model.predict(features)[0]

        # Send prediction back to ESP32
        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
