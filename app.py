from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Machine Learning model
model = joblib.load("landslide_model.pkl")


# Store the latest sensor data
latest_data = {
    "soil_moisture": None,
    "rain_level": None,
    "tilt": None,
    "vibration": None,
    "prediction": "Waiting for sensor data..."
}


# Home page - Dashboard
@app.route("/")
def home():
    return render_template("index.html")


# Receive sensor data from ESP32
@app.route("/predict", methods=["POST"])
def predict():

    global latest_data

    try:

        # Receive JSON data
        data = request.get_json()

        # Get sensor values
        soil_moisture = float(data["soil_moisture"])
        rain_level = float(data["rain_level"])
        tilt = float(data["tilt"])
        vibration = float(data["vibration"])

        # Arrange features for ML model
        features = np.array([[
            soil_moisture,
            rain_level,
            tilt,
            vibration
        ]])

        # ML Prediction
        prediction = model.predict(features)[0]

        # Save latest sensor data for dashboard
        latest_data = {
            "soil_moisture": soil_moisture,
            "rain_level": rain_level,
            "tilt": tilt,
            "vibration": vibration,
            "prediction": str(prediction)
        }

        # Send prediction back to ESP32
        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# Dashboard gets latest sensor data
@app.route("/data", methods=["GET"])
def get_data():

    return jsonify(latest_data)


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000)
