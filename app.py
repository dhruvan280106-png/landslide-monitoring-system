from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the newly trained model
model = joblib.load("landslide_model.pkl")

# Store latest sensor data
latest_data = {
    "soil_moisture": None,
    "rain_level": None,
    "soil_wetness": None,
    "rain_wetness": None,
    "tilt": None,
    "vibration": None,
    "prediction": "Waiting for sensor data..."
}


# Dashboard home page
@app.route("/")
def home():
    return render_template("index.html")


# Receive ESP32 sensor data
@app.route("/predict", methods=["POST"])
def predict():

    global latest_data

    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        # Get raw sensor readings
        soil_raw = float(data["soil_moisture"])
        rain_raw = float(data["rain_level"])
        tilt = float(data["tilt"])
        vibration = float(data["vibration"])

        # Validate ADC readings
        if not (0 <= soil_raw <= 4095):
            raise ValueError("Invalid soil ADC reading")

        if not (0 <= rain_raw <= 4095):
            raise ValueError("Invalid rain ADC reading")

        # Convert raw ADC into wetness scale
        # 0 = dry, 4095 = wet
        soil_wetness = 4095 - soil_raw
        rain_wetness = 4095 - rain_raw

        # Prepare ML features
        features = np.array([[
            soil_wetness,
            rain_wetness,
            tilt,
            vibration
        ]])

        # Predict landslide risk
        prediction = model.predict(features)[0]

        # Store latest readings
        latest_data = {
            "soil_moisture": soil_raw,
            "rain_level": rain_raw,
            "soil_wetness": soil_wetness,
            "rain_wetness": rain_wetness,
            "tilt": tilt,
            "vibration": vibration,
            "prediction": str(prediction)
        }

        # Send prediction to ESP32
        return jsonify({
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Dashboard data endpoint
@app.route("/data", methods=["GET"])
def get_data():
    return jsonify(latest_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
