from flask import Flask, render_template, request
import json
import os
import joblib
import numpy as np

app = Flask(__name__)
MODEL_PATH = "house_price_model.pkl"
METRICS_PATH = "model_metrics.json"

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    metrics = load_metrics()

    # Default values are example inputs from the dataset's feature ranges.
    values = {
        "medinc": "5.0",
        "houseage": "25",
        "averooms": "5.0",
        "avebedrms": "1.0",
        "population": "1000",
        "aveoccup": "3.0",
        "latitude": "34.05",
        "longitude": "-118.25",
    }

    if request.method == "POST":
        values = {key: request.form.get(key, "").strip() for key in values}
        try:
            nums = [float(values[k]) for k in values]
            if not all(np.isfinite(nums)):
                raise ValueError("Please enter valid finite numbers.")
            if nums[0] <= 0 or nums[1] < 0 or nums[2] <= 0 or nums[3] <= 0 or nums[4] < 0 or nums[5] <= 0:
                raise ValueError("Income, rooms, bedrooms and occupancy must be positive; age and population cannot be negative.")
            if not (32 <= nums[6] <= 42 and -125 <= nums[7] <= -114):
                raise ValueError("For this demo dataset, latitude should be 32–42 and longitude should be -125 to -114 (California area).")

            model = load_model()
            if model is None:
                raise RuntimeError("The model file is missing. Run train_model.py first.")
            # Feature order must match training:
            # MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
            predicted_units = float(model.predict([nums])[0])
            price_usd = max(0, predicted_units * 100000)
            prediction = {
                "usd": price_usd,
                "formatted": f"${price_usd:,.0f}",
                "note": "Estimated median value for a California census district, not an appraisal of an individual house."
            }
        except (ValueError, TypeError) as exc:
            error = str(exc)
        except RuntimeError as exc:
            error = str(exc)

    return render_template("index.html", prediction=prediction, error=error, values=values, metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True)
