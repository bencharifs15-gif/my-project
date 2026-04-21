from flask import Blueprint, request, jsonify
import numpy as np
import joblib

hyperspectral_bp = Blueprint("hyperspectral", __name__)

model = joblib.load("models/hyperspectral/model.pkl")
scaler = joblib.load("models/hyperspectral/scaler.pkl")

@hyperspectral_bp.route("/hyperspectral/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]

        if len(data) != 128:
            return jsonify({"error": "You must send 128 values"}), 400

        arr = np.array(data).reshape(1, -1)
        arr = scaler.transform(arr)

        pred = model.predict(arr)

        return jsonify({
            "prediction": int(pred[0]),
            "message": f"Adulteration level: {pred[0]}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)})