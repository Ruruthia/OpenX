import numpy as np
from flask import Flask, request, jsonify

from src.utils import load_models

app = Flask(__name__)
models = load_models()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    point = data["input"]
    model_type = data["model_type"]

    model = models.get(model_type, None)
    if model is None:
        return jsonify(
            {
                "error": "model_type should be one of: simple_heuristic, knn, random_forest, neural_network"
            }
        )
    prediction = model.predict(np.array(point).reshape(1, -1)).item()
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
