"""Flask web app and JSON prediction API."""
from flask import Flask, jsonify, render_template, request
from models.predict import predict_language

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "The 'text' field must be a string."}), 400
    if not text.strip():
        return jsonify({"error": "Please enter some text."}), 400
    result = predict_language(text)
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
