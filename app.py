from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Spam Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        message = data.get("message", "")

        if message.strip() == "":
            return jsonify({"error": "Empty message provided"}), 400

        # Transform message using the saved vectorizer
        features = vectorizer.transform([message])

        # Predict
        prediction = model.predict(features)[0]

        result = "Ham Mail" if prediction == 1 else "Spam Mail"

        return jsonify({
            "input_message": message,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run the app
    app.run(host="0.0.0.0", port=5000, debug=True)
