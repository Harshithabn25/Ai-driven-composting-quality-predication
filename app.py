from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))  # File name should be in quotes
scaler = pickle.load(open("scaler.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

# Home route - renders a simple HTML form
@app.route("/")
def home():
    return render_template("index.html")  # File name should be in quotes

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from form
        temp = float(request.form["temperature"])  
        humidity = float(request.form["humidity"])  
        microbes = float(request.form["microbial_activity"])  

        # Prepare input for model
        input_data = np.array([[temp, humidity, microbes]])
        input_scaled = scaler.transform(input_data)

        # Get prediction
        prediction = model.predict(input_scaled)[0]
        result = "Good" if prediction == 1 else "Bad"  # Strings need quotes

        return render_template("index.html", prediction_text=f"Compost Quality: {result}")

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == "__main__":  # Use double quotes
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=8501)  # Streamlit uses port 8501

