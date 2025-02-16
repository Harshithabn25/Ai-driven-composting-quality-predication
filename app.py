import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit UI
st.title("Composting Quality Predictor")

# User inputs
temperature = st.number_input("Enter Temperature (°C)", min_value=-10.0, max_value=100.0, step=0.1)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
microbial_activity = st.number_input("Enter Microbial Activity Level", min_value=0.0, max_value=100.0, step=0.1)

# Prediction button
if st.button("Predict Compost Quality"):
    try:
        # Prepare input for model
        input_data = np.array([[temperature, humidity, microbial_activity]])
        input_scaled = scaler.transform(input_data)

        # Get prediction
        prediction = model.predict(input_scaled)[0]
        result = "Good" if prediction == 1 else "Bad"

        # Show result
        st.success(f"Compost Quality: {result}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
