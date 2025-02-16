import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Example test input: Temperature=32°C, Humidity=65%, Microbial Activity=7
sample_input = np.array([[32, 65, 7]])
sample_input_scaled = scaler.transform(sample_input)

# Get prediction
prediction = model.predict(sample_input_scaled)
print("Predicted Compost Quality:", "Good" if prediction[0] == 1 else "Bad")
