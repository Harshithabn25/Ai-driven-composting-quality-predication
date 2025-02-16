import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# 🔹 Simulating Sensor Data (Example Data)
data_dict = {
    "Temperature": [30, 25, 40, 50, 20, 35, 45, 27, 33, 48],
    "Humidity": [60, 55, 70, 80, 50, 65, 75, 58, 62, 78],
    "Microbial_Activity": [7, 6, 8, 9, 5, 7, 9, 6, 7, 8],
    "Compost_Quality": ["Good", "Bad", "Good", "Good", "Bad", "Good", "Good", "Bad", "Good", "Good"]
}

# Convert to DataFrame
df = pd.DataFrame(data_dict)

# 🔹 Convert Target Variable to Binary (Good=1, Bad=0)
df["Compost_Quality"] = df["Compost_Quality"].map({"Good": 1, "Bad": 0})

# 🔹 Split Data into Features and Target
X = df.drop("Compost_Quality", axis=1)
y = df["Compost_Quality"]

# 🔹 Split into Training & Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Normalize Data (Scaling Sensor Readings)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔹 Save Model and Scaler
pickle.dump(model, open("compost_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# 🔹 Test Prediction
sample_input = np.array([[32, 65, 7]])  # Example input (Temp, Humidity, Microbial Activity)
sample_input = scaler.transform(sample_input)
prediction = model.predict(sample_input)

print("Predicted Compost Quality:", "Good" if prediction[0] == 1 else "Bad")
print("Model Training Successful!")
