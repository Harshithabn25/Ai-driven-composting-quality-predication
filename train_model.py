import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("compost_quality.csv")  # Ensure this file exists

# Convert target variable to numeric (Good = 1, Bad = 0)
df["Compost_Quality"] = df["Compost_Quality"].map({"Good": 1, "Bad": 0})

# Split into features (X) and target (y)
X = df.drop("Compost_Quality", axis=1)
y = df["Compost_Quality"]

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save trained model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model Training Completed Successfully!")
