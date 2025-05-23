# main.py

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Save model and scaler
joblib.dump(model, "models/iris_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully in 'models/' folder.")
