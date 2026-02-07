# Mileage Prediction using Regression Models
# Author: Saurabh
# Description: Predicts vehicle mileage using Linear Regression and Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------
# Replace 'data.csv' with your dataset file
data = pd.read_csv("data.csv")

print("Dataset Preview:")
print(data.head())

# -------------------------------
# Data Preprocessing
# -------------------------------

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = label_encoder.fit_transform(data[col])

# Separate features and target
X = data.drop("mileage", axis=1)   # target column: mileage
y = data["mileage"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# Linear Regression Model
# -------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# -------------------------------
# Random Forest Regressor
# -------------------------------
rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# -------------------------------
# Model Evaluation
# -------------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("MSE :", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R²  :", r2_score(y_true, y_pred))


evaluate_model("Linear Regression", y_test, lr_pred)
evaluate_model("Random Forest Regressor", y_test, rf_pred)

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(10, 5))
plt.scatter(y_test, lr_pred, label="Linear Regression", alpha=0.6)
plt.scatter(y_test, rf_pred, label="Random Forest", alpha=0.6)
plt.xlabel("Actual Mileage")
plt.ylabel("Predicted Mileage")
plt.title("Actual vs Predicted Mileage")
plt.legend()
plt.show()
