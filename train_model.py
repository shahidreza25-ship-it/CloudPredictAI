import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ==========================================
# CLOUDPREDICT AI - MODEL TRAINING
# ==========================================

# Sample cloud workload data
data = {
    "cpu_usage": [
        20, 25, 30, 35, 40,
        45, 50, 55, 60, 65,
        70, 75, 80, 85, 90,
        22, 32, 42, 52, 62,
        72, 82, 92, 28, 48
    ],

    "memory_usage": [
        30, 35, 40, 45, 50,
        55, 60, 65, 70, 75,
        80, 82, 85, 88, 92,
        32, 42, 52, 62, 72,
        78, 86, 94, 38, 58
    ],

    # Expected future CPU workload
    "predicted_cpu_usage": [
        25, 30, 35, 40, 45,
        50, 55, 60, 65, 70,
        75, 80, 85, 90, 95,
        27, 37, 47, 57, 67,
        77, 87, 97, 34, 54
    ]
}


# Create DataFrame
df = pd.DataFrame(data)


# ==========================================
# INPUT FEATURES
# ==========================================

X = df[
    [
        "cpu_usage",
        "memory_usage"
    ]
]


# ==========================================
# TARGET
# ==========================================

y = df["predicted_cpu_usage"]


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# TEST MODEL
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("========================================")
print("       CloudPredict AI")
print("       Model Training")
print("========================================")

print("MAE  :", round(mae, 2))
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 2))


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "cloud_model.joblib"
)

print()
print("Model trained successfully!")
print("Model saved as: cloud_model.joblib")