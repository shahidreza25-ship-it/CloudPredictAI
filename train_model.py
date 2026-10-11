
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Generate sample cloud workload data
np.random.seed(42)
n = 1000
time = np.arange(n)

cpu = np.clip(
    45 + 20 * np.sin(time / 35) + np.random.normal(0, 7, n),
    5, 98
)
memory = np.clip(
    50 + 0.45 * (cpu - 45) + np.random.normal(0, 6, n),
    10, 98
)

next_cpu = np.roll(cpu, -1)
next_cpu[-1] = cpu[-1]

df = pd.DataFrame({
    "cpu_usage": cpu,
    "memory_usage": memory,
    "next_cpu_usage": next_cpu
})

X = df[["cpu_usage", "memory_usage"]]
y = df["next_cpu_usage"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("CLOUDPREDICT AI")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

joblib.dump(model, "cloud_model.joblib")
print("Model saved successfully!")

sample = pd.DataFrame({
    "cpu_usage": [75],
    "memory_usage": [80]
})
print("Predicted next CPU usage:",
      round(model.predict(sample)[0], 2), "%")
