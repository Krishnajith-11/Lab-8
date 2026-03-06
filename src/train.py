import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset

data = pd.read_csv("data/housing.csv")

# Remove missing values

data = data.dropna()

# Features and target

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Convert categorical columns

X = pd.get_dummies(X)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Train model

model = LinearRegression()
model.fit(X_train, y_train)

# Predict

y_pred = model.predict(X_test)

# Metrics

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

metrics = {
"dataset_size": len(data),
"rmse": float(rmse),
"r2": float(r2)
}

# Save metrics

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training Complete")
print(metrics)
