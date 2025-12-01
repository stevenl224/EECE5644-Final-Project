import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

# -----------------------------
# 1. Prepare dataset (same as before)
# -----------------------------
df = pd.read_csv("merged_flight_weather_data.csv")
df = df[df["CANCELLED"] == 0]
df = df.dropna(subset=["AIR_TIME", "DISTANCE", "wind_dir", "wind_mph"])

# Convert wind_dir strings → degrees
wind_map = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}
df["wind_dir_deg"] = df["wind_dir"].map(wind_map)
df["wind_dir_deg"] = pd.to_numeric(df["wind_dir_deg"], errors="coerce")
df = df.dropna(subset=["wind_dir_deg", "wind_mph"])

# Compute east–west wind component
wind_rad = np.radians(df["wind_dir_deg"])
df["wind_east"] = df["wind_mph"] * np.sin(wind_rad)

# Features + target
X = df[["DISTANCE", "wind_east"]]
y = df["AIR_TIME"]

# -----------------------------
# 2. Grid search for alpha
# -----------------------------
alphas = np.logspace(-3, 3, 13)  # 0.001, 0.01, 0.1, 1, 10, ... 1000
ridge = Ridge()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {"alpha": alphas}

grid = GridSearchCV(ridge, param_grid, cv=kf, scoring="r2")
grid.fit(X, y)

# -----------------------------
# 3. Results
# -----------------------------
best_alpha = grid.best_params_["alpha"]
best_r2 = grid.best_score_

print(f"Best alpha: {best_alpha}")
print(f"Cross-validated R²: {best_r2:.4f}")

# Fit final Ridge model with best alpha
ridge_best = Ridge(alpha=best_alpha)
ridge_best.fit(X, y)

print("Final coefficients (DISTANCE, wind_east):", ridge_best.coef_)
print("Intercept:", ridge_best.intercept_)
print("R² on full data:", ridge_best.score(X, y))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1. 3-D Surface Plot
# -----------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Create grid for surface
distance_grid = np.linspace(X['DISTANCE'].min(), X['DISTANCE'].max(), 50)
wind_grid = np.linspace(X['wind_east'].min(), X['wind_east'].max(), 50)
distance_mesh, wind_mesh = np.meshgrid(distance_grid, wind_grid)

# Predict AIR_TIME on grid
air_time_pred = ridge_best.predict(
    np.c_[distance_mesh.ravel(), wind_mesh.ravel()]
).reshape(distance_mesh.shape)

# Plot surface
ax.plot_surface(distance_mesh, wind_mesh, air_time_pred, alpha=0.6, cmap='viridis')
ax.scatter(X['DISTANCE'], X['wind_east'], y, c='r', s=10, label='Data points')
ax.set_xlabel('DISTANCE (miles)')
ax.set_ylabel('Wind East Component (mph)')
ax.set_zlabel('AIR_TIME (minutes)')
ax.set_title('Ridge Regression Prediction Surface')
plt.show()

# -----------------------------
# 2. 2-D scatter plot: wind_east vs air_time
# -----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X['wind_east'], y, alpha=0.5)
plt.xlabel('Wind East Component (mph)')
plt.ylabel('AIR_TIME (minutes)')
plt.title('Effect of East-West Wind on Air Time')
plt.grid(True)
plt.show()


