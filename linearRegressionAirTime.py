import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------
# Load data
df = pd.read_csv("merged_flight_weather_data.csv")
# ----------------------------------------------------

# Remove canceled flights
df = df[df["CANCELLED"] == 0]

# Drop rows missing AIR_TIME or DISTANCE
df = df.dropna(subset=["AIR_TIME", "DISTANCE"])

# =====================================================
# Convert wind_dir (strings like "SW") → degrees
# =====================================================

wind_map = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
}

# Convert:
df["wind_dir_deg"] = df["wind_dir"].map(wind_map)

# If values are numeric strings, convert them
df["wind_dir_deg"] = pd.to_numeric(df["wind_dir_deg"], errors="coerce")

# Drop rows where direction is unknown
df = df.dropna(subset=["wind_dir_deg", "wind_mph"])

# Convert degrees → radians
wind_rad = np.radians(df["wind_dir_deg"])

# =====================================================
# Compute east–west wind component
# =====================================================
df["wind_east"] = df["wind_mph"] * np.sin(wind_rad)

# =====================================================
# Linear regression using DISTANCE + east–west wind
# =====================================================
X = df[["DISTANCE", "wind_east"]]
y = df["AIR_TIME"]

model = LinearRegression()
model.fit(X, y)

# Print results
print("Coefficients (DISTANCE, wind_east):", model.coef_)
print("Intercept:", model.intercept_)
print("R²:", model.score(X, y))

# =====================================================
# Plot DISTANCE vs AIR_TIME
# =====================================================
plt.figure(figsize=(10, 6))
plt.scatter(df["DISTANCE"], df["AIR_TIME"], alpha=0.5)
plt.title("DISTANCE vs AIR_TIME")
plt.xlabel("DISTANCE (miles)")
plt.ylabel("AIR_TIME (minutes)")
plt.grid(True)
plt.show()
