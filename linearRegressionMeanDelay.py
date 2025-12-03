import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv('merged_flight_weather_data.csv')

# Remove extreme outliers in DEP_DELAY
delay_threshold = data['DEP_DELAY'].quantile(0.95)
data = data[data['DEP_DELAY'] <= delay_threshold]

# Keep only non-cancelled flights
data = data[data['CANCELLED'] == 0]

# Convert CRS_DEP_TIME (HHMM) → minutes since midnight
data['dep_minutes'] = (data['CRS_DEP_TIME'] // 100) * 60 + (data['CRS_DEP_TIME'] % 100)

# 20-minute segments
data['dep_20min_seg'] = (data['dep_minutes'] // 20) * 20

# Replace NaN weather delay (shouldn't matter)
data['DEP_DELAY'] = data['DEP_DELAY'].fillna(0)

# Convert minutes to readable HH:MM
def minutes_to_hhmm(minutes):
    h = minutes // 60
    m = minutes % 60
    return f"{int(h):02d}:{int(m):02d}"

conditions = data['condition_text'].dropna().unique()

# Create directory for plots if it doesn't exist
import os
if not os.path.exists('meanDelayPlots'):
    os.makedirs('meanDelayPlots')
# Loop through each weather condition
for cond in conditions:
    subset = data[data['condition_text'] == cond]

    # -----------------------------
    # Standard deviation and counts
    # -----------------------------
    grouped = subset.groupby('dep_20min_seg')['DEP_DELAY']
    mean_delay = grouped.mean().reset_index()     # mean
    counts = grouped.count().reset_index().rename(columns={'DEP_DELAY': 'count'})  # N
    merged = mean_delay.merge(counts, on='dep_20min_seg')

    # Remove bins passed 23:00 (1380 minutes)
    merged = merged[merged['dep_20min_seg'] <= 1380]
    # Drop NaN mean (bins with only 1 flight)
    merged = merged.dropna(subset=['DEP_DELAY'])

    # Skip if too few bins for modeling
    if len(merged) < 5:
        print(f"Skipping {cond}: too few valid bins ({len(merged)}).")
        continue

    merged['HHMM'] = merged['dep_20min_seg'].apply(minutes_to_hhmm)

    # -----------------------------
    # Prepare X and y
    # -----------------------------
    X = merged['dep_20min_seg'].values.reshape(-1, 1)
    y = merged['DEP_DELAY'].values

    # -----------------------------
    # Linear regression
    # -----------------------------
    lin = LinearRegression()
    lin.fit(X, y)
    y_pred_lin = lin.predict(X)
    r2_lin = r2_score(y, y_pred_lin)

    # Extract equation: y = a*x + b
    a_lin = lin.coef_[0]
    b_lin = lin.intercept_

    # K-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_lin = []
    for train_idx, test_idx in kf.split(X):
        lin.fit(X[train_idx], y[train_idx])
        cv_r2_lin.append(r2_score(y[test_idx], lin.predict(X[test_idx])))
    cv_r2_lin_mean = np.mean(cv_r2_lin)

    # -----------------------------
    # Quadratic model
    # -----------------------------
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    quad = LinearRegression()
    quad.fit(X_poly, y)
    y_pred_quad = quad.predict(X_poly)
    r2_quad = r2_score(y, y_pred_quad)

    # Equation: y = ax² + bx + c
    a2 = quad.coef_[2]
    b2 = quad.coef_[1]
    c2 = quad.intercept_

    # K-fold CV for quadratic
    cv_r2_quad = []
    for train_idx, test_idx in kf.split(X):
        quad.fit(X_poly[train_idx], y[train_idx])
        cv_r2_quad.append(r2_score(y[test_idx], quad.predict(X_poly[test_idx])))
    cv_r2_quad_mean = np.mean(cv_r2_quad)

    # -----------------------------
    # Plotting
    # -----------------------------
    plt.figure(figsize=(14, 7))
    sc = plt.scatter(
        merged['dep_20min_seg'], 
        merged['DEP_DELAY'], 
        c=merged['count'], 
        cmap='viridis',
        alpha=0.8,
        s=60
    )

    # Sort for smooth line plotting
    sort_idx = np.argsort(X.flatten())
    x_sorted = X.flatten()[sort_idx]

    # Plot trendlines
    plt.plot(
        x_sorted,
        y_pred_lin[sort_idx],
        label=f"Linear (R²={r2_lin:.3f}, CV={cv_r2_lin_mean:.3f})\n"
              f"y = {a_lin:.6f}·x + {b_lin:.3f}",
        linewidth=2
    )

    plt.plot(
        x_sorted,
        y_pred_quad[sort_idx],
        label=f"Quadratic (R²={r2_quad:.3f}, CV={cv_r2_quad_mean:.3f})\n"
              f"y = {a2:.10f}·x² + {b2:.6f}·x + {c2:.3f}",
        linewidth=2
    )

    # Axis formatting
    plt.xticks(
        ticks=np.arange(0, 1441, 60),
        labels=[minutes_to_hhmm(m) for m in np.arange(0, 1441, 60)],
        rotation=45
    )

    plt.title(f"Mean of Weather Delays vs Time of Day\nWeather Condition: {cond}", fontsize=16)
    plt.xlabel("Scheduled Departure Time", fontsize=14)
    plt.ylabel("Mean of Weather Delay (minutes)", fontsize=14)
    plt.colorbar(sc, label='Number of Flights in 20-min Bin')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()

    # Save plot in meanDelayPlots folder
    plt.savefig(f'meanDelayPlots/{cond}_mean_delay.png')
    plt.close()
    