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

# Remove extreme outliers in TAXI_OUT
delay_threshold = data['TAXI_OUT'].quantile(0.95)
data = data[data['TAXI_OUT'] <= delay_threshold]

# Keep only non-cancelled flights
data = data[data['CANCELLED'] == 0]

# Convert CRS_DEP_TIME (HHMM) → minutes since midnight
data['dep_minutes'] = (data['CRS_DEP_TIME'] // 100) * 60 + (data['CRS_DEP_TIME'] % 100)

# 20-minute segments
data['dep_20min_seg'] = (data['dep_minutes'] // 20) * 20

# Replace NaN weather delay (shouldn't matter)
data['TAXI_OUT'] = data['TAXI_OUT'].fillna(0)

# Convert minutes (relative to 5:00 AM) to readable HH:MM
def minutes_to_hhmm(minutes_from_5am):
    # Add back the 5 AM offset (300 minutes) to get actual clock time
    actual_minutes = (minutes_from_5am + 300) % 1440
    h = actual_minutes // 60
    m = actual_minutes % 60
    return f"{int(h):02d}:{int(m):02d}"

conditions = data['condition_text'].dropna().unique()


# Create directory for plots if it doesn't exist
import os
if not os.path.exists('meanTaxiOutPlots'):
    os.makedirs('meanTaxiOutPlots')

# Loop through each weather condition
for cond in conditions:
    subset = data[data['condition_text'] == cond]

    # -----------------------------
    # Standard deviation and counts
    # -----------------------------
    grouped = subset.groupby('dep_20min_seg')['TAXI_OUT']
    mean_delay = grouped.mean().reset_index()     # mean
    counts = grouped.count().reset_index().rename(columns={'TAXI_OUT': 'count'})  # N
    merged = mean_delay.merge(counts, on='dep_20min_seg')

    # Remove bins passed 23:00 (1380 minutes from midnight = 1080 minutes from 5 AM)
    # 23:00 = 23*60 = 1380 minutes from midnight
    # 1380 - 300 (5 AM offset) = 1080 minutes from 5 AM
    merged = merged[merged['dep_20min_seg'] <= 1080]
    # Drop NaN mean (bins with only 1 flight)
    merged = merged.dropna(subset=['TAXI_OUT'])

    # Skip if too few bins for modeling
    if len(merged) < 5:
        print(f"Skipping {cond}: too few valid bins ({len(merged)}).")
        continue

    merged['HHMM'] = merged['dep_20min_seg'].apply(minutes_to_hhmm)

    # -----------------------------
    # Prepare X and y
    # -----------------------------
    X = merged['dep_20min_seg'].values.reshape(-1, 1)
    y = merged['TAXI_OUT'].values

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
    # Quadratic model (degree 2)
    # -----------------------------
    poly2 = PolynomialFeatures(degree=2)
    X_poly2 = poly2.fit_transform(X)

    quad = LinearRegression()
    quad.fit(X_poly2, y)
    y_pred_quad = quad.predict(X_poly2)
    r2_quad = r2_score(y, y_pred_quad)

    # Equation: y = ax² + bx + c
    a2 = quad.coef_[2]
    b2 = quad.coef_[1]
    c2 = quad.intercept_

    # K-fold CV for quadratic
    cv_r2_quad = []
    for train_idx, test_idx in kf.split(X):
        quad.fit(X_poly2[train_idx], y[train_idx])
        cv_r2_quad.append(r2_score(y[test_idx], quad.predict(X_poly2[test_idx])))
    cv_r2_quad_mean = np.mean(cv_r2_quad)

    # -----------------------------
    # Cubic model (degree 3)
    # -----------------------------
    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(X)

    cubic = LinearRegression()
    cubic.fit(X_poly3, y)
    y_pred_cubic = cubic.predict(X_poly3)
    r2_cubic = r2_score(y, y_pred_cubic)

    # Extract coefficients: y = ax³ + bx² + cx + d
    a3 = cubic.coef_[3]  # x³ coefficient
    b3 = cubic.coef_[2]  # x² coefficient
    c3 = cubic.coef_[1]  # x coefficient
    d3 = cubic.intercept_  # intercept

    # K-fold CV for cubic
    cv_r2_cubic = []
    for train_idx, test_idx in kf.split(X):
        cubic.fit(X_poly3[train_idx], y[train_idx])
        cv_r2_cubic.append(r2_score(y[test_idx], cubic.predict(X_poly3[test_idx])))
    cv_r2_cubic_mean = np.mean(cv_r2_cubic)

    # -----------------------------
    # Quartic model (degree 4)
    # -----------------------------
    poly4 = PolynomialFeatures(degree=4)
    X_poly4 = poly4.fit_transform(X)

    quart = LinearRegression()
    quart.fit(X_poly4, y)
    y_pred_quart = quart.predict(X_poly4)
    r2_quart = r2_score(y, y_pred_quart)

    # Extract coefficients: y = ax⁴ + bx³ + cx² + dx + e
    a4 = quart.coef_[4]  # x⁴ coefficient
    b4 = quart.coef_[3]  # x³ coefficient
    c4 = quart.coef_[2]  # x² coefficient
    d4 = quart.coef_[1]  # x coefficient
    e4 = quart.intercept_  # intercept

    # K-fold CV for quartic
    cv_r2_quart = []
    for train_idx, test_idx in kf.split(X):
        quart.fit(X_poly4[train_idx], y[train_idx])
        cv_r2_quart.append(r2_score(y[test_idx], quart.predict(X_poly4[test_idx])))
    cv_r2_quart_mean = np.mean(cv_r2_quart)

    # -----------------------------
    # Plotting
    # -----------------------------
    plt.figure(figsize=(14, 7))
    sc = plt.scatter(
        merged['dep_20min_seg'], 
        merged['TAXI_OUT'], 
        c=merged['count'], 
        cmap='viridis',
        alpha=0.8,
        s=60,
        label='Data (colored by count)',
        zorder=3
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
        linewidth=2,
        linestyle='--',
        alpha=0.8
    )

    plt.plot(
        x_sorted,
        y_pred_quad[sort_idx],
        label=f"Quadratic (R²={r2_quad:.3f}, CV={cv_r2_quad_mean:.3f})\n"
              f"y = {a2:.10f}·x² + {b2:.6f}·x + {c2:.3f}",
        linewidth=2,
        linestyle='-.',
        alpha=0.8
    )

    plt.plot(
        x_sorted,
        y_pred_cubic[sort_idx],
        label=f"Cubic (R²={r2_cubic:.3f}, CV={cv_r2_cubic_mean:.3f})\n"
              f"y = {a3:.12f}·x³ + {b3:.10f}·x² + {c3:.6f}·x + {d3:.3f}",
        linewidth=2,
        linestyle=':',
        alpha=0.8
    )

    plt.plot(
        x_sorted,
        y_pred_quart[sort_idx],
        label=f"Quartic (R²={r2_quart:.3f}, CV={cv_r2_quart_mean:.3f})\n"
              f"y = {a4:.15f}·x⁴ + {b4:.12f}·x³ + {c4:.10f}·x² + {d4:.6f}·x + {e4:.3f}",
        linewidth=2.5,
        linestyle='-',
        alpha=0.9
    )

    # Axis formatting (show times from 5 AM through the day)
    # Create ticks every hour from t=0 (5 AM) to t=1080 (11 PM)
    tick_positions = np.arange(0, 1081, 60)  # Every 60 minutes from 5 AM
    plt.xticks(
        ticks=tick_positions,
        labels=[minutes_to_hhmm(m) for m in tick_positions],
        rotation=45
    )

    plt.title(f"Mean Taxi-Out Time vs Time of Day (t=0 at 5:00 AM)\nWeather Condition: {cond}", fontsize=16)
    plt.xlabel("Actual Departure Time (t=0 corresponds to 5:00 AM)", fontsize=14)
    plt.ylabel("Mean Taxi-Out Time (minutes)", fontsize=14)
    plt.colorbar(sc, label='Number of Flights in 20-min Bin')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc='best')
    plt.tight_layout()
    
    # Save plot to meanTaxiOutPlots folder
    plt.savefig(f'meanTaxiOutPlots/{cond}_mean_taxi_out.png')

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Weather Condition: {cond}")
    print(f"{'='*60}")
    print(f"Linear    - R²: {r2_lin:.4f}, CV R²: {cv_r2_lin_mean:.4f}")
    print(f"Quadratic - R²: {r2_quad:.4f}, CV R²: {cv_r2_quad_mean:.4f}")
    print(f"Cubic     - R²: {r2_cubic:.4f}, CV R²: {cv_r2_cubic_mean:.4f}")
    print(f"Quartic   - R²: {r2_quart:.4f}, CV R²: {cv_r2_quart_mean:.4f}")
    print(f"Number of time bins: {len(merged)}")
    print(f"Total flights: {merged['count'].sum()}")

