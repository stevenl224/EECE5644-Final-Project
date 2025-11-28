import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load and filter data
# -----------------------------
data = pd.read_csv('merged_flight_weather_data.csv')
conditions = ["Moderate snow", "Moderate or heavy snow showers", "Light freezing rain"]
df = data[data["condition_text"].isin(conditions)].copy()

# Create 2-class target: 0 = not canceled, 1 = canceled
df['status'] = 0
df.loc[df['CANCELLED'] == 1, 'status'] = 1

# Features
features = ['wind_mph', 'precip_mm', 'snow_cm', 'vis_km', 'humidity']
X = df[features]
y = df['status']

# -----------------------------
# 2. Setup 5-fold CV + SMOTE
# -----------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_auc = 0
best_model = None
fold_num = 1

for train_idx, test_idx in kf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Apply SMOTE on training fold only
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Naive Bayes pipeline
    preprocessor = ColumnTransformer(transformers=[("num", "passthrough", features)])
    model = Pipeline([
        ('preprocess', preprocessor),
        ('nb', GaussianNB())
    ])
    
    # Fit model
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability for class 1

    # Metrics
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n=== Fold {fold_num} ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))
    print(f"AUC: {auc:.3f}")

    # Track best model
    if auc > best_auc:
        best_auc = auc
        best_model = model
    
    fold_num += 1

# -----------------------------
# 3. Best model parameters
# -----------------------------
print("\n=== Best Fold Model Parameters ===")
print(best_model.named_steps['nb'].get_params())
print(f"Best AUC: {best_auc:.3f}")
