import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

# 1. Data Prep (Reusing logic)
print("Loading and preparing data...")
df = pd.read_csv('ds_13.csv')
cols_to_drop = ['Unnamed: 0', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 
                'cvtd_timestamp', 'new_window', 'num_window']
df = df.drop(columns=cols_to_drop, errors='ignore')
threshold = 0.5 * len(df)
df = df.dropna(thresh=threshold, axis=1)
df = df.dropna()
X = df.drop(columns=['class'])
y = df['class']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
iso = IsolationForest(contamination=0.05, random_state=42)
yhat = iso.fit_predict(X)
mask = yhat != -1
X_clean = X[mask]
y_clean = y_encoded[mask]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X_scaled)

print(f"Data ready. Shape: {X_pca.shape}")

# 2. Ensemble Classifiers Experiments
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')

# Random Forest
print("\n--- Random Forest Experiment ---")
rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring=f1_scorer, n_jobs=-1)
rf_grid.fit(X_pca, y_clean)
print(f"Best Random Forest Params: {rf_grid.best_params_}")
print(f"Best Random Forest F1 Score: {rf_grid.best_score_:.4f}")

# Gradient Boosting
print("\n--- Gradient Boosting Experiment ---")
# Using a smaller grid to save time, as GB is slow
gb_params = {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.2]}
gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring=f1_scorer, n_jobs=-1)
gb_grid.fit(X_pca, y_clean)
print(f"Best Gradient Boosting Params: {gb_grid.best_params_}")
print(f"Best Gradient Boosting F1 Score: {gb_grid.best_score_:.4f}")
