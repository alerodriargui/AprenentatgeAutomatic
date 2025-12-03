import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
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

# 2. Single Classifiers Experiments
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')

# k-NN
print("\n--- k-NN Experiment ---")
knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn, knn_params, cv=cv, scoring=f1_scorer, n_jobs=-1)
knn_grid.fit(X_pca, y_clean)
print(f"Best k-NN Params: {knn_grid.best_params_}")
print(f"Best k-NN F1 Score: {knn_grid.best_score_:.4f}")

# Decision Tree
print("\n--- Decision Tree Experiment ---")
dt_params = {'max_depth': [None, 10, 20, 30], 'criterion': ['gini', 'entropy']}
dt = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt, dt_params, cv=cv, scoring=f1_scorer, n_jobs=-1)
dt_grid.fit(X_pca, y_clean)
print(f"Best Decision Tree Params: {dt_grid.best_params_}")
print(f"Best Decision Tree F1 Score: {dt_grid.best_score_:.4f}")
