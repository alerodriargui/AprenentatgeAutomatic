import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, make_scorer

# 1. Data Prep
print("--- Data Prep ---")
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

# 2. Best Single Classifier (k-NN)
print("\n--- Best Single Classifier (k-NN) ---")
best_knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, average='weighted')
scores = cross_val_score(best_knn, X_pca, y_clean, cv=cv, scoring=f1_scorer)
print(f"k-NN CV F1-Scores: {scores}")
print(f"Average F1-Score: {scores.mean():.4f}")
best_knn.fit(X_pca, y_clean)
y_pred = best_knn.predict(X_pca)
print("Confusion Matrix (Full Dataset):")
print(confusion_matrix(y_clean, y_pred))

# 3. Best Ensemble (Random Forest)
print("\n--- Best Ensemble (Random Forest) ---")
best_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
scores_rf = cross_val_score(best_rf, X_pca, y_clean, cv=cv, scoring=f1_scorer)
print(f"Random Forest CV F1-Scores: {scores_rf}")
print(f"Average F1-Score: {scores_rf.mean():.4f}")
best_rf.fit(X_pca, y_clean)
y_pred_rf = best_rf.predict(X_pca)
print("Confusion Matrix (Full Dataset):")
print(confusion_matrix(y_clean, y_pred_rf))

# 4. Common.csv
print("\n--- Common.csv Processing ---")
try:
    common_df = pd.read_csv('common.csv')
    print("common.csv found.")
except FileNotFoundError:
    print("common.csv not found (expected).")
