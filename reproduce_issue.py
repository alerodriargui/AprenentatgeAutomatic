import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

try:
    print("--- Loading ds_13.csv to get X.columns ---")
    df = pd.read_csv('ds_13.csv')
    cols_to_drop = ['Unnamed: 0', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 
                    'cvtd_timestamp', 'new_window', 'num_window']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Remove columns with excessive missing data (as in notebook)
    threshold = 0.5 * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    df = df.dropna()
    
    X = df.drop(columns=['class'])
    print(f"X shape: {X.shape}")
    print("X dtypes:")
    # print(X.dtypes)

    print("\n--- Loading common.csv ---")
    common_df = pd.read_csv('common.csv')
    print(f"common.csv shape: {common_df.shape}")
    
    print("\n--- Preprocessing common.csv ---")
    common_df_clean = common_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Ensure alignment
    # This might fail if common.csv is missing columns present in X
    # or if we try to select columns that don't exist.
    # But the error is 'isnan', so it's likely about content.
    
    # Check if all X columns are in common_df_clean
    missing_cols = set(X.columns) - set(common_df_clean.columns)
    if missing_cols:
        print(f"Warning: Missing columns in common.csv: {missing_cols}")
    
    # Select columns
    # We only select columns that exist in both to avoid KeyError, 
    # but the notebook does `common_df_clean = common_df_clean[X.columns]`
    # which implies they must exist.
    common_df_clean = common_df_clean[X.columns]
    
    print("common_df_clean dtypes before fillna:")
    print(common_df_clean.dtypes[common_df_clean.dtypes == 'object'])
    
    common_df_clean = common_df_clean.fillna(0)
    
    print("\n--- Checking for non-numeric columns ---")
    non_numeric_cols = common_df_clean.select_dtypes(include=['object']).columns
    if len(non_numeric_cols) > 0:
        print(f"Non-numeric columns found: {non_numeric_cols.tolist()}")
        for col in non_numeric_cols:
            print(f"Unique values in {col}: {common_df_clean[col].unique()}")
            
    print("\n--- Attempting StandardScaler ---")
    scaler = StandardScaler()
    # We need to fit scaler on X first to match notebook
    scaler.fit(X)
    
    X_common_scaled = scaler.transform(common_df_clean)
    print("StandardScaler successful.")
    
except Exception as e:
    print(f"\nCaught Exception: {e}")
    import traceback
    traceback.print_exc()
