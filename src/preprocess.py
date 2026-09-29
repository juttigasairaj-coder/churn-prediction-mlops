import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW / 'churn.csv')
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    X, y = df.drop('Churn', axis=1), df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))
    np.save(PROCESSED / 'X_train_final.npy', X_train_final)
    np.save(PROCESSED / 'X_test_final.npy', X_test_final)
    np.save(PROCESSED / 'y_train.npy', y_train.values)
    np.save(PROCESSED / 'y_test.npy', y_test.values)
    joblib.dump(scaler, MODELS / 'scaler.pkl')
    joblib.dump(ohe, MODELS / 'ohe.pkl')
    metadata = {'dataset_name':'Telco Customer Churn','train_shape':list(X_train_final.shape),'test_shape':list(X_test_final.shape),'numerical_features':num_cols,'categorical_features':cat_cols}
    with open(PROCESSED / 'dataset_metadata.json','w') as f: json.dump(metadata,f,indent=4)
    print(f'Preprocessing complete: train={X_train_final.shape}, test={X_test_final.shape}')

if __name__ == '__main__': main()
