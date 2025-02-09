import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def generate_realistic_data(samples=50000):
    np.random.seed(42)
    return pd.DataFrame({
        'tenure': np.random.poisson(24, samples),
        'monthly_charges': np.clip(np.random.normal(70, 15, samples), 20, 120),
        'total_charges': np.abs(np.random.normal(2500, 1000, samples)),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                   samples, p=[0.6, 0.3, 0.1]),
        'online_security': np.random.choice([0, 1], samples, p=[0.65, 0.35]),
        'tech_support': np.random.choice([0, 1], samples, p=[0.6, 0.4]),
        'streaming_tv': np.random.choice([0, 1], samples, p=[0.55, 0.45]),
        'churn': np.random.choice([0, 1], samples, p=[0.75, 0.25])
    })

def preprocess_data():
    df = generate_realistic_data()
    df = pd.get_dummies(df, columns=['contract'], drop_first=True)

    df['value_score'] = (df['total_charges'] / (df['tenure'] + 1)) * df['online_security']
    df['engagement_ratio'] = df['streaming_tv'] / (df['tenure'] + 1)

    X = df.drop('churn', axis=1)
    y = df['churn']

    smote = SMOTE(sampling_strategy=0.67, random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)

    return X_scaled, y_res, scaler

if __name__ == "__main__":
    X, y, scaler = preprocess_data()
    print("Data Preprocessing Completed")
