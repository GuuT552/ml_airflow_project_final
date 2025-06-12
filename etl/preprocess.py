import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame) -> tuple:
    df = df.drop(columns=['ID'])
    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y