# scripts/data_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df = df.dropna()  # Handle missing values
    X = df.drop(columns=['MEDV'])  # Features
    y = df['MEDV']  # Target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = load_data('../data/BostonHousing.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print("Data preprocessing complete.")
