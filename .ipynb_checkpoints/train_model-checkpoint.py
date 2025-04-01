import joblib
import os
from sklearn.linear_model import LinearRegression
from data_preprocessing import load_data, preprocess_data

# Function to train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Load and preprocess the data
    df = load_data('../data/BostonHousing.csv')  # Load data from CSV
    X_train, X_test, y_train, y_test = preprocess_data(df)  # Preprocess data (split, scale, etc.)

    # Train the model
    model = train_model(X_train, y_train)

    # Ensure the directory exists where the model will be saved
    model_dir = '../models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory if it doesn't exist

    # Save the trained model to disk
    joblib.dump(model, os.path.join(model_dir, 'linear_regression_model.pkl'))
    print("Model training complete. Model saved.")
