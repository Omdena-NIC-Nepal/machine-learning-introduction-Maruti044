import joblib
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import load_data, preprocess_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-Squared: {r2}")

if __name__ == "__main__":
    df = load_data('../data/BostonHousing.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = joblib.load('../models/linear_regression_model.pkl')
    evaluate_model(model, X_test, y_test)