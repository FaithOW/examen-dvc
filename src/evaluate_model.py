import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def main():
    #load test data and trained model 
    X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
    y_test = pd.read_csv('data/processed_data/y_test.csv')
    model = joblib.load('models/model.pkl')

    #generate predictions and save them
    preds = model.predict(X_test)
    pd.DataFrame({'predictions': preds}).to_csv('data/predictions.csv', index=False)

    #compute metrics: mse (prediction error) and r2 (overall model fit)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    #write metrics to JSON for tracking with DVC
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/scores.json', 'w') as f:
        json.dump({'MSE': mse, 'R2': r2}, f)


if __name__ == "__main__":
    main()