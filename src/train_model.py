import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

def main():
    #load normalized training data and retrieve best param.
    X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()

    best_params = joblib.load('models/best_params.pkl')

    #instantiate and train model with optimal param.; save trained model
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')


if __name__ == "__main__":
    main()