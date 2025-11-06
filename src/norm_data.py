import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

def main():
    #load previously split datasets
    X_train = pd.read_csv('data/processed_data/X_train.csv')
    X_test = pd.read_csv('data/processed_data/X_test.csv')

    #drop datetime columns to avoid string to float ValueError
    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])

    #create scaler to transform datasets; only fit X_train (prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #convert arrays back to dataframes and save
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed_data/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed_data/X_test_scaled.csv', index=False)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')


if __name__ == "__main__":
    main()