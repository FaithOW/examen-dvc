import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def main():
    #load normalized training data; reshape y_train into 1D array (needed for sklearn)
    X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv').values.ravel()

    #define a search space for hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    #run cross-validation and find a hyperparameter combination to max. R2; save best param.
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    joblib.dump(grid_search.best_params_, 'models/best_params.pkl')


if __name__ == "__main__":
    main()