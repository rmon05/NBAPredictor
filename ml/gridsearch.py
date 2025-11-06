import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from pathlib import Path
import time

data_dir = Path(__file__).parent / "../data"
input_path_parquet = data_dir / f"rolling/parquet/gamesRolling.parquet"


def normalize_rolling(X_train, X_test, y_train, y_test, home_train, home_test):
    # Z scale everything else besides Home which is binary feature
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Add Home back unscaled
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_train_scaled["Home"] = home_train.values
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    X_test_scaled["Home"] = home_test.values

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1))
    y_train_scaled = pd.DataFrame(y_train_scaled, columns=["Result"])
    y_test_scaled  = pd.DataFrame(y_test_scaled, columns=["Result"])

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

def run_grid_search():
    df = pd.read_parquet(input_path_parquet)

    # Separate X and y
    X = df.drop(columns=["Result"])
    y = df["Result"]

    # Normalize rolling datapoints
    X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    home_train, home_test = X_train["Home"], X_test["Home"]
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = normalize_rolling(X_train, X_test, y_train, y_test, home_train, home_test)


    # Try smaller inputs
    cols = ["Home", "WinPct1", "WinPct2", "OppWinPct1", "OppWinPct2", "PTSDiff1", "PTSDiff2", 
            "Streak1", "Streak2"
            ]
    for i in range(5):
        cols.append(f"Starter{i}_BPM1")
        cols.append(f"Starter{i}_BPM2")
    X_train_spreads = X_train_scaled.loc[:, "Spread"]
    X_test_spreads = X_test_scaled.loc[:, "Spread"]
    X_train_scaled = X_train_scaled.loc[:, cols]
    X_test_scaled = X_test_scaled.loc[:, cols]


    # GRID for in house model
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    model = xgb.XGBRegressor(
        random_state=42,
        n_estimators=500,
        tree_method="hist"
    )
    grid_search_inhouse = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,               # 5-fold cross-validation
        verbose=2,          # print progress
        n_jobs=-1           # use all cores
    )
    grid_search_inhouse.fit(X_train_scaled, y_train_scaled)
    best_model_inhouse = grid_search_inhouse.best_estimator_

    # GRID for book model
    grid_search_book = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,               # 5-fold cross-validation
        verbose=2,          # print progress
        n_jobs=-1           # use all cores
    )
    grid_search_book.fit(X_train_spreads, y_train_scaled)
    best_model_book = grid_search_book.best_estimator_


    
    # Predict ONCE with best model
    y_pred_inhouse =  best_model_inhouse.predict(X_test_scaled)
    inhouse_mse = mean_squared_error(y_test_scaled, y_pred_inhouse)
    print("In house test MSE:", inhouse_mse)
    print("In house test RMSE:", inhouse_mse ** 0.5)

    y_pred_book =  best_model_book.predict(X_test_spreads)
    book_mse = mean_squared_error(y_test_scaled, y_pred_book)
    print("Book test MSE:", book_mse)
    print("Book test RMSE:", book_mse ** 0.5)

        


if __name__ == "__main__":
    total_start = time.time()

    run_grid_search()

    total_end = time.time()
    print(f"Completed in {total_end - total_start:.2f}s")
