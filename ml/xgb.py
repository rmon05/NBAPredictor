import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from normalize import normalize
from pathlib import Path
import time

data_dir = Path(__file__).parent / "../data"
input_path_parquet = data_dir / f"rolling/parquet/gamesRolling.parquet"


def compute_baseline_mse(y_train, y_test):
    y_train_mean = y_train.mean()
    y_pred_baseline = [y_train_mean] * len(y_test)
    baseline_mse = mean_squared_error(y_test, y_pred_baseline)
    return baseline_mse

def kfold_cross_validation(k=5):
    df = pd.read_parquet(input_path_parquet)

    # Separate X and y
    X = df.drop(columns=["Result"])
    y = df["Result"]


    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_mses = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold+1}:")

        # Normalize
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        home_train, home_test = X_train["Home"], X_test["Home"]
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = normalize(X_train, X_test, y_train, y_test, home_train, home_test)


        # Try smaller inputs
        cols = ["Home", "WinPct1", "WinPct2", "PTSDiff1", "PTSDiff2", 
                "Streak1", "Streak2"
                ]
        for i in range(5):
            cols.append(f"Starter{i}_BPM1")
            cols.append(f"Starter{i}_BPM2")
            
        X_train_scaled = X_train_scaled.loc[:, cols]
        X_test_scaled = X_test_scaled.loc[:, cols]


        # Compute baseline
        baseline_mse = compute_baseline_mse(y_train_scaled, y_test_scaled)
        print(f"Baseline MSE (predicting mean outcome): {baseline_mse:.4f}")

        # xg boost regressor
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train
        model.fit(X_train_scaled, y_train_scaled, eval_set=[(X_test_scaled, y_test_scaled)], 
                  verbose=False)
        
        # Predict
        y_pred = model.predict(X_test_scaled)

        # Compute MSE
        mse = np.mean((y_test_scaled.values.ravel() - y_pred)**2)
        fold_mses.append(mse)
        print(f"Fold {fold+1} MSE: {mse:.4f}")

        # Compute sign accuracy for game prediction
        # NOTE This is not entirely accurate since Z scaling results can change sign
        win_loss_accuracy = np.mean(np.sign(y_test_scaled.values.ravel()) == np.sign(y_pred))
        print(f"Fold {fold+1} Win/Loss Accuracy: {win_loss_accuracy*100:.2f}%")

        # Original scale
        mse_original = mse.item() * (y.std() ** 2)
        rmse_original = mse_original ** 0.5
        print(f"Fold {fold+1} MSE (original scale): {mse_original:.4f}")
        print(f"Fold {fold+1} RMSE (original scale): {rmse_original:.4f}")
    
    # cumulative
    print(f"\nAverage MSE across {k} folds: {np.mean(fold_mses):.4f}")
        


if __name__ == "__main__":
    total_start = time.time()

    kfold_cross_validation(k=5)

    total_end = time.time()
    print(f"Completed in {total_end - total_start:.2f}s")
