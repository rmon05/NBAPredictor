import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

from normalize import normalize
from pathlib import Path
import time

data_dir = Path(__file__).parent / "../data"
input_path_parquet = data_dir / f"rolling/parquet/gamesRolling.parquet"

class FNN(nn.Module):
    def __init__(self, input_size):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

def compute_baseline_mse(y_train, y_test):
    y_train_mean = y_train.mean()
    y_pred_baseline = [y_train_mean] * len(y_test)
    baseline_mse = mean_squared_error(y_test, y_pred_baseline)
    return baseline_mse

def kfold_cross_validation(k=5, epochs=50, batch_size=64, lr=0.001):
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

        # Compute baseline
        baseline_mse = compute_baseline_mse(y_train_scaled, y_test_scaled)
        print(f"Baseline MSE (predicting mean outcome): {baseline_mse:.4f}")
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled.values, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Init model
        model = FNN(input_size=X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            if epoch % 10 == 0 or epoch == epochs-1:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            for X_batch, y_batch in test_loader:
                preds = model(X_batch)
                y_pred.extend(preds.squeeze().tolist())
                y_true.extend(y_batch.squeeze().tolist())

            # Convert to tensors for MSE computation
            y_pred_tensor = torch.tensor(y_pred)
            y_true_tensor = torch.tensor(y_true)

            # Compute MSE
            mse = nn.functional.mse_loss(y_pred_tensor, y_true_tensor)
            fold_mses.append(mse.item())
            print(f"Fold {fold+1} MSE: {mse:.4f}")

            # Compute sign accuracy for game prediction
            # NOTE This is not entirely accurate since Z scaling results can change sign
            win_loss_correct = sum(
                (torch.sign(y_pred_tensor) == torch.sign(y_true_tensor)).int()
            ).item()
            win_loss_accuracy = win_loss_correct / len(y_true_tensor)
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

    kfold_cross_validation(k=5, epochs=50)

    total_end = time.time()
    print(f"Completed in {total_end - total_start:.2f}s")
