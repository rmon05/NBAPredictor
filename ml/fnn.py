import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#  TEMPORARY
data_dir = Path(__file__).parent / "../data"
train_X_parquet_file = data_dir / f"normalized/parquet/gamesTrainX.parquet"
test_X_parquet_file = data_dir / f"normalized/parquet/gamesTestX.parquet"
train_y_parquet_file = data_dir / f"normalized/parquet/gamesTrainY.parquet"
test_y_parquet_file = data_dir / f"normalized/parquet/gamesTestY.parquet"
X_train = pd.read_parquet(train_X_parquet_file)
X_test = pd.read_parquet(test_X_parquet_file)
y_train = pd.read_parquet(train_y_parquet_file)
y_test = pd.read_parquet(test_y_parquet_file)

# === Convert to tensors ===
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# === Wrap into PyTorch Datasets and Loaders ===
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# === Define a simple Feedforward Neural Network ===
class NBAFNN(nn.Module):
    def __init__(self, input_size):
        super(NBAFNN, self).__init__()
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

# === Initialize model, loss, optimizer ===
input_size = X_train.shape[1]
model = NBAFNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training loop ===
epochs = 50
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
    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []

    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        y_pred.extend(preds.squeeze().tolist())
        y_true.extend(y_batch.squeeze().tolist())

mse = nn.functional.mse_loss(torch.tensor(y_pred), torch.tensor(y_true))
print(f"\nTest MSE: {mse:.4f}")