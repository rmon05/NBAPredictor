from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np


# Paths
GAMES_WINDOW_SZ = 10
data_dir = Path(__file__).parent / "../data"
    
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size)
        )

    def forward(self, input):
        return self.net(input)
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
    
    def forward(self, input_seq):
        _, hidden = self.gru(input_seq)
        # (1, batch, hidden_size) --> (batch, hidden_size)
        hidden = hidden.squeeze(0)
        return hidden
    
class HybridNN(nn.Module):
    def __init__(self, fnn_input_size, fnn_hidden_size, gru_input_size, gru_hidden_size):
        super().__init__()
        self.gru_home = GRU(gru_input_size, gru_hidden_size)
        self.gru_away = GRU(gru_input_size, gru_hidden_size)
        self.fnn = FNN(fnn_input_size, fnn_hidden_size)
        self.fc_top = nn.Linear(2*gru_hidden_size+fnn_hidden_size, 1)

    def forward(self, home_seq, away_seq, static_input):
        hidden_fnn = self.fnn(static_input)
        hidden_home = self.gru_home(home_seq)
        hidden_away = self.gru_away(away_seq)
        
        # feed all into final top fc layer: (batch, hidden_size*2)
        combined = torch.cat([hidden_home, hidden_away, hidden_fnn], dim=1)  
        out = self.fc_top(combined)

        # (batch,1) --> (batch,)
        return out.squeeze(1)

    
# class TwoGRUModel(nn.Module):
#     # TBD TBD TBD TBD TBD TBD TBD
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.gru_home = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
#         self.gru_away = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size*2, 1)  # regression output

#     def forward(self, home_seq, away_seq):
#         _, h_home = self.gru_home(home_seq)
#         _, h_away = self.gru_away(away_seq)
#         h_home = h_home.squeeze(0)  # (1, batch, hidden_size) --> (batch, hidden_size)
#         h_away = h_away.squeeze(0)
#         combined = torch.cat([h_home, h_away], dim=1)  # (batch, hidden_size*2)
#         out = self.fc(combined)
#         return out.squeeze(1)  # (batch,1) --> (batch,)

    
class WindowedDataset:
    # Clean this up!
    def __init__(self):
        self.X_train_list = []
        self.y_train_list = []
        self.X_test_list = []
        self.y_test_list = []

    def add_games(self, df_games, train_ratio=0.5):
        # Because this is time series, we need to: 
        # 1: split by data threshold, 2: normalize, 
        # 3. group games by team and window

        # split into train & test
        train_thresh = int(len(df_games)*train_ratio)
        df_train_unwindowed = df_games.head(train_thresh)
        df_test_unwindowed = df_games.tail(len(df_games)-train_thresh)

        # take only relevant cols
        cols = ["Date", "Home", "Opp", "Result"]
        scale_cols = ["Result"]
        df_train_unwindowed = df_train_unwindowed.loc[:,cols]
        df_test_unwindowed = df_test_unwindowed.loc[:,cols]

        # normalize just this batch of games
        ct = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), scale_cols)
            ],
            remainder="passthrough"
        )
        ct.fit(df_train_unwindowed)
        df_train_unwindowed_normalized = ct.transform(df_train_unwindowed)
        df_test_unwindowed_normalized = ct.transform(df_test_unwindowed)

        # Convert back to pandas df for columns 
        all_columns = list(scale_cols) + [c for c in df_train_unwindowed.columns if c not in scale_cols]
        df_train_unwindowed_normalized = pd.DataFrame(
            df_train_unwindowed_normalized,
            columns=all_columns,
            index=df_train_unwindowed.index
        )
        df_test_unwindowed_normalized = pd.DataFrame(
            df_test_unwindowed_normalized,
            columns=all_columns,
            index=df_test_unwindowed.index
        )

        team_games = {}
        # window and write TRAINING data
        for _, row in df_train_unwindowed_normalized.iterrows():
            home = row["Home"]
            away = row["Opp"]
            if home not in team_games:
                team_games[home] = []
            if away not in team_games:
                team_games[away] = []

            # Windowed datapoint
            if len(team_games[home]) >= GAMES_WINDOW_SZ and len(team_games[away]) >= GAMES_WINDOW_SZ:
                # TODO match up the game details
                # windowed_datapoint["Date"] = row["Date"]
                # windowed_datapoint["Home"] = home
                # windowed_datapoint["Away"] = away
                # Store sequenced data for both teams
                feature_cols = ["Result"]
                home_window = np.array([
                    [game[col] for col in feature_cols] for game in team_games[home][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                away_window = np.array([
                    [game[col] for col in feature_cols] for game in team_games[away][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                windowed_datapoint = np.stack([home_window, away_window], axis=0)
                self.X_train_list.append(windowed_datapoint)
                self.y_train_list.append(row["Result"])
            
            # Store only subset of stats
            home_data = {
                "Result": row["Result"]
            }
            away_data = {
                "Result": -row["Result"]
            }
            team_games[home].append(home_data)
            team_games[away].append(away_data)
        # window and write TEST data
        for _, row in df_test_unwindowed_normalized.iterrows():
            home = row["Home"]
            away = row["Opp"]
            if home not in team_games:
                team_games[home] = []
            if away not in team_games:
                team_games[away] = []

            # Windowed datapoint
            if len(team_games[home]) >= GAMES_WINDOW_SZ and len(team_games[away]) >= GAMES_WINDOW_SZ:
                # TODO match up the game details later
                # windowed_datapoint["Date"] = row["Date"]
                # windowed_datapoint["Home"] = home
                # windowed_datapoint["Away"] = away
                # Store sequenced data for both teams
                feature_cols = ["Result"]
                home_window = np.array([
                    [game[col] for col in feature_cols] for game in team_games[home][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                away_window = np.array([
                    [game[col] for col in feature_cols] for game in team_games[away][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                windowed_datapoint = np.stack([home_window, away_window], axis=0)
                self.X_test_list.append(windowed_datapoint)
                self.y_test_list.append(row["Result"])
            
            # Store only subset of stats
            home_data = {
                "Result": row["Result"]
            }
            away_data = {
                "Result": -row["Result"]
            }
            team_games[home].append(home_data)
            team_games[away].append(away_data)


    def to_tensor(self):
        # convert list to numpy array as intermediary
        X_train = np.stack(self.X_train_list)
        y_train = np.stack(self.y_train_list)
        X_test = np.stack(self.X_test_list)
        y_test = np.stack(self.y_test_list)

        # convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


        
def main():
    years = [year for year in range(2015, 2026)]
    total_start = time.time()

    # Init windowed dataset
    wd = WindowedDataset()

    # Add data per year
    for year in years:
        start_year = time.time()
        joined_games_path = data_dir / f"joined/parquet/{year}/gamesJoined.parquet"
        df_games = pd.read_parquet(joined_games_path)
        wd.add_games(df_games, train_ratio=0.5)
        end_year = time.time()
        print(f"Finished window process for year {year} | Total time: {end_year - start_year:.2f}s\n")
        

    # Train and test
    # For GRU split into home and away sequences
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = wd.to_tensor()

    X_train_home_tensor = X_train_tensor[:, 0, :, :]
    X_train_away_tensor = X_train_tensor[:, 1, :, :]
    X_test_home_tensor = X_test_tensor[:, 0, :, :]
    X_test_away_tensor = X_test_tensor[:, 1, :, :]
    train_dataset = TensorDataset(X_train_home_tensor, X_train_away_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_home_tensor, X_test_away_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    

    # TBD TBD TBD TBD TBD TBD TBD
    # NEED to feed some static data into FNN to use
    
    # Init network
    EPOCHS = 50
    LR = 0.001
    model = TwoGRUModel(input_size=1, hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for home_batch, away_batch, y_batch in train_loader:   # <-- two inputs now
            optimizer.zero_grad()
            outputs = model(home_batch, away_batch)  # forward pass
            loss = criterion(outputs, y_batch)      # compute MSE
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * home_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        if epoch % 10 == 0 or epoch == EPOCHS-1:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for home_batch, away_batch, y_batch in test_loader:
            preds = model(home_batch, away_batch)
            y_pred.extend(preds.tolist())
            y_true.extend(y_batch.tolist())

        # Compute MSE
        y_pred_tensor = torch.tensor(y_pred)
        y_true_tensor = torch.tensor(y_true)
        mse = nn.functional.mse_loss(y_pred_tensor, y_true_tensor)
        print(f"MSE: {mse:.4f}")

    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")



if __name__ == "__main__":
    main()