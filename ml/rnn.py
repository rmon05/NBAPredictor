from pathlib import Path
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np


# Paths
GAMES_WINDOW_SZ = 5
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
        self.fnn = FNN(fnn_input_size, fnn_hidden_size)
        self.gru_team = GRU(gru_input_size, gru_hidden_size)
        self.gru_opp = GRU(gru_input_size, gru_hidden_size)
        self.fc_top = nn.Linear(2*gru_hidden_size+fnn_hidden_size, 1)

    def forward(self, static_input, team_seq, opp_seq):
        hidden_fnn = self.fnn(static_input)
        hidden_team = self.gru_team(team_seq)
        hidden_opp = self.gru_opp(opp_seq)
        
        # feed all into final top fc layer: (batch, hidden_size*2)
        combined = torch.cat([hidden_fnn, hidden_team, hidden_opp], dim=1)  
        out = self.fc_top(combined)

        # DONT squeeze for now since y_batch is (batch, 1) 
        return out
        # (batch,1) --> (batch,)
        # return out.squeeze(1)

    
class WindowedDataset:
    # Clean this up!
    def __init__(self):
        self.X_train_static = np.array([])
        self.X_train_seq_team = np.array([])
        self.X_train_seq_opp = np.array([])
        self.y_train = np.array([])
        self.X_test_static = np.array([])
        self.X_test_seq_team = np.array([])
        self.X_test_seq_opp = np.array([])
        self.y_test = np.array([])

        self.X_list_train_static = []
        self.X_list_test_static = []
        self.X_list_train_seq_team = []
        self.X_list_test_seq_team = []
        self.X_list_train_seq_opp = []
        self.X_list_test_seq_opp = []
        self.y_list_train = []
        self.y_list_test = []

    def _np_concat(self, curr, new):
        if curr is None or curr.size == 0:
            return new
        return np.concatenate([curr, new], axis=0)

    def add_games(self, df_games):
        # In order to allow shuffling, we need to:
        # 1. group games by team and window
        # 2. split randomly, 3. normalize

        # Convert box scores to list of maps
        df_games["HomeBoxScores"] = df_games["HomeBoxScores"].apply(json.loads)
        df_games["AwayBoxScores"] = df_games["AwayBoxScores"].apply(json.loads)

        # window data
        X_list_static = []
        X_list_seq_team = []
        X_list_seq_opp = []
        y_list = []
        team_games = {}
        team_stats = {}
        player_stats = {}
        for _, row in df_games.iterrows():
            home = row["Home"]
            away = row["Opp"]

            home_box_scores = row["HomeBoxScores"]
            away_box_scores = row["AwayBoxScores"]
            
            # init STATIC data
            if not (home in team_stats):
                team_stats[home] = {}               
                team_stats[home]["PTSDiffTot"] = 0
                team_stats[home]["Played"] = 0
            if not (away in team_stats):
                team_stats[away] = {}
                team_stats[away]["PTSDiffTot"] = 0
                team_stats[away]["Played"] = 0

            # Init sequence
            if home not in team_games:
                team_games[home] = []
            if away not in team_games:
                team_games[away] = []

            # Windowed datapoint
            has_rest_data = not (pd.isna(row["RestHome"]) or pd.isna(row["RestOpp"]))
            if len(team_games[home]) >= GAMES_WINDOW_SZ and len(team_games[away]) >= GAMES_WINDOW_SZ and has_rest_data:
                # TODO match up the game details
                # windowed_datapoint["Date"] = row["Date"]
                # windowed_datapoint["Home"] = home
                # windowed_datapoint["Away"] = away

                # Static data for both teams
                home_location = 1
                away_location = 0
                home_pts_diff = team_stats[home]["PTSDiffTot"]/team_stats[home]["Played"]
                away_pts_diff = team_stats[away]["PTSDiffTot"]/team_stats[away]["Played"]
                home_rest = row["RestHome"]
                away_rest = row["RestOpp"]
                home_static_features = [
                    home_location, home_pts_diff, away_pts_diff, home_rest, away_rest
                ]
                away_static_features = [
                    away_location, away_pts_diff, home_pts_diff, away_rest, home_rest
                ]
                # Assume 5 highest minute players are the starters
                # Write data for both sides
                # Make sure the player orders are correct for each datapoint!
                for i in range(5):
                    player_team = home_box_scores[i]
                    player_opp = away_box_scores[i]
                    pid_team = player_team["Player-additional"]
                    pid_opp = player_opp["Player-additional"]
                    if pid_team in player_stats:
                        home_static_features.append(player_stats[pid_team]["TotBPM"]/player_stats[pid_team]["GamesPlayed"])
                    else:
                        home_static_features.append(0)
                    if pid_opp in player_stats:
                        away_static_features.append(player_stats[pid_opp]["TotBPM"]/player_stats[pid_opp]["GamesPlayed"])
                    else:
                        away_static_features.append(0)

                for i in range(5):
                    player_team = away_box_scores[i]
                    player_opp = home_box_scores[i]
                    pid_team = player_team["Player-additional"]
                    pid_opp = player_opp["Player-additional"]
                    if pid_team in player_stats:
                        home_static_features.append(player_stats[pid_team]["TotBPM"]/player_stats[pid_team]["GamesPlayed"])
                    else:
                        home_static_features.append(0)
                    if pid_opp in player_stats:
                        away_static_features.append(player_stats[pid_opp]["TotBPM"]/player_stats[pid_opp]["GamesPlayed"])
                    else:
                        away_static_features.append(0)

                home_static = np.array(home_static_features)
                away_static = np.array(away_static_features)

                # Store sequenced data for both teams
                # The normalization will have to be tweaked as well
                seq_features = ["Result"]
                home_window = np.array([
                    [game[col] for col in seq_features] for game in team_games[home][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                away_window = np.array([
                    [game[col] for col in seq_features] for game in team_games[away][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                # Write datapoint for both sides
                X_list_static.append(home_static)
                X_list_seq_team.append(home_window)
                X_list_seq_opp.append(away_window)
                y_list.append(row["Result"])
                X_list_static.append(away_static)
                X_list_seq_team.append(away_window)
                X_list_seq_opp.append(home_window)
                y_list.append(-row["Result"])
            
            # UPDATE
            # Store only subset of stats
            home_data = {
                "Result": row["Result"]
            }
            away_data = {
                "Result": -row["Result"]
            }
            team_games[home].append(home_data)
            team_games[away].append(away_data)

            # Update static accumulated
            team_stats[home]["PTSDiffTot"] += row["Result"]
            team_stats[home]["Played"] += 1
            team_stats[away]["PTSDiffTot"] += -row["Result"]
            team_stats[away]["Played"] += 1

            # Box score stats
            for player in home_box_scores:
                pid = player["Player-additional"]
                if not (pid in player_stats):
                    player_stats[pid] = {}
                    player_stats[pid]["GamesPlayed"] = 0
                    player_stats[pid]["TotBPM"] = 0

                player_stats[pid]["GamesPlayed"] += 1
                player_stats[pid]["TotBPM"] += player["BPM"]

                # team_stats[home]["AST"] += player["AST"]
                # team_stats[home]["ORB"] += player["ORB"]
                # team_stats[away]["OppDRB"] += player["DRB"]
                # team_stats[home]["STL"] += player["STL"]
                # team_stats[home]["BLK"] += player["BLK"]
                # team_stats[home]["TOV"] += player["TOV"]
            
            for player in away_box_scores:
                pid = player["Player-additional"]
                if not (pid in player_stats):
                    player_stats[pid] = {}
                    player_stats[pid]["GamesPlayed"] = 0
                    player_stats[pid]["TotBPM"] = 0
                
                player_stats[pid]["GamesPlayed"] += 1
                player_stats[pid]["TotBPM"] += player["BPM"]

                # team_stats[away]["AST"] += player["AST"]
                # team_stats[away]["ORB"] += player["ORB"]
                # team_stats[home]["OppDRB"] += player["DRB"]
                # team_stats[away]["STL"] += player["STL"]
                # team_stats[away]["BLK"] += player["BLK"]
                # team_stats[away]["TOV"] += player["TOV"]

        # Shuffled split 
        (
            X_list_train_static,
            X_list_test_static,
            X_list_train_seq_team,
            X_list_test_seq_team,
            X_list_train_seq_opp,
            X_list_test_seq_opp,
            y_list_train,
            y_list_test
        ) = train_test_split(
            X_list_static,
            X_list_seq_team,
            X_list_seq_opp,
            y_list,
            test_size=0.2,
            shuffle=True,
            random_state=42
        )

        # Temp for comparison
        self.X_list_train_static.extend(X_list_train_static)
        self.X_list_test_static.extend(X_list_test_static)
        self.X_list_train_seq_team.extend(X_list_train_seq_team)
        self.X_list_test_seq_team.extend(X_list_test_seq_team)
        self.X_list_train_seq_opp.extend(X_list_train_seq_opp)
        self.X_list_test_seq_opp.extend(X_list_test_seq_opp)
        self.y_list_train.extend(y_list_train)
        self.y_list_test.extend(y_list_test)


        # # Normalize
        # scaler_X_static = StandardScaler()
        # X_train_static_scaled = scaler_X_static.fit_transform(np.stack(X_list_train_static, axis=0))
        # X_test_static_scaled = scaler_X_static.transform(np.stack(X_list_test_static, axis=0))

        # scaler_y = StandardScaler()
        # y_train_scaled = scaler_y.fit_transform(np.array(y_list_train).reshape(-1, 1))
        # y_test_scaled = scaler_y.transform(np.array(y_list_test).reshape(-1, 1))

        # # Stack, Flatten, scale, unflatten for seq data
        # X_train_seq_team_tbd = np.stack(X_list_train_seq_team, axis=0)
        # X_test_seq_team_tbd = np.stack(X_list_test_seq_team, axis=0)
        # X_train_seq_opp_tbd = np.stack(X_list_train_seq_opp, axis=0)
        # X_test_seq_opp_tbd = np.stack(X_list_test_seq_opp, axis=0)

        # N_train, seq_len, num_feat = X_train_seq_team_tbd.shape
        # N_test, seq_len, num_feat = X_test_seq_team_tbd.shape

        # X_train_seq_team_tbd = X_train_seq_team_tbd.reshape(-1, 1)
        # X_test_seq_team_tbd = X_test_seq_team_tbd.reshape(-1, 1)
        # X_train_seq_opp_tbd = X_train_seq_opp_tbd.reshape(-1, 1)
        # X_test_seq_opp_tbd = X_test_seq_opp_tbd.reshape(-1, 1)

        # X_train_seq_team_scaled = scaler_y.transform(X_train_seq_team_tbd)
        # X_test_seq_team_scaled = scaler_y.transform(X_test_seq_team_tbd)
        # X_train_seq_opp_scaled = scaler_y.transform(X_train_seq_opp_tbd)
        # X_test_seq_opp_scaled = scaler_y.transform(X_test_seq_opp_tbd)

        # X_train_seq_team_scaled = X_train_seq_team_scaled.reshape(N_train, seq_len, num_feat)
        # X_test_seq_team_scaled = X_test_seq_team_scaled.reshape(N_test, seq_len, num_feat)
        # X_train_seq_opp_scaled = X_train_seq_opp_scaled.reshape(N_train, seq_len, num_feat)
        # X_test_seq_opp_scaled = X_test_seq_opp_scaled.reshape(N_test, seq_len, num_feat)

        # # concat with rest of data
        # self.X_train_static    = self._np_concat(self.X_train_static, X_train_static_scaled)
        # self.X_train_seq_team  = self._np_concat(self.X_train_seq_team, X_train_seq_team_scaled)
        # self.X_train_seq_opp   = self._np_concat(self.X_train_seq_opp, X_train_seq_opp_scaled)
        # self.y_train           = self._np_concat(self.y_train, y_train_scaled)
        # self.X_test_static     = self._np_concat(self.X_test_static, X_test_static_scaled)
        # self.X_test_seq_team   = self._np_concat(self.X_test_seq_team, X_test_seq_team_scaled)
        # self.X_test_seq_opp    = self._np_concat(self.X_test_seq_opp, X_test_seq_opp_scaled)
        # self.y_test            = self._np_concat(self.y_test, y_test_scaled)

    def to_tensor(self):
        # Normalize ALL
        scaler_X_static = StandardScaler()
        X_train_static_scaled = scaler_X_static.fit_transform(np.stack(self.X_list_train_static, axis=0))
        X_test_static_scaled = scaler_X_static.transform(np.stack(self.X_list_test_static, axis=0))

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(np.array(self.y_list_train).reshape(-1, 1))
        y_test_scaled = scaler_y.transform(np.array(self.y_list_test).reshape(-1, 1))

        # Stack, Flatten, scale, unflatten for seq data
        X_train_seq_team_tbd = np.stack(self.X_list_train_seq_team, axis=0)
        X_test_seq_team_tbd = np.stack(self.X_list_test_seq_team, axis=0)
        X_train_seq_opp_tbd = np.stack(self.X_list_train_seq_opp, axis=0)
        X_test_seq_opp_tbd = np.stack(self.X_list_test_seq_opp, axis=0)

        N_train, seq_len, num_feat = X_train_seq_team_tbd.shape
        N_test, seq_len, num_feat = X_test_seq_team_tbd.shape

        X_train_seq_team_tbd = X_train_seq_team_tbd.reshape(-1, 1)
        X_test_seq_team_tbd = X_test_seq_team_tbd.reshape(-1, 1)
        X_train_seq_opp_tbd = X_train_seq_opp_tbd.reshape(-1, 1)
        X_test_seq_opp_tbd = X_test_seq_opp_tbd.reshape(-1, 1)

        X_train_seq_team_scaled = scaler_y.transform(X_train_seq_team_tbd)
        X_test_seq_team_scaled = scaler_y.transform(X_test_seq_team_tbd)
        X_train_seq_opp_scaled = scaler_y.transform(X_train_seq_opp_tbd)
        X_test_seq_opp_scaled = scaler_y.transform(X_test_seq_opp_tbd)

        X_train_seq_team_scaled = X_train_seq_team_scaled.reshape(N_train, seq_len, num_feat)
        X_test_seq_team_scaled = X_test_seq_team_scaled.reshape(N_test, seq_len, num_feat)
        X_train_seq_opp_scaled = X_train_seq_opp_scaled.reshape(N_train, seq_len, num_feat)
        X_test_seq_opp_scaled = X_test_seq_opp_scaled.reshape(N_test, seq_len, num_feat)

        # concat with rest of data
        self.X_train_static    = self._np_concat(self.X_train_static, X_train_static_scaled)
        self.X_train_seq_team  = self._np_concat(self.X_train_seq_team, X_train_seq_team_scaled)
        self.X_train_seq_opp   = self._np_concat(self.X_train_seq_opp, X_train_seq_opp_scaled)
        self.y_train           = self._np_concat(self.y_train, y_train_scaled)
        self.X_test_static     = self._np_concat(self.X_test_static, X_test_static_scaled)
        self.X_test_seq_team   = self._np_concat(self.X_test_seq_team, X_test_seq_team_scaled)
        self.X_test_seq_opp    = self._np_concat(self.X_test_seq_opp, X_test_seq_opp_scaled)
        self.y_test            = self._np_concat(self.y_test, y_test_scaled)

        # convert to tensors
        X_train_static_tensor = torch.tensor(self.X_train_static, dtype=torch.float32)
        X_train_seq_team_tensor = torch.tensor(self.X_train_seq_team, dtype=torch.float32)
        X_train_seq_opp_tensor = torch.tensor(self.X_train_seq_opp, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        X_test_static_tensor = torch.tensor(self.X_test_static, dtype=torch.float32)
        X_test_seq_team_tensor = torch.tensor(self.X_test_seq_team, dtype=torch.float32)
        X_test_seq_opp_tensor = torch.tensor(self.X_test_seq_opp, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

        return \
        X_train_static_tensor, X_train_seq_team_tensor, X_train_seq_opp_tensor, y_train_tensor, \
        X_test_static_tensor, X_test_seq_team_tensor, X_test_seq_opp_tensor, y_test_tensor


        
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
        wd.add_games(df_games)
        end_year = time.time()
        print(f"Finished window process for year {year} | Total time: {end_year - start_year:.2f}s\n")
        

    # Train and test
    # For GRU split into home and away sequences
    X_train_static_tensor, X_train_seq_team_tensor, X_train_seq_opp_tensor, y_train_tensor, \
    X_test_static_tensor, X_test_seq_team_tensor, X_test_seq_opp_tensor, y_test_tensor = wd.to_tensor()


    # TEST OUTPUT PRINT
    print(torch.isnan(X_train_static_tensor).any())
    print(torch.isnan(X_test_static_tensor).any())

    def sanity_check_tensor(name, tensor):
        print(f"\n{name}:")
        print(f"  shape: {tensor.shape}")
        # print(f"  dtype: {tensor.dtype}")
        # try:
        #     print(f"  min:   {tensor.min().item():.4f}")
        #     print(f"  max:   {tensor.max().item():.4f}")
        # except:
        #     print("  (min/max not available)")
        # print(f"  sample: {tensor[0]}\n")


    print("=== Sanity Check: Train Tensors ===")
    sanity_check_tensor("X_train_static_tensor", X_train_static_tensor)
    sanity_check_tensor("X_train_seq_team_tensor", X_train_seq_team_tensor)
    sanity_check_tensor("X_train_seq_opp_tensor", X_train_seq_opp_tensor)
    sanity_check_tensor("y_train_tensor", y_train_tensor)

    print("=== Sanity Check: Test Tensors ===")
    sanity_check_tensor("X_test_static_tensor", X_test_static_tensor)
    sanity_check_tensor("X_test_seq_team_tensor", X_test_seq_team_tensor)
    sanity_check_tensor("X_test_seq_opp_tensor", X_test_seq_opp_tensor)
    sanity_check_tensor("y_test_tensor", y_test_tensor)


    train_dataset = TensorDataset(
        X_train_static_tensor,
        X_train_seq_team_tensor,
        X_train_seq_opp_tensor,
        y_train_tensor
    )
    test_dataset = TensorDataset(
        X_test_static_tensor,
        X_test_seq_team_tensor,
        X_test_seq_opp_tensor,
        y_test_tensor
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Init network
    EPOCHS = 30
    LR = 0.001
    static_input_size = X_train_static_tensor.shape[1]
    seq_input_size = X_train_seq_team_tensor.shape[2]
    model = HybridNN(
        fnn_input_size=static_input_size, fnn_hidden_size=8, 
        gru_input_size=seq_input_size, gru_hidden_size=8
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for static_batch, team_seq_batch, opp_seq_batch, y_batch in train_loader:
            # reset grads and forward pass
            optimizer.zero_grad()
            outputs = model(static_batch, team_seq_batch, opp_seq_batch)
            # compute MSE as criterion and backprop
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * team_seq_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        if epoch % 10 == 0 or epoch == EPOCHS-1:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = []
        y_true = []
        for static_batch, team_seq_batch, opp_seq_batch, y_batch in test_loader:
            preds = model(static_batch, team_seq_batch, opp_seq_batch)
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