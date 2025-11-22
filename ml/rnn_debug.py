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
GAMES_WINDOW_SZ = 5
data_dir = Path(__file__).parent / "../data"

def normalize(X_train, X_test, y_train, y_test, home_train, home_test):
    # # Z scale everything else besides Home which is binary feature
    # scaler_X = StandardScaler()
    # X_train_scaled = scaler_X.fit_transform(X_train)
    # X_test_scaled = scaler_X.transform(X_test)

    # # Add Home back unscaled
    # X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    # X_train_scaled["Home"] = home_train.values
    # X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    # X_test_scaled["Home"] = home_test.values

    # scaler_y = StandardScaler()
    # y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    # y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1))
    # y_train_scaled = pd.DataFrame(y_train_scaled, columns=["Result"])
    # y_test_scaled  = pd.DataFrame(y_test_scaled, columns=["Result"])

    # return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled



    # Keep all features as-is (no scaling)
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Add Home back (if it was separated before)
    X_train_scaled["Home"] = home_train.values
    X_test_scaled["Home"] = home_test.values

    # Keep y as DataFrame with correct shape
    y_train_scaled = pd.DataFrame(y_train.values.reshape(-1, 1), columns=["Result"])
    y_test_scaled  = pd.DataFrame(y_test.values.reshape(-1, 1), columns=["Result"])

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
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

        # (batch,1) --> (batch,)
        return out.squeeze(1)

    
class WindowedDataset:
    # Clean this up!
    def __init__(self):
        self.X_train_list_static = []
        self.X_train_list_seq_team = []
        self.X_train_list_seq_opp = []
        self.y_train_list = []
        self.X_test_list_static = []
        self.X_test_list_seq_team = []
        self.X_test_list_seq_opp = []
        self.y_test_list = []

    def add_games(self, df_games, train_ratio=0.5):
        # SOMETHING PROBABLY WRONG HERE

        # Because this is time series, we need to: 
        # 1: split by data threshold, 2: normalize, 
        # 3. group games by team and window

        # split into train & test
        train_thresh = int(len(df_games)*train_ratio)
        df_train_unwindowed = df_games.head(train_thresh)
        df_test_unwindowed = df_games.tail(len(df_games)-train_thresh)

        # take only relevant cols
        cols = ["Date", "Home", "Opp", "Result", "RestHome", "RestOpp"]
        scale_cols = ["Result", "RestHome", "RestOpp"]
        df_train_unwindowed = df_train_unwindowed.loc[:,cols]
        df_test_unwindowed = df_test_unwindowed.loc[:,cols]

        # normalize just this batch of games (1 season)
        ct = ColumnTransformer(
            transformers=[
                ("scale", StandardScaler(), scale_cols)
            ],
            remainder="passthrough"
        )
        ct.fit(df_train_unwindowed)
        # df_train_unwindowed_normalized = ct.transform(df_train_unwindowed)
        # df_test_unwindowed_normalized = ct.transform(df_test_unwindowed)
        df_train_unwindowed_normalized = df_train_unwindowed
        df_test_unwindowed_normalized = df_test_unwindowed

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
        team_stats = {}
        player_stats = {}
        # window and write TRAINING data
        for _, row in df_train_unwindowed_normalized.iterrows():
            home = row["Home"]
            away = row["Opp"]

            # home_box_scores = row["HomeBoxScores"]
            # away_box_scores = row["AwayBoxScores"]
            
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
                home_static = np.array([
                    home_location,
                ])
                away_static = np.array([
                    away_location,
                ])

                # Store sequenced data for both teams
                seq_features = ["Result"]
                home_window = np.array([
                    [game[col] for col in seq_features] for game in team_games[home][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                away_window = np.array([
                    [game[col] for col in seq_features] for game in team_games[away][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)

                # Write datapoint for both sides
                self.X_train_list_static.append(home_static)
                self.X_train_list_seq_team.append(home_window)
                self.X_train_list_seq_opp.append(away_window)
                self.y_train_list.append(row["Result"])
                self.X_train_list_static.append(away_static)
                self.X_train_list_seq_team.append(away_window)
                self.X_train_list_seq_opp.append(home_window)
                self.y_train_list.append(-row["Result"])
            
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


        # window and write TEST data
        for _, row in df_test_unwindowed_normalized.iterrows():
            home = row["Home"]
            away = row["Opp"]

            # home_box_scores = row["HomeBoxScores"]
            # away_box_scores = row["AwayBoxScores"]
            
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
                home_static = np.array([
                    home_location,
                ])
                away_static = np.array([
                    away_location,
                ])

                # Store sequenced data for both teams
                seq_features = ["Result"]
                home_window = np.array([
                    [game[col] for col in seq_features] for game in team_games[home][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                away_window = np.array([
                    [game[col] for col in seq_features] for game in team_games[away][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)

                # Write datapoint for both sides
                self.X_test_list_static.append(home_static)
                self.X_test_list_seq_team.append(home_window)
                self.X_test_list_seq_opp.append(away_window)
                self.y_test_list.append(row["Result"])
                self.X_test_list_static.append(away_static)
                self.X_test_list_seq_team.append(away_window)
                self.X_test_list_seq_opp.append(home_window)
                self.y_test_list.append(-row["Result"])
            
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


    def to_tensor(self):
        # convert list to numpy array as intermediary
        X_train_static = np.stack(self.X_train_list_static)
        X_train_seq_team = np.stack(self.X_train_list_seq_team)
        X_train_seq_opp = np.stack(self.X_train_list_seq_opp)
        y_train = np.stack(self.y_train_list)
        X_test_static = np.stack(self.X_test_list_static)
        X_test_seq_team = np.stack(self.X_test_list_seq_team)
        X_test_seq_opp = np.stack(self.X_test_list_seq_opp)
        y_test = np.stack(self.y_test_list)

        # convert to tensors
        X_train_static_tensor = torch.tensor(X_train_static, dtype=torch.float32)
        X_train_seq_team_tensor = torch.tensor(X_train_seq_team, dtype=torch.float32)
        X_train_seq_opp_tensor = torch.tensor(X_train_seq_opp, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_static_tensor = torch.tensor(X_test_static, dtype=torch.float32)
        X_test_seq_team_tensor = torch.tensor(X_test_seq_team, dtype=torch.float32)
        X_test_seq_opp_tensor = torch.tensor(X_test_seq_opp, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

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
        wd.add_games(df_games, train_ratio=0.8)
        end_year = time.time()
        # print(f"Finished window process for year {year} | Total time: {end_year - start_year:.2f}s\n")
        

    # Train and test
    # For GRU split into home and away sequences
    X_train_static_tensor, X_train_seq_team_tensor, X_train_seq_opp_tensor, y_train_tensor, \
    X_test_static_tensor, X_test_seq_team_tensor, X_test_seq_opp_tensor, y_test_tensor = wd.to_tensor()


    # TEST OUTPUT PRINT
    # print(torch.isnan(X_train_static_tensor).any())
    # print(torch.isnan(X_test_static_tensor).any())

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
    # sanity_check_tensor("X_train_seq_team_tensor", X_train_seq_team_tensor)
    # sanity_check_tensor("X_train_seq_opp_tensor", X_train_seq_opp_tensor)
    sanity_check_tensor("y_train_tensor", y_train_tensor)

    print("=== Sanity Check: Test Tensors ===")
    sanity_check_tensor("X_test_static_tensor", X_test_static_tensor)
    # sanity_check_tensor("X_test_seq_team_tensor", X_test_seq_team_tensor)
    # sanity_check_tensor("X_test_seq_opp_tensor", X_test_seq_opp_tensor)
    sanity_check_tensor("y_test_tensor", y_test_tensor)


    # train_dataset = TensorDataset(
    #     X_train_static_tensor,
    #     X_train_seq_team_tensor,
    #     X_train_seq_opp_tensor,
    #     y_train_tensor
    # )
    # test_dataset = TensorDataset(
    #     X_test_static_tensor,
    #     X_test_seq_team_tensor,
    #     X_test_seq_opp_tensor,
    #     y_test_tensor
    # )
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # # Init network
    # EPOCHS = 50
    # LR = 0.001
    # static_input_size = X_train_static_tensor.shape[1]
    # seq_input_size = X_train_seq_team_tensor.shape[2]
    # model = HybridNN(
    #     fnn_input_size=static_input_size, fnn_hidden_size=1, 
    #     gru_input_size=seq_input_size, gru_hidden_size=32
    # )
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # # Training loop
    # for epoch in range(EPOCHS):
    #     model.train()
    #     total_loss = 0
    #     for static_batch, team_seq_batch, opp_seq_batch, y_batch in train_loader:
    #         # reset grads and forward pass
    #         optimizer.zero_grad()
    #         outputs = model(static_batch, team_seq_batch, opp_seq_batch)
    #         # compute MSE as criterion and backprop
    #         loss = criterion(outputs, y_batch)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item() * team_seq_batch.size(0)

    #     avg_loss = total_loss / len(train_loader.dataset)
    #     if epoch % 10 == 0 or epoch == EPOCHS-1:
    #         print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # # Evaluation
    # model.eval()
    # with torch.no_grad():
    #     y_pred = []
    #     y_true = []
    #     for static_batch, team_seq_batch, opp_seq_batch, y_batch in test_loader:
    #         preds = model(static_batch, team_seq_batch, opp_seq_batch)
    #         y_pred.extend(preds.tolist())
    #         y_true.extend(y_batch.tolist())

    #     # Compute MSE
    #     y_pred_tensor = torch.tensor(y_pred)
    #     y_true_tensor = torch.tensor(y_true)
    #     mse = nn.functional.mse_loss(y_pred_tensor, y_true_tensor)
    #     print(f"MSE: {mse:.4f}")


    # Test standalone FNN
    data_dir2 = Path(__file__).parent / "../data"
    input_path_parquet2 = data_dir2 / f"rolling/parquet/gamesRolling.parquet"
    df = pd.read_parquet(input_path_parquet2)
    X = df.drop(columns=["Result"])
    y = df["Result"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # train_ratio = 0.8
    # train_thresh = int(len(df)*train_ratio)
    # X_train, X_test = X.head(train_thresh), X.tail(len(df)-train_thresh)
    # y_train, y_test = y.head(train_thresh), y.tail(len(df)-train_thresh)


    # SANITY CHECK for y
    y_all = pd.concat([y_train, y_test])
    mean_y = y_all.mean()
    std_y = y_all.std()
    print(f"\n\n\n\n\n\n\nMean of y: {mean_y:.4f}")
    print(f"Std of y: {std_y:.4f}")

    home_train, home_test = X_train["Home"], X_test["Home"]
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = normalize(X_train, X_test, y_train, y_test, home_train, home_test)
    
    # column truncate
    cols = ["Home",]
    X_train_scaled = X_train_scaled.loc[:, cols]
    X_test_scaled = X_test_scaled.loc[:, cols]
    
    # convert
    X_train_static_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled.values, dtype=torch.float32)
    X_test_static_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled.values, dtype=torch.float32)


    print("\n\n\n\n=== Sanity Check 2: Train Tensors ===")
    sanity_check_tensor("X_train_static_tensor", X_train_static_tensor)
    # sanity_check_tensor("X_train_seq_team_tensor", X_train_seq_team_tensor)
    # sanity_check_tensor("X_train_seq_opp_tensor", X_train_seq_opp_tensor)
    sanity_check_tensor("y_train_tensor", y_train_tensor)

    print("=== Sanity Check 2: Test Tensors ===")
    sanity_check_tensor("X_test_static_tensor", X_test_static_tensor)
    # sanity_check_tensor("X_test_seq_team_tensor", X_test_seq_team_tensor)
    # sanity_check_tensor("X_test_seq_opp_tensor", X_test_seq_opp_tensor)
    sanity_check_tensor("y_test_tensor", y_test_tensor)

    train_dataset = TensorDataset(
        X_train_static_tensor,
        y_train_tensor
    )
    test_dataset = TensorDataset(
        X_test_static_tensor,
        y_test_tensor
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    EPOCHS = 30
    LR = 0.001
    static_input_size = X_train_static_tensor.shape[1]
    model = FNN(input_size=static_input_size, hidden_size=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            # reset grads and forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            # compute MSE as criterion and backprop
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 10 == 0 or epoch == EPOCHS-1:
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_tensor = []
        y_true_tensor = []
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            y_pred_tensor.append(preds)
            y_true_tensor.append(y_batch)

        # Concatenate along batch dimension
        y_pred_tensor = torch.cat(y_pred_tensor)
        y_true_tensor = torch.cat(y_true_tensor)

        print("y_true_tensor mean: " + str(float(torch.mean(y_true_tensor))))
        print("y_true_tensor std: " + str(float(torch.std(y_true_tensor))))

        mse = nn.functional.mse_loss(y_pred_tensor, y_true_tensor)
        print(f"MSE: {mse:.4f}")

    
    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")



if __name__ == "__main__":
    main()