from pathlib import Path
import time
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from joblib import dump
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from team_strengths import TeamStrengthCalculator


# Paths
# NOTE: increasing data_threshold seems to hurt performance, look into it
DATA_THRESHOLD = 10
GAMES_WINDOW_SZ = 10
DATA_DIR = Path(__file__).parent / "../data"
SCALERS_DIR = Path("E:/NBAPredictor/ml/scalers")
MODELS_DIR = Path("E:/NBAPredictor/ml/models")
    
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
    def __init__(self, 
                 static_input_size, static_hidden_size, 
                 metric_input_size, metric_hidden_size,
                 starter_input_size, starter_hidden_size, 
                 gru_input_size, gru_hidden_size):
        super().__init__()
        self.static = nn.Linear(static_input_size, static_hidden_size)
        self.metric = nn.Linear(metric_input_size, metric_hidden_size)
        self.starter = nn.Linear(starter_input_size, starter_hidden_size)
        self.gru_team = GRU(gru_input_size, gru_hidden_size)
        self.gru_opp = GRU(gru_input_size, gru_hidden_size)
        self.fc_top = nn.Linear(2*gru_hidden_size+metric_hidden_size+static_hidden_size+starter_hidden_size, 1)

    def forward(self, static_input, metric_input, starter_input, team_seq, opp_seq):
        hidden_static = self.static(static_input)
        hidden_metric = self.metric(metric_input)
        hidden_starter = self.starter(starter_input)    
        hidden_team = self.gru_team(team_seq)
        hidden_opp = self.gru_opp(opp_seq)

        
        # feed into final top fc layer
        combined = torch.cat([hidden_static, hidden_metric, hidden_starter, hidden_team, hidden_opp], dim=1)  
        out = self.fc_top(combined)

        # DONT squeeze for now since y_batch is (batch, 1) 
        # # (batch,1) --> (batch,)
        # # return out.squeeze(1)
        return out

    
class WindowedDataset:
    # Clean this up!
    def __init__(self):
        self.X_train_static = np.array([])
        self.X_train_metric = np.array([])
        self.X_train_starter = np.array([])
        self.X_train_seq_team = np.array([])
        self.X_train_seq_opp = np.array([])
        self.y_train = np.array([])
        self.X_test_static = np.array([])
        self.X_test_metric = np.array([])
        self.X_test_starter = np.array([])
        self.X_test_seq_team = np.array([])
        self.X_test_seq_opp = np.array([])
        self.y_test = np.array([])

        self.X_list_train_static = []
        self.X_list_train_metric = []
        self.X_list_train_starter = []
        self.X_list_train_seq_team = []
        self.X_list_train_seq_opp = []
        self.y_list_train = []
        self.X_list_test_static = []
        self.X_list_test_metric = []
        self.X_list_test_starter = []
        self.X_list_test_seq_team = []
        self.X_list_test_seq_opp = []
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

        # Datapoints
        X_list_static = []
        X_list_metric = []
        X_list_starter = []
        X_list_seq_team = []
        X_list_seq_opp = []
        y_list = []

        # window data
        games_history = {}
        # static data
        team_stats = {}
        player_stats = {}
        # computed data
        team_strengths = {team : 0 for team in df_games["Home"].unique()}
        home_adv = 1.5
        rest_effect = 1.5
        tsc = TeamStrengthCalculator(df_games)
        for _, row in df_games.iterrows():
            home = row["Home"]
            away = row["Opp"]
            home_box_scores = row["HomeBoxScores"]
            away_box_scores = row["AwayBoxScores"]
            result_home = row["Result"]
            result_away = -row["Result"]
            
            # init STATIC data
            if not (home in team_stats):
                team_stats[home] = {}               
                team_stats[home]["PTSDiffTot"] = 0
                team_stats[home]["Played"] = 0
                team_stats[home]["W"] = 0
                team_stats[home]["L"] = 0
                team_stats[home]["BenchAvgBPM"] = 0
                team_stats[home]["BenchTotBPM"] = 0
                team_stats[home]["BenchMinutes"] = 0
            if not (away in team_stats):
                team_stats[away] = {}
                team_stats[away]["PTSDiffTot"] = 0
                team_stats[away]["Played"] = 0
                team_stats[away]["W"] = 0
                team_stats[away]["L"] = 0
                team_stats[away]["BenchAvgBPM"] = 0
                team_stats[away]["BenchTotBPM"] = 0
                team_stats[away]["BenchMinutes"] = 0

            # Init sequence
            if home not in games_history:
                games_history[home] = []
            if away not in games_history:
                games_history[away] = []

            # Backfill team strengths
            if row["Date"] > tsc.get_next_date():
                tsc.process_day()
                team_strengths, home_adv, rest_effect = tsc.calc_team_strengths()

            # Windowed datapoint
            has_rest_data = not (pd.isna(row["RestHome"]) or pd.isna(row["RestOpp"]))
            has_bet_data = not pd.isna(row["PregameSpread"])
            if len(games_history[home]) >= GAMES_WINDOW_SZ \
                and len(games_history[away]) >= GAMES_WINDOW_SZ \
                and has_rest_data \
                and has_bet_data \
                and team_stats[home]["Played"] >= DATA_THRESHOLD:

                # Static data for both teams
                home_location = 1
                away_location = -1
                home_pts_diff = team_stats[home]["PTSDiffTot"]/team_stats[home]["Played"]
                away_pts_diff = team_stats[away]["PTSDiffTot"]/team_stats[away]["Played"]
                home_strength = team_strengths[home]
                away_strength = team_strengths[away]
                home_bench_bpm_pg = team_stats[home]["BenchTotBPM"]/team_stats[home]["Played"]
                away_bench_bpm_pg = team_stats[away]["BenchTotBPM"]/team_stats[away]["Played"]
                home_bench_min_pg = team_stats[home]["BenchMinutes"]/team_stats[home]["Played"]
                away_bench_min_pg = team_stats[away]["BenchMinutes"]/team_stats[away]["Played"]
                home_bench_avgbpm_pg = team_stats[home]["BenchAvgBPM"]/team_stats[home]["Played"]
                away_bench_avgbpm_pg = team_stats[away]["BenchAvgBPM"]/team_stats[away]["Played"]
                home_rest = row["RestHome"]
                away_rest = row["RestOpp"]
                home_spread_ks = float(row["PregameSpread"])
                away_spread_ks = -float(row["PregameSpread"])

                # Grouped features (NOTE home must go at end to be unscaled)
                home_static_features = [
                    home_rest, away_rest, home_location
                ]
                away_static_features = [
                    away_rest, home_rest, away_location
                ]
                home_metric_features = [
                    # home_pts_diff, away_pts_diff,
                    home_strength, away_strength,
                    home_bench_avgbpm_pg, away_bench_avgbpm_pg,
                    home_bench_min_pg, away_bench_min_pg,
                    # home_bench_bpm_pg, away_bench_bpm_pg
                    # home_spread_ks
                ]
                away_metric_features = [
                    # away_pts_diff, home_pts_diff,
                    away_strength, home_strength,
                    away_bench_avgbpm_pg, home_bench_avgbpm_pg,
                    away_bench_min_pg, home_bench_min_pg,
                    # away_bench_bpm_pg, home_bench_bpm_pg
                    # away_spread_ks
                ]
                # Assume 5 highest minute players are the starters
                # Write data for both sides
                # Make sure the player orders are correct for each datapoint!
                home_starter_features = []
                away_starter_features = []
                for i in range(5):
                    player_team = home_box_scores[i]
                    player_opp = away_box_scores[i]
                    pid_team = player_team["Player-additional"]
                    pid_opp = player_opp["Player-additional"]
                    if pid_team in player_stats:
                        home_starter_features.append(player_stats[pid_team]["TotBPM"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotPTS"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotREB"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotAST"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotBLK"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotSTL"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotTOV"]/player_stats[pid_team]["GamesPlayed"])
                    else:
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                    if pid_opp in player_stats:
                        away_starter_features.append(player_stats[pid_opp]["TotBPM"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotPTS"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotREB"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotAST"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotBLK"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotSTL"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotTOV"]/player_stats[pid_opp]["GamesPlayed"])
                    else:
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)

                for i in range(5):
                    player_team = away_box_scores[i]
                    player_opp = home_box_scores[i]
                    pid_team = player_team["Player-additional"]
                    pid_opp = player_opp["Player-additional"]
                    if pid_team in player_stats:
                        home_starter_features.append(player_stats[pid_team]["TotBPM"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotPTS"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotREB"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotAST"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotBLK"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotSTL"]/player_stats[pid_team]["GamesPlayed"])
                        home_starter_features.append(player_stats[pid_team]["TotTOV"]/player_stats[pid_team]["GamesPlayed"])
                    else:
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                        home_starter_features.append(0)
                    if pid_opp in player_stats:
                        away_starter_features.append(player_stats[pid_opp]["TotBPM"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotPTS"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotREB"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotAST"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotBLK"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotSTL"]/player_stats[pid_opp]["GamesPlayed"])
                        away_starter_features.append(player_stats[pid_opp]["TotTOV"]/player_stats[pid_opp]["GamesPlayed"])
                    else:
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)
                        away_starter_features.append(0)

                home_static = np.array(home_static_features)
                away_static = np.array(away_static_features)
                home_metric = np.array(home_metric_features)
                away_metric = np.array(away_metric_features)
                home_starter = np.array(home_starter_features)
                away_starter = np.array(away_starter_features)

                # Store sequenced data for both teams
                seq_features = ["Result", "PregameStrength", "OppPregameStrength", "Rest", "OppRest", "Home"]
                home_window = np.array([
                    [game[col] for col in seq_features] for game in games_history[home][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                away_window = np.array([
                    [game[col] for col in seq_features] for game in games_history[away][-GAMES_WINDOW_SZ:]
                ], dtype=np.float32)
                # Write datapoint for both sides
                X_list_static.append(home_static)
                X_list_metric.append(home_metric)
                X_list_starter.append(home_starter)
                X_list_seq_team.append(home_window)
                X_list_seq_opp.append(away_window)
                y_list.append(result_home)
                X_list_static.append(away_static)
                X_list_metric.append(away_metric)
                X_list_starter.append(away_starter)
                X_list_seq_team.append(away_window)
                X_list_seq_opp.append(home_window)
                y_list.append(result_away)
            
            # UPDATE
            # Store only subset of stats
            # Note that rest may be NaN, use 1 as a replacement for now
            if not has_rest_data:
                row["RestHome"] = 1
                row["RestOpp"] = 1
            home_data = {
                "Result": result_home,
                "PregameStrength": team_strengths[home],
                "OppPregameStrength": team_strengths[away],
                "Home": 1,
                "Rest": row["RestHome"],
                "OppRest": row["RestOpp"],
            }
            away_data = {
                "Result": result_away,
                "PregameStrength": team_strengths[away],
                "OppPregameStrength": team_strengths[home],
                "Home": -1,
                "Rest": row["RestHome"],
                "OppRest": row["RestOpp"],
            }
            games_history[home].append(home_data)
            games_history[away].append(away_data)


            # Update static accumulated
            team_stats[home]["PTSDiffTot"] += result_home
            team_stats[home]["Played"] += 1
            team_stats[home]["W"] += (1 if result_home > 0 else 0)
            team_stats[home]["L"] += (1 if result_home< 0 else 0)

            team_stats[away]["PTSDiffTot"] += result_away
            team_stats[away]["Played"] += 1
            team_stats[away]["W"] += (1 if result_away > 0 else 0)
            team_stats[away]["L"] += (1 if result_away < 0 else 0)

            # Box score stats
            home_bench_bpm_tot = 0
            home_bench_players = 0
            home_bench_minutes = 0
            for i in range(len(home_box_scores)):
                player = home_box_scores[i]
                pid = player["Player-additional"]
                if not (pid in player_stats):
                    player_stats[pid] = {}
                    player_stats[pid]["GamesPlayed"] = 0
                    player_stats[pid]["TotBPM"] = 0
                    player_stats[pid]["TotPTS"] = 0
                    player_stats[pid]["TotAST"] = 0
                    player_stats[pid]["TotREB"] = 0
                    player_stats[pid]["TotOREB"] = 0
                    player_stats[pid]["TotSTL"] = 0
                    player_stats[pid]["TotBLK"] = 0
                    player_stats[pid]["TotTOV"] = 0


                player_stats[pid]["GamesPlayed"] += 1
                player_stats[pid]["TotBPM"] += player["BPM"]
                player_stats[pid]["TotPTS"] += player["PTS"]
                player_stats[pid]["TotAST"] += player["AST"]
                player_stats[pid]["TotREB"] += player["TRB"]
                player_stats[pid]["TotOREB"] += player["ORB"]
                player_stats[pid]["TotSTL"] += player["STL"]
                player_stats[pid]["TotBLK"] += player["BLK"]
                player_stats[pid]["TotTOV"] += player["TOV"]
                if i >= 5:
                    # Bench player 
                    # TBD add this stat to rolling if it makes an impact!
                    home_bench_bpm_tot += player["BPM"]
                    home_bench_players += 1
                    home_bench_minutes += player["MP"]
            team_stats[home]["BenchAvgBPM"] += home_bench_bpm_tot/home_bench_players
            team_stats[home]["BenchTotBPM"] += home_bench_bpm_tot
            team_stats[home]["BenchMinutes"] += home_bench_minutes

            away_bench_bpm_tot = 0
            away_bench_players = 0
            away_bench_minutes = 0
            for i in range(len(away_box_scores)):
                player = away_box_scores[i]
                pid = player["Player-additional"]
                if not (pid in player_stats):
                    player_stats[pid] = {}
                    player_stats[pid]["GamesPlayed"] = 0
                    player_stats[pid]["TotBPM"] = 0
                    player_stats[pid]["TotPTS"] = 0
                    player_stats[pid]["TotAST"] = 0
                    player_stats[pid]["TotREB"] = 0
                    player_stats[pid]["TotOREB"] = 0
                    player_stats[pid]["TotSTL"] = 0
                    player_stats[pid]["TotBLK"] = 0
                    player_stats[pid]["TotTOV"] = 0

                
                player_stats[pid]["GamesPlayed"] += 1
                player_stats[pid]["TotBPM"] += player["BPM"]
                player_stats[pid]["TotPTS"] += player["PTS"]
                player_stats[pid]["TotAST"] += player["AST"]
                player_stats[pid]["TotREB"] += player["TRB"]
                player_stats[pid]["TotOREB"] += player["ORB"]
                player_stats[pid]["TotSTL"] += player["STL"]
                player_stats[pid]["TotBLK"] += player["BLK"]
                player_stats[pid]["TotTOV"] += player["TOV"]
                if i >= 5:
                    # Bench player
                    # TBD add this stat to rolling if it makes an impact!
                    away_bench_bpm_tot += player["BPM"]
                    away_bench_players += 1
                    away_bench_minutes += player["MP"]
            team_stats[away]["BenchAvgBPM"] += away_bench_bpm_tot/away_bench_players
            team_stats[away]["BenchTotBPM"] += away_bench_bpm_tot
            team_stats[away]["BenchMinutes"] += away_bench_minutes

        # Shuffled split 
        (
            X_list_train_static,
            X_list_test_static,
            X_list_train_metric,
            X_list_test_metric,
            X_list_train_starter,
            X_list_test_starter,
            X_list_train_seq_team,
            X_list_test_seq_team,
            X_list_train_seq_opp,
            X_list_test_seq_opp,
            y_list_train,
            y_list_test
        ) = train_test_split(
            X_list_static,
            X_list_metric,
            X_list_starter,
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
        self.X_list_train_metric.extend(X_list_train_metric)
        self.X_list_test_metric.extend(X_list_test_metric)
        self.X_list_train_starter.extend(X_list_train_starter)
        self.X_list_test_starter.extend(X_list_test_starter)
        self.X_list_train_seq_team.extend(X_list_train_seq_team)
        self.X_list_test_seq_team.extend(X_list_test_seq_team)
        self.X_list_train_seq_opp.extend(X_list_train_seq_opp)
        self.X_list_test_seq_opp.extend(X_list_test_seq_opp)
        self.y_list_train.extend(y_list_train)
        self.y_list_test.extend(y_list_test)

    def to_tensor(self):
        # Make sure to ignore the Home column (last col) when scaling static
        scaler_X_static = StandardScaler()
        X_train_static_tbd = np.stack(self.X_list_train_static, axis=0)
        X_test_static_tbd = np.stack(self.X_list_test_static, axis=0)
        X_train_static_scaled = scaler_X_static.fit_transform(X_train_static_tbd[:, :-1])
        X_test_static_scaled = scaler_X_static.transform(X_test_static_tbd[:, :-1])

        # Add home back
        X_train_static_scaled = np.concatenate([X_train_static_scaled, X_train_static_tbd[:, -1:]], axis=1)
        X_test_static_scaled = np.concatenate([X_test_static_scaled, X_test_static_tbd[:, -1:]], axis=1)

        # Scale X metric normally
        scaler_X_metric = StandardScaler()
        X_train_metric_scaled = scaler_X_metric.fit_transform(np.stack(self.X_list_train_metric, axis=0))
        X_test_metric_scaled = scaler_X_metric.transform(np.stack(self.X_list_test_metric, axis=0))

        scaler_X_starter = StandardScaler()
        X_train_starter_scaled = scaler_X_starter.fit_transform(np.stack(self.X_list_train_starter, axis=0))
        X_test_starter_scaled = scaler_X_starter.transform(np.stack(self.X_list_test_starter, axis=0))

        # Stack, Flatten, scale, unflatten for seq data
        scaler_X_seq_team = StandardScaler()
        scaler_X_seq_opp = StandardScaler()
        X_train_seq_team_tbd = np.stack(self.X_list_train_seq_team, axis=0)
        X_test_seq_team_tbd = np.stack(self.X_list_test_seq_team, axis=0)
        X_train_seq_opp_tbd = np.stack(self.X_list_train_seq_opp, axis=0)
        X_test_seq_opp_tbd = np.stack(self.X_list_test_seq_opp, axis=0)

        N_train, seq_len, num_feat = X_train_seq_team_tbd.shape
        N_test, seq_len, num_feat = X_test_seq_team_tbd.shape

        X_train_seq_team_tbd = X_train_seq_team_tbd.reshape(N_train*seq_len, num_feat)
        X_test_seq_team_tbd = X_test_seq_team_tbd.reshape(N_test*seq_len, num_feat)
        X_train_seq_opp_tbd = X_train_seq_opp_tbd.reshape(N_train*seq_len, num_feat)
        X_test_seq_opp_tbd = X_test_seq_opp_tbd.reshape(N_test*seq_len, num_feat)

        # Make sure to ignore the Home column (last col) when scaling        
        X_train_seq_team_scaled = scaler_X_seq_team.fit_transform(X_train_seq_team_tbd[:, :-1])
        X_test_seq_team_scaled = scaler_X_seq_team.transform(X_test_seq_team_tbd[:, :-1])
        X_train_seq_opp_scaled = scaler_X_seq_opp.fit_transform(X_train_seq_opp_tbd[:, :-1])
        X_test_seq_opp_scaled = scaler_X_seq_opp.transform(X_test_seq_opp_tbd[:, :-1])
        # Add home back
        X_train_seq_team_scaled = np.concatenate([X_train_seq_team_scaled, X_train_seq_team_tbd[:, -1:]], axis=1)
        X_test_seq_team_scaled = np.concatenate([X_test_seq_team_scaled, X_test_seq_team_tbd[:, -1:]], axis=1)
        X_train_seq_opp_scaled = np.concatenate([X_train_seq_opp_scaled, X_train_seq_opp_tbd[:, -1:]], axis=1)
        X_test_seq_opp_scaled = np.concatenate([X_test_seq_opp_scaled, X_test_seq_opp_tbd[:, -1:]], axis=1)

        X_train_seq_team_scaled = X_train_seq_team_scaled.reshape(N_train, seq_len, num_feat)
        X_test_seq_team_scaled = X_test_seq_team_scaled.reshape(N_test, seq_len, num_feat)
        X_train_seq_opp_scaled = X_train_seq_opp_scaled.reshape(N_train, seq_len, num_feat)
        X_test_seq_opp_scaled = X_test_seq_opp_scaled.reshape(N_test, seq_len, num_feat)

        # Scale y normally
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(np.array(self.y_list_train).reshape(-1, 1))
        y_test_scaled = scaler_y.transform(np.array(self.y_list_test).reshape(-1, 1))

        # concat with rest of data
        self.X_train_static    = self._np_concat(self.X_train_static, X_train_static_scaled)
        self.X_train_metric    = self._np_concat(self.X_train_metric, X_train_metric_scaled)
        self.X_train_starter   = self._np_concat(self.X_train_starter, X_train_starter_scaled)
        self.X_train_seq_team  = self._np_concat(self.X_train_seq_team, X_train_seq_team_scaled)
        self.X_train_seq_opp   = self._np_concat(self.X_train_seq_opp, X_train_seq_opp_scaled)
        self.y_train           = self._np_concat(self.y_train, y_train_scaled)
        self.X_test_static     = self._np_concat(self.X_test_static, X_test_static_scaled)
        self.X_test_metric     = self._np_concat(self.X_test_metric, X_test_metric_scaled)
        self.X_test_starter    = self._np_concat(self.X_test_starter, X_test_starter_scaled)
        self.X_test_seq_team   = self._np_concat(self.X_test_seq_team, X_test_seq_team_scaled)
        self.X_test_seq_opp    = self._np_concat(self.X_test_seq_opp, X_test_seq_opp_scaled)
        self.y_test            = self._np_concat(self.y_test, y_test_scaled)

        # convert to tensors
        X_train_static_tensor = torch.tensor(self.X_train_static, dtype=torch.float32)
        X_train_metric_tensor = torch.tensor(self.X_train_metric, dtype=torch.float32)
        X_train_starter_tensor = torch.tensor(self.X_train_starter, dtype=torch.float32)
        X_train_seq_team_tensor = torch.tensor(self.X_train_seq_team, dtype=torch.float32)
        X_train_seq_opp_tensor = torch.tensor(self.X_train_seq_opp, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        X_test_static_tensor = torch.tensor(self.X_test_static, dtype=torch.float32)
        X_test_metric_tensor = torch.tensor(self.X_test_metric, dtype=torch.float32)
        X_test_starter_tensor = torch.tensor(self.X_test_starter, dtype=torch.float32)
        X_test_seq_team_tensor = torch.tensor(self.X_test_seq_team, dtype=torch.float32)
        X_test_seq_opp_tensor = torch.tensor(self.X_test_seq_opp, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

        # Store scalers
        dump(scaler_X_static, SCALERS_DIR / "scaler_X_static.joblib")
        dump(scaler_X_metric, SCALERS_DIR / "scaler_X_metric.joblib")
        dump(scaler_X_starter, SCALERS_DIR / "scaler_X_starter.joblib")
        dump(scaler_X_seq_team, SCALERS_DIR / "scaler_X_seq_team.joblib")
        dump(scaler_X_seq_opp, SCALERS_DIR / "scaler_X_seq_opp.joblib")
        dump(scaler_y, SCALERS_DIR / "scaler_y.joblib")

        return \
        X_train_static_tensor, X_train_metric_tensor, X_train_starter_tensor, \
        X_train_seq_team_tensor, X_train_seq_opp_tensor, y_train_tensor, \
        X_test_static_tensor, X_test_metric_tensor, X_test_starter_tensor, \
        X_test_seq_team_tensor, X_test_seq_opp_tensor, y_test_tensor


        
def main():
    years = [year for year in range(2015, 2026)]
    total_start = time.time()

    # Init windowed dataset
    wd = WindowedDataset()

    # Add data per year
    for year in years:
        start_year = time.time()
        joined_games_path = DATA_DIR / f"joined/parquet/{year}/gamesJoined.parquet"
        df_games = pd.read_parquet(joined_games_path)
        wd.add_games(df_games)
        end_year = time.time()
        print(f"Finished window process for year {year} | Total time: {end_year - start_year:.2f}s\n")
        

    # Train and test
    # For GRU split into home and away sequences
    X_train_static_tensor, X_train_metric_tensor, X_train_starter_tensor, \
    X_train_seq_team_tensor, X_train_seq_opp_tensor, y_train_tensor, \
    X_test_static_tensor, X_test_metric_tensor, X_test_starter_tensor, \
    X_test_seq_team_tensor, X_test_seq_opp_tensor, y_test_tensor = wd.to_tensor()


    # # TEST OUTPUT PRINT
    # print(torch.isnan(X_train_static_tensor).any())
    # print(torch.isnan(X_test_static_tensor).any())

    # def sanity_check_tensor(name, tensor):
    #     print(f"\n{name}:")
    #     print(f"  shape: {tensor.shape}")
    #     # print(f"  dtype: {tensor.dtype}")
    #     # try:
    #     #     print(f"  min:   {tensor.min().item():.4f}")
    #     #     print(f"  max:   {tensor.max().item():.4f}")
    #     # except:
    #     #     print("  (min/max not available)")
    #     # print(f"  sample: {tensor[0]}\n")


    # print("=== Sanity Check: Train Tensors ===")
    # sanity_check_tensor("X_train_static_tensor", X_train_static_tensor)
    # sanity_check_tensor("X_train_seq_team_tensor", X_train_seq_team_tensor)
    # sanity_check_tensor("X_train_seq_opp_tensor", X_train_seq_opp_tensor)
    # sanity_check_tensor("y_train_tensor", y_train_tensor)

    # print("=== Sanity Check: Test Tensors ===")
    # sanity_check_tensor("X_test_static_tensor", X_test_static_tensor)
    # sanity_check_tensor("X_test_seq_team_tensor", X_test_seq_team_tensor)
    # sanity_check_tensor("X_test_seq_opp_tensor", X_test_seq_opp_tensor)
    # sanity_check_tensor("y_test_tensor", y_test_tensor)


    train_dataset = TensorDataset(
        X_train_static_tensor,
        X_train_metric_tensor,
        X_train_starter_tensor,
        X_train_seq_team_tensor,
        X_train_seq_opp_tensor,
        y_train_tensor
    )
    test_dataset = TensorDataset(
        X_test_static_tensor,
        X_test_metric_tensor,
        X_test_starter_tensor,
        X_test_seq_team_tensor,
        X_test_seq_opp_tensor,
        y_test_tensor
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Init network
    EPOCHS = 30
    LR = 0.001
    static_input_size = X_train_static_tensor.shape[1]
    metric_input_size = X_train_metric_tensor.shape[1]
    starter_input_size = X_train_starter_tensor.shape[1]
    seq_input_size = X_train_seq_team_tensor.shape[2]
    model = HybridNN(
        static_input_size=static_input_size, static_hidden_size=1, 
        metric_input_size=metric_input_size, metric_hidden_size=1,
        starter_input_size=starter_input_size, starter_hidden_size=1, 
        gru_input_size=seq_input_size, gru_hidden_size=1
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for static_batch, metric_batch, starter_batch, team_seq_batch, opp_seq_batch, y_batch in train_loader:
            # reset grads and forward pass
            optimizer.zero_grad()
            outputs = model(static_batch, metric_batch, starter_batch, team_seq_batch, opp_seq_batch)
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

        for static_batch, metric_batch, starter_batch, team_seq_batch, opp_seq_batch, y_batch in test_loader:
            preds = model(static_batch, metric_batch, starter_batch, team_seq_batch, opp_seq_batch)
            preds_list = preds.tolist()
            true_list = y_batch.tolist()
            y_pred.extend(preds_list)
            y_true.extend(true_list)

        # Compute MSE
        y_pred_tensor = torch.tensor(y_pred)
        y_true_tensor = torch.tensor(y_true)
        mse = nn.functional.mse_loss(y_pred_tensor, y_true_tensor)
        print(f"MSE: {mse:.4f}")

    # Save model
    torch.save(model, MODELS_DIR / "test_full.pt")
    # torch.save(model.state_dict(), MODELS_DIR / "ensemble_linear_1.pt")

    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")



if __name__ == "__main__":
    main()