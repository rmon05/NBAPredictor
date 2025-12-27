import torch
from datetime import date
import pandas as pd
import numpy as np
from team_strengths import TeamStrengthCalculator
from pathlib import Path
from joblib import load
from unidecode import unidecode
import json
import psycopg2
from rnn import HybridNN, GRU
import os
from dotenv import load_dotenv

load_dotenv()
SEASON = 2026
DATE = date.today()
SCALERS_DIR = Path("E:/NBAPredictor/ml/scalers")
MODELS_DIR = Path("E:/NBAPredictor/ml/models")
DATA_DIR = Path("E:/NBAPredictor/data")
RAW_DIR = DATA_DIR / "nba_live_extracted"
BOX_SCORE_FILE = DATA_DIR / f"clean/parquet/{SEASON}/boxScoreClean.parquet"
UNAVAILABLE_PID = "pid_not_available"
GAMES_WINDOW_SZ = 10
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

class PredictionInputGenerator:
    def __init__(self):
        # window data
        self.games_history = {}
        # static data
        self.team_stats = {}
        self.player_stats = {}
        # computed data
        self.team_strengths = {}

        # For prediction purposes
        self.games = []
        self.X_list_static = []
        self.X_list_metric = []
        self.X_list_starter = []
        self.X_list_seq_team = []
        self.X_list_seq_opp = []
    
    # Should take in a dataframe of the JOINED games for the current season
    def add_games(self, df_games):
        # computed data
        self.team_strengths = {team : 0 for team in df_games["Home"].unique()}

        # Convert box scores to list of maps
        df_games["HomeBoxScores"] = df_games["HomeBoxScores"].apply(json.loads)
        df_games["AwayBoxScores"] = df_games["AwayBoxScores"].apply(json.loads)

        # Compute
        tsc = TeamStrengthCalculator(df_games)
        for _, row in df_games.iterrows():
            home = row["Home"]
            away = row["Opp"]
            home_box_scores = row["HomeBoxScores"]
            away_box_scores = row["AwayBoxScores"]
            result_home = row["Result"]
            result_away = -row["Result"]
            
            # init STATIC data
            if not (home in self.team_stats):
                self.team_stats[home] = {}               
                self.team_stats[home]["PTSDiffTot"] = 0
                self.team_stats[home]["Played"] = 0
                self.team_stats[home]["W"] = 0
                self.team_stats[home]["L"] = 0
                self.team_stats[home]["BenchAvgBPM"] = 0
                self.team_stats[home]["BenchTotBPM"] = 0
                self.team_stats[home]["BenchMinutes"] = 0
            if not (away in self.team_stats):
                self.team_stats[away] = {}
                self.team_stats[away]["PTSDiffTot"] = 0
                self.team_stats[away]["Played"] = 0
                self.team_stats[away]["W"] = 0
                self.team_stats[away]["L"] = 0
                self.team_stats[away]["BenchAvgBPM"] = 0
                self.team_stats[away]["BenchTotBPM"] = 0
                self.team_stats[away]["BenchMinutes"] = 0

            # Init sequence
            if home not in self.games_history:
                self.games_history[home] = []
            if away not in self.games_history:
                self.games_history[away] = []

            # Backfill team strengths
            if row["Date"] > tsc.get_next_date():
                tsc.process_day()
                self.team_strengths, home_adv, rest_effect = tsc.calc_team_strengths()

            # UPDATE
            # Store only subset of stats
            # Note that rest may be NaN, use 1 as a replacement for now
            has_rest_data = not (pd.isna(row["RestHome"]) or pd.isna(row["RestOpp"]))
            if not has_rest_data:
                row["RestHome"] = 1
                row["RestOpp"] = 1
            home_data = {
                "Result": result_home,
                "PregameStrength": self.team_strengths[home],
                "OppPregameStrength": self.team_strengths[away],
                "Home": 1,
                "Rest": row["RestHome"],
                "OppRest": row["RestOpp"],
            }
            away_data = {
                "Result": result_away,
                "PregameStrength": self.team_strengths[away],
                "OppPregameStrength": self.team_strengths[home],
                "Home": -1,
                "Rest": row["RestHome"],
                "OppRest": row["RestOpp"],
            }
            self.games_history[home].append(home_data)
            self.games_history[away].append(away_data)


            # Update static accumulated
            self.team_stats[home]["PTSDiffTot"] += result_home
            self.team_stats[home]["Played"] += 1
            self.team_stats[home]["W"] += (1 if result_home > 0 else 0)
            self.team_stats[home]["L"] += (1 if result_home < 0 else 0)

            self.team_stats[away]["PTSDiffTot"] += result_away
            self.team_stats[away]["Played"] += 1
            self.team_stats[away]["W"] += (1 if result_away > 0 else 0)
            self.team_stats[away]["L"] += (1 if result_away < 0 else 0)

            # Box score stats
            home_bench_bpm_tot = 0
            home_bench_players = 0
            home_bench_minutes = 0
            for i in range(len(home_box_scores)):
                player = home_box_scores[i]
                pid = player["Player-additional"]
                if not (pid in self.player_stats):
                    self.player_stats[pid] = {}
                    self.player_stats[pid]["GamesPlayed"] = 0
                    self.player_stats[pid]["TotBPM"] = 0
                    self.player_stats[pid]["TotPTS"] = 0
                    self.player_stats[pid]["TotAST"] = 0
                    self.player_stats[pid]["TotREB"] = 0
                    self.player_stats[pid]["TotOREB"] = 0
                    self.player_stats[pid]["TotSTL"] = 0
                    self.player_stats[pid]["TotBLK"] = 0
                    self.player_stats[pid]["TotTOV"] = 0


                self.player_stats[pid]["GamesPlayed"] += 1
                self.player_stats[pid]["TotBPM"] += player["BPM"]
                self.player_stats[pid]["TotPTS"] += player["PTS"]
                self.player_stats[pid]["TotAST"] += player["AST"]
                self.player_stats[pid]["TotREB"] += player["TRB"]
                self.player_stats[pid]["TotOREB"] += player["ORB"]
                self.player_stats[pid]["TotSTL"] += player["STL"]
                self.player_stats[pid]["TotBLK"] += player["BLK"]
                self.player_stats[pid]["TotTOV"] += player["TOV"]
                if i >= 5:
                    # Bench player 
                    # TBD add this stat to rolling if it makes an impact!
                    home_bench_bpm_tot += player["BPM"]
                    home_bench_players += 1
                    home_bench_minutes += player["MP"]
            self.team_stats[home]["BenchAvgBPM"] += home_bench_bpm_tot/home_bench_players
            self.team_stats[home]["BenchTotBPM"] += home_bench_bpm_tot
            self.team_stats[home]["BenchMinutes"] += home_bench_minutes

            away_bench_bpm_tot = 0
            away_bench_players = 0
            away_bench_minutes = 0
            for i in range(len(away_box_scores)):
                player = away_box_scores[i]
                pid = player["Player-additional"]
                if not (pid in self.player_stats):
                    self.player_stats[pid] = {}
                    self.player_stats[pid]["GamesPlayed"] = 0
                    self.player_stats[pid]["TotBPM"] = 0
                    self.player_stats[pid]["TotPTS"] = 0
                    self.player_stats[pid]["TotAST"] = 0
                    self.player_stats[pid]["TotREB"] = 0
                    self.player_stats[pid]["TotOREB"] = 0
                    self.player_stats[pid]["TotSTL"] = 0
                    self.player_stats[pid]["TotBLK"] = 0
                    self.player_stats[pid]["TotTOV"] = 0

                
                self.player_stats[pid]["GamesPlayed"] += 1
                self.player_stats[pid]["TotBPM"] += player["BPM"]
                self.player_stats[pid]["TotPTS"] += player["PTS"]
                self.player_stats[pid]["TotAST"] += player["AST"]
                self.player_stats[pid]["TotREB"] += player["TRB"]
                self.player_stats[pid]["TotOREB"] += player["ORB"]
                self.player_stats[pid]["TotSTL"] += player["STL"]
                self.player_stats[pid]["TotBLK"] += player["BLK"]
                self.player_stats[pid]["TotTOV"] += player["TOV"]
                if i >= 5:
                    # Bench player
                    # TBD add this stat to rolling if it makes an impact!
                    away_bench_bpm_tot += player["BPM"]
                    away_bench_players += 1
                    away_bench_minutes += player["MP"]
            self.team_stats[away]["BenchAvgBPM"] += away_bench_bpm_tot/away_bench_players
            self.team_stats[away]["BenchTotBPM"] += away_bench_bpm_tot
            self.team_stats[away]["BenchMinutes"] += away_bench_minutes
    
    def add_prediction_input(
            self, 
            home, home_pids, home_rest, 
            away, away_pids, away_rest
        ):
        # get features
        home_location = 1
        # away_location = -1
        home_strength = self.team_strengths[home]
        away_strength = self.team_strengths[away]
        home_bench_min_pg = self.team_stats[home]["BenchMinutes"]/self.team_stats[home]["Played"]
        away_bench_min_pg = self.team_stats[away]["BenchMinutes"]/self.team_stats[away]["Played"]
        home_bench_avgbpm_pg = self.team_stats[home]["BenchAvgBPM"]/self.team_stats[home]["Played"]
        away_bench_avgbpm_pg = self.team_stats[away]["BenchAvgBPM"]/self.team_stats[away]["Played"]

        # group features
        home_static_features = [
            home_rest, away_rest, home_location
        ]
        # away_static_features = [
        #     away_rest, home_rest, away_location
        # ]
        home_metric_features = [
            home_strength, away_strength,
            home_bench_avgbpm_pg, away_bench_avgbpm_pg,
            home_bench_min_pg, away_bench_min_pg,
        ]
        # away_metric_features = [
        #     away_strength, home_strength,
        #     away_bench_avgbpm_pg, home_bench_avgbpm_pg,
        #     away_bench_min_pg, home_bench_min_pg,
        # ]
        # Assume 5 highest minute players are the starters
        # Write data for both sides TBD
        # Make sure the player orders are correct for each datapoint!
        home_starter_features = []
        away_starter_features = []
        for i in range(5):
            pid_team = home_pids[i]
            pid_opp = away_pids[i]
            if pid_team in self.player_stats:
                home_starter_features.append(self.player_stats[pid_team]["TotBPM"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotPTS"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotREB"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotAST"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotBLK"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotSTL"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotTOV"]/self.player_stats[pid_team]["GamesPlayed"])
            else:
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
            if pid_opp in self.player_stats:
                away_starter_features.append(self.player_stats[pid_opp]["TotBPM"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotPTS"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotREB"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotAST"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotBLK"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotSTL"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotTOV"]/self.player_stats[pid_opp]["GamesPlayed"])
            else:
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)

        for i in range(5):
            pid_team = away_pids[i]
            pid_opp = home_pids[i]
            if pid_team in self.player_stats:
                home_starter_features.append(self.player_stats[pid_team]["TotBPM"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotPTS"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotREB"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotAST"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotBLK"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotSTL"]/self.player_stats[pid_team]["GamesPlayed"])
                home_starter_features.append(self.player_stats[pid_team]["TotTOV"]/self.player_stats[pid_team]["GamesPlayed"])
            else:
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
                home_starter_features.append(0)
            if pid_opp in self.player_stats:
                away_starter_features.append(self.player_stats[pid_opp]["TotBPM"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotPTS"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotREB"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotAST"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotBLK"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotSTL"]/self.player_stats[pid_opp]["GamesPlayed"])
                away_starter_features.append(self.player_stats[pid_opp]["TotTOV"]/self.player_stats[pid_opp]["GamesPlayed"])
            else:
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)
                away_starter_features.append(0)

        
        # Store
        home_static = np.array(home_static_features)
        # away_static = np.array(away_static_features)
        home_metric = np.array(home_metric_features)
        # away_metric = np.array(away_metric_features)
        home_starter = np.array(home_starter_features)
        # away_starter = np.array(away_starter_features)
        # Store sequenced data for both teams
        seq_features = ["Result", "PregameStrength", "OppPregameStrength", "Rest", "OppRest", "Home"]
        home_window = np.array([
            [game[col] for col in seq_features] for game in self.games_history[home][-GAMES_WINDOW_SZ:]
        ], dtype=np.float32)
        away_window = np.array([
            [game[col] for col in seq_features] for game in self.games_history[away][-GAMES_WINDOW_SZ:]
        ], dtype=np.float32)

        # Append to prediction input list
        self.games.append({
            "Home": home,
            "Away": away
        })
        self.X_list_static.append(home_static)
        self.X_list_metric.append(home_metric)
        self.X_list_starter.append(home_starter)
        self.X_list_seq_team.append(home_window)
        self.X_list_seq_opp.append(away_window)

        # TBD do we want an away copy?
        # self.games.append({
        #     "Home": home,
        #     "Away": away
        # })
        # self.X_list_static.append(away_static)
        # self.X_list_metric.append(away_metric)
        # self.X_list_starter.append(away_starter)
        # self.X_list_seq_team.append(away_window)
        # self.X_list_seq_opp.append(home_window)
    
    def to_tensor(self, scaler_X_static, scaler_X_metric, scaler_X_starter, 
            scaler_X_seq_team, scaler_X_seq_opp):
        X_static_stacked = np.stack(self.X_list_static, axis=0)
        X_metric_stacked = np.stack(self.X_list_metric, axis=0)
        X_starter_stacked = np.stack(self.X_list_starter, axis=0)
        X_seq_team_stacked = np.stack(self.X_list_seq_team, axis=0)
        X_seq_opp_stacked = np.stack(self.X_list_seq_opp, axis=0)
    
        # Scale down input with scaler_X
        # Make sure to ignore the Home column (last col) when scaling static
        X_static_scaled = scaler_X_static.transform(X_static_stacked[:, :-1])
        X_static_scaled = np.concatenate([X_static_scaled, X_static_stacked[:, -1:]], axis=1)
        X_metric_scaled = scaler_X_metric.transform(X_metric_stacked)
        X_starter_scaled = scaler_X_starter.transform(X_starter_stacked)

        # Stack, Flatten, scale, unflatten for seq data
        N_games, seq_len, num_feat = X_seq_team_stacked.shape
        X_seq_team_stacked = X_seq_team_stacked.reshape(N_games*seq_len, num_feat)
        X_seq_opp_stacked = X_seq_opp_stacked.reshape(N_games*seq_len, num_feat)
        # Make sure to ignore the Home column (last col) when scaling
        X_seq_team_scaled = scaler_X_seq_team.transform(X_seq_team_stacked[:, :-1])
        X_seq_opp_scaled = scaler_X_seq_opp.transform(X_seq_opp_stacked[:, :-1])
        X_seq_team_scaled = np.concatenate([X_seq_team_scaled, X_seq_team_stacked[:, -1:]], axis=1)
        X_seq_opp_scaled = np.concatenate([X_seq_opp_scaled, X_seq_opp_stacked[:, -1:]], axis=1)
        X_seq_team_scaled = X_seq_team_scaled.reshape(N_games, seq_len, num_feat)
        X_seq_opp_scaled = X_seq_opp_scaled.reshape(N_games, seq_len, num_feat)

        # convert to tensors
        X_static_tensor = torch.tensor(X_static_scaled, dtype=torch.float32)
        X_metric_tensor = torch.tensor(X_metric_scaled, dtype=torch.float32)
        X_starter_tensor = torch.tensor(X_starter_scaled, dtype=torch.float32)
        X_seq_team_tensor = torch.tensor(X_seq_team_scaled, dtype=torch.float32)
        X_seq_opp_tensor = torch.tensor(X_seq_opp_scaled, dtype=torch.float32)

        return X_static_tensor, X_metric_tensor, X_starter_tensor, X_seq_team_tensor, X_seq_opp_tensor

def is_same_player(player1: str, player2: str):
    # Identified special cases
    player_name_exceptions = {
        "Jimmy Butler" : "Jimmy Butler III",
        "Egor Diomin" : "Egor Demin"
    }

    # Strip periods and special chars so that names like A.J. and AJ are the same
    player1 = player1.replace(".", "")
    player2 = player2.replace(".", "")
    player1 = unidecode(player1)
    player2 = unidecode(player2)
    if player1 == player2:
        return True

    # Check exceptions
    if player1 in player_name_exceptions and player_name_exceptions[player1] == player2:
        return True
    if player2 in player_name_exceptions and player_name_exceptions[player2] == player1:
        return True

    return False

def round_to_half_pts(x: float):
    return round(x*2)/2

def main():
    # Preload existing games up to this point
    joined_games_path = DATA_DIR / f"joined/parquet/{SEASON}/gamesJoined.parquet"
    df_games = pd.read_parquet(joined_games_path)
    generator = PredictionInputGenerator()
    generator.add_games(df_games)

    # Read games to be predicted
    # Read raw box scores to find all available pids
    df_boxscores = pd.read_parquet(BOX_SCORE_FILE).loc[:, ["Player", "Team", "Player-additional"]]
    for game_file in RAW_DIR.iterdir():
        away_pids = []
        home_pids = []
        with open(game_file, "r", encoding="utf-8") as f:
            away = f.readline().strip()
            for i in range(5):
                player = f.readline().strip()
                matched_row = df_boxscores[
                    df_boxscores.apply(
                        lambda row: is_same_player(row["Player"], player) and row["Team"] == away,
                        axis=1
                    )
                ]
                if matched_row.empty:
                    print(player + " COULD NOT FIND pid!")
                    away_pids.append(UNAVAILABLE_PID)
                else:
                    away_pids.append(matched_row.iloc[0]["Player-additional"])

            home = f.readline().strip()
            for i in range(5):
                player = f.readline().strip()
                matched_row = df_boxscores[
                    df_boxscores.apply(
                        lambda row: is_same_player(row["Player"], player) and row["Team"] == home,
                        axis=1
                    )
                ]
                if matched_row.empty:
                    print(player + " COULD NOT FIND pid!")
                    home_pids.append(UNAVAILABLE_PID)
                else:
                    home_pids.append(matched_row.iloc[0]["Player-additional"])

            # TBD figure out how to get live rest
            home_rest = 1
            away_rest = 1
            generator.add_prediction_input(home, home_pids, home_rest, away, away_pids, away_rest)
    
    # Preload scalers
    scaler_X_static = load(SCALERS_DIR / "scaler_X_static.joblib")
    scaler_X_metric = load(SCALERS_DIR / "scaler_X_metric.joblib")
    scaler_X_starter = load(SCALERS_DIR / "scaler_X_starter.joblib")
    scaler_X_seq_team = load(SCALERS_DIR / "scaler_X_seq_team.joblib")
    scaler_X_seq_opp = load(SCALERS_DIR / "scaler_X_seq_opp.joblib")
    scaler_y = load(SCALERS_DIR / "scaler_y.joblib")

    # Get input tensors
    X_static_tensor, X_metric_tensor, X_starter_tensor, X_seq_team_tensor, X_seq_opp_tensor = generator.to_tensor(
        scaler_X_static, scaler_X_metric, scaler_X_starter, scaler_X_seq_team, scaler_X_seq_opp
    )
    
    # Load model and predict
    # TBD update weights only
    model = torch.load(MODELS_DIR / "test_full.pt", map_location="cpu", weights_only=False)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_static_tensor, X_metric_tensor, X_starter_tensor, X_seq_team_tensor, X_seq_opp_tensor)

    # Scale y up
    y_pred_upscaled = scaler_y.inverse_transform(y_pred).tolist()

    # output to DB test
    # WORKS?!
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME
    )
    cur = conn.cursor()
    test_games = generator.games
    print(f"TOTAL games: {len(test_games)}")
    print(f"TOTAL predictions: {len(y_pred_upscaled)}")
    for i in range(len(test_games)):
        home = test_games[i]['Home']
        away = test_games[i]['Away']
        spread_pred = round_to_half_pts(-y_pred_upscaled[i][0])
        print(f"Home: {home}")
        print(f"Away: {away}")
        print(f"Game date: {DATE}")
        print(f"Predicted Spread (Opposite Result): {spread_pred}\n")
        cur.execute("""
            INSERT INTO predictions (home, away, game_date, spread_prediction) 
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (home, game_date) 
            DO UPDATE SET away = EXCLUDED.away, spread_prediction = EXCLUDED.spread_prediction;
        """, 
        (home, away, DATE, spread_pred))

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()