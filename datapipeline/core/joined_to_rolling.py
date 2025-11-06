import pandas as pd
from pathlib import Path
import time
import os
import json

# min games before writing data
DATA_THRESHOLD = 40

# paths
data_dir = Path(__file__).parent / "../../data"
output_path_parquet = data_dir / f"rolling/parquet/gamesRolling.parquet"
output_path_csv = data_dir / f"rolling/csv/gamesRolling.csv"
output_path_parquet.parent.mkdir(parents=True, exist_ok=True)
output_path_csv.parent.mkdir(parents=True, exist_ok=True)
init_year = 2015

def calc_agg4FR(data_row, is_home):
    sfx = ".1" if not is_home else ""
    home_box_scores = data_row["HomeBoxScores"]
    away_box_scores = data_row["AwayBoxScores"]

    # Extract TOV, ORB, oppDRB from box scores
    TOV = 0
    ORB = 0
    oppDRB = 0
    if is_home:
        for player in home_box_scores:
            TOV += player["TOV"]
            ORB += player["ORB"]
        for player in away_box_scores:
            oppDRB += player["DRB"]
    else:
        for player in away_box_scores:
            TOV += player["TOV"]
            ORB += player["ORB"]
        for player in home_box_scores:
            oppDRB += player["DRB"]

    eFG = (data_row["FG"+sfx]+0.5*data_row["3P"+sfx])/data_row["FGA"+sfx]
    TOVpct = TOV/(data_row["FGA"+sfx]+0.44*data_row["FTA"+sfx]+TOV)
    ORBpct = ORB/(ORB+oppDRB)
    FTrate = data_row["FTA"]/data_row["FGA"]
    agg4FR = 0.5*eFG-0.3*TOVpct+0.15*ORBpct+0.05*FTrate
    return agg4FR

def roll_year(year: int):
    start_year = time.time()
    joined_games_path = data_dir / f"joined/parquet/{year}/gamesJoined.parquet"
    df_games = pd.read_parquet(joined_games_path)
    df_out = pd.DataFrame([])

    # Convert box scores to list of maps
    df_games["HomeBoxScores"] = df_games["HomeBoxScores"].apply(json.loads)
    df_games["AwayBoxScores"] = df_games["AwayBoxScores"].apply(json.loads)

    # Initialize team stats & player stats
    team_stats = {}
    player_stats = {}
    for _, row in df_games.iterrows():
        home = row["Home"]
        away = row["Opp"]
        home_box_scores = row["HomeBoxScores"]
        away_box_scores = row["AwayBoxScores"]
        # bet data
        has_bet_data = row["DaysRest"] is not None and row["DaysRest"] != "-" \
            and row["PregameSpread"] != "NaN" and row["PregameTotal"] != "NaN"
        

        # init
        if not (home in team_stats):
            team_stats[home] = {}
            team_stats[home]["W"] = 0
            team_stats[home]["L"] = 0
            team_stats[home]["OppW"] = 0
            team_stats[home]["OppL"] = 0
            team_stats[home]["HomeW"] = 0
            team_stats[home]["HomeL"] = 0
            team_stats[home]["AwayW"] = 0
            team_stats[home]["AwayL"] = 0
            team_stats[home]["PTS"] = 0
            team_stats[home]["FG"] = 0
            team_stats[home]["FGA"] = 0
            team_stats[home]["2P"] = 0
            team_stats[home]["2PA"] = 0
            team_stats[home]["3P"] = 0
            team_stats[home]["3PA"] = 0
            team_stats[home]["FT"] = 0
            team_stats[home]["FTA"] = 0
            
            team_stats[home]["PTSDiffTot"] = 0
            team_stats[home]["agg4FRDiffTot"] = 0
            team_stats[home]["Streak"] = 0
            team_stats[home]["AST"] = 0
            team_stats[home]["ORB"] = 0
            team_stats[home]["OppDRB"] = 0
            team_stats[home]["STL"] = 0
            team_stats[home]["BLK"] = 0
            team_stats[home]["TOV"] = 0

        if not (away in team_stats):
            team_stats[away] = {}
            team_stats[away]["W"] = 0
            team_stats[away]["L"] = 0
            team_stats[away]["OppW"] = 0
            team_stats[away]["OppL"] = 0
            team_stats[away]["HomeW"] = 0
            team_stats[away]["HomeL"] = 0
            team_stats[away]["AwayW"] = 0
            team_stats[away]["AwayL"] = 0
            team_stats[away]["PTS"] = 0
            team_stats[away]["FG"] = 0
            team_stats[away]["FGA"] = 0
            team_stats[away]["2P"] = 0
            team_stats[away]["2PA"] = 0
            team_stats[away]["3P"] = 0
            team_stats[away]["3PA"] = 0
            team_stats[away]["FT"] = 0
            team_stats[away]["FTA"] = 0
            
            team_stats[away]["PTSDiffTot"] = 0
            team_stats[away]["agg4FRDiffTot"] = 0
            team_stats[away]["Streak"] = 0
            team_stats[away]["AST"] = 0
            team_stats[away]["ORB"] = 0
            team_stats[away]["OppDRB"] = 0
            team_stats[away]["STL"] = 0
            team_stats[away]["BLK"] = 0
            team_stats[away]["TOV"] = 0
        
        # Write to datapoint if sufficient data
        if team_stats[home]["W"]+team_stats[home]["L"] >= DATA_THRESHOLD and has_bet_data:
            home_datapoint = {}
            away_datapoint = {}
            home_played = team_stats[home]["W"]+team_stats[home]["L"]
            away_played = team_stats[away]["W"]+team_stats[away]["L"]
            home_rest = row["DaysRest"].split("&")[0]
            away_rest = row["DaysRest"].split("&")[1]
            pregame_spread = float(row["PregameSpread"]) # Note this is from the home perspective
            pregame_total = row["PregameTotal"]


            home_datapoint["Home"] = 1
            home_datapoint["Result"] = row["PTS"]-row["PTS.1"] # Note this is OPPOSITE of spread
            home_datapoint["Spread"] = pregame_spread
            home_datapoint["Total"] = pregame_total
            # home_datapoint["Team1"] = home
            home_datapoint["WinPct1"] = team_stats[home]["W"]/(team_stats[home]["W"]+team_stats[home]["L"])
            home_datapoint["OppWinPct1"] = team_stats[home]["OppW"]/(team_stats[home]["OppW"]+team_stats[home]["OppL"])
            home_datapoint["LocationWinPct1"] = team_stats[home]["HomeW"]/(team_stats[home]["HomeW"]+team_stats[home]["HomeL"])
            home_datapoint["PTS1"] = team_stats[home]["PTS"]/home_played
            home_datapoint["FG1"] = team_stats[home]["FG"]/home_played
            home_datapoint["FGA1"] = team_stats[home]["FGA"]/home_played
            home_datapoint["2P1"] = team_stats[home]["2P"]/home_played
            home_datapoint["2PA1"] = team_stats[home]["2PA"]/home_played
            home_datapoint["3P1"] = team_stats[home]["3P"]/home_played
            home_datapoint["3PA1"] = team_stats[home]["3PA"]/home_played
            home_datapoint["FT1"] = team_stats[home]["FT"]/home_played
            home_datapoint["FTA1"] = team_stats[home]["FTA"]/home_played
            home_datapoint["PTSDiff1"] = team_stats[home]["PTSDiffTot"]/home_played
            home_datapoint["agg4FRDiff1"] = team_stats[home]["agg4FRDiffTot"]/home_played
            home_datapoint["Streak1"] = team_stats[home]["Streak"]
            home_datapoint["AST1"] = team_stats[home]["AST"]/home_played
            home_datapoint["STL1"] = team_stats[home]["STL"]/home_played
            home_datapoint["BLK1"] = team_stats[home]["BLK"]/home_played
            home_datapoint["TOV1"] = team_stats[home]["TOV"]/home_played
            home_datapoint["Rest1"] = home_rest
            # Assume 5 highest minute players are the starters
            starter_tot_bpm = 0
            for i in range(5):
                player = home_box_scores[i]
                pid = player["Player-additional"]
                if pid in player_stats:
                    home_datapoint[f"Starter{i}_BPM1"] = player_stats[pid]["TotBPM"]/player_stats[pid]["GamesPlayed"]
                else:
                    home_datapoint[f"Starter{i}_BPM1"] = 0
                starter_tot_bpm += home_datapoint[f"Starter{i}_BPM1"]
            home_datapoint[f"Starters_avg_BPM1"] = starter_tot_bpm/5

            # home_datapoint["Team2"] = away
            home_datapoint["WinPct2"] = team_stats[away]["W"]/(team_stats[away]["W"]+team_stats[away]["L"])
            home_datapoint["OppWinPct2"] = team_stats[away]["OppW"]/(team_stats[away]["OppW"]+team_stats[away]["OppL"])
            home_datapoint["LocationWinPct2"] = team_stats[away]["AwayW"]/(team_stats[away]["AwayW"]+team_stats[away]["AwayL"])
            home_datapoint["PTS2"] = team_stats[away]["PTS"]/away_played
            home_datapoint["FG2"] = team_stats[away]["FG"]/away_played
            home_datapoint["FGA2"] = team_stats[away]["FGA"]/away_played
            home_datapoint["2P2"] = team_stats[away]["2P"]/away_played
            home_datapoint["2PA2"] = team_stats[away]["2PA"]/away_played
            home_datapoint["3P2"] = team_stats[away]["3P"]/away_played
            home_datapoint["3PA2"] = team_stats[away]["3PA"]/away_played
            home_datapoint["FT2"] = team_stats[away]["FT"]/away_played
            home_datapoint["FTA2"] = team_stats[away]["FTA"]/away_played
            home_datapoint["PTSDiff2"] = team_stats[away]["PTSDiffTot"]/away_played
            home_datapoint["agg4FRDiff2"] = team_stats[away]["agg4FRDiffTot"]/away_played
            home_datapoint["Streak2"] = team_stats[away]["Streak"]
            home_datapoint["AST2"] = team_stats[away]["AST"]/away_played
            home_datapoint["STL2"] = team_stats[away]["STL"]/away_played
            home_datapoint["BLK2"] = team_stats[away]["BLK"]/away_played
            home_datapoint["TOV2"] = team_stats[away]["TOV"]/away_played
            home_datapoint["Rest2"] = away_rest
            # Assume 5 highest minute players are the starters
            starter_tot_bpm = 0
            for i in range(5):
                player = away_box_scores[i]
                pid = player["Player-additional"]
                if pid in player_stats:
                    home_datapoint[f"Starter{i}_BPM2"] = player_stats[pid]["TotBPM"]/player_stats[pid]["GamesPlayed"]
                else:
                    home_datapoint[f"Starter{i}_BPM2"] = 0
                starter_tot_bpm += home_datapoint[f"Starter{i}_BPM2"]
            home_datapoint[f"Starters_avg_BPM2"] = starter_tot_bpm/5


            away_datapoint["Home"] = 0
            away_datapoint["Result"] = row["PTS.1"]-row["PTS"]
            away_datapoint["Spread"] = -pregame_spread
            away_datapoint["Total"] = pregame_total
            # away_datapoint["Team1"] = away
            away_datapoint["WinPct1"] = team_stats[away]["W"]/(team_stats[away]["W"]+team_stats[away]["L"])
            away_datapoint["OppWinPct1"] = team_stats[away]["OppW"]/(team_stats[away]["OppW"]+team_stats[away]["OppL"])
            away_datapoint["LocationWinPct1"] = team_stats[away]["AwayW"]/(team_stats[away]["AwayW"]+team_stats[away]["AwayL"])
            away_datapoint["PTS1"] = team_stats[away]["PTS"]/away_played
            away_datapoint["FG1"] = team_stats[away]["FG"]/away_played
            away_datapoint["FGA1"] = team_stats[away]["FGA"]/away_played
            away_datapoint["2P1"] = team_stats[away]["2P"]/away_played
            away_datapoint["2PA1"] = team_stats[away]["2PA"]/away_played
            away_datapoint["3P1"] = team_stats[away]["3P"]/away_played
            away_datapoint["3PA1"] = team_stats[away]["3PA"]/away_played
            away_datapoint["FT1"] = team_stats[away]["FT"]/away_played
            away_datapoint["FTA1"] = team_stats[away]["FTA"]/away_played
            away_datapoint["PTSDiff1"] = team_stats[away]["PTSDiffTot"]/away_played
            away_datapoint["agg4FRDiff1"] = team_stats[away]["agg4FRDiffTot"]/away_played
            away_datapoint["Streak1"] = team_stats[away]["Streak"]
            away_datapoint["AST1"] = team_stats[away]["AST"]/away_played
            away_datapoint["STL1"] = team_stats[away]["STL"]/away_played
            away_datapoint["BLK1"] = team_stats[away]["BLK"]/away_played
            away_datapoint["TOV1"] = team_stats[away]["TOV"]/away_played
            away_datapoint["Rest1"] = away_rest
            # Assume 5 highest minute players are the starters
            starter_tot_bpm = 0
            for i in range(5):
                player = away_box_scores[i]
                pid = player["Player-additional"]
                if pid in player_stats:
                    away_datapoint[f"Starter{i}_BPM1"] = player_stats[pid]["TotBPM"]/player_stats[pid]["GamesPlayed"]
                else:
                    away_datapoint[f"Starter{i}_BPM1"] = 0
                starter_tot_bpm += away_datapoint[f"Starter{i}_BPM1"]
            away_datapoint[f"Starters_avg_BPM1"] = starter_tot_bpm/5

            # away_datapoint["Team2"] = home
            away_datapoint["WinPct2"] = team_stats[home]["W"]/(team_stats[home]["W"]+team_stats[home]["L"])
            away_datapoint["OppWinPct2"] = team_stats[home]["OppW"]/(team_stats[home]["OppW"]+team_stats[home]["OppL"])
            away_datapoint["LocationWinPct2"] = team_stats[home]["HomeW"]/(team_stats[home]["HomeW"]+team_stats[home]["HomeL"])
            away_datapoint["PTS2"] = team_stats[home]["PTS"]/home_played
            away_datapoint["FG2"] = team_stats[home]["FG"]/home_played
            away_datapoint["FGA2"] = team_stats[home]["FGA"]/home_played
            away_datapoint["2P2"] = team_stats[home]["2P"]/home_played
            away_datapoint["2PA2"] = team_stats[home]["2PA"]/home_played
            away_datapoint["3P2"] = team_stats[home]["3P"]/home_played
            away_datapoint["3PA2"] = team_stats[home]["3PA"]/home_played
            away_datapoint["FT2"] = team_stats[home]["FT"]/home_played
            away_datapoint["FTA2"] = team_stats[home]["FTA"]/home_played
            away_datapoint["PTSDiff2"] = team_stats[home]["PTSDiffTot"]/home_played
            away_datapoint["agg4FRDiff2"] = team_stats[home]["agg4FRDiffTot"]/home_played
            away_datapoint["Streak2"] = team_stats[home]["Streak"]
            away_datapoint["AST2"] = team_stats[home]["AST"]/home_played
            away_datapoint["STL2"] = team_stats[home]["STL"]/home_played
            away_datapoint["BLK2"] = team_stats[home]["BLK"]/home_played
            away_datapoint["TOV2"] = team_stats[home]["TOV"]/home_played
            away_datapoint["Rest2"] = home_rest
            # Assume 5 highest minute players are the starters
            starter_tot_bpm = 0
            for i in range(5):
                player = home_box_scores[i]
                pid = player["Player-additional"]
                if pid in player_stats:
                    away_datapoint[f"Starter{i}_BPM2"] = player_stats[pid]["TotBPM"]/player_stats[pid]["GamesPlayed"]
                else:
                    away_datapoint[f"Starter{i}_BPM2"] = 0
                starter_tot_bpm += away_datapoint[f"Starter{i}_BPM2"]
            away_datapoint[f"Starters_avg_BPM2"] = starter_tot_bpm/5
            
            df_out = pd.concat([df_out, pd.DataFrame([home_datapoint])], ignore_index=True)
            df_out = pd.concat([df_out, pd.DataFrame([away_datapoint])], ignore_index=True)


        # Update record
        result = row["Result"][0]
        team_stats[home]["OppW"] += team_stats[away]["W"]
        team_stats[home]["OppL"] += team_stats[away]["L"]
        team_stats[away]["OppW"] += team_stats[home]["W"]
        team_stats[away]["OppL"] += team_stats[home]["L"]
        if result == "W":
            team_stats[home]["W"] += 1
            team_stats[home]["HomeW"] += 1
            team_stats[home]["Streak"] = max(1, team_stats[home]["Streak"]+1)
            team_stats[away]["L"] += 1
            team_stats[away]["AwayL"] += 1
            team_stats[away]["Streak"] = min(-1, team_stats[away]["Streak"]-1)
        else:
            team_stats[home]["L"] += 1
            team_stats[home]["HomeL"] += 1
            team_stats[home]["Streak"] = min(-1, team_stats[home]["Streak"]-1)
            team_stats[away]["W"] += 1
            team_stats[away]["AwayW"] += 1
            team_stats[away]["Streak"] = max(1, team_stats[away]["Streak"]+1)

        # core stats
        team_stats[home]["PTS"] += row["PTS"]
        team_stats[away]["PTS"] += row["PTS.1"]
        team_stats[home]["FG"] += row["FG"]
        team_stats[away]["FG"] += row["FG.1"]
        team_stats[home]["FGA"] += row["FGA"]
        team_stats[away]["FGA"] += row["FGA.1"]
        team_stats[home]["2P"] += row["2P"]
        team_stats[away]["2P"] += row["2P.1"]
        team_stats[home]["2PA"] += row["2PA"]
        team_stats[away]["2PA"] += row["2PA.1"]
        team_stats[home]["3P"] += row["3P"]
        team_stats[away]["3P"] += row["3P.1"]
        team_stats[home]["3PA"] += row["3PA"]
        team_stats[away]["3PA"] += row["3PA.1"]
        team_stats[home]["FT"] += row["FT"]
        team_stats[away]["FT"] += row["FT.1"]
        team_stats[home]["FTA"] += row["FTA"]
        team_stats[away]["FTA"] += row["FTA.1"]

        # derived stats
        team_stats[home]["PTSDiffTot"] += row["PTS"] - row["PTS.1"]
        team_stats[away]["PTSDiffTot"] += row["PTS.1"] - row["PTS"]
        agg4FRHome = calc_agg4FR(row, True)
        agg4FRAway = calc_agg4FR(row, False)
        team_stats[home]["agg4FRDiffTot"] += agg4FRHome - agg4FRAway
        team_stats[away]["agg4FRDiffTot"] += agg4FRAway - agg4FRHome

        # Box score stats
        for player in home_box_scores:
            pid = player["Player-additional"]
            if not (pid in player_stats):
                player_stats[pid] = {}
                player_stats[pid]["GamesPlayed"] = 0
                player_stats[pid]["TotBPM"] = 0

            player_stats[pid]["GamesPlayed"] += 1
            player_stats[pid]["TotBPM"] += player["BPM"]

            team_stats[home]["AST"] += player["AST"]
            team_stats[home]["ORB"] += player["ORB"]
            team_stats[away]["OppDRB"] += player["DRB"]
            team_stats[home]["STL"] += player["STL"]
            team_stats[home]["BLK"] += player["BLK"]
            team_stats[home]["TOV"] += player["TOV"]
        
        for player in away_box_scores:
            pid = player["Player-additional"]
            if not (pid in player_stats):
                player_stats[pid] = {}
                player_stats[pid]["GamesPlayed"] = 0
                player_stats[pid]["TotBPM"] = 0
            
            player_stats[pid]["GamesPlayed"] += 1
            player_stats[pid]["TotBPM"] += player["BPM"]

            team_stats[away]["AST"] += player["AST"]
            team_stats[away]["ORB"] += player["ORB"]
            team_stats[home]["OppDRB"] += player["DRB"]
            team_stats[away]["STL"] += player["STL"]
            team_stats[away]["BLK"] += player["BLK"]
            team_stats[away]["TOV"] += player["TOV"]

    # Write out
    if not os.path.exists(output_path_csv) or year==init_year:
        df_out.to_csv(output_path_csv, mode='w', header=True, index=False)
    else:
        df_out.to_csv(output_path_csv, mode='a', header=False, index=False)

    end_year = time.time()
    print(f"Finished generating rolling dataset for year {year} | Total time: {end_year - start_year:.2f}s\n")

def main():
    years = [year for year in range(2015, 2026)]
    total_start = time.time()

    for year in years:
        roll_year(year)

    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")

    # copy csv to parquet
    df_all = pd.read_csv(output_path_csv)
    df_all.to_parquet(output_path_parquet)

if __name__ == "__main__":
    main()