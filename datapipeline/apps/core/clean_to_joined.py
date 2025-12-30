import pandas as pd
from pathlib import Path
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = Path("/usr/local/airflow/data")

def join_year(year: int):
    start_year = time.time()
    raw_games_path = DATA_DIR / f"clean/parquet/{year}/gamesClean.parquet"
    raw_box_path = DATA_DIR / f"clean/parquet/{year}/boxScoreClean.parquet"
    raw_odds_path = DATA_DIR / f"killersports_formatted/parquet/{year}/betData.parquet"
    raw_sbr_path = DATA_DIR / f"sportsbookreview_formatted/parquet/{year}/formatted.parquet"
    output_path_parquet = DATA_DIR / f"joined/parquet/{year}/gamesJoined.parquet"
    output_path_csv = DATA_DIR / f"joined/csv/{year}/gamesJoined.csv"
    output_path_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_path_csv.parent.mkdir(parents=True, exist_ok=True)
    df_games = pd.read_parquet(raw_games_path)
    df_box = pd.read_parquet(raw_box_path)
    # TBD add below back once 2026 data is good to go
    # df_killersports = pd.read_parquet(raw_odds_path)
    # df_sbr = pd.read_parquet(raw_sbr_path)

    
    # JOIN GAME and BOX SCORES
    # agg on key (date, team)
    # limit to top 10 contributors for now
    df_home_agg = (
        df_box.groupby(["Date", "Team"])
        .apply(lambda g: g[g["Unnamed: 5"] != "@"].sort_values("MP", ascending=False).head(10).to_dict(orient="records"), include_groups=False)
        .reset_index(name="HomeBoxScores")
    )
    df_away_agg = (
        df_box.groupby(["Date", "Opp"])
        .apply(lambda g: g[g["Unnamed: 5"] == "@"].sort_values("MP", ascending=False).head(10).to_dict(orient="records"), include_groups=False)
        .reset_index(name="AwayBoxScores")
    )
    df_away_agg.to_csv(output_path_csv, index=False)
    df_away_agg.rename(columns={"Opp": "Team", "Team": "Opp"}, inplace=True)

    # join on home then away games
    df_joined = df_games.merge(df_home_agg, on=["Date", "Team"])
    df_joined = df_joined.merge(df_away_agg, on=["Date", "Team"])

    # Rewrite Results as the pts diff
    df_joined["Result"] = df_joined["PTS"] - df_joined["PTS.1"]
    
    # TBD add this back once 2026 data is good to go
    # # LEFT JOIN GAME+BOX with KILLERSPORTS BET DATA (some rows missing)
    # df_killersports = df_killersports.drop(columns=["Result", "Site", "Opp"])
    # df_joined = pd.merge(df_joined, df_killersports, on=["Date", "Team"], how='left')
    # df_joined.rename(columns={"RestTeam": "RestHome"}, inplace=True)

    # TBD add this back once 2026 data is good to go
    # # LEFT JOIN GAME+BOX with SPORTSBOOKREVIEW BET DATA (some rows missing)
    # df_sbr = df_sbr.drop(columns=["gid", "Opp"])
    # df_joined = pd.merge(df_joined, df_sbr, on=["Date", "Team"], how='left')


    # TEMP placeholder values for 2026 year
    if year == 2026:
        df_joined["RestHome"] = 1
        df_joined["RestOpp"] = 1

    # rename home move boxscore data to end and json dump
    df_joined.rename(columns={"Team": "Home"}, inplace=True)
    df_joined = df_joined[
        [c for c in df_joined.columns if (c != "HomeBoxScores" and c != "AwayBoxScores")] + ["HomeBoxScores"] + ["AwayBoxScores"]
    ]
    df_joined["HomeBoxScores"] = df_joined["HomeBoxScores"].apply(lambda x: json.dumps(x))
    df_joined["AwayBoxScores"] = df_joined["AwayBoxScores"].apply(lambda x: json.dumps(x))
    
    # write out
    df_joined.to_parquet(output_path_parquet, index=False)
    df_joined.to_csv(output_path_csv, index=False)

    end_year = time.time()
    print(f"Finished joining year {year} | Total time: {end_year - start_year:.2f}s\n")



def main():
    # years = [year for year in range(2015, 2027)]
    years = [2026]
    total_start = time.time()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(join_year, year) for year in years]

        for f in as_completed(futures):
            f.result()

    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")

if __name__ == "__main__":
    main()