import pandas as pd
from pathlib import Path
import time
import os

# paths
data_dir = Path(__file__).parent / "../../data"
output_path_books_parquet = data_dir / f"books/parquet/books.parquet"
output_path_books_csv = data_dir / f"books/csv/books.csv"
output_path_books_parquet.parent.mkdir(parents=True, exist_ok=True)
output_path_books_csv.parent.mkdir(parents=True, exist_ok=True)
init_year = 2020

def roll_year(year: int):
    start_year = time.time()
    joined_games_path = data_dir / f"joined/parquet/{year}/gamesJoined.parquet"
    df_games = pd.read_parquet(joined_games_path)
    df_books = pd.DataFrame([])

    for _, row in df_games.iterrows():        
        # Write to sbr books results
        if row["bet365"] is not None and row["Pinnacle"] is not None and row["PregameSpread"] is not None:
            book_datapoint = {
                "Date": row["Date"],
                "Home": row["Home"],
                "Away": row["Opp"],
                "Spread_Actual": row["PTS.1"]-row["PTS"],
                "Spread_Killersports": row["PregameSpread"],
                "Spread_bet365": row["bet365"],
                "Spread_pinnacle": row["Pinnacle"]
            }
            df_books = pd.concat([df_books, pd.DataFrame([book_datapoint])], ignore_index=True)

    # Write out
    if not os.path.exists(output_path_books_csv) or year==init_year:
        df_books.to_csv(output_path_books_csv, mode='w', header=True, index=False)
    else:
        df_books.to_csv(output_path_books_csv, mode='a', header=False, index=False)

    end_year = time.time()
    print(f"Finished generating rolling dataset for year {year} | Total time: {end_year - start_year:.2f}s\n")

def main():
    years = [year for year in range(2020, 2026)]
    total_start = time.time()

    for year in years:
        roll_year(year)

    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")

    # copy csv to parquet
    df_all = pd.read_csv(output_path_books_csv)
    df_all.to_parquet(output_path_books_parquet)

if __name__ == "__main__":
    main()