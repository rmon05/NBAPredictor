import pandas as pd
from pathlib import Path
import time

DATA_DIR = Path("/usr/local/airflow/data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "clean"

# List of expected CSV files per year
CSV_FILES = ["boxScoreRaw.csv", "gamesRaw.csv"]

def ingest_year(year: int):
    """Read CSVs for a given year and write them as Parquet, tracking time."""
    start_year = time.time()
    
    year_raw_path = RAW_DIR / str(year)

    for csv_file in CSV_FILES:
        csv_start = time.time()
        
        csv_path = year_raw_path / csv_file
        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        if "Date" in df.columns:
            df["Date"] = df["Date"].astype(str)
        
        # Duplicate column, fix MP
        if csv_file == "boxScoreRaw.csv" and "MP.1" in df.columns:
            df = df.drop(columns=["MP"])
            df = df.rename(columns={"MP.1": "MP"}) 
            print(f"Replaced MP.1 for {csv_path}")
        
        # Duplicate column, and fix PTS
        if csv_file == "gamesRaw.csv" and "PTS.2" in df.columns:
            df['PTS.1'], df['PTS.2'] = df['PTS.2'], df['PTS.1']
            df = df.drop(columns=["PTS.2"]) 
            print(f"Replaced PTS.1 with PTS.2 for {csv_path}")

        # Fix Unnamed columns
        # TBD: make this @ column less fragile
        if csv_file == "boxScoreRaw.csv":
            for i in range(len(df.columns)):
                if "Unnamed" in df.columns[i]:
                    df = df.rename(columns={df.columns[i]: f"Unnamed: {i}"})
        
        
        output_parquet_path = PROCESSED_DIR / f"parquet/{year}"
        output_csv_path = PROCESSED_DIR / f"csv/{year}"
        output_parquet_path.mkdir(parents=True, exist_ok=True)
        output_csv_path.mkdir(parents=True, exist_ok=True)

        csv_file = csv_file.replace('Raw', 'Clean')
        parquet_file = csv_file.replace('.csv', '.parquet')
        df.to_parquet(output_parquet_path / parquet_file, index=False)
        df.to_csv(output_csv_path / csv_file, index=False)
        
        csv_end = time.time()
        print(f"Converted in time: {csv_end - csv_start:.2f}s")

    end_year = time.time()
    print(f"Finished ingesting year {year} | Total time: {end_year - start_year:.2f}s\n")


def main():
    # years = [year for year in range(2015, 2027)]
    years = [2026]
    total_start = time.time()
    for year in years:
        ingest_year(year)
    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")


if __name__ == "__main__":
    main()
