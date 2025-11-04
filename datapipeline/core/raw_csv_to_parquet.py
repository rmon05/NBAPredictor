import pandas as pd
from pathlib import Path
import time

RAW_DIR = Path(__file__).parent / "../../data/raw/csv"
PROCESSED_DIR = Path(__file__).parent / "../../data/raw/parquet"

# List of expected CSV files per year
CSV_FILES = ["boxScoreRaw.csv", "gamesRaw.csv", "playersRaw.csv"]

def ingest_year(year: int):
    """Read CSVs for a given year and write them as Parquet, tracking time."""
    start_year = time.time()
    
    year_raw_path = RAW_DIR / str(year)
    year_processed_path = PROCESSED_DIR / str(year)
    year_processed_path.mkdir(parents=True, exist_ok=True)

    for csv_file in CSV_FILES:
        csv_start = time.time()
        
        csv_path = year_raw_path / csv_file
        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        
        # Duplicate column
        if csv_file == "gamesRaw.csv" and "PTS.2" in df.columns:
            df['PTS'], df['PTS.2'] = df['PTS.2'], df['PTS']
            df = df.drop(columns=["PTS.2"]) 
            print(f"Replaced PTS with PTS.2 for {csv_path}")
        
        parquet_path = year_processed_path / f"{csv_file.replace('.csv', '.parquet')}"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
        
        csv_end = time.time()
        print(f"Converted {csv_path} → {parquet_path} | Time: {csv_end - csv_start:.2f}s")

    end_year = time.time()
    print(f"Finished ingesting year {year} | Total time: {end_year - start_year:.2f}s\n")


def main():
    years = [year for year in range(2015, 2026)]
    total_start = time.time()
    for year in years:
        ingest_year(year)
    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")


if __name__ == "__main__":
    main()
