import pandas as pd
from pathlib import Path
import time

RAW_DIR = Path(__file__).parent / "../../data/sportsbookreview_extracted"
PROCESSED_DIR = Path(__file__).parent / "../../data/sportsbookreview_formatted"


def ingest_year(year: int):
    """Read CSVs for a given year and write them as Parquet, tracking time."""
    start_year = time.time()
    
    year_raw_path = RAW_DIR / str(year)
    year_processed_csv = PROCESSED_DIR / "csv" / str(year)
    year_processed_parquet = PROCESSED_DIR / "parquet" / str(year)
    year_processed_csv.mkdir(parents=True, exist_ok=True)
    year_processed_parquet.mkdir(parents=True, exist_ok=True)
    input_file = "extracted.csv"    
    input_path = year_raw_path / input_file

    # format data as just string
    df = pd.read_csv(input_path, dtype={"Date": str})    

    # convert PK (pickem) to 0
    books = ["Betway",
            "Sports Interaction",
            "bet365",
            "Pinnacle",
            "Tonybet",
            "Betano",
            "Caesars"]
    for book in books:
        df[book] = df[book].replace("PK", "0").astype(float)

    output_file = "formatted.csv"
    output_csv_path = year_processed_csv / output_file
    df.to_csv(output_csv_path, index=False)
    output_parquet_path = year_processed_parquet / f"{output_file.replace('.csv', '.parquet')}"
    df.to_parquet(output_parquet_path, index=False)


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
