import pandas as pd
from pathlib import Path
import time
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).parent / "../../data/killersports_extracted"
PROCESSED_DIR = Path(__file__).parent / "../../data/killersports_formatted"

team_to_abbrv = {
    "Hawks": "ATL",
    "Celtics": "BOS",
    "Nets": "BRK",
    "Bulls": "CHI",
    "Hornets": "CHO",
    "Cavaliers": "CLE",
    "Mavericks": "DAL",
    "Nuggets": "DEN",
    "Pistons": "DET",
    "Warriors": "GSW",
    "Rockets": "HOU",
    "Pacers": "IND",
    "Clippers": "LAC",
    "Lakers": "LAL",
    "Grizzlies": "MEM",
    "Heat": "MIA",
    "Bucks": "MIL",
    "Timberwolves": "MIN",
    "Pelicans": "NOP",
    "Knicks": "NYK",
    "Thunder": "OKC",
    "Magic": "ORL",
    "Seventysixers": "PHI",
    "Suns": "PHO",
    "Trailblazers": "POR",
    "Kings": "SAC",
    "Spurs": "SAS",
    "Raptors": "TOR",
    "Jazz": "UTA",
    "Wizards": "WAS",
}


def ingest_year(year: int):
    """Read CSVs for a given year and write them as Parquet, tracking time."""
    start_time = time.time()
    year_raw_path = RAW_DIR / str(year)
    year_processed_path_csv = PROCESSED_DIR / "csv" /str(year)
    year_processed_path_parquet = PROCESSED_DIR / "parquet" /str(year)
    year_processed_path_csv.mkdir(parents=True, exist_ok=True)
    year_processed_path_parquet.mkdir(parents=True, exist_ok=True)

    data_file = year_raw_path / "extracted.txt"
    if not data_file.exists():
        raise FileNotFoundError(f"File not found: {data_file}")

    # Read the entire HTML-like file
    with open(data_file, "r", encoding="utf-8") as f:
        raw_html = f.read()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")

    rows = []
    for tr in soup.find_all("tr", class_="qry-tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) >= 10:
            rows.append(tds[:10])  # keep only the first 10 columns

    # Define the expected columns
    columns = [
        "Date",
        "DayOfWeek",
        "Season",
        "Team",
        "Opp",
        "Site",
        "Result",
        "DaysRest",
        "PregameSpread",
        "PregameTotal",
    ]

    # Build the DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Clean columns
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y").dt.strftime("%Y-%m-%d")
    df = df.drop(columns=['DayOfWeek', "Season"])
    df["Team"] = df["Team"].apply(lambda team_name: team_to_abbrv[team_name])
    df["Opp"] = df["Opp"].apply(lambda team_name: team_to_abbrv[team_name])

    # Clean rest data
    df[["RestTeam", "RestOpp"]] = (
        df["DaysRest"]
        .str.split("&", expand=True)
        .replace("-", None)
        .fillna("5")
        .astype(float)
    )

    # Write to CSV and Parquet
    csv_path = year_processed_path_csv / f"betData.csv"
    parquet_path = year_processed_path_parquet / f"betData.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    end_time = time.time()
    print(f"Processed {year}: {len(df)} rows in {end_time - start_time:.2f}s")
    return df

        



def main():
    years = [year for year in range(2015, 2026)]
    total_start = time.time()
    for year in years:
        ingest_year(year)
    total_end = time.time()
    print(f"All years processed in {total_end - total_start:.2f}s")


if __name__ == "__main__":
    main()
