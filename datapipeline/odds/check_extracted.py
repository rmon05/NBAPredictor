from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter

# === CONFIG ===
for year in range(2015, 2026):
    html_file = Path(f"../../data/killersports_extracted/{year}/extracted.txt")
    csv_file = Path(f"../../data/raw/csv/{year}/gamesRaw.csv")

    # === PARSE HTML FILE ===
    with html_file.open("r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Extract all rows with class 'qry-tr'
    rows = soup.find_all("tr", class_="qry-tr")

    # Extract and normalize the date column (first <td>)
    html_dates = []
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if not cols:
            continue
        date_text = cols[0]  # e.g. "Oct 28, 2014"
        # Parse to yyyy-mm-dd using pandas
        try:
            parsed_date = pd.to_datetime(date_text).strftime("%Y-%m-%d")
            html_dates.append(parsed_date)
        except Exception:
            pass

    html_counts = Counter(html_dates)

    # === PARSE CSV FILE ===
    df = pd.read_csv(csv_file)
    csv_counts = Counter(df["Date"])

    # === COMPARE COUNTS ===
    all_dates = sorted(set(html_counts) | set(csv_counts))

    print(f"{'Date':<12} {'HTML':>6} {'CSV':>6} {'Diff':>6}")
    print("-" * 30)
    for date in all_dates:
        html_count = html_counts.get(date, 0)
        csv_count = csv_counts.get(date, 0)
        diff = csv_count - html_count
        if diff != 0:
            print(f"{date:<12} {html_count:>6} {csv_count:>6} {diff:>6}")

    # === OPTIONAL ===
    # If you just want missing dates:
    missing_in_html = [d for d in all_dates if csv_counts.get(d, 0) > html_counts.get(d, 0)]
    if missing_in_html:
        print("\nDates missing from HTML:")
        for d in missing_in_html:
            print(f"  {d} ({csv_counts[d] - html_counts.get(d, 0)} games missing)")
