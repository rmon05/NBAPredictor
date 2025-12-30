from playwright.sync_api import TimeoutError
import time
from dotenv import load_dotenv
import os
from pathlib import Path
from playwright_client import PWScraper
import pandas as pd
import io

load_dotenv()

DATA_DIR = Path("E:/NBAPredictor/data")
OUTPUT_DIR = DATA_DIR / "killersports_html"
CURR_YEAR = 2026

# TBD BELOWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW
    

def scrape_games(browser_context):
    page = browser_context.new_page()

    # Navigate to site
    page.goto("https://killersports.com/query?filter=NBA")

    # Filter Query
    yyyymmdd = int(current.strftime("%Y%m%d"))
    query = f"date={yyyymmdd} and site=home"

    page.get_by_text("Export Data", exact=True).click()
    page.get_by_role("button", name="Get table as CSV (for Excel)").click()

    # Extract html
    csv_text = page.locator("#csv_stats").inner_text()
    rk_index = csv_text.find("Rk,")
    if rk_index != -1:
        csv_text = csv_text[rk_index:]
    else:
        raise Exception("Games header not found, please manually review the data")

    output_path = OUTPUT_DIR / str(CURR_YEAR)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "gamesRaw.csv"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(csv_text + "\n")

    # while next page exists keep going!
    try:
        while True:
            page.get_by_role("link", name="Next Page").click()
            # Export Data to CSV
            page.get_by_text("Export Data", exact=True).click()
            page.get_by_role("button", name="Get table as CSV (for Excel)").click()

            # Extract
            csv_text = page.locator("#csv_stats").inner_text()
            rk_index = csv_text.find("Rk,")
            if rk_index != -1:
                csv_text = csv_text[rk_index:]
            else:
                raise Exception("Games header not found, please manually review the data")

            # Remove header for subsequent writes
            csv_io = io.StringIO(csv_text)
            df = pd.read_csv(csv_io)
            df.to_csv(output_file, mode="a", header=False, index=False)

    except TimeoutError as e:
        print("All pages scraped!")

def scrape_box_scores(browser_context):
    page = browser_context.new_page()

    # Navigate to site
    page.goto("https://www.sports-reference.com/stathead/basketball/player-game-finder.cgi?request=1&order_by_asc=1&order_by=date&timeframe=seasons&year_min=2026&year_max=2026")

    # Export Data to CSV
    page.get_by_text("Export Data", exact=True).click()
    page.get_by_role("button", name="Get table as CSV (for Excel)").click()

    # Extract
    csv_text = page.locator("#csv_stats").inner_text()
    rk_index = csv_text.find("Rk,")
    if rk_index != -1:
        csv_text = csv_text[rk_index:]
    else:
        raise Exception("Box Score header not found, please manually review the data")

    output_path = OUTPUT_DIR / str(CURR_YEAR)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "boxScoreRaw.csv"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(csv_text + "\n")

    # while next page exists keep going!
    try:
        while True:
            page.get_by_role("link", name="Next Page").click()
            # Export Data to CSV
            page.get_by_text("Export Data", exact=True).click()
            page.get_by_role("button", name="Get table as CSV (for Excel)").click()

            # Extract
            csv_text = page.locator("#csv_stats").inner_text()
            rk_index = csv_text.find("Rk,")
            if rk_index != -1:
                csv_text = csv_text[rk_index:]
            else:
                raise Exception("Box Score header not found, please manually review the data")

            # Remove header for subsequent writes
            csv_io = io.StringIO(csv_text)
            df = pd.read_csv(csv_io)
            df.to_csv(output_file, mode="a", header=False, index=False)

    except TimeoutError as e:
        print("All pages scraped!")

def main():
    scraper = PWScraper()
    scraper.add_task(scrape_games)
    scraper.add_task(scrape_box_scores)
    # Set headless to false for debugging, true for production
    scraper.scrape(headless=True) 

if __name__ == "__main__":
    main()