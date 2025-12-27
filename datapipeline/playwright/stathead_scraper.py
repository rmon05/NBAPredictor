from playwright.sync_api import TimeoutError
import time
from dotenv import load_dotenv
import os
from pathlib import Path
from playwright_client import PWScraper
import pandas as pd
import io

load_dotenv()

STATHEAD_USERNAME = os.getenv("STATHEAD_USERNAME")
STATHEAD_PASSWORD = os.getenv("STATHEAD_PASSWORD")
DATA_DIR = Path("E:/NBAPredictor/data")
OUTPUT_DIR = DATA_DIR / "raw"
CURR_YEAR = 2026 # TBD this needs to be updated to not be hardcoded!

def login(browser_context):
    # Open a new page
    page = browser_context.new_page()

    # 2. Login to a website
    page.goto("https://www.sports-reference.com/users/login.cgi")
    
    # # Fill login form
    page.fill("#username", STATHEAD_USERNAME)
    page.fill("#password", STATHEAD_PASSWORD)
    # Wait for the button popup to disappear
    time.sleep(10)
    page.locator("#sh-submit-button").click()
    
    # disable cookies
    page.get_by_role("link", name="Cookie Preferences").click()
    page.locator("button.osano-cm-denyAll").click()
    

def scrape_games(browser_context):
    page = browser_context.new_page()

    # Navigate to site
    page.goto("https://www.sports-reference.com/stathead/basketball/team-game-finder.cgi?request=1&team_game_max=84&home_away_neutral=h&comp_id=NBA&year_max=2026&team_game_min=1&match=team_game&order_by=pts&comp_type=reg&timeframe=seasons&year_min=2026")

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
    scraper.add_task(login)
    scraper.add_task(scrape_games)
    scraper.add_task(scrape_box_scores)
    # Set headless to false for debugging, true for production
    scraper.scrape(headless=False) 

if __name__ == "__main__":
    main()