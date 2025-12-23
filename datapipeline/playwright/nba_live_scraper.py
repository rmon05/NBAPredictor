from playwright_client import PWScraper
from pathlib import Path

DATA_DIR = Path("E:/NBAPredictor/data")
RAW_OUTPUT_DIR = DATA_DIR / "nba_live_extracted"

nba_to_stathead_abbrv = {
    "BKN": "BRK",
    "CHA": "CHO",
    "PHX": "PHO",
}

def scrape(browser_context):
    # Clear existing games
    for prev_game_file in RAW_OUTPUT_DIR.iterdir():
        prev_game_file.unlink()

    page = browser_context.new_page()
    page.goto("https://www.nba.com/players/todays-lineups")

    # Find all games (1 per div)
    page.wait_for_selector("div[class*='DailyLineup_dl'][class*='LineupsView_game']")
    game_divs = page.locator("div[class*='DailyLineup_dl'][class*='LineupsView_game']")    
    for i in range(game_divs.count()):
        game_div = game_divs.nth(i)
        # Extract the matchup header
        matchup_header = game_div.locator("h1[class*='Text_text'][class*='Block_blockTitleText']")
        teams = matchup_header.text_content().upper().split("VS")
        away_nba = teams[0].strip()
        home_nba = teams[1].strip()
        away_stathead = nba_to_stathead_abbrv[away_nba] if away_nba in nba_to_stathead_abbrv.keys() else away_nba
        home_stathead = nba_to_stathead_abbrv[home_nba] if home_nba in nba_to_stathead_abbrv.keys() else home_nba
        
        # Write one file per game
        lineups_file = f"{away_stathead}_AT_{home_stathead}.txt"
        with open(RAW_OUTPUT_DIR / lineups_file, "w", encoding="utf-8") as f:
            f.write(away_stathead + "\n")
            # Extract away lineups
            away_lineup = game_div.locator("span[class*='DailyLineup_dlName']")
            for j in range(5):
                f.write(away_lineup.nth(j).text_content() + "\n")
            
            # Click on home team button for next run
            page.click(f"button:has-text('{home_nba}')")
            
            f.write(home_stathead + "\n")
            # Extract home lineups
            home_lineup = game_div.locator("span[class*='DailyLineup_dlName']")
            for j in range(5):
                f.write(home_lineup.nth(j).text_content() + "\n")




   
        





def main():
    scraper = PWScraper()
    scraper.add_task(scrape)
    # Set headless to false for debugging, true for production
    scraper.scrape(headless=False) 

if __name__ == "__main__":
    main()