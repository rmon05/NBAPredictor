import re
import time
from datetime import timedelta
from pathlib import Path
import os
from bs4 import BeautifulSoup

team_abbrv_convert = {
    "BK": "BRK",
    "CHA": "CHO",
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
}

def extract_data(year):
    total_start = time.time()

    input_folder = f"../../data/sportsbookreview_html/{year}"
    output_folder = f"../../data/sportsbookreview_extracted/{year}"
    output_file = os.path.join(output_folder, "extracted.csv")
    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write("gid,Opp,Team,Date,Betway,Sports Interaction,bet365,Pinnacle,Tonybet,Betano,Caesars\n")

    # No data pre 2020 currently
    if year < 2020:
        return
    
    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        with open(input_file, "r", encoding="utf-8") as fin:
            content = fin.read()
            soup = BeautifulSoup(content, "html.parser")

            # match game id
            game_id_pattern = re.compile(
                r'<div id="game-([0-9]+)">',
                re.DOTALL
            )
            game_ids = game_id_pattern.findall(content)

            # match game block for rest of data to be isolated in
            for gid in game_ids:
                game_block = soup.find("div", id=f"game-{gid}")

                # match team names
                team_links = game_block.find_all(
                    "a",
                    href=f"/scores/nba-basketball/matchup/{gid}/",
                    class_="d-flex overflow-hidden align-items-center fw-bold fs-9"
                )
                team_abbrvs = [link.get_text(strip=True) for link in team_links]
                if len(team_abbrvs) != 2:
                    print(f"ERROR! team_abbrvs does not have length 2")
                    for team in team_abbrvs:
                        print(team)         
                    print("\n")
                    continue
                else:
                    team1 = team_abbrvs[0]
                    team2 = team_abbrvs[1]  
                    if team1 in team_abbrv_convert.keys():
                        team1 = team_abbrv_convert[team1]
                    if team2 in team_abbrv_convert.keys():
                        team2 = team_abbrv_convert[team2]
                
                # match sportsbooks
                books_links = game_block.find_all(
                    "a",
                    attrs={"data-aatracker": lambda x: x and "Odds Table - Odds Cell CTA" in x}
                )

                # match odds for each book
                books_data = {
                    "betway" : {},
                    "sportsinteraction" : {},
                    "bet365" : {},
                    "pinnacle" : {},
                    "tonybet" : {},
                    "betano" : {},
                    "caesars" : {},
                }
                for link in books_links:
                    # get book name
                    tracker = link.get("data-aatracker", "")
                    book_name = tracker.split(" - ")[-1].strip()
                    
                    # match odds blocks with class "OddsTableMobile_odds__thxLF"
                    odds_divs = link.find_all("div", class_="OddsTableMobile_odds__thxLF")   
                    if len(odds_divs) == 2:
                        # away team is listed first
                        div_away = odds_divs[0]
                        spans_away = div_away.find_all("span")
                        if len(spans_away) == 2:
                            spread = spans_away[0].get_text(strip=True)
                            odds = spans_away[1].get_text(strip=True)
                            books_data[book_name]["away_spread"] = spread
                            books_data[book_name]["away_odds"] = odds
                        else:
                            print("Malformed away spans!")

                        # home team is listed second
                        div_home = odds_divs[1]
                        spans_home = div_home.find_all("span")
                        if len(spans_home) == 2:
                            spread = spans_home[0].get_text(strip=True)
                            odds = spans_home[1].get_text(strip=True)
                            books_data[book_name]["home_spread"] = spread
                            books_data[book_name]["home_odds"] = odds
                        else:
                            print("Malformed home spans!")

                # Write out
                date = filename.split(".")[0]
                with open(output_file, "a", encoding="utf-8") as fout:
                    fout.write(gid + "," + team1 + "," + team2 + "," + date)
                    for book in books_data.keys():
                        if books_data[book] != {}:
                            fout.write("," + books_data[book]["home_spread"])
                        else:
                            fout.write(",")
                    fout.write("\n")

    total_end = time.time()
    total_seconds = total_end - total_start
    td = timedelta(seconds=total_seconds)
    print(f"Extracted {year} in {td}")


def main():
    for year in range(2015, 2026):
        extract_data(year)


if __name__ == "__main__":
    main()