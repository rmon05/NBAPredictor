import os
import time
import pandas as pd
import psycopg2
from dotenv import load_dotenv

print("Started loading data for 2015 to 2025 seasons")
start_time = time.time()

load_dotenv()
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
BASE_DIR = os.path.dirname(__file__)
GAME_TYPE = "Regular" # only have regular season data for now (sample)


# DB Connection
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

# Load teams (same 30 teams over last 10 years)
csv_path = os.path.join(BASE_DIR, 'shared/teamsRaw.csv')
df = pd.read_csv(csv_path)
for _, row in df.iterrows():
    cur.execute(
        """
        INSERT INTO teams (tid, tname)
        VALUES (%s, %s)
        """,
        (row["Abbreviation"], row["Team"])
    )


# will map game_date, tid to its unique gid to be used in box score later
gid_index = 1
gid_of = dict()
pid_exists = dict()

# Load data for last 10 years
for season in range(2015, 2026, 1):
    print(f"Started loading data for season: {season}")
    SEASON_DIR = BASE_DIR + f"/{season}"

    # Load players + player team relationships
    csv_path = os.path.join(SEASON_DIR, 'playersRaw.csv')
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        # Only add player if unadded before
        if row["Player-additional"] not in pid_exists.keys():
            cur.execute(
                """
                INSERT INTO players (pid, pname)
                VALUES (%s, %s)
                """,
                (row["Player-additional"], row["Player"])
            )
            pid_exists[row["Player-additional"]] = True

        # players may be traded mid season so may play for 2+ teams in season
        teams = row["Team"]
        team_ind = 0
        while team_ind < len(teams):
            cur.execute(
                """
                INSERT INTO plays_for (pid, tid, season)
                VALUES (%s, %s, %s)
                """,
                (row["Player-additional"], teams[team_ind:team_ind+3], season)
            )
            team_ind += 3


    # Load games
    csv_path = os.path.join(SEASON_DIR, 'gamesRaw.csv')
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        # Team = Home team, Opp = Away team
        winner = row["Team"] if row["Result"] == "W" else row["Opp"]
        cur.execute(
            """
            INSERT INTO games (season, game_date, tid_home, tid_away, winner, game_type)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (season, row["Date"], row["Team"], row["Opp"], winner, GAME_TYPE)
        )

        gid_of[(row["Date"], row["Team"])] = gid_index
        gid_of[(row["Date"], row["Opp"])] = gid_index
        gid_index +=1

    # Load box scores
    csv_path = os.path.join(SEASON_DIR, 'boxScoreClean.csv')
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        cur.execute(
            """
            INSERT INTO box_score (
                pid, gid, mins, pts, reb, ast, stl, blk, tov,
                fta, ft, fga, fg, fg3a, fg3, plus_minus
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["Player-additional"],
                gid_of[(row["Date"], row["Team"])],
                row["MP"],
                row["PTS"],
                row["TRB"],
                row["AST"],
                row["STL"],
                row["BLK"],
                row["TOV"],
                row["FTA"],
                row["FT"],
                row["FGA"],
                row["FG"],
                row["3PA"],
                row["3P"],
                row["+/-"]
            )
        )
    print(f"Finished loading data for season: {season}")

conn.commit()
cur.close()
conn.close()


# End time tracking
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)
print(f"\nDone Loading! Total time elapsed: {int(mins)} minutes and {int(secs)} seconds.")
