from pathlib import Path
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

# THIS FILE IS TBD!

data_dir = Path(__file__).parent / "../data"

class PlayerStrengthCalculator:
    def __init__(self, df_players, df_games):
        self.df_games = df_games
        self.players = sorted(list(df_players["Player-additional"].unique()))
        self.ind_of_player = {player : ind for ind, player in enumerate(self.players)}
        self.game_dates = sorted(list(df_games["Date"].unique()))
        self.next_date_ind = 0
        self.input_rows_X = []
        self.results_y = []

    def process_day(self):
        if self.next_date_ind < len(self.game_dates):
            for _, row in self.df_games[self.df_games["Date"] == self.game_dates[self.next_date_ind]].iterrows():
                # -2 for home court (always 1), -1 for rest differential
                has_rest_data = not (pd.isna(row["RestHome"]) or pd.isna(row["RestOpp"]))
                if has_rest_data:
                    row_X = np.zeros(len(self.players)+2)
                    home_box_scores = row["HomeBoxScores"]
                    away_box_scores = row["AwayBoxScores"]

                    # Box Score Players
                    for i in range(len(home_box_scores)):
                        player_team = home_box_scores[i]
                        pid_team = player_team["Player-additional"]
                        player_ind_team = self.ind_of_player[pid_team]
                        row_X[player_ind_team] = 1
                    for i in range(len(away_box_scores)):
                        player_opp = away_box_scores[i]
                        pid_opp = player_opp["Player-additional"]
                        player_ind_opp = self.ind_of_player[pid_opp]
                        row_X[player_ind_opp] = -1


                    row_X[-2] = 1
                    row_X[-1] = row["RestHome"] - row["RestOpp"]
                    self.input_rows_X.append(row_X)
                    self.results_y.append(row["Result"])
            self.next_date_ind += 1
            return True
        else:
            return False
        
    def calc_team_strengths(self):
        # Fit ridge only on current batch of games
        X = np.vstack(self.input_rows_X)
        y = np.array(self.results_y)    

        # use low alpha for better fit
        alpha = 0.1
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X, y)
        coefs = model.coef_

        team_strengths = coefs[:30]
        team_strengths_str = {team : team_strengths[self.ind_of_team[team]] for team in self.teams}
        home_adv = coefs[30]
        rest_effect = coefs[31]

        # for i in range(30):
        #     print(f"Team {self.teams[i]} has strength: {team_strengths[i]}")
        # print(f"\nEstimated Home-Court Advantage: {home_adv:.2f}")
        # print(f"Estimated Rest Effect (per day): {rest_effect:.2f}")
        return team_strengths_str, home_adv, rest_effect

    def get_next_date(self):
        return self.game_dates[self.next_date_ind]
        
    

# Below for test purposes only, the main workflow will import TeamStrengthCalculator
def process_year(year):
    joined_games_path = data_dir / f"joined/parquet/{year}/gamesJoined.parquet"
    df_games = pd.read_parquet(joined_games_path)

    # Test
    tsc = TeamStrengthCalculator(df_games)
    while tsc.process_day():
        pass
    tsc.calc_team_strengths()


def main():
    years = [year for year in range(2025, 2026)]

    # Process data per year
    for year in years:
        start_year = time.time()
        process_year(year)
        end_year = time.time()
        print(f"Finished window process for year {year} | Total time: {end_year - start_year:.2f}s\n")
        

if __name__ == "__main__":
    main()