import pandas as pd
import json
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm  # optional, for progress bar

load_dotenv()

# -------------------------
# Configuration
# -------------------------
data_dir = Path(__file__).parent / "../../data"
ml_dir = Path(__file__).parent / "../ml/llm"

test_X_parquet_file = data_dir / "llm/parquet/gamesTestX.parquet"
test_y_parquet_file = data_dir / "llm/parquet/gamesTestY.parquet"

MODEL_NAME = "gpt-4.1-mini-2025-04-14"  # replace with your fine-tuned model ID

# -------------------------
# Load test data
# -------------------------
X_test = pd.read_parquet(test_X_parquet_file)
y_test = pd.read_parquet(test_y_parquet_file)

client = OpenAI()

mse_list = []
winner_correct_list = []
predictions = []

# -------------------------
# Loop over each test row
# -------------------------
for idx, row in tqdm(X_test.iterrows(), total=20, desc="Testing"):
    home_team = "Team1" if row["Home"] == 1 else "Team2"
    team1_loc = "home" if row["Home"] == 1 else "away"
    team2_loc = "away" if row["Home"] == 1 else "home"
    team1_streak = "winning" if row["Streak1"] > 0 else "losing"
    team2_streak = "winning" if row["Streak2"] > 0 else "losing"

    # Team 1 stats
    team1_stats = [
        f"Team1 has a {row['WinPct1']:.5f} overall win percentage",
        f"Team1 has a {row['LocationWinPct1']:.5f} win percentage as the {team1_loc} team",
        f"Team1 is on a {row['Streak1']} game {team1_streak} streak",
        f"Team1 scores {row['PTS1']} points per game",
        f"Team1 makes on average {row['FG1']} field goals out of {row['FGA1']} attempts per game ",
        f"Team1 makes on average {row['2P1']} two-pointers out of {row['2PA1']} attempts per game",
        f"Team1 makes on average {row['3P1']} three-pointers out of {row['3PA1']} attempts per game",
        f"Team1 makes on average {row['FT1']} free throws out of {row['FTA1']} attempts per game",
        f"Team1 has an average point differential of {row['PTSDiff1']} per game",
        f"Team1 averages {row['AST1']} assists per game",
        f"Team1 averages {row['ORB1']} offensive rebounds and {row['DRB1']} defensive rebounds per game",
        f"Team1 averages {row['STL1']} steals and {row['BLK1']} blocks per game",
        f"Team1 averages {row['TOV1']} turnovers per game"
    ]

    # Team 2 stats
    team2_stats = [
        f"Team2 has a {row['WinPct2']:.5f} overall win percentage",
        f"Team2 has a {row['LocationWinPct2']:.5f} win percentage as the {team2_loc} team",
        f"Team2 is on a {row['Streak2']} game {team2_streak} streak",
        f"Team2 scores {row['PTS2']} points per game",
        f"Team2 makes on average {row['FG2']} field goals out of {row['FGA2']} attempts per game",
        f"Team2 makes on average {row['2P2']} two-pointers out of {row['2PA2']} attempts per game",
        f"Team2 makes on average {row['3P2']} three-pointers out of {row['3PA2']} attempts per game",
        f"Team2 makes on average {row['FT2']} free throws out of {row['FTA2']} attempts per game",
        f"Team2 has an average point differential of {row['PTSDiff2']} per game",
        f"Team2 averages {row['AST2']} assists per game",
        f"Team2 averages {row['ORB2']} offensive rebounds and {row['DRB2']} defensive rebounds per game",
        f"Team2 averages {row['STL2']} steals and {row['BLK2']} blocks per game",
        f"Team2 averages {row['TOV2']} turnovers per game"
    ]
    
    # Build JSONL message
    prompt_text = (
        f"Predict the NBA game outcome of Team1 and Team2 given these scaled stats:\n"
        f"{home_team} has home court advantage\n\n"
        f"\n".join(team1_stats) + "\n\n"
        f"\n".join(team2_stats) + "\n\n"
        f"Respond with JSON: {{\"winner\": \"...\", \"score_margin_for_team_1\": ...}}"
    )

    # Call the fine-tuned model
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0
        )
        output_text = resp.choices[0].message.content
        output_json = json.loads(output_text)
        predicted_margin = float(output_json["score_margin_for_team_1"])
        predicted_winner = output_json["winner"]
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        predicted_margin = np.nan
        predicted_winner = None

    # True data
    true_margin = float(y_test.iloc[idx]["Result"])
    true_winner = "Team1" if true_margin > 0 else "Team2"

    # Track MSE
    if not np.isnan(predicted_margin):
        mse_list.append((predicted_margin - true_margin) ** 2)

    # Track winner accuracy
    winner_correct_list.append(predicted_winner == true_winner)

    predictions.append({
        "predicted_margin": predicted_margin,
        "true_margin": true_margin,
        "predicted_winner": predicted_winner,
        "true_winner": true_winner
    })

    # Optional: print running stats every 5 rows
    if (idx + 1) % 10 == 0:
        running_mse = np.mean(mse_list)
        running_acc = np.mean(winner_correct_list)
        print(f"Processed {idx+1}/{len(X_test)} rows, running MSE: {running_mse:.4f}, accuracy: {running_acc:.4f}")

    if idx > 20:
        break

# Final metrics
final_mse = np.mean(mse_list)
final_acc = np.mean(winner_correct_list)
print(f"\nFinal MSE over {len(mse_list)} predictions: {final_mse:.4f}")
print(f"Final winner accuracy: {final_acc:.4f}")

# Save predictions for analysis
pred_df = pd.DataFrame(predictions)
pred_df.to_csv(ml_dir / "llm_test_predictions.csv", index=False)