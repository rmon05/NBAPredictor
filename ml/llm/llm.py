from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import json
from pathlib import Path

load_dotenv()


# Paths
data_dir = Path(__file__).parent / "../../data"
ml_dir = Path(__file__).parent / "../ml/llm"

train_X_parquet_file = data_dir / f"normalized/parquet/gamesTrainX.parquet"
test_X_parquet_file = data_dir / f"normalized/parquet/gamesTestX.parquet"
train_y_parquet_file = data_dir / f"normalized/parquet/gamesTrainY.parquet"
test_y_parquet_file = data_dir / f"normalized/parquet/gamesTestY.parquet"
X_train = pd.read_parquet(train_X_parquet_file)
X_test = pd.read_parquet(test_X_parquet_file)
y_train = pd.read_parquet(train_y_parquet_file)
y_test = pd.read_parquet(test_y_parquet_file)


# -------------------------
# Convert to JSONL for OpenAI fine-tuning
# -------------------------
jsonl_lines = []
columns_team1 = [
    "WinPct1","LocationWinPct1","PTS1","FG1","FGA1","2P1","2PA1","3P1","3PA1",
    "FT1","FTA1","PTSDiff1","Streak1","AST1","ORB1","DRB1","STL1","BLK1","TOV1"
]
columns_team2 = [
    "WinPct2","LocationWinPct2","PTS2","FG2","FGA2","2P2","2PA2","3P2","3PA2",
    "FT2","FTA2","PTSDiff2","Streak2","AST2","ORB2","DRB2","STL2","BLK2","TOV2"
]

for idx, row in X_train.iterrows():
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
    
    # Corresponding label / target
    y_row = y_train.iloc[idx]
    winner = "Team1" if y_row["Result"] > 0 else "Team2"
    score_margin = float(y_row["Result"])
    
    # Build JSONL message
    prompt_text = (
        f"Predict the NBA game outcome of Team1 and Team2 given these scaled stats:\n"
        f"{home_team} has home court advantage\n\n"
        f"\n".join(team1_stats) + "\n\n"
        f"\n".join(team2_stats) + "\n\n"
        f"Respond with JSON: {{\"winner\": \"...\", \"score_margin_for_team_1\": ...}}"
    )
    
    completion_text = json.dumps({
        "winner": winner,
        "score_margin_for_team_1": score_margin
    })
    
    jsonl_lines.append({
        "messages": [
            {"role": "system", "content": "You are an NBA game prediction model."},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": completion_text}
        ]
    })

# -------------------------
# Save JSONL to file
# -------------------------
jsonl_file = ml_dir / "nba_train.jsonl"
with open(jsonl_file, "w", encoding="utf-8") as f:
    for item in jsonl_lines:
        f.write(json.dumps(item) + "\n")

print(f"JSONL fine-tuning data saved to {jsonl_file}")

# -------------------------
# Upload and create fine-tuning job
# -------------------------
client = OpenAI()  # make sure OPENAI_API_KEY is set in environment

# Upload the file first
uploaded_file = client.files.create(
    file=open(jsonl_file, "rb"),
    purpose="fine-tune"
)

print("Uploaded file id:", uploaded_file.id)

# Start fine-tuning job
fine_tune_job = client.fine_tuning.jobs.create(
    training_file=uploaded_file.id,
    model="gpt-4.1-mini-2025-04-14"  # fine-tunable GPT-4.1 variant
)

print("Fine-tuning job started:", fine_tune_job)

