import os
import time
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

def clean_boxscore(season):
    input_file = os.path.join(BASE_DIR, f'{season}/boxScoreRaw.csv')
    output_file = os.path.join(BASE_DIR, f'{season}/boxScoreClean.csv')

    # 1: Read file lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 2: Find the header line index
    header_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Rk,"):
            header_index = i
            break

    if header_index is None:
        raise ValueError("Could not find header line starting with 'Rk'")

    # 3: Load CSV starting from header
    df = pd.read_csv(input_file, skiprows=header_index)

    # 4: Remove rows where Rk is not a number (non-data junk)
    df = df[df['Rk'].apply(lambda x: str(x).isdigit())]

    # 5: Drop duplicates based on Rk
    df = df.drop_duplicates(subset='Rk')

    # 6: Drop duplicates based on KEY
    # IDENTIFY the duplicates first (submit ticket)
    duplicates = df[df.duplicated(subset=['Player-additional', 'Date'], keep='first')]
    if len(duplicates):
        print(f"Found {len(duplicates)} duplicate rows in season: {season}")
        print(duplicates[['Player-additional', 'Date']])

    before = len(df)
    df = df.drop_duplicates(subset=['Player-additional', 'Date'])
    after = len(df)
    if before-after:
        print(f"Dropped {before-after} duplicate rows based on KEY in season: {season}.")

    # 7: Reset Rk to sequential values starting from 1
    df = df.reset_index(drop=True)
    df['Rk'] = range(1, len(df) + 1)

    # 8: Write cleaned file
    df.to_csv(output_file, index=False)


# Start time tracking
print("Started cleaning data for 2015 to 2025 seasons")
start_time = time.time()

for season in range(2015, 2026):
    clean_boxscore(season)

# End time tracking
end_time = time.time()
elapsed = end_time - start_time
print(f"\nDone Cleaning! Total time elapsed: {elapsed:.3f} seconds.")