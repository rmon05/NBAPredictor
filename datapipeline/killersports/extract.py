import re
import time
from datetime import timedelta
from pathlib import Path
import os

def extract_data(year):
    total_start = time.time()

    input_folder = f"../../data/killersports_html/{year}"
    output_folder = f"../../data/killersports_extracted/{year}"
    output_file = os.path.join(output_folder, "extracted.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        # reset
        pass

    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
            pattern = re.compile(r'<tr class="qry-tr">.*?</tr>', re.DOTALL)
            matches = pattern.findall(content)
        with open(output_file, "a", encoding="utf-8") as f:
            for match in matches:
                f.write(match + "\n")


    total_end = time.time()
    total_seconds = total_end - total_start
    td = timedelta(seconds=total_seconds)
    print(f"Extracted {year} in {td}")


def main():
    for year in range(2015, 2026):
        extract_data(year)


if __name__ == "__main__":
    main()