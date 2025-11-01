import os


def check(year):
    folder_path = f"../../data/killersports_html/{year}"
    good = 0
    bad = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if not ('<tr class="qry-tr">' in content) and not  ('No Results Found' in content):
                print(f"❌ '{filename}' does NOT contain results")
                bad += 1
            else:
                good += 1
    print(f"{good} files are good, {bad} files are bad for year: {year}\n")

def main():
    for year in range(2015, 2026):
        check(year)


if __name__ == "__main__":
    main()
