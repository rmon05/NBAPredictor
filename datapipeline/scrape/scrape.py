from datetime import date, timedelta
import time
import pyautogui
import pyperclip
import re

def extract_html(year, date, query):
    # switch to screen
    pyautogui.hotkey('alt', 'tab')
    time.sleep(0.1)

    # query box
    pyautogui.click(330, 550)
    time.sleep(0.1)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.1)

    # paste query and enter
    pyperclip.copy(query)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)
    pyautogui.hotkey("enter")
    time.sleep(10)


    for i in range(2):
        # html
        pyautogui.click(1170, 215)
        time.sleep(0.5)

        # dropdown
        pyautogui.click(1132, 210)
        time.sleep(0.5)

        # copy & copy element
        pyautogui.click(1225, 481)
        time.sleep(0.5)
        pyautogui.click(1566, 488)
        time.sleep(0.5)

    # Write out
    pyautogui.hotkey('alt', 'tab')
    clipboard_content = pyperclip.paste()
    with open(f"../../data/killersports_html/{year}/{date}.txt", "w", encoding="utf-8") as f:
        f.write(clipboard_content)
    time.sleep(0.1)
    
    # Check
    with open(f"../../data/killersports_html/{year}/{date}.txt", "r", encoding="utf-8") as f:
        content = f.read()
        if not ('<tr class="qry-tr">' in content) and not  ('No Results Found' in content):
            # refresh and try again
            pyautogui.hotkey('alt', 'tab')
            time.sleep(0.1)
            pyautogui.hotkey('ctrl', 'r')
            time.sleep(0.1)
            pyautogui.hotkey('alt', 'tab')
            time.sleep(0.1)
            extract_html(year, date, query)

    # print
    print(f"Finished: {query}")

def extract_data():
    input_file = "raw.txt"
    output_file = "extracted.txt"

    # Read input file
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Regex pattern to match: <tr class="qry-tr"> ... </tr>
    pattern = re.compile(r'<tr class="qry-tr">.*?</tr>', re.DOTALL)
    matches = pattern.findall(text)

    # Write matches to output file
    with open(output_file, "a", encoding="utf-8") as f:
        for match in matches:
            f.write(match + "\n")

    # Clear the original file
    open(input_file, "w").close()


start_end_dates = {
    "2015": [date(2014, 10, 28), date(2015, 4, 15)],
    "2016": [date(2015, 10, 27), date(2016, 4, 13)],
    "2017": [date(2016, 10, 25), date(2017, 4, 12)],
    "2018": [date(2017, 10, 17), date(2018, 4, 11)],
    "2019": [date(2018, 10, 16), date(2019, 4, 10)],
    "2020": [date(2019, 10, 22), date(2020, 8, 14)],
    "2021": [date(2020, 12, 22), date(2021, 5, 16)],
    "2022": [date(2021, 10, 19), date(2022, 4, 10)],
    "2023": [date(2022, 10, 18), date(2023, 4, 9)],
    "2024": [date(2023, 10, 24), date(2024, 4, 14)],
    "2025": [date(2024, 10, 22), date(2025, 4, 13)],
}
def main():
    print("Preparing to scrape in 5 seconds!")
    time.sleep(5)

    total_start = time.time()

    # Iterate through all days
    # NOTE start flow on raw.txt
    for year, (start_date, end_date) in start_end_dates.items():
        current = start_date
        while current <= end_date:
            yyyymmdd = int(current.strftime("%Y%m%d"))
            query = f"date={yyyymmdd} and site=home"
            extract_html(year, current, query)
            current += timedelta(days=1)

    total_end = time.time()
    total_seconds = total_end - total_start
    td = timedelta(seconds=total_seconds)
    print(f"Scraped {start_date} to {end_date} in {td}")


if __name__ == "__main__":
    main()
