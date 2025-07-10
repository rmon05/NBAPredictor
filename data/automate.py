import pyautogui
import time
import random

def click_near(x, y, x_range=5, y_range=0):
    rand_x = x + random.randint(-x_range, x_range)
    rand_y = y + random.randint(-y_range, y_range)
    pyautogui.click(rand_x, rand_y)

def perform_iteration(pageNum):
    click_near(2097, 735, x_range=0) # widescreen

    click_near(337, 693, x_range=5)     # Export Data
    time.sleep(0.5)

    click_near(351, 806, x_range=15)     # Get Table as CSV
    time.sleep(0.5)

    click_near(1556, 738, x_range=5, y_range=5)  # Click Off
    time.sleep(0.5)

    pyautogui.hotkey('ctrl', 'a')  # Select all
    time.sleep(0.5)

    pyautogui.hotkey('ctrl', 'c')  # Copy
    time.sleep(0.5)

    pyautogui.hotkey('alt', 'tab')  # Switch to .csv file
    time.sleep(0.5)

    pyautogui.hotkey('ctrl', 'v')  # Paste
    time.sleep(0.5)

    pyautogui.hotkey('alt', 'tab')  # Switch back to sports reference
    time.sleep(0.5)

    for _ in range(10):  # Scroll down
        pyautogui.scroll(-1000)
        time.sleep(0.02)

    for _ in range(23):  # Scroll up 1 tick from bottom
        pyautogui.press('up')

    if pageNum:
        click_near(300, 244, x_range=10)  # previous page, THEN next page
    else:
        click_near(100, 244, x_range=10)  # next page (CHANGE 100 BACK TO 300)
    time.sleep(10)  # wait to load


# Optional: Give yourself time to switch to the right starting window
print("Starting in 10 seconds...")
time.sleep(10)

pyautogui.hotkey('alt', 'tab')  # Switch window initially

# Start time tracking
start_time = time.time()

# Run the loop 150 times
for i in range(150):
    print(f"Iteration {i+1}")
    perform_iteration(i)
    time.sleep(1)

# End time tracking
end_time = time.time()
elapsed = end_time - start_time
# Print elapsed time in minutes and seconds
mins, secs = divmod(elapsed, 60)
print(f"\nDone Automated Extraction! Total time elapsed: {int(mins)} minutes and {int(secs)} seconds.")
