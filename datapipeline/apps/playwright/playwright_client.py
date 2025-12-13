from playwright.sync_api import sync_playwright

class PWScraper:
    def __init__(self):
        # list of functions to run in a new tab
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def scrape(self, headless=True):
        with sync_playwright() as p:
            # Boot up browser and new session
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context()

            # Run tasks
            for task in self.tasks:
                task(context)

            browser.close()

