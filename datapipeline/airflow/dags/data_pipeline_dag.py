from datetime import datetime
import os
from pathlib import Path
from airflow.decorators import dag, task



APPS_FOLDER = Path("/opt/apps")
STATHEAD_SCRAPER_FILE = APPS_FOLDER / "playwright/stathead_scraper.py"
NBA_SCRAPER_FILE = APPS_FOLDER / "playwright/nba_live_scraper.py"
RAW_TO_CLEAN_FILE = APPS_FOLDER / "core/raw_to_clean.py"
CLEAN_TO_JOINED_FILE = APPS_FOLDER / "core/clean_to_joined.py"

@dag(
    dag_id="data_pipeline_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["test", "scraper"],
)
def stathead_scraper_dag():
    # Use airflow 3 style
    @task.bash
    def scrape_stathead():
        return f"python {STATHEAD_SCRAPER_FILE}"

    @task.bash
    def scrape_nba():
        return f"python {NBA_SCRAPER_FILE}"

    @task.bash
    def raw_to_clean():
        return f"python {RAW_TO_CLEAN_FILE}"

    @task.bash
    def clean_to_joined():
        return f"python {CLEAN_TO_JOINED_FILE}"

    scrape_stathead() >> scrape_nba() >> raw_to_clean() >> clean_to_joined()

stathead_scraper_dag()