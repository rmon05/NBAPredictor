from datetime import datetime
import os
from pathlib import Path
from airflow.decorators import dag, task
from airflow.providers.standard.operators.bash import BashOperator



APPS_FOLDER = Path("/opt/apps")
SCRAPER_FILE = APPS_FOLDER / "playwright/stathead_scraper.py"

@dag(
    dag_id="stathead_scraper_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["test", "scraper"],
)
def StatheadScraperDAG():
    task = BashOperator(
        task_id="scrape",
        bash_command=f"python {SCRAPER_FILE}",
        # bash_command=f"echo 'Hello Airflow'",
    )

StatheadScraperDAG()