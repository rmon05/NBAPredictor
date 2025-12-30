from airflow.decorators import dag, task
from datetime import datetime

@dag(
    dag_id="test_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["test"],
)
def hello_world_dag():
    @task.bash
    def hello_world():
        return "echo 'Hello World from Airflow 3!'"
    
    hello_world()

hello_world_dag()