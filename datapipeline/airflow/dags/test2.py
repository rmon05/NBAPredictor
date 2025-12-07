from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

with DAG(
    dag_id="example_dag_2",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    task = BashOperator(
        task_id="print_hello",
        bash_command="echo 'Hello Airflow Again!'"
    )
