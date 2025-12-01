FROM apache/airflow:3.1.3-python3.11

USER root

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean


COPY requirements.txt requirements.txt
COPY ./airflow /opt/airflow
RUN chown -R airflow: /opt/airflow
RUN chown -R airflow: requirements.txt

USER airflow
RUN pip install --no-cache-dir -r requirements.txt