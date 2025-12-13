FROM apache/airflow:3.1.3-python3.11

USER root

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        libglib2.0-0 \
        libnspr4 \
        libnss3 \
        libdbus-1-3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libatspi2.0-0 \
        libxcomposite1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxrandr2 \
        libgbm1 \
        libxkbcommon0 \
        libasound2 \
        wget curl && \
    apt-get clean


COPY requirements.txt requirements.txt
COPY ./apps /opt/apps

USER airflow
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m playwright install chromium
