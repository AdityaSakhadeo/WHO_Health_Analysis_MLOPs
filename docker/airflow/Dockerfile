FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

# Install project dependencies (repo will be mounted at runtime; this is only for pip)
# We keep this tolerant: if requirements.txt changes, rebuild the image.
COPY requirements.txt /opt/airflow/project-requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/project-requirements.txt

