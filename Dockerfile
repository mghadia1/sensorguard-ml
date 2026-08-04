FROM python:3.12-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir \
        "joblib>=1.4" \
        "numpy>=2.0" \
        "pandas>=2.2" \
        "scikit-learn>=1.6" \
        "xgboost-cpu>=3.0,<4" \
    && pip install --no-cache-dir --no-deps .

ENTRYPOINT ["sensorguard"]
CMD ["--help"]
