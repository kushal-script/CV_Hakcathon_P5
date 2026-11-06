FROM rust:1.99-slim-bookworm AS gateway
WORKDIR /build
COPY gateway/Cargo.toml gateway/Cargo.lock ./
COPY gateway/src ./src
RUN cargo build --release --locked

FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt && python -c "from rapidocr import RapidOCR; RapidOCR()"

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .
COPY app ./app
COPY configs ./configs
COPY models ./models
COPY web ./web
COPY --from=gateway /build/target/release/vahana-gateway /usr/local/bin/vahana-gateway
COPY deploy/entrypoint.sh /usr/local/bin/entrypoint
RUN mkdir -p /app/data && chown app:app /app/data && rm -rf /app/web/_dev

ENV P5_DATA_DIR=/app/data P5_WEB_DIR=/app/web P5_CONFIG=/app/configs/app.yaml P5_INVESTIGATION_CONFIG=/app/configs/investigation.yaml P5_ALLOWED_ORIGINS="" P5_UPSTREAM=http://127.0.0.1:8001 P5_LISTEN=0.0.0.0:8080 P5_USERS_FILE=/run/secrets/users.json
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health')"
ENTRYPOINT ["entrypoint"]
