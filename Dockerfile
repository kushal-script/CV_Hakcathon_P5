FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 app
WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app src ./src
RUN pip install --no-deps .
COPY --chown=app:app app ./app
COPY --chown=app:app configs ./configs
COPY --chown=app:app models ./models

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
