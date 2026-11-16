FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORTAL_DATA_DIR=/data

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY training_portal /app/training_portal
COPY samples /app/samples

RUN mkdir -p /data && useradd --create-home --uid 10001 portal \
    && chown -R portal:portal /data /app

USER portal

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=4)"

CMD ["uvicorn", "training_portal.main:app", "--host", "0.0.0.0", "--port", "8000"]
