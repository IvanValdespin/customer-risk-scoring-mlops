FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8080

CMD ["python","-m","uvicorn","src.api.main:app","--host","0.0.0.0","--port","8080"]