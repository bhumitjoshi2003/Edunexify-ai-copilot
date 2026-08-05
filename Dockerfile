FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8090

# Must bind 0.0.0.0, not 127.0.0.1 — the host/other containers can't reach a
# loopback-only bind from outside this container's network namespace.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
