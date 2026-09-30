# Image de l'API RAG (Phase 5). Lourde (~2 Go, torch inclus) : voir requirements.txt.
# La base Postgres reste un service séparé (docker-compose.yml) — pas de DB dans cette image.
FROM python:3.12-slim

WORKDIR /app

# Dépendances d'abord (couche cache Docker séparée du code, qui change plus souvent).
# Timeout pip relevé : torch (~800 Mo) peut dépasser le timeout par défaut (15s) sur un réseau lent.
ENV PIP_DEFAULT_TIMEOUT=120
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY data/corpus.jsonl data/corpus.jsonl
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
