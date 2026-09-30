#!/usr/bin/env bash
# Point d'entrée du conteneur API : charge les embeddings s'ils manquent, puis démarre l'API.
# `docker compose up` doit répondre à une requête sans étape manuelle (spec Phase 5).
set -euo pipefail
cd /app

python scripts/ensure_embedded.py
exec uvicorn rag.api:app --host 0.0.0.0 --port 8000
