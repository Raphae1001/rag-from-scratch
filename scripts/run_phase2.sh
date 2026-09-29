#!/usr/bin/env bash
# Exécute TOUTE la Phase 2 avec le vrai modèle : installation, corpus, tests, Postgres, embeddings, benchmark.
#   bash scripts/run_phase2.sh               # avec Docker (Docker Desktop doit être lancé)
#   NO_DOCKER=1 bash scripts/run_phase2.sh   # sans Docker : Postgres+pgvector local via `pgserver`
set -euo pipefail
cd "$(dirname "$0")/.."

DB_ARGS=()
if [ "${NO_DOCKER:-0}" = "1" ]; then
  DB_ARGS=(--pgdata data/pgdata)
  echo ">> mode sans Docker (pgserver)"
else
  command -v docker >/dev/null || { echo "ERREUR : Docker introuvable. Installe Docker Desktop, ou lance avec NO_DOCKER=1."; exit 1; }
  docker info >/dev/null 2>&1 || { echo "ERREUR : Docker ne tourne pas. Ouvre Docker Desktop, attends qu'il soit prêt, puis relance."; exit 1; }
fi

echo ">> 1/6 environnement Python"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt pytest rank-bm25
# pgserver est OPTIONNEL (Postgres jetable pour les tests) : s'il n'existe pas pour cette machine, les tests de base
# utilisent le Postgres de Docker dans une base dédiée. On masque donc l'erreur de pip, qui n'en est pas une ici.
if ! pip install -q pgserver >/dev/null 2>&1; then
  echo "   (pgserver indisponible sur cette machine : optionnel, les tests de base utiliseront le Postgres de Docker)"
fi

echo ">> 2/6 corpus"
if [ ! -f data/corpus.jsonl ]; then
  bash scripts/fetch_docs.sh
  python scripts/build_corpus.py
fi

echo ">> 3/6 Postgres + pgvector"
if [ "${NO_DOCKER:-0}" != "1" ]; then
  docker compose up -d --wait
fi

echo ">> 4/6 tests (base dédiée rag_test_pytest : tes données ne sont pas touchées)"
python -m pytest -q

echo ">> 5/6 embeddings (le modèle se télécharge au premier lancement)"
python scripts/embed_corpus.py ${DB_ARGS[@]+"${DB_ARGS[@]}"}

echo ">> 6/6 benchmark"
python scripts/benchmark.py ${DB_ARGS[@]+"${DB_ARGS[@]}"}

echo
echo "=== TERMINÉ : résultats dans results/phase2_benchmark.md (colle-le dans la conversation) ==="
