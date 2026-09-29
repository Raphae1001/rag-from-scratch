"""Utilitaires partagés par les scripts (chemins, corpus, set d'évaluation, connexion Postgres)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "data" / "corpus.jsonl"
EVAL = ROOT / "data" / "eval" / "queries.json"


def load_corpus() -> list[dict]:
    return [json.loads(l) for l in CORPUS.open(encoding="utf-8")]


def load_eval() -> list[dict]:
    return json.loads(EVAL.read_text(encoding="utf-8"))


def doc_key(doc: dict) -> str:
    return f"{doc['path']}#{doc['section']}"


def add_db_args(parser):
    parser.add_argument("--dsn", help="URL Postgres (défaut : $DATABASE_URL ou le conteneur Docker du docker-compose)")
    parser.add_argument("--pgdata", help="DEV SANS DOCKER : dossier d'un Postgres+pgvector local géré par le paquet pip `pgserver`")


def get_dsn(args) -> str | None:
    if args.pgdata:
        import pgserver
        return pgserver.get_server(args.pgdata, cleanup_mode="stop").get_uri()
    return args.dsn
