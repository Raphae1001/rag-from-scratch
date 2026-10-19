"""Utilitaires partagés par les scripts (chemins, corpus, set d'évaluation, connexion Postgres)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CORPUS = ROOT / "data" / "corpus.jsonl"
EVAL = ROOT / "data" / "eval" / "queries.json"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


def safe_generate_answer(client, query: str, passages: list[tuple[str, str]]) -> dict:
    """Comme rag.generate.generate_answer, mais ne lève jamais : une réponse mal formée (ex. le modèle pose
    une question de clarification en texte libre au lieu de répondre en JSON, observé en pratique sur une
    requête courte style mot-clé) devient un cas "error" isolé plutôt que de faire planter toute une mesure
    par lots (phase3_check.py, phase4_check.py) sur une seule requête problématique."""
    from rag.generate import generate_answer
    try:
        return generate_answer(client, query, passages)
    except ValueError as e:
        return {"answerable": None, "answer": None, "citations": [], "error": str(e)}


def make_parser(doc: str | None) -> argparse.ArgumentParser:
    """Parser commun : la docstring du script devient l'aide, mise en forme telle quelle."""
    return argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)


def add_embedder_args(parser, fake_help: str = "embeddings factices : plomberie seulement, résultats NON valides"):
    parser.add_argument("--fake", action="store_true", help=fake_help)
    parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)


def build_embedder(args):
    """HashingEmbedder en mode --fake (aucun modèle téléchargé), sinon le vrai modèle d'embeddings."""
    from rag.embed import HashingEmbedder, SentenceTransformerEmbedder
    return HashingEmbedder() if args.fake else SentenceTransformerEmbedder(args.model)
