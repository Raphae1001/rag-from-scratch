"""Charge les embeddings si absents ou incomplets — idempotent, ne fait rien s'ils sont déjà à jour.
Utilisé par le conteneur Docker de l'API au démarrage (voir docker-entrypoint.sh) pour que
`docker compose up` réponde à une requête de bout en bout sans étape manuelle.
"""
import subprocess
import sys

from _common import add_db_args, get_dsn, load_corpus

from rag import db


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    add_db_args(ap)
    args = ap.parse_args()

    docs = load_corpus()
    conn = db.connect(get_dsn(args))
    n_docs, _ = db.counts(conn)
    if n_docs == len(docs):
        print(f"embeddings déjà chargés ({n_docs} documents), rien à faire")
        return
    print(f"embeddings absents ou incomplets ({n_docs}/{len(docs)} documents) : chargement...")
    subprocess.run([sys.executable, "scripts/embed_corpus.py"], check=True)


if __name__ == "__main__":
    main()
