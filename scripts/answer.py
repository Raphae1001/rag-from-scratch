"""CLI Phase 3 : pose une question au système RAG complet (hybride -> reranking -> génération sourcée).

Usage : python scripts/answer.py "how do I upload a file to the server"
Prérequis : scripts/embed_corpus.py déjà exécuté, et $ANTHROPIC_API_KEY dans l'environnement (.env).
"""
import argparse

from _common import add_db_args, get_dsn, load_corpus

from rag import db
from rag.embed import SentenceTransformerEmbedder
from rag.generate import AnthropicClient
from rag.index import build_index
from rag.pipeline import answer
from rag.rerank import CrossEncoderReranker


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5, help="nombre de passages gardés après reranking")
    ap.add_argument("--top-n", type=int, default=20, dest="top_n", help="nombre de résultats hybrides passés au reranker")
    add_db_args(ap)
    args = ap.parse_args()

    docs = load_corpus()
    index = build_index([d["text"] for d in docs])
    conn = db.connect(get_dsn(args))
    n_docs, _ = db.counts(conn)
    assert n_docs == len(docs), "embeddings non chargés : lancer scripts/embed_corpus.py"

    result = answer(args.query, conn, index, SentenceTransformerEmbedder(), CrossEncoderReranker(),
                     AnthropicClient(), k=args.k, top_n=args.top_n)

    if not result["answerable"]:
        print("Je ne sais pas : le contexte récupéré ne permet pas de répondre à cette question.")
    else:
        print(result["answer"])
        print("\nSources :")
        for c in result["citations"]:
            print(f"  - {c}")

    lat = result["latency_ms"]
    print(f"\n[latence] retrieval {lat['retrieval']:.0f} ms | reranking {lat['reranking']:.0f} ms | "
          f"génération {lat['generation']:.0f} ms")


if __name__ == "__main__":
    main()
