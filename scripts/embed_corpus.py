"""Découpe le corpus en passages, calcule les embeddings et les charge dans Postgres/pgvector.

Usage (avec le modèle réel, Docker démarré) :
    docker compose up -d
    python scripts/embed_corpus.py
Test hors ligne (embeddings factices, NE PAS utiliser pour évaluer) :
    python scripts/embed_corpus.py --fake --pgdata data/pgdata
"""
import time

from _common import add_db_args, add_embedder_args, build_embedder, get_dsn, load_corpus, make_parser

from rag import db
from rag.chunking import make_chunks


def main():
    ap = make_parser(__doc__)
    add_db_args(ap)
    add_embedder_args(ap, "embeddings factices (HashingEmbedder) : plomberie seulement")
    ap.add_argument("--chunk-size", type=int, default=120, help="mots par passage")
    ap.add_argument("--overlap", type=int, default=20, help="mots de chevauchement")
    args = ap.parse_args()

    docs = load_corpus()
    pieces = [(d["id"], i, p) for d in docs for i, p in enumerate(make_chunks(d, args.chunk_size, args.overlap))]
    print(f"{len(docs)} documents -> {len(pieces)} passages ({args.chunk_size} mots, chevauchement {args.overlap})")

    embedder = build_embedder(args)
    t0 = time.perf_counter()
    vectors = embedder.encode([p for _, _, p in pieces])
    print(f"embeddings : {vectors.shape} en {time.perf_counter() - t0:.1f} s ({'FACTICES' if args.fake else args.model})")

    conn = db.connect(get_dsn(args))
    db.init_schema(conn)
    db.reset_data(conn)
    db.load_corpus(conn, docs, [(d, i, p, v) for (d, i, p), v in zip(pieces, vectors)])
    n_docs, n_chunks = db.counts(conn)
    print(f"chargé dans Postgres : {n_docs} documents, {n_chunks} passages"
          f"{'  [EMBEDDINGS FACTICES : ne pas évaluer]' if args.fake else ''}")


if __name__ == "__main__":
    main()
